"""Core embedding service for generating code embeddings."""

import structlog
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .exceptions import EmbeddingError, ModelLoadError
from .models import CodeSnippet, Embedding

logger = structlog.get_logger(__name__)


class CodeEmbeddingService:
    """Service for generating L2-normalized code embeddings using transformers.

    This service handles model loading, tokenization, and embedding generation
    with mean pooling and L2 normalization for cosine similarity.

    The model is lazy-loaded on first use to improve startup time and allow
    validation without loading heavy models.

    Args:
        model_name: HuggingFace model identifier (e.g., 'microsoft/codebert-base')
        max_length: Maximum token length for truncation
        device: Device for inference ('cpu', 'cuda', 'mps', or 'auto')

    Example:
        >>> service = CodeEmbeddingService('microsoft/codebert-base')
        >>> snippets = [CodeSnippet(
        ...     id='1', file_path='test.py', language='python',
        ...     content='print("hello")'
        ... )]
        >>> embeddings = service.embed_batch(snippets)
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        max_length: int = 512,
        device: str = "auto",
    ) -> None:
        """Initialize the embedding service.

        Args:
            model_name: HuggingFace model identifier
            max_length: Maximum token length (64-2048)
            device: Device selection ('cpu', 'cuda', 'mps', or 'auto')
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = self._resolve_device(device)

        # Lazy loading - model and tokenizer are None until first use
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

        logger.info(
            "initialized_embedding_service",
            model=model_name,
            max_length=max_length,
            device=str(self.device),
        )

    def _resolve_device(self, device: str) -> torch.device:
        """Resolve device string to torch.device.

        Python Note: This handles platform-specific device selection.
        macOS has 'mps' (Metal Performance Shaders), Linux/Windows have 'cuda'.

        Args:
            device: Device string ('cpu', 'cuda', 'mps', or 'auto')

        Returns:
            Resolved torch.device
        """
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _load_model(self) -> None:
        """Lazy load model and tokenizer on first use.

        Raises:
            ModelLoadError: If model or tokenizer loading fails
        """
        if self._model is not None and self._tokenizer is not None:
            return  # Already loaded

        try:
            logger.info("loading_model", model=self.model_name)

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()  # Set to evaluation mode

            logger.info("model_loaded_successfully", model=self.model_name)

        except Exception as e:
            logger.error(
                "model_load_failed",
                model=self.model_name,
                error=str(e),
                exc_info=True,
            )
            msg = f"Failed to load model '{self.model_name}': {e}"
            raise ModelLoadError(
                msg
            ) from e

    def _mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply mean pooling over token embeddings with attention mask weighting.

        Python/PyTorch Note: This uses broadcasting and tensor operations.
        - unsqueeze(-1): Adds dimension for broadcasting
        - expand(): Repeats the mask to match embedding dimensions
        - torch.sum(): Aggregates over token dimension (dim=1)
        - torch.clamp(): Prevents division by zero

        Args:
            token_embeddings: Token-level embeddings [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Mean-pooled embeddings [batch_size, hidden_dim]
        """
        # Expand attention mask to match embedding dimensions
        # Shape: [batch_size, seq_len] -> [batch_size, seq_len, hidden_dim]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        # Sum embeddings weighted by attention mask
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        # Sum attention mask (number of non-padding tokens per sample)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Compute mean: sum / count
        return sum_embeddings / sum_mask

    def embed_batch(self, snippets: list[CodeSnippet]) -> list[Embedding]:
        """Generate L2-normalized embeddings for a batch of code snippets.

        This method:
        1. Tokenizes code with padding and truncation
        2. Generates token embeddings using the transformer model
        3. Applies mean pooling over tokens (weighted by attention mask)
        4. Normalizes embeddings with L2 norm for cosine similarity

        Python/Async Note: This is a synchronous (blocking) method because
        transformer inference is CPU/GPU-bound. Call via asyncio.to_thread()
        from async code to avoid blocking the event loop.

        Args:
            snippets: List of code snippets to embed

        Returns:
            List of embeddings with L2-normalized vectors

        Raises:
            EmbeddingError: If embedding generation fails
            ModelLoadError: If model loading fails
        """
        if not snippets:
            return []

        # Lazy load model on first use
        self._load_model()

        assert self._model is not None  # Type checker hint
        assert self._tokenizer is not None  # Type checker hint

        try:
            logger.debug(
                "generating_embeddings",
                batch_size=len(snippets),
                model=self.model_name,
            )

            # Tokenize code snippets with padding and truncation
            # Python Note: List comprehension extracts content from each snippet
            encoded = self._tokenizer(
                [s.content for s in snippets],
                padding=True,  # Pad to max length in batch
                truncation=True,  # Truncate to max_length
                max_length=self.max_length,
                return_tensors="pt",  # Return PyTorch tensors
            )

            # Move tensors to the same device as model
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Generate embeddings (no gradient computation needed)
            # Python Note: torch.no_grad() disables autograd for efficiency
            with torch.no_grad():
                outputs = self._model(**encoded)

                # Extract token embeddings from last hidden state
                # Shape: [batch_size, sequence_length, hidden_dim]
                token_embeddings = outputs.last_hidden_state

                # Apply mean pooling (average over tokens)
                embeddings = self._mean_pooling(
                    token_embeddings,
                    encoded["attention_mask"],
                )

                # L2 normalization for cosine similarity
                # Python/PyTorch Note: F.normalize with p=2 normalizes to unit length
                embeddings = torch.nn.functional.normalize(
                    embeddings,
                    p=2,
                    dim=1,
                )

            # Create Embedding objects
            # Python Note: zip() pairs each snippet with its embedding
            results = [
                Embedding(
                    snippet=snippet,
                    vector=embedding.cpu(),  # Move to CPU for serialization
                    model_name=self.model_name,
                    normalized=True,
                )
                for snippet, embedding in zip(snippets, embeddings, strict=False)
            ]

            logger.debug(
                "embeddings_generated",
                count=len(results),
                dim=embeddings.shape[1],
            )

            return results

        except Exception as e:
            logger.error(
                "embedding_generation_failed",
                batch_size=len(snippets),
                error=str(e),
                exc_info=True,
            )
            msg = f"Failed to generate embeddings: {e}"
            raise EmbeddingError(
                msg
            ) from e
