"""Unit tests for CodeEmbeddingService."""

from unittest.mock import Mock, patch

import pytest
import torch

from code_transformer.domain.embedding_service import CodeEmbeddingService
from code_transformer.domain.exceptions import EmbeddingError, ModelLoadError
from code_transformer.domain.models import CodeSnippet, Embedding

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_snippets() -> list[CodeSnippet]:
    """Create sample code snippets for testing."""
    return [
        CodeSnippet(
            id="test_001",
            file_path="test1.py",
            language="python",
            content="def hello(): return 'world'"
        ),
        CodeSnippet(
            id="test_002",
            file_path="test2.js",
            language="javascript",
            content="function greet() { return 'hi'; }"
        ),
        CodeSnippet(
            id="test_003",
            file_path="test3.go",
            language="go",
            content='func main() { fmt.Println("hello") }'
        )
    ]


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = Mock()
    # Mock tokenizer returns dict with input_ids and attention_mask
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
    }
    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock transformer model."""
    model = Mock()
    # Mock model output with last_hidden_state
    output = Mock()
    # Shape: [batch_size=1, seq_len=5, hidden_dim=8]
    output.last_hidden_state = torch.randn(1, 5, 8)
    model.return_value = output
    model.eval = Mock()
    model.to = Mock(return_value=model)
    return model


# ============================================================================
# CONSTRUCTOR TESTS
# ============================================================================

class TestCodeEmbeddingServiceInit:
    """Test CodeEmbeddingService initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        service = CodeEmbeddingService()
        assert service.model_name == "microsoft/codebert-base"
        assert service.max_length == 512
        assert isinstance(service.device, torch.device)

    def test_init_with_custom_model(self) -> None:
        """Test initialization with custom model name."""
        service = CodeEmbeddingService(model_name="custom/model")
        assert service.model_name == "custom/model"

    def test_init_with_custom_max_length(self) -> None:
        """Test initialization with custom max_length."""
        service = CodeEmbeddingService(max_length=1024)
        assert service.max_length == 1024

    def test_init_lazy_loads_model(self) -> None:
        """Test that model is not loaded during initialization."""
        service = CodeEmbeddingService()
        assert service._model is None
        assert service._tokenizer is None


# ============================================================================
# DEVICE RESOLUTION TESTS
# ============================================================================

class TestDeviceResolution:
    """Test device resolution logic."""

    def test_resolve_device_cpu(self) -> None:
        """Test explicit CPU device selection."""
        service = CodeEmbeddingService(device="cpu")
        assert service.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_resolve_device_cuda(self) -> None:
        """Test explicit CUDA device selection."""
        service = CodeEmbeddingService(device="cuda")
        assert service.device == torch.device("cuda")

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_resolve_device_mps(self) -> None:
        """Test explicit MPS device selection."""
        service = CodeEmbeddingService(device="mps")
        assert service.device == torch.device("mps")

    def test_resolve_device_auto_fallback_to_cpu(self) -> None:
        """Test auto device resolution falls back to CPU when no accelerator."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            service = CodeEmbeddingService(device="auto")
            assert service.device == torch.device("cpu")

    def test_resolve_device_auto_prefers_cuda(self) -> None:
        """Test auto device resolution prefers CUDA over MPS."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            service = CodeEmbeddingService(device="auto")
            assert service.device == torch.device("cuda")


# ============================================================================
# MEAN POOLING TESTS
# ============================================================================

class TestMeanPooling:
    """Test mean pooling implementation."""

    def test_mean_pooling_single_sequence(self) -> None:
        """Test mean pooling on single sequence."""
        service = CodeEmbeddingService()

        # Create simple tensors for testing
        # Shape: [batch_size=1, seq_len=3, hidden_dim=4]
        token_embeddings = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0],
             [2.0, 3.0, 4.0, 5.0],
             [3.0, 4.0, 5.0, 6.0]]
        ])
        # All tokens are valid (no padding)
        attention_mask = torch.tensor([[1, 1, 1]])

        result = service._mean_pooling(token_embeddings, attention_mask)

        # Expected: mean of the 3 token embeddings
        # [1+2+3, 2+3+4, 3+4+5, 4+5+6] / 3 = [2, 3, 4, 5]
        expected = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
        assert torch.allclose(result, expected)

    def test_mean_pooling_with_padding(self) -> None:
        """Test mean pooling with padded tokens (attention_mask = 0)."""
        service = CodeEmbeddingService()

        # Same embeddings but last token is padding
        token_embeddings = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0],
             [2.0, 3.0, 4.0, 5.0],
             [0.0, 0.0, 0.0, 0.0]]  # This is padding
        ])
        # Last token is padding (mask = 0)
        attention_mask = torch.tensor([[1, 1, 0]])

        result = service._mean_pooling(token_embeddings, attention_mask)

        # Expected: mean of only first 2 tokens
        # [1+2, 2+3, 3+4, 4+5] / 2 = [1.5, 2.5, 3.5, 4.5]
        expected = torch.tensor([[1.5, 2.5, 3.5, 4.5]])
        assert torch.allclose(result, expected)

    def test_mean_pooling_batch(self) -> None:
        """Test mean pooling with batch of sequences."""
        service = CodeEmbeddingService()

        # Batch of 2 sequences
        token_embeddings = torch.tensor([
            # First sequence
            [[1.0, 2.0], [2.0, 3.0]],
            # Second sequence
            [[3.0, 4.0], [4.0, 5.0]]
        ])
        attention_mask = torch.tensor([[1, 1], [1, 1]])

        result = service._mean_pooling(token_embeddings, attention_mask)

        # Expected means
        expected = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
        assert torch.allclose(result, expected)


# ============================================================================
# EMBED_BATCH TESTS
# ============================================================================

class TestEmbedBatch:
    """Test embed_batch method."""

    def test_embed_batch_empty_list(self) -> None:
        """Test that empty batch returns empty list."""
        service = CodeEmbeddingService()
        result = service.embed_batch([])
        assert result == []

    @patch("code_transformer.domain.embedding_service.AutoModel")
    @patch("code_transformer.domain.embedding_service.AutoTokenizer")
    def test_embed_batch_loads_model_once(
        self,
        mock_tokenizer_class,
        mock_model_class,
        sample_snippets,
        mock_tokenizer,
        mock_model
    ) -> None:
        """Test that model is loaded only once (lazy loading)."""
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        service = CodeEmbeddingService(device="cpu")

        # First call loads model
        service.embed_batch([sample_snippets[0]])
        assert mock_model_class.from_pretrained.call_count == 1

        # Second call reuses model
        service.embed_batch([sample_snippets[1]])
        assert mock_model_class.from_pretrained.call_count == 1  # Still 1!

    @patch("code_transformer.domain.embedding_service.AutoModel")
    @patch("code_transformer.domain.embedding_service.AutoTokenizer")
    def test_embed_batch_returns_embeddings(
        self,
        mock_tokenizer_class,
        mock_model_class,
        sample_snippets,
        mock_tokenizer,
        mock_model
    ) -> None:
        """Test that embed_batch returns Embedding objects."""
        # Setup mocks for batch of 3
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]] * 3),
            "attention_mask": torch.tensor([[1, 1, 1]] * 3)
        }
        output = Mock()
        output.last_hidden_state = torch.randn(3, 3, 8)
        mock_model.return_value = output

        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        service = CodeEmbeddingService(device="cpu")
        results = service.embed_batch(sample_snippets)

        # Verify results
        assert len(results) == 3
        assert all(isinstance(e, Embedding) for e in results)
        assert all(e.normalized is True for e in results)
        assert all(e.model_name == service.model_name for e in results)

    @patch("code_transformer.domain.embedding_service.AutoModel")
    @patch("code_transformer.domain.embedding_service.AutoTokenizer")
    def test_embed_batch_vectors_are_normalized(
        self,
        mock_tokenizer_class,
        mock_model_class,
        sample_snippets,
        mock_tokenizer,
        mock_model
    ) -> None:
        """Test that embedding vectors are L2-normalized."""
        # Setup mocks
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        output = Mock()
        output.last_hidden_state = torch.randn(1, 3, 8)
        mock_model.return_value = output

        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        service = CodeEmbeddingService(device="cpu")
        results = service.embed_batch([sample_snippets[0]])

        # Verify L2 norm is ~1.0 (normalized)
        vector = results[0].vector
        norm = torch.norm(vector, p=2)
        assert torch.isclose(norm, torch.tensor(1.0), atol=1e-6)

    @patch("code_transformer.domain.embedding_service.AutoModel")
    @patch("code_transformer.domain.embedding_service.AutoTokenizer")
    def test_embed_batch_preserves_snippet_data(
        self,
        mock_tokenizer_class,
        mock_model_class,
        sample_snippets,
        mock_tokenizer,
        mock_model
    ) -> None:
        """Test that original snippet data is preserved in embeddings."""
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        output = Mock()
        output.last_hidden_state = torch.randn(1, 3, 8)
        mock_model.return_value = output

        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        service = CodeEmbeddingService(device="cpu")
        results = service.embed_batch([sample_snippets[0]])

        # Verify snippet is preserved
        embedding = results[0]
        assert embedding.snippet.id == "test_001"
        assert embedding.snippet.file_path == "test1.py"
        assert embedding.snippet.language == "python"
        assert embedding.snippet.content == "def hello(): return 'world'"

    @patch("code_transformer.domain.embedding_service.AutoTokenizer")
    def test_embed_batch_raises_model_load_error_on_failure(
        self,
        mock_tokenizer_class,
        sample_snippets
    ) -> None:
        """Test that ModelLoadError is raised when model loading fails."""
        # Make tokenizer loading fail
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Model not found")

        service = CodeEmbeddingService(device="cpu")

        with pytest.raises(ModelLoadError, match="Failed to load model"):
            service.embed_batch([sample_snippets[0]])

    @patch("code_transformer.domain.embedding_service.AutoModel")
    @patch("code_transformer.domain.embedding_service.AutoTokenizer")
    def test_embed_batch_raises_embedding_error_on_inference_failure(
        self,
        mock_tokenizer_class,
        mock_model_class,
        sample_snippets,
        mock_tokenizer
    ) -> None:
        """Test that EmbeddingError is raised when inference fails."""
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = Mock()

        # Make model inference fail
        mock_tokenizer.side_effect = Exception("Tokenization failed")

        service = CodeEmbeddingService(device="cpu")

        with pytest.raises(EmbeddingError, match="Failed to generate embeddings"):
            service.embed_batch(sample_snippets)
