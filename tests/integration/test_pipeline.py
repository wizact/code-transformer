"""Integration tests for the full embedding pipeline.

These tests verify that the complete system works end-to-end:
- JSONLReader reads code snippets
- CodeEmbeddingService generates embeddings
- EmbeddingWriter writes output
- Full pipeline orchestration
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from code_transformer.adapters.embedding_writer import EmbeddingWriter
from code_transformer.adapters.jsonl_reader import JSONLReader
from code_transformer.application.pipeline import EmbeddingPipeline
from code_transformer.domain.embedding_service import CodeEmbeddingService
from code_transformer.domain.exceptions import InvalidInputError

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def valid_input_file(tmp_path: Path) -> Path:
    """Create a valid input JSONL file with 5 code snippets."""
    file_path = tmp_path / "input.jsonl"

    snippets = [
        {
            "id": "snippet_001",
            "file_path": "src/utils.py",
            "language": "python",
            "content": "def add(a, b):\n    return a + b",
            "chunk_name": "add",
            "chunk_type": "function"
        },
        {
            "id": "snippet_002",
            "file_path": "src/math.py",
            "language": "python",
            "content": "def multiply(x, y):\n    return x * y",
            "chunk_name": "multiply",
            "chunk_type": "function"
        },
        {
            "id": "snippet_003",
            "file_path": "src/handler.go",
            "language": "go",
            "content": "func Handle() error {\n  return nil\n}",
            "chunk_name": "Handle",
            "chunk_type": "function"
        },
        {
            "id": "snippet_004",
            "file_path": "src/app.js",
            "language": "javascript",
            "content": "function greet(name) { return `Hello ${name}`; }",
            "chunk_name": "greet",
            "chunk_type": "function"
        },
        {
            "id": "snippet_005",
            "file_path": "src/Calculator.java",
            "language": "java",
            "content": "public int sum(int a, int b) { return a + b; }",
            "chunk_name": "sum",
            "chunk_type": "method"
        }
    ]

    with open(file_path, "w") as f:
        for snippet in snippets:
            f.write(json.dumps(snippet) + "\n")

    return file_path


@pytest.fixture
def invalid_input_file(tmp_path: Path) -> Path:
    """Create an invalid JSONL file (malformed JSON)."""
    file_path = tmp_path / "invalid_input.jsonl"
    file_path.write_text("not valid json\n{broken:")
    return file_path


@pytest.fixture
def missing_fields_file(tmp_path: Path) -> Path:
    """Create JSONL with missing required fields."""
    file_path = tmp_path / "missing_fields.jsonl"

    # Missing 'content' field
    snippet = {
        "id": "bad_001",
        "file_path": "test.py",
        "language": "python"
    }

    with open(file_path, "w") as f:
        f.write(json.dumps(snippet) + "\n")

    return file_path


@pytest.fixture
def empty_input_file(tmp_path: Path) -> Path:
    """Create an empty JSONL file."""
    file_path = tmp_path / "empty_input.jsonl"
    file_path.write_text("")
    return file_path


@pytest.fixture
def output_file_path(tmp_path: Path) -> Path:
    """Create output file path in temp directory."""
    return tmp_path / "output.jsonl"


@pytest.fixture
def mock_transformer_model():
    """Mock HuggingFace model that returns predictable embeddings."""
    model = Mock()

    # Mock the forward pass to return realistic tensor shapes
    def mock_forward(*args, **kwargs):
        # Determine batch size from input_ids
        batch_size = kwargs["input_ids"].shape[0] if "input_ids" in kwargs else args[0].shape[0]

        # Return mock output with last_hidden_state
        # Shape: [batch_size, seq_len=10, hidden_dim=768]
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(batch_size, 10, 768)
        return mock_output

    model.side_effect = mock_forward
    model.eval = Mock(return_value=model)
    model.to = Mock(return_value=model)

    return model


@pytest.fixture
def mock_tokenizer():
    """Mock HuggingFace tokenizer that returns realistic token tensors."""
    tokenizer = Mock()

    def mock_tokenize(texts, **_kwargs):
        batch_size = len(texts) if isinstance(texts, list) else 1
        # Return mock tokenizer output
        return {
            "input_ids": torch.randint(0, 1000, (batch_size, 10)),
            "attention_mask": torch.ones(batch_size, 10, dtype=torch.long)
        }

    tokenizer.side_effect = mock_tokenize
    return tokenizer


# ============================================================================
# FULL PIPELINE SUCCESS TESTS
# ============================================================================

class TestPipelineSuccess:
    """Test successful end-to-end pipeline execution."""

    @pytest.mark.asyncio
    @patch("code_transformer.domain.embedding_service.AutoModel")
    @patch("code_transformer.domain.embedding_service.AutoTokenizer")
    async def test_pipeline_end_to_end_success(
        self,
        mock_tokenizer_class,
        mock_model_class,
        valid_input_file,
        output_file_path,
        mock_tokenizer,
        mock_transformer_model
    ):
        """Test complete pipeline: read → embed → write."""
        # Setup mocks
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_transformer_model

        # Create real components
        input_adapter = JSONLReader(valid_input_file)
        output_adapter = EmbeddingWriter(output_file_path)
        embedding_service = CodeEmbeddingService(
            model_name="microsoft/codebert-base",
            device="cpu"
        )

        # Create pipeline with small batch size for testing batching
        pipeline = EmbeddingPipeline(
            input_adapter=input_adapter,
            embedding_service=embedding_service,
            output_adapter=output_adapter,
            batch_size=2  # 5 snippets → 3 batches (2, 2, 1)
        )

        # Run pipeline
        await pipeline.process()

        # Verify output file was created
        assert output_file_path.exists()

        # Read and verify output
        with open(output_file_path) as f:
            lines = f.readlines()

        # Should have 5 embeddings (one per input snippet)
        assert len(lines) == 5

        # Verify each line is valid JSON with expected fields
        for _i, line in enumerate(lines):
            data = json.loads(line)

            # Original snippet fields preserved
            assert "id" in data
            assert "file_path" in data
            assert "language" in data
            assert "content" in data

            # Embedding fields added
            assert "embedding" in data
            assert "model" in data
            assert "dim" in data
            assert "normalized" in data

            # Verify embedding is a list of floats
            assert isinstance(data["embedding"], list)
            assert len(data["embedding"]) == 768  # CodeBERT dimension
            assert all(isinstance(x, (int, float)) for x in data["embedding"])

            # Verify model metadata
            assert data["model"] == "microsoft/codebert-base"
            assert data["dim"] == 768
            assert data["normalized"] is True

    @pytest.mark.asyncio
    @patch("code_transformer.domain.embedding_service.AutoModel")
    @patch("code_transformer.domain.embedding_service.AutoTokenizer")
    async def test_pipeline_embeddings_are_normalized(
        self,
        mock_tokenizer_class,
        mock_model_class,
        valid_input_file,
        output_file_path,
        mock_tokenizer,
        mock_transformer_model
    ):
        """Verify that output embeddings are L2 normalized."""
        # Setup
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_transformer_model

        input_adapter = JSONLReader(valid_input_file)
        output_adapter = EmbeddingWriter(output_file_path)
        embedding_service = CodeEmbeddingService(device="cpu")

        pipeline = EmbeddingPipeline(
            input_adapter=input_adapter,
            embedding_service=embedding_service,
            output_adapter=output_adapter,
            batch_size=2
        )

        await pipeline.process()

        # Read output and check normalization
        with open(output_file_path) as f:
            for line in f:
                data = json.loads(line)
                vector = torch.tensor(data["embedding"])

                # L2 norm should be ~1.0 (within floating point tolerance)
                norm = torch.norm(vector, p=2)
                assert torch.isclose(norm, torch.tensor(1.0), atol=1e-4)

    @pytest.mark.asyncio
    @patch("code_transformer.domain.embedding_service.AutoModel")
    @patch("code_transformer.domain.embedding_service.AutoTokenizer")
    async def test_pipeline_batching_behavior(
        self,
        mock_tokenizer_class,
        mock_model_class,
        valid_input_file,
        output_file_path,
        mock_tokenizer,
        mock_transformer_model
    ):
        """Test that pipeline correctly processes snippets in batches."""
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_transformer_model

        input_adapter = JSONLReader(valid_input_file)
        output_adapter = EmbeddingWriter(output_file_path)
        embedding_service = CodeEmbeddingService(device="cpu")

        # Use batch_size=3 with 5 snippets → should create 2 batches (3, 2)
        pipeline = EmbeddingPipeline(
            input_adapter=input_adapter,
            embedding_service=embedding_service,
            output_adapter=output_adapter,
            batch_size=3
        )

        await pipeline.process()

        # Verify correct number of outputs
        with open(output_file_path) as f:
            lines = f.readlines()

        assert len(lines) == 5  # All snippets processed despite batching


# ============================================================================
# ERROR SCENARIO TESTS
# ============================================================================

class TestPipelineErrors:
    """Test pipeline error handling."""

    @pytest.mark.asyncio
    async def test_pipeline_invalid_input_raises_error(
        self,
        invalid_input_file,
        output_file_path
    ):
        """Test that pipeline raises InvalidInputError for invalid JSONL."""
        input_adapter = JSONLReader(invalid_input_file)
        output_adapter = EmbeddingWriter(output_file_path)
        embedding_service = CodeEmbeddingService(device="cpu")

        pipeline = EmbeddingPipeline(
            input_adapter=input_adapter,
            embedding_service=embedding_service,
            output_adapter=output_adapter,
            batch_size=2
        )

        # Should fail during validation
        with pytest.raises(InvalidInputError, match="Input validation failed"):
            await pipeline.process()

    @pytest.mark.asyncio
    async def test_pipeline_missing_required_fields_fails_validation(
        self,
        missing_fields_file,
        output_file_path
    ):
        """Test that snippets with missing fields fail validation."""
        input_adapter = JSONLReader(missing_fields_file)
        output_adapter = EmbeddingWriter(output_file_path)
        embedding_service = CodeEmbeddingService(device="cpu")

        pipeline = EmbeddingPipeline(
            input_adapter=input_adapter,
            embedding_service=embedding_service,
            output_adapter=output_adapter,
            batch_size=2
        )

        with pytest.raises(InvalidInputError):
            await pipeline.process()

    @pytest.mark.asyncio
    @patch("code_transformer.domain.embedding_service.AutoModel")
    @patch("code_transformer.domain.embedding_service.AutoTokenizer")
    async def test_pipeline_empty_input_creates_empty_output(
        self,
        mock_tokenizer_class,
        mock_model_class,
        empty_input_file,
        output_file_path,
        mock_tokenizer,
        mock_transformer_model
    ):
        """Test that empty input creates empty output without errors."""
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_transformer_model

        input_adapter = JSONLReader(empty_input_file)
        output_adapter = EmbeddingWriter(output_file_path)
        embedding_service = CodeEmbeddingService(device="cpu")

        pipeline = EmbeddingPipeline(
            input_adapter=input_adapter,
            embedding_service=embedding_service,
            output_adapter=output_adapter,
            batch_size=2
        )

        # Should complete without error
        await pipeline.process()

        # Empty input means write([]) is called, creating empty file
        assert output_file_path.exists()
        assert output_file_path.stat().st_size == 0


# ============================================================================
# METADATA PRESERVATION TESTS
# ============================================================================

class TestMetadataPreservation:
    """Test that input metadata is preserved in output."""

    @pytest.mark.asyncio
    @patch("code_transformer.domain.embedding_service.AutoModel")
    @patch("code_transformer.domain.embedding_service.AutoTokenizer")
    async def test_pipeline_preserves_optional_fields(
        self,
        mock_tokenizer_class,
        mock_model_class,
        tmp_path,
        output_file_path,
        mock_tokenizer,
        mock_transformer_model
    ):
        """Test that optional fields from input are preserved in output."""
        # Create input with ALL optional fields
        input_file = tmp_path / "rich_input.jsonl"
        snippet = {
            "id": "rich_001",
            "file_path": "src/handler.go",
            "language": "go",
            "content": "func Handle() {}",
            "version": "v1.2.3",
            "git_hash": "abc123def456",
            "chunk_name": "Handle",
            "chunk_type": "function",
            "metadata": {
                "start_line": 10,
                "end_line": 12,
                "receiver": None,
                "type_kind": "function"
            },
            "created_at": "2025-01-15T10:30:00Z"
        }

        with open(input_file, "w") as f:
            f.write(json.dumps(snippet) + "\n")

        # Setup and run pipeline
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_transformer_model

        input_adapter = JSONLReader(input_file)
        output_adapter = EmbeddingWriter(output_file_path)
        embedding_service = CodeEmbeddingService(device="cpu")

        pipeline = EmbeddingPipeline(
            input_adapter=input_adapter,
            embedding_service=embedding_service,
            output_adapter=output_adapter,
            batch_size=1
        )

        await pipeline.process()

        # Verify all fields preserved
        with open(output_file_path) as f:
            data = json.loads(f.readline())

        # All original fields should be present
        assert data["id"] == "rich_001"
        assert data["version"] == "v1.2.3"
        assert data["git_hash"] == "abc123def456"
        assert data["chunk_name"] == "Handle"
        assert data["chunk_type"] == "function"
        assert data["metadata"]["start_line"] == 10
        assert data["created_at"] == "2025-01-15T10:30:00Z"

        # Plus embedding fields
        assert "embedding" in data
        assert "model" in data


# ============================================================================
# BATCH SIZE TESTS
# ============================================================================

class TestBatchSizes:
    """Test pipeline with different batch sizes."""

    @pytest.mark.asyncio
    @patch("code_transformer.domain.embedding_service.AutoModel")
    @patch("code_transformer.domain.embedding_service.AutoTokenizer")
    async def test_pipeline_large_batch_size(
        self,
        mock_tokenizer_class,
        mock_model_class,
        valid_input_file,
        output_file_path,
        mock_tokenizer,
        mock_transformer_model
    ):
        """Test pipeline with batch_size larger than input (single batch)."""
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_transformer_model

        input_adapter = JSONLReader(valid_input_file)
        output_adapter = EmbeddingWriter(output_file_path)
        embedding_service = CodeEmbeddingService(device="cpu")

        # batch_size=100 with 5 snippets → 1 batch of 5
        pipeline = EmbeddingPipeline(
            input_adapter=input_adapter,
            embedding_service=embedding_service,
            output_adapter=output_adapter,
            batch_size=100
        )

        await pipeline.process()

        # Verify all processed
        with open(output_file_path) as f:
            lines = f.readlines()
        assert len(lines) == 5

    @pytest.mark.asyncio
    @patch("code_transformer.domain.embedding_service.AutoModel")
    @patch("code_transformer.domain.embedding_service.AutoTokenizer")
    async def test_pipeline_batch_size_one(
        self,
        mock_tokenizer_class,
        mock_model_class,
        valid_input_file,
        output_file_path,
        mock_tokenizer,
        mock_transformer_model
    ):
        """Test pipeline processes correctly with batch_size=1."""
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_transformer_model

        input_adapter = JSONLReader(valid_input_file)
        output_adapter = EmbeddingWriter(output_file_path)
        embedding_service = CodeEmbeddingService(device="cpu")

        # batch_size=1: each snippet processed individually
        pipeline = EmbeddingPipeline(
            input_adapter=input_adapter,
            embedding_service=embedding_service,
            output_adapter=output_adapter,
            batch_size=1
        )

        await pipeline.process()

        with open(output_file_path) as f:
            lines = f.readlines()
        assert len(lines) == 5
