"""Unit tests for EmbeddingWriter adapter."""

import json
from pathlib import Path

import pytest
import torch

from code_transformer.adapters.embedding_writer import EmbeddingWriter
from code_transformer.domain.models import CodeSnippet, Embedding

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_snippet() -> CodeSnippet:
    """Create a sample code snippet for testing."""
    return CodeSnippet(
        id="test_001",
        file_path="test.py",
        language="python",
        content="def hello(): return 'hi'",
        version="v1.0.0",
        metadata={"start_line": 1, "end_line": 2}
    )


@pytest.fixture
def sample_embedding(sample_snippet: CodeSnippet) -> Embedding:
    """Create a sample embedding for testing."""
    # Create a small embedding vector (768-dim would be too large for testing)
    vector = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    return Embedding(
        snippet=sample_snippet,
        vector=vector,
        model_name="test-model",
        normalized=True
    )


@pytest.fixture
def multiple_embeddings() -> list[Embedding]:
    """Create multiple embeddings for batch testing."""
    embeddings = []
    for i in range(3):
        snippet = CodeSnippet(
            id=f"test_{i:03d}",
            file_path=f"test_{i}.py",
            language="python",
            content=f"def func_{i}(): pass"
        )
        vector = torch.tensor([float(i), float(i+1), float(i+2)])
        embeddings.append(
            Embedding(
                snippet=snippet,
                vector=vector,
                model_name="test-model",
                normalized=True
            )
        )
    return embeddings


# ============================================================================
# CONSTRUCTOR TESTS
# ============================================================================

class TestEmbeddingWriterInit:
    """Test EmbeddingWriter initialization."""

    def test_init_with_string_path(self, tmp_path: Path) -> None:
        """Test initialization with string file path."""
        file_path = tmp_path / "output.jsonl"
        writer = EmbeddingWriter(str(file_path))
        assert writer.file_path == file_path
        assert isinstance(writer.file_path, Path)

    def test_init_with_path_object(self, tmp_path: Path) -> None:
        """Test initialization with Path object."""
        file_path = tmp_path / "output.jsonl"
        writer = EmbeddingWriter(file_path)
        assert writer.file_path == file_path

    def test_init_with_empty_string_raises_error(self) -> None:
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            EmbeddingWriter("")

    def test_init_with_none_raises_error(self) -> None:
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="file_path must be a non-empty string"):
            EmbeddingWriter(None)


# ============================================================================
# WRITE METHOD TESTS
# ============================================================================

class TestEmbeddingWriterWrite:
    """Test EmbeddingWriter.write() method."""

    @pytest.mark.asyncio
    async def test_write_single_embedding(
        self, tmp_path: Path, sample_embedding: Embedding
    ) -> None:
        """Test writing a single embedding to file."""
        file_path = tmp_path / "single.jsonl"
        writer = EmbeddingWriter(file_path)

        await writer.write([sample_embedding])

        # Verify file exists
        assert file_path.exists()

        # Read and verify content
        with open(file_path) as f:
            lines = f.readlines()
            assert len(lines) == 1

            data = json.loads(lines[0])
            assert data["id"] == "test_001"
            assert data["file_path"] == "test.py"
            assert data["language"] == "python"
            assert data["model"] == "test-model"
            assert data["normalized"] is True
            assert "embedding" in data
            assert isinstance(data["embedding"], list)

    @pytest.mark.asyncio
    async def test_write_multiple_embeddings(
        self, tmp_path: Path, multiple_embeddings: list[Embedding]
    ) -> None:
        """Test writing multiple embeddings to file."""
        file_path = tmp_path / "multiple.jsonl"
        writer = EmbeddingWriter(file_path)

        await writer.write(multiple_embeddings)

        # Verify file content
        with open(file_path) as f:
            lines = f.readlines()
            assert len(lines) == 3

            for i, line in enumerate(lines):
                data = json.loads(line)
                assert data["id"] == f"test_{i:03d}"
                assert data["file_path"] == f"test_{i}.py"
                assert data["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_write_empty_list(self, tmp_path: Path) -> None:
        """Test writing empty list creates empty file."""
        file_path = tmp_path / "empty.jsonl"
        writer = EmbeddingWriter(file_path)

        await writer.write([])

        # File should exist but be empty
        assert file_path.exists()
        assert file_path.stat().st_size == 0

    @pytest.mark.asyncio
    async def test_write_preserves_metadata(
        self, tmp_path: Path, sample_embedding: Embedding
    ) -> None:
        """Test that write preserves all snippet metadata."""
        file_path = tmp_path / "metadata.jsonl"
        writer = EmbeddingWriter(file_path)

        await writer.write([sample_embedding])

        # Verify metadata is preserved
        with open(file_path) as f:
            data = json.loads(f.readline())
            assert data["version"] == "v1.0.0"
            assert data["metadata"]["start_line"] == 1
            assert data["metadata"]["end_line"] == 2

    @pytest.mark.asyncio
    async def test_write_overwrites_existing_file(
        self, tmp_path: Path, sample_embedding: Embedding
    ) -> None:
        """Test that write overwrites existing file."""
        file_path = tmp_path / "overwrite.jsonl"

        # Write initial content
        file_path.write_text("old content\n")

        writer = EmbeddingWriter(file_path)
        await writer.write([sample_embedding])

        # Verify old content is gone
        with open(file_path) as f:
            lines = f.readlines()
            assert len(lines) == 1
            assert "old content" not in lines[0]


# ============================================================================
# WRITE_BATCH METHOD TESTS
# ============================================================================

class TestEmbeddingWriterWriteBatch:
    """Test EmbeddingWriter.write_batch() async generator method."""

    @pytest.mark.asyncio
    async def test_write_batch_from_async_generator(
        self, tmp_path: Path, multiple_embeddings: list[Embedding]
    ) -> None:
        """Test writing batches from async generator."""
        file_path = tmp_path / "batch.jsonl"
        writer = EmbeddingWriter(file_path)

        # Create async generator that yields batches
        async def embedding_generator():
            # Yield embeddings in batches of 2
            yield multiple_embeddings[0:2]
            yield multiple_embeddings[2:3]

        await writer.write_batch(embedding_generator())

        # Verify all embeddings were written
        with open(file_path) as f:
            lines = f.readlines()
            assert len(lines) == 3

            for i, line in enumerate(lines):
                data = json.loads(line)
                assert data["id"] == f"test_{i:03d}"

    @pytest.mark.asyncio
    async def test_write_batch_empty_generator(self, tmp_path: Path) -> None:
        """Test writing from empty async generator."""
        file_path = tmp_path / "empty_batch.jsonl"
        writer = EmbeddingWriter(file_path)

        # Empty generator
        async def empty_generator():
            if False:
                yield []

        await writer.write_batch(empty_generator())

        # File should exist but be empty
        assert file_path.exists()
        assert file_path.stat().st_size == 0

    @pytest.mark.asyncio
    async def test_write_batch_single_batch(
        self, tmp_path: Path, multiple_embeddings: list[Embedding]
    ) -> None:
        """Test writing single batch from async generator."""
        file_path = tmp_path / "single_batch.jsonl"
        writer = EmbeddingWriter(file_path)

        async def single_batch_generator():
            yield multiple_embeddings

        await writer.write_batch(single_batch_generator())

        # Verify content
        with open(file_path) as f:
            lines = f.readlines()
            assert len(lines) == 3
