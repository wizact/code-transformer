"""Domain models for code snippets and embeddings."""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class CodeSnippet:
    """Immutable code snippet with comprehensive metadata.

    This model represents a code chunk with all necessary metadata for
    embedding generation and downstream processing.

    Args:
        id: Unique identifier for the chunk
        file_path: Path to source file
        language: Programming language
        content: Code snippet content
        version: Semantic version tag (optional)
        git_hash: Git commit hash (optional)
        chunk_name: Function/class/method name (optional)
        chunk_type: Type of chunk (function, class, method, etc.) (optional)
        metadata: Language-specific and positional metadata (optional)
        created_at: ISO 8601 timestamp (optional)
    """

    id: str
    file_path: str
    language: str
    content: str
    version: str | None = None
    git_hash: str | None = None
    chunk_name: str | None = None
    chunk_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None

    def validate(self) -> bool:
        """Validate that required fields are present and non-empty.

        Returns:
            True if all required fields are valid, False otherwise
        """
        return bool(
            self.id
            and self.file_path
            and self.language
            and self.content
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with all non-None fields
        """
        result: dict[str, Any] = {
            "id": self.id,
            "file_path": self.file_path,
            "language": self.language,
            "content": self.content,
        }

        # Add optional fields if present
        if self.version is not None:
            result["version"] = self.version
        if self.git_hash is not None:
            result["git_hash"] = self.git_hash
        if self.chunk_name is not None:
            result["chunk_name"] = self.chunk_name
        if self.chunk_type is not None:
            result["chunk_type"] = self.chunk_type
        if self.metadata:
            result["metadata"] = self.metadata
        if self.created_at is not None:
            result["created_at"] = self.created_at

        return result


@dataclass(frozen=True)
class Embedding:
    """Code embedding with L2-normalized vector and metadata.

    This model represents the output of the embedding process, containing
    the original snippet, the embedding vector, and generation metadata.

    Args:
        snippet: Original code snippet
        vector: L2-normalized embedding vector (PyTorch tensor)
        model_name: Model identifier used for embedding
        normalized: Whether embeddings are L2-normalized (always True)
    """

    snippet: CodeSnippet
    vector: torch.Tensor
    model_name: str
    normalized: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSONL serialization.

        Returns:
            Dictionary with all snippet fields plus embedding metadata
        """
        result = self.snippet.to_dict()

        # Add embedding-specific fields
        result["embedding"] = self.vector.cpu().tolist()
        result["model"] = self.model_name
        result["dim"] = self.vector.shape[0]
        result["normalized"] = self.normalized

        return result
