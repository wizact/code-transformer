"""
Embedding writer adapter for writing embeddings to a file.
"""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import aiofiles
import structlog

from ..domain.models import Embedding
from .ports import OutputPort

logger = structlog.get_logger(__name__)

class EmbeddingWriter(OutputPort):
    """Write embeddings to a JSONL file."""

    def __init__(self, file_path: str | Path) -> None:
        if not file_path:
            msg = "file_path must be a non-empty string"
            raise ValueError(msg)

        self.file_path = Path(file_path)
        logger.info("initialized_embedding_writer", path=str(self.file_path))

    async def write(self, embeddings: list[Embedding]) -> None:
        """
        Write embeddings to a file in JSONL format.

        Args:
            embeddings: List of embedding objects to write to the jsonl file
        """
        try:
            async with aiofiles.open(self.file_path, mode="w") as f:
                for embedding in embeddings:
                    value = embedding.to_dict()
                    await f.write(json.dumps(value) + "\n")
        except (OSError, PermissionError) as e:
            logger.error("file_permission_error", path=str(self.file_path), error=str(e))
            raise

    async def write_batch(self, embeddings: AsyncIterator[list[Embedding]]) -> None:
        """
        Write embeddings to a file in JSONL format asynchronously

        Args:
            embeddings: Async generator yielding embedding batches
        """
        try:
            async with aiofiles.open(self.file_path, mode="w") as f:
                async for batch in embeddings:
                    for embedding in batch:
                        values = embedding.to_dict()
                        await f.write(json.dumps(values) + "\n")
        except (OSError, PermissionError) as e:
            logger.error("file_permission_error", path=str(self.file_path), error=str(e))
            raise
