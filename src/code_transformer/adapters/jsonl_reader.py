"""JSONL reader adapter for reading code snippets from JSON Lines files."""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import aiofiles
import structlog

from ..domain.exceptions import InvalidInputError
from ..domain.models import CodeSnippet
from .ports import InputPort

logger = structlog.get_logger(__name__)

class JSONLReader(InputPort):
    """JSONL reader adapter."""

    def __init__(self, file_path: str | Path) -> None:
        if not file_path:
            msg = "file_path must be a non-empty string"
            raise ValueError(msg)

        self.file_path = Path(file_path)

    async def read(self) -> AsyncIterator[CodeSnippet]:
        """Read code snippets from a JSON Lines file.

        Yields:
            AsyncIterator[CodeSnippet]: An async iterator of CodeSnippet objects.
        """
        logger.debug("reading_snippets", path=str(self.file_path))
        try:
            async with aiofiles.open(self.file_path) as f:
                async for line in f:
                    data = json.loads(line)
                    value = CodeSnippet(**data)
                    yield value
        except json.JSONDecodeError as err:
            msg = f"The file {self.file_path} is not a valid JSONL file."
            raise InvalidInputError(msg) from err
        except FileNotFoundError as err:
            msg = f"The file {self.file_path} does not exist."
            raise InvalidInputError(msg) from err


    async def validate(self) -> bool:
        """Validate the JSONL file format by checking the first few lines.

        Returns:
            bool: True if validation is successful, False otherwise.
        """
        logger.debug("reading_snippets", path=str(self.file_path))
        try:
            if not self.file_path.exists():
                logger.error("file_not_found", path=str(self.file_path))
                return False

            async with aiofiles.open(self.file_path) as f:
                for _ in range(5):
                    line = await f.readline()
                    if not line or not line.strip():
                        break

                    data = json.loads(line)
                    value = CodeSnippet(**data)
                    if not value.validate():
                        return False
            return True
        except (json.JSONDecodeError, TypeError) as e:
            logger.error("validation_failed", path=str(self.file_path), error=str(e))
            return False


