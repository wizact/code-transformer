"""Port for embedding service input and output interfaces."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..domain.models import CodeSnippet, Embedding


class InputPort(ABC):
    """Input port for embedding service."""

    @abstractmethod
    async def read(self) -> AsyncIterator[CodeSnippet]:
        """
        Read a the code chunk and yield CodeSnippet objects.

        Yields:
            AsyncIterator[CodeSnippet]: An async iterator of CodeSnippet objects.
        """
        if False:
            yield  # Make this an async generator, not a coroutine

    @abstractmethod
    async def validate(self) -> bool:
       """Validate the input format by reading and validating the schema.
       This method does not have any side effects.

        Returns:
            bool: True if validation is successful, False otherwise.
        """


class OutputPort(ABC):
    """Output port for embedding service."""

    @abstractmethod
    async def write(self, embeddings: list[Embedding]) -> None:
        """
        Write the embedding to the output destination.

        Args:
            embeddings (list[Embedding]): A list of Embedding objects.
        """

    @abstractmethod
    async def write_batch(self, embeddings: AsyncIterator[list[Embedding]]) -> None:
        """
        Write embeddings in batches to the output destination.

        Args:
            embeddings (AsyncIterator[list[Embedding]]): An async iterator of list of
                Embedding objects.
        """
