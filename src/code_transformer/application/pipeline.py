import asyncio
from collections.abc import AsyncIterator

import structlog

from ..adapters.ports import InputPort, OutputPort
from ..domain.embedding_service import CodeEmbeddingService
from ..domain.exceptions import EmbeddingError, InvalidInputError
from ..domain.models import CodeSnippet

logger = structlog.get_logger(__name__)

class EmbeddingPipeline:
    """
    A pipeline for orchestrating input reading, embedding generation, and output writing.

    This pipelines:
    1. Reads code snippets from an input source using an InputPort.
    2. Streams code snippets in batches.
    3. Generates embeddings for each batch using an EmbeddingService.
    4. Writes the generated embeddings to an output destination using an OutputPort.

    Args:
        input_port (InputPort): The input port for reading code snippets.
        output_port (OutputPort): The output port for writing embeddings.
        embedding_service (EmbeddingService): The service for generating embeddings.
        batch_size (int): The number of code snippets to process in each batch.
    """

    def __init__(
           self,
           input_adapter: InputPort,
           embedding_service: CodeEmbeddingService,
           output_adapter: OutputPort,
           batch_size: int = 16
    ) -> None:
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.embedding_service = embedding_service
        self.batch_size = batch_size
        logger.info(
          "initialized_pipeline",
          batch_size=batch_size,
        )

    async def process(self) -> None:
        """
        Main processing pipeline.

        Orchestrates the full flow:
        - Validate input
        - Stream and batch snippets
        - Generate embeddings
        - Write results

        Raises:
        InvalidInputError: If input validation fails
        EmbeddingError: If embedding generation fails
        """
        try:
            if not await self.input_adapter.validate():
                msg = "Input validation failed."
                raise InvalidInputError(msg)
            await self._process_batches()
        except EmbeddingError as e:
            logger.error("embedding_generation_failed", error=str(e))
            raise


    async def _process_batches(self) -> None:
        """
        Process snippets in batches and write embeddings.
        """
        batch_num = 0
        all_embeddings = []

        async for snippets in self._batch_snippets():
            batch_num += 1
            logger.info("processing_batch", batch_number=batch_num, batch_size=len(snippets))
            embeddings = await asyncio.to_thread(self.embedding_service.embed_batch, snippets)
            all_embeddings.extend(embeddings)

            logger.debug(
              "batch_completed",
              batch_num=batch_num,
              embeddings_count=len(embeddings),
            )

        # Write all embeddings at once
        await self.output_adapter.write(all_embeddings)
        logger.info("pipeline_completed", total_batches=batch_num)



    async def _batch_snippets(self) -> AsyncIterator[list[CodeSnippet]]:
        """
        Batch code snippets for efficient processing.

        It accumulates snippets into batches of size self.batch_size.

        Yields:
            Batches of code snippets
        """

        batch = []
        async for snippet in self.input_adapter.read():
            batch.append(snippet)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []

        if batch:
            yield batch
