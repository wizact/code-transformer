"""CLI commands for code-transformer using typer."""

import asyncio
from pathlib import Path
from typing import Annotated, Literal

import structlog
import typer

from ..adapters.embedding_writer import EmbeddingWriter
from ..adapters.jsonl_reader import JSONLReader
from ..application.config import EmbeddingConfig
from ..application.logging import setup_logging
from ..application.pipeline import EmbeddingPipeline
from ..domain.embedding_service import CodeEmbeddingService
from ..domain.exceptions import CodeTransformerError

logger = structlog.get_logger(__name__)

app = typer.Typer(
    name="code-transformer",
    help="Generate code embeddings using transformer models",
    add_completion=False,
)


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        typer.echo("code-transformer version 0.1.0")
        raise typer.Exit()


@app.command()
def embed(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to input JSONL file containing code snippets",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("in/input.jsonl"),
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to output JSONL file for embeddings",
        ),
    ] = Path("out/embeddings.jsonl"),
    model_name: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="HuggingFace model name for embeddings",
        ),
    ] = "microsoft/codebert-base",
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Batch size for processing (1-128)",
            min=1,
            max=128,
        ),
    ] = 16,
    max_length: Annotated[
        int,
        typer.Option(
            "--max-length",
            help="Maximum token length (64-2048)",
            min=64,
            max=2048,
        ),
    ] = 512,
    device: Annotated[
        Literal["cpu", "cuda", "mps", "auto"],
        typer.Option(
            "--device",
            help="Device to use for inference (cpu, cuda, mps, auto)",
        ),
    ] = "auto",
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose logging",
        ),
    ] = False,
    _version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ] = None,
) -> None:
    """
    Generate embeddings for code snippets.

    Reads code snippets from a JSONL file, generates embeddings using a
    transformer model, and writes the results to an output JSONL file.

    Example:
        code-transformer embed -i input.jsonl -o output.jsonl
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)

    try:
        # Create configuration
        config = EmbeddingConfig(
            input_path=input_path,
            output_path=output_path,
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )

        logger.info(
            "starting_embedding_pipeline",
            input=str(config.input_path),
            output=str(config.output_path),
            model=config.model_name,
            batch_size=config.batch_size,
        )

        # Run async pipeline
        asyncio.run(_run_pipeline(config))

        typer.echo(f"✓ Embeddings written to {output_path}")
        logger.info("pipeline_completed_successfully")

    except CodeTransformerError as e:
        logger.error("pipeline_failed", error=str(e))
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        logger.error("unexpected_error", error=str(e), exc_info=True)
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(code=1) from e


async def _run_pipeline(config: EmbeddingConfig) -> None:
    """
    Run the embedding pipeline with dependency injection.

    Args:
        config: Validated configuration
    """
    # Create output directory if it doesn't exist
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Wire up dependencies (dependency injection)
    input_adapter = JSONLReader(config.input_path)
    output_adapter = EmbeddingWriter(config.output_path)
    embedding_service = CodeEmbeddingService(
        model_name=config.model_name,
        max_length=config.max_length,
        device=config.device,
    )

    # Create and run pipeline
    pipeline = EmbeddingPipeline(
        input_adapter=input_adapter,
        embedding_service=embedding_service,
        output_adapter=output_adapter,
        batch_size=config.batch_size,
    )

    # Run with progress indication
    typer.echo("Processing embeddings...")
    await pipeline.process()


@app.command()
def validate(
    input_path: Annotated[
        Path,
        typer.Argument(
            help="Path to JSONL file to validate",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose logging",
        ),
    ] = False,
) -> None:
    """
    Validate a JSONL input file.

    Checks that the file is properly formatted and contains valid code snippets.

    Example:
        code-transformer validate input.jsonl
    """
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)

    try:
        logger.info("validating_input", path=str(input_path))

        reader = JSONLReader(input_path)
        is_valid = asyncio.run(reader.validate())

        if is_valid:
            typer.echo(f"✓ {input_path} is valid")
            logger.info("validation_successful", path=str(input_path))
        else:
            typer.echo(f"✗ {input_path} is invalid", err=True)
            logger.error("validation_failed", path=str(input_path))
            raise typer.Exit(code=1)

    except Exception as e:
        logger.error("validation_error", path=str(input_path), error=str(e))
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
