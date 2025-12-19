"""Configuration models using pydantic for validation."""
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation.

    Attributes:
        input_path: Path to the input JSONL file
        output_path: Path to the output JSONL file
        model_name: Name of the embedding model to use
        batch_size: Number of code snippets to process in a batch
        max_length: Maximum length of code snippets to consider
        device: Device to run the embedding model on (cpu, cuda, mps, auto)
    """

    # File paths
    input_path: Path = Field(default=Path("in/input.jsonl"))
    output_path: Path = Field(default=Path("out/embeddings.jsonl"))

    # Model configuration
    model_name: str = Field(default="microsoft/codebert-base")
    batch_size: int = Field(default=16, ge=1, le=128)
    max_length: int = Field(default=512, ge=64, le=2048)
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(default="auto")

    class Config:
        frozen = True # Make instances immutable

    @field_validator("input_path")
    @classmethod
    def validate_input_exists(cls, v: Path) -> Path:
        """
        Validate that input path exists.

        Args:
            v: the input_path value to validate

        Returns:
            Validated path

        Raises:
            ValueError: If the input file does not exist
        """
        if not v.exists():
            msg = f"input path {v} does not exist"
            raise ValueError(msg)
        return v

    @field_validator("output_path")
    @classmethod
    def validate_output_parent_exists(cls, v: Path) -> Path:
        """
        Validate the output path parent exists.

        Args:
            v: the output path to validate

        Returns:
            Validated path

        Raises:
            ValueError: If the parent path does not exist
        """
        if not v.parent.exists():
            msg = f"output path parent {v.parent} does not exist"
            raise ValueError(msg)

        return v

