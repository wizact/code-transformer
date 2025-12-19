"""Custom exceptions for the code transformer domain."""


class CodeTransformerError(Exception):
    """Base exception for all code transformer errors."""



class ModelLoadError(CodeTransformerError):
    """Raised when model loading fails."""



class InvalidInputError(CodeTransformerError):
    """Raised when input validation fails."""



class EmbeddingError(CodeTransformerError):
    """Raised when embedding generation fails."""

