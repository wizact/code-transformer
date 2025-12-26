# Code Transformer

Transform source code into semantic embeddings for AI-powered code understanding.

## Overview

Code Transformer processes code snippets from any programming language and generates normalized vector embeddings using state-of-the-art transformer models. These embeddings enable semantic code search, similarity detection, and AI-assisted development tools.

**Key Features:**
- Streaming architecture: Process codebases of any size with constant memory usage (< 500MB)
- Transformer-based: Uses CodeBERT and other HuggingFace models
- Mean pooling + L2 normalization: Ready for cosine similarity without additional processing
- Async I/O: Non-blocking file operations with async/await
- Hexagonal architecture: Swappable input/output adapters for extensibility
- Multi-language: Python, Go, JavaScript, Java, and more
- ✅ 95%+ test coverage: Comprehensive unit and integration tests

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/wizact/code-transformer.git
cd code-transformer

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Basic Usage

```python
import asyncio
from code_transformer.adapters.jsonl_reader import JSONLReader
from code_transformer.adapters.embedding_writer import EmbeddingWriter
from code_transformer.domain.embedding_service import CodeEmbeddingService
from code_transformer.application.pipeline import EmbeddingPipeline

async def main():
    # Create components
    input_adapter = JSONLReader("input.jsonl")
    output_adapter = EmbeddingWriter("output.jsonl")
    embedding_service = CodeEmbeddingService(
        model_name="microsoft/codebert-base",
        device="auto"
    )

    # Create and run pipeline
    pipeline = EmbeddingPipeline(
        input_adapter=input_adapter,
        embedding_service=embedding_service,
        output_adapter=output_adapter,
        batch_size=16
    )

    await pipeline.process()

asyncio.run(main())
```

### Input Format (JSONL)

Each line is a JSON object with code snippet information:

```json
{"id": "001", "file_path": "src/utils.py", "language": "python", "content": "def add(a, b):\n    return a + b"}
{"id": "002", "file_path": "src/math.go", "language": "go", "content": "func Multiply(x, y int) int { return x * y }"}
```

**Required fields:**
- `id`: Unique identifier
- `file_path`: Source file path
- `language`: Programming language
- `content`: Code content

**Optional fields:** `version`, `git_hash`, `chunk_name`, `chunk_type`, `metadata`

### Output Format (JSONL)

Output preserves all input fields and adds embedding information:

```json
{
  "id": "001",
  "file_path": "src/utils.py",
  "language": "python",
  "content": "def add(a, b):\n    return a + b",
  "embedding": [0.023, -0.041, ..., 0.067],
  "model": "microsoft/codebert-base",
  "dim": 768,
  "normalized": true
}
```

## Documentation

### Constitutional Documents
Long-term principles and decisions that rarely change:

- **[CLAUDE.md](CLAUDE.md)**: Comprehensive architectural principles (20 core principles)
- **[Product Vision](docs/constitution/product.md)**: Mission, scope, success metrics, roadmap
- **[Technical Standards](docs/constitution/tech.md)**: Technology stack, code style, CI/CD
- **[Principles Summary](docs/constitution/principles.md)**: Quick reference to CLAUDE.md

### Feature Specifications
Detailed specifications for implemented features:

- **[Features Overview](docs/features/README.md)**: Specification pattern and template usage
- **[F001: Streaming Pipeline](docs/features/f001-streaming-pipeline/)**: Core embedding pipeline
  - [Requirements](docs/features/f001-streaming-pipeline/requirements.md)
  - [Design](docs/features/f001-streaming-pipeline/design.md)
  - [Tasks](docs/features/f001-streaming-pipeline/tasks.md)

### Templates for New Features
Use these templates when creating new feature specifications:

- **[Requirements Template](docs/features/TEMPLATES/requirements.md)**: EARS notation, user stories, acceptance criteria
- **[Design Template](docs/features/TEMPLATES/design.md)**: Architecture, components, data flow, algorithms
- **[Tasks Template](docs/features/TEMPLATES/tasks.md)**: Implementation breakdown, progress tracking, lessons learned

## Development

### Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dev dependencies
uv sync --dev
```

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# With coverage
pytest --cov=code_transformer --cov-report=html
```

### Code Quality

```bash
# Format code
ruff format

# Lint code
ruff check --fix

# Type check
mypy src/

# Run all checks
ruff format && ruff check --fix && mypy src/ && pytest
```
