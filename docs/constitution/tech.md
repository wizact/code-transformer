# Technical Decisions & Standards

## Technology Stack

### Core Technologies
- **Python**: 3.11+ (modern async, type hints, performance improvements)
- **PyTorch**: 2.0+ (deep learning framework for embeddings)
- **Transformers**: 4.30+ (HuggingFace model library)

**Why Python 3.11+?**
- Native async/await support for I/O
- Improved type hints (Self, TypeVarTuple)
- Better performance (10-60% faster than 3.10)
- Industry standard for ML/AI work

**Why PyTorch over TensorFlow?**
- Better Python integration
- More intuitive API
- Stronger research community
- HuggingFace ecosystem built on PyTorch

### Development Tools
- **uv**: Fast dependency management (10-100x faster than pip)
- **ruff**: Linting and formatting (10-100x faster than black + flake8)
- **mypy**: Static type checking (catches bugs before runtime)
- **pytest**: Testing framework (async support, fixtures, plugins)
- **pytest-asyncio**: Async test support
- **structlog**: Structured logging with JSON output

**Why uv over pip/poetry?**
- Rust-based, extremely fast
- Drop-in replacement for pip
- Better resolver than pip
- Simpler than poetry for our use case

**Why ruff over black + flake8?**
- Single tool instead of two
- 10-100x faster (Rust-based)
- Covers both formatting and linting
- Auto-fix for many issues

### Architecture

Code Transformer follows **Hexagonal Architecture** (Ports & Adapters):

```
┌─────────────────────────────────────────┐
│          Application Layer              │
│         (EmbeddingPipeline)             │
│  • Orchestrates data flow                │
│  • Manages batching                      │
│  • Coordinates async/sync boundary       │
└────────────┬──────────────┬─────────────┘
             │              │
      ┌──────▼──────┐  ┌───▼──────────┐
      │   Domain    │  │   Adapters   │
      │  Services   │  │ (Ports impl) │
      │             │  │              │
      │ Embedding   │  │ JSONLReader  │
      │  Service    │  │ Writer       │
      └─────────────┘  └──────────────┘
```

**Core Components:**
- **Domain**: `CodeEmbeddingService` (pure business logic, no dependencies)
- **Ports**: `InputAdapter`, `OutputAdapter` (interface definitions)
- **Adapters**: `JSONLReader`, `EmbeddingWriter` (I/O implementations)
- **Application**: `EmbeddingPipeline` (orchestration)

**Key Patterns:**
- **Async I/O**: aiofiles for non-blocking file operations
- **Thread Pool**: asyncio.to_thread() for CPU-bound inference
- **Structured Logging**: structlog with JSON output

See [CLAUDE.md](../../CLAUDE.md) for comprehensive architectural principles.

## Project Structure

```
code-transformer/
├── docs/                    # Documentation
│   ├── constitution/        # Long-term principles
│   ├── features/            # Feature specifications
│   │   ├── TEMPLATES/       # Templates for new features
│   │   └── f001-*/          # Implemented features
├── src/code_transformer/
│   ├── domain/              # Core business logic
│   ├── adapters/            # I/O implementations
│   ├── application/         # Orchestration
│   └── cli/                 # Command-line interface
├── tests/
│   ├── unit/                # Isolated component tests
│   └── integration/         # End-to-end tests
├── CLAUDE.md                # Constitutional principles
├── README.md                # Quick start guide
└── pyproject.toml           # Project configuration
```

## Design Standards

### Code Style
- **Line Length**: 100 characters max
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Organized by stdlib, third-party, local
- **Naming**:
  - Classes: PascalCase
  - Functions/methods: snake_case
  - Constants: UPPER_SNAKE_CASE
  - Private: leading underscore (_private_method)

### Type Hints
- Required on all public functions
- Required on all class attributes
- Use specific types over Any
- Use typing.Protocol for structural subtyping
- Use dataclasses with frozen=True for immutability

Example:
```python
from typing import Protocol
from dataclasses import dataclass

@dataclass(frozen=True)
class CodeSnippet:
    id: str
    content: str
    language: str

class InputAdapter(Protocol):
    async def read(self) -> AsyncGenerator[CodeSnippet, None]:
        ...
```

### Docstrings
- **Style**: Google style
- **Required for**: All public classes, functions, methods
- **Not required for**: Private methods, obvious property getters
- **Include**: Args, Returns, Raises, Examples

Example:
```python
def embed_batch(self, snippets: list[CodeSnippet]) -> list[Embedding]:
    """Generate L2-normalized embeddings for a batch of code snippets.

    This method tokenizes code, generates embeddings using the transformer
    model, applies mean pooling, and L2-normalizes the results.

    Args:
        snippets: List of code snippets to embed

    Returns:
        List of embeddings with L2-normalized vectors

    Raises:
        EmbeddingError: If embedding generation fails
        ModelLoadError: If model loading fails
    """
```

### Testing Standards

#### Coverage
- **Target**: 95%+ total coverage
- **Required**: 100% coverage for domain logic
- **Allowed lower**: Integration test glue code

#### Test Organization
```
tests/
├── unit/              # Fast, isolated, mocked dependencies
│   ├── test_models.py
│   ├── test_embedding_service.py
│   └── test_adapters.py
└── integration/       # Slower, real components, minimal mocking
    └── test_pipeline.py
```

#### Test Naming
- **Pattern**: `test_<method>_<scenario>_<expected_outcome>`
- **Example**: `test_embed_batch_empty_list_returns_empty`

#### Async Tests
- Mark with `@pytest.mark.asyncio`
- Use async fixtures where needed
- Test both success and error paths

## Dependencies

### Production Dependencies
```toml
[project.dependencies]
torch = ">=2.0.0"
transformers = ">=4.30.0"
pydantic = ">=2.0.0"
structlog = ">=23.1.0"
typer = ">=0.9.0"
aiofiles = ">=23.1.0"
```

### Development Dependencies
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "mypy>=1.4.0",
    "ruff>=0.0.280",
    "uv>=0.1.0",
]
```

### Dependency Policy
- **Minimum versions**: Specify minimum version that works
- **No upper bounds**: Let users choose latest compatible version
- **Security updates**: Update dependencies when CVEs are found
- **Breaking changes**: Major version bumps require code updates

## CI/CD Standards

### Pre-commit Checks
Run locally before committing:
```bash
make format    # ruff format
make lint      # ruff check
make typecheck # mypy
make test      # pytest (fast tests only)
```

### CI Pipeline
Run on every PR and push to main:
1. **Linting**: ruff check --fix
2. **Formatting**: ruff format --check
3. **Type Checking**: mypy --strict
4. **Testing**: pytest with coverage
5. **Coverage Report**: Fail if < 95%

### Release Process
1. Update version in pyproject.toml
2. Update CHANGELOG.md
3. Create git tag (v0.1.0, v1.0.0, etc.)
4. Push tag to trigger release workflow
5. Publish to PyPI (automated)

## Performance Targets

### Throughput
- **CPU**: 100+ snippets/second
- **GPU**: 500+ snippets/second
- **Memory**: < 500MB regardless of input size

### Latency
- **Per snippet**: < 1 second (CPU)
- **Per batch**: < 0.5 seconds (GPU, batch size 32)

### Scalability
- **Input size**: Support files up to 1GB
- **Codebase size**: Support 10M+ lines of code
- **Concurrent requests**: N/A (batch processing in v1.0)

## Security Standards

### Input Validation
- Validate all input at system boundaries
- Use Pydantic for configuration validation
- Sanitize file paths (prevent path traversal)
- Limit input size (prevent DoS)

### Secrets Management
- **Never** commit secrets to git
- Use environment variables for sensitive config
- Document required environment variables
- Provide .env.example (not .env)

### Logging
- **Never** log sensitive data (tokens, API keys, PII)
- Use structured logging for easy filtering
- Log at appropriate levels (DEBUG, INFO, WARNING, ERROR)
- Include request IDs for traceability

### Dependencies
- Run `uv pip check` to check for conflicts
- Monitor security advisories
- Update dependencies quarterly (or when CVEs found)

## Code Review Standards

### Review Checklist
- [ ] Code follows style guide (ruff passes)
- [ ] Type hints are comprehensive (mypy passes)
- [ ] Tests are included and passing
- [ ] Coverage meets 95% threshold
- [ ] Documentation is updated
- [ ] No secrets in code
- [ ] Error handling is appropriate
- [ ] Performance is acceptable

### Review Guidelines
- Review within 24 hours
- Be constructive and specific
- Approve if minor comments only
- Request changes if blocking issues
- Use GitHub suggestions for quick fixes

## Decision Records

### Why Frozen Dataclasses?
**Decision**: Use frozen=True for all domain models

**Rationale**:
- Prevents accidental mutation
- Makes code easier to reason about
- Enables hashability for use in sets/dicts
- Thread-safe by default

**Alternatives**: Regular dataclasses (too error-prone), NamedTuples (less flexible)

### Why No ORM?
**Decision**: No SQLAlchemy or other ORM in v1.0

**Rationale**:
- JSONL I/O only in v1.0
- Adds significant complexity
- Not needed for current use case
- Can add in v2.0 if database output needed

**Alternatives**: SQLAlchemy (overkill), raw SQL (premature)

### Why Async I/O?
**Decision**: Use async/await for file operations

**Rationale**:
- Non-blocking I/O prevents bottlenecks
- Better throughput for large files
- Composable with other async code
- Modern Python standard

**Alternatives**: Sync I/O (blocks event loop), Threading (complex)

### Why Mean Pooling over [CLS] Token?
**Decision**: Use mean pooling with attention mask weighting

**Rationale**:
- More robust semantic representation
- Uses information from all tokens
- Empirically better results
- Standard in sentence embedding research

**Alternatives**: [CLS] token (less robust), Max pooling (loses information)

## Monitoring & Observability

### Metrics to Track (Future)
- Embeddings generated per second
- Average batch processing time
- Memory usage per batch
- Error rate by error type
- Input file sizes

### Logging Strategy
- **DEBUG**: Internal state for debugging
- **INFO**: Normal operations (files processed, batches completed)
- **WARNING**: Recoverable errors (invalid snippets skipped)
- **ERROR**: Unrecoverable errors (model loading failed)

### Structured Logging Format
```python
logger.info(
    "embeddings_generated",
    count=len(results),
    batch_size=len(snippets),
    model=self.model_name,
    duration_ms=elapsed * 1000,
)
```

## Backwards Compatibility

### Versioning
- Follow Semantic Versioning (SemVer)
- MAJOR: Breaking API changes
- MINOR: New features, backwards compatible
- PATCH: Bug fixes, backwards compatible

### Deprecation Policy
1. Mark feature as deprecated in docs
2. Add deprecation warning in code
3. Wait at least 1 minor version
4. Remove in next major version

Example:
```python
import warnings

def old_function():
    warnings.warn(
        "old_function is deprecated, use new_function instead",
        DeprecationWarning,
        stacklevel=2
    )
```

## Updates to This Document

This document should be updated when:
- Adopting new major technology
- Changing code standards
- Making architectural decisions
- Learning lessons that affect all development

Last updated: 2025-01-XX
