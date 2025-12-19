# Code Transformer - Constitutional Principles

This document defines the core architectural and development principles for the Code Transformer project. These principles should guide all implementation decisions and future enhancements.

---

## Architecture Principles

### 1. Hexagonal Architecture (Ports & Adapters)
**Principle**: Separate business logic from external dependencies using ports and adapters pattern.

**Implementation**:
- **Domain Layer**: Pure business logic with no external dependencies
- **Ports**: Interface definitions (ABC) that define contracts between layers
- **Adapters**: Concrete implementations of ports for specific technologies
- **Application Layer**: Orchestration only, delegates to domain and adapters

**Benefits**:
- Storage-agnostic design enables swapping I/O implementations
- Testable core domain without external dependencies
- Clear separation of concerns

**Example**: Input port defines `read()` contract; JSONL adapter implements it; future database adapter can replace it without touching domain logic.

---

### 2. Dependency Injection
**Principle**: Pass dependencies through constructors, never use global state.

**Implementation**:
- All components receive dependencies via `__init__()`
- No singletons or global mutable state
- Configuration passed as immutable objects

**Benefits**:
- Testability through dependency injection
- Explicit dependency graph
- No hidden coupling

---

### 3. Immutability by Default
**Principle**: Domain models and configuration should be immutable.

**Implementation**:
- Dataclasses with `frozen=True` for domain models
- Pydantic models with `frozen=True` for configuration
- Return new instances rather than mutating existing ones

**Benefits**:
- Thread-safe by default
- Easier to reason about state
- Prevents accidental side effects

---

## Data Processing Principles

### 4. Streaming and Batching
**Principle**: Process data in streams and batches to minimize memory footprint.

**Implementation**:
- Use async generators for streaming input
- Batch processing for efficient model inference
- **Accumulate batches before writing**: Collect all batch results, then write once
- Never load entire dataset into memory

**Benefits**:
- Memory-efficient for large codebases
- Better GPU/CPU utilization through batching
- Scalable to datasets of any size

**Critical Pattern**:
```python
# CORRECT: Accumulate all batches, write once
all_embeddings = []
async for snippets in self._batch_snippets():
    embeddings = await asyncio.to_thread(self.embedding_service.embed_batch, snippets)
    all_embeddings.extend(embeddings)  # Accumulate
await self.output_adapter.write(all_embeddings)  # Write once

# WRONG: Writing per batch overwrites file
async for snippets in self._batch_snippets():
    embeddings = await asyncio.to_thread(self.embedding_service.embed_batch, snippets)
    await self.output_adapter.write(embeddings)  # Overwrites!
```

---

### 5. Mean Pooling with L2 Normalization
**Principle**: Generate embeddings using mean pooling over token embeddings with L2 normalization for cosine similarity.

**Implementation**:
- Mean pooling: Average all token embeddings weighted by attention mask
- Avoid special tokens (CLS) for more robust semantic representation
- L2 normalization: Normalize vectors to unit length for cosine similarity

**Benefits**:
- More robust than single-token representations
- Ready for cosine similarity without additional processing
- Better semantic representation for code chunks

**Code Pattern**:
```python
# Mean pooling with attention mask
attention_mask = encoded['attention_mask']
token_embeddings = outputs.last_hidden_state
input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
embeddings = sum_embeddings / sum_mask

# L2 normalization
embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
```

---

## Concurrency Principles

### 6. Async I/O with Sync Model Inference
**Principle**: Use async for I/O operations, run CPU/GPU-bound model inference in thread pool.

**Implementation**:
- File I/O: Use `aiofiles` for async reading/writing
- Model inference: Run in thread pool via `asyncio.to_thread()`
- Never block the event loop with CPU-bound operations

**Benefits**:
- Non-blocking I/O for better throughput
- Efficient use of CPU/GPU during inference
- Scalable concurrency model

**Pattern**:
```python
# Async I/O
async with aiofiles.open(path, 'r') as f:
    async for line in f:
        yield process(line)

# Sync model in thread pool
embeddings = await asyncio.to_thread(
    self.embedding_service.embed_batch,
    batch
)
```

---

## Type Safety Principles

### 7. Comprehensive Type Hints
**Principle**: Use type hints on all functions and validate with static type checker.

**Implementation**:
- Type hints on all function signatures
- Type hints for class attributes
- Use `mypy` in strict mode for validation
- Prefer specific types over `Any`

**Benefits**:
- Catch type errors before runtime
- Better IDE support (autocomplete, refactoring)
- Self-documenting code

---

### 8. Runtime Validation at Boundaries
**Principle**: Validate data at system boundaries (input, config), trust internal types.

**Implementation**:
- Use Pydantic for configuration validation
- Validate input data in adapters (input port implementations)
- Domain layer assumes valid data from adapters
- No redundant validation in business logic

**Benefits**:
- Fail fast with clear error messages
- No performance overhead in hot paths
- Clean domain logic without validation clutter

---

## Code Quality Principles

### 9. Single Responsibility
**Principle**: Each component has one clear responsibility.

**Implementation**:
- `EmbeddingService`: Generate embeddings only
- `Pipeline`: Orchestrate data flow only
- Adapters: Handle I/O only
- CLI: Parse arguments and wire dependencies only

**Benefits**:
- Easier to test in isolation
- Easier to understand and modify
- Natural modularity

---

### 10. Explicit Over Implicit
**Principle**: Prefer explicit, obvious code over clever implicit solutions.

**Implementation**:
- Explicit error handling (no silent failures)
- Clear variable names (no abbreviations)
- Explicit imports (no `from x import *`)
- Docstrings explain "why" not just "what"

**Benefits**:
- Easier for senior engineers to skim and understand
- Reduces cognitive load
- Maintainable by future developers

---

### 11. Context-Aware Linting
**Principle**: Apply linting rules contextually based on code purpose.

**Implementation**:
- **Tests**: Allow `open()` for simplicity, ignore annotation requirements
- **CLI**: Allow many function arguments (naturally needed for CLI options)
- **Core code**: Enforce all rules strictly
- Use `pyproject.toml` per-file-ignores for context-specific rules

**Benefits**:
- Pragmatic code quality without fighting the linter
- Rules match real-world usage patterns
- Less noise, more signal in linting output

**Pattern**:
```toml
[tool.ruff.lint.per-file-ignores]
"tests/**" = ["PTH123", "ANN"]  # Allow open(), skip annotations
"src/code_transformer/cli/**" = ["PLR0913"]  # Allow many CLI args
```

---

## Testing Principles

### 12. Test at the Right Level
**Principle**: Unit test domain logic, integration test the pipeline with strategic mocking.

**Implementation**:
- **Unit tests**: Domain logic with mocked dependencies
- **Integration tests**: Full pipeline with real components, mock only expensive/slow operations
- **Strategic mocking**: Mock model downloads/loading, keep I/O, batching, orchestration real
- Use fixtures for repeatable test data

**Benefits**:
- Fast unit tests for rapid feedback
- Fast integration tests (< 5s) that test 95% of real code paths
- High confidence without slow model downloads

**Pattern**:
```python
# Mock only the slow parts
@patch('code_transformer.domain.embedding_service.AutoModel')
@patch('code_transformer.domain.embedding_service.AutoTokenizer')
async def test_pipeline_end_to_end(mock_model, mock_tokenizer):
    # Use real JSONLReader, EmbeddingWriter, Pipeline, CodeEmbeddingService
    # Only model loading is mocked
```

---

### 13. Async Test Support
**Principle**: Test async code naturally with pytest-asyncio.

**Implementation**:
- Mark async tests with `@pytest.mark.asyncio`
- Configure pytest with `asyncio_mode = auto`
- Test async generators and streams

**Benefits**:
- Natural async test syntax
- Catches async-specific bugs
- Tests match production code patterns

---

## Development Workflow Principles

### 14. Modern Python Tooling
**Principle**: Use modern, fast Python tools for development.

**Implementation**:
- `uv` for dependency management (faster than pip)
- `ruff` for linting and formatting (faster than black + flake8)
- `mypy` for static type checking
- `pytest` for testing

**Benefits**:
- Fast feedback cycles
- Consistent code style
- Early error detection

---

### 15. Package with pyproject.toml
**Principle**: Use modern Python packaging standards (PEP 621).

**Implementation**:
- All metadata in `pyproject.toml`
- No `setup.py` or `setup.cfg`
- Declare dependencies with version ranges
- Separate dev dependencies

**Benefits**:
- Standard, modern packaging
- Better tool integration
- Clear dependency management

---

### 16. Makefile for Development Commands
**Principle**: Use Makefile to provide convenient, discoverable development commands.

**Implementation**:
- `make test`, `make test-unit`, `make test-integration` - Run tests
- `make lint`, `make format`, `make type-check` - Code quality
- `make clean`, `make install`, `make dev` - Project setup
- All commands use `uv run` for consistency

**Benefits**:
- Standard interface across Python projects
- Self-documenting (`make help`)
- Easy onboarding for new developers
- Muscle memory from other projects

**Pattern**:
```makefile
test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/unit/ -v

lint:
	uv run ruff check src/ tests/
```

---

## Schema Design Principles

### 17. Rich, Extensible Schemas
**Principle**: Design schemas to capture comprehensive metadata with extensibility for future needs.

**Implementation**:
- Required fields: Minimal core fields only (`id`, `file_path`, `language`, `content`)
- Optional fields: Rich metadata for flexibility
- `metadata` dict: Language-specific fields without breaking schema
- Preserve input fields in output (pass-through)

**Benefits**:
- Forward compatibility
- Language-specific metadata without schema changes
- Full traceability from input to output

**Schema Structure**:
- Common fields: `start_line`, `end_line`, `start_byte`, `end_byte`
- Go-specific: `receiver`, `type_kind`
- Python-specific: `decorator`, `is_async`
- Java-specific: `access_modifier`, `return_type`

---

### 18. Metadata Preservation
**Principle**: Preserve all input metadata in output for traceability.

**Implementation**:
- Output includes all input fields
- Add embedding-specific fields: `embedding`, `model`, `dim`, `normalized`
- Never discard input metadata

**Benefits**:
- Full audit trail
- Downstream systems have full context
- Debugging and analysis enabled

---

## Error Handling Principles

### 19. Structured Logging
**Principle**: Use structured logging (JSON) for production observability.

**Implementation**:
- Use `structlog` for structured logging
- Log with context (batch_size, model, file_path)
- JSON output for easy parsing
- Include timestamps and log levels

**Benefits**:
- Easy to parse and analyze logs
- Better debugging in production
- Integrates with monitoring systems

---

### 20. Custom Domain Exceptions
**Principle**: Define domain-specific exceptions with proper error chaining.

**Implementation**:
- Base exception: `CodeTransformerError`
- Specific exceptions: `ModelLoadError`, `InvalidInputError`, `EmbeddingError`
- **Always chain exceptions**: Use `raise ... from err` to preserve context
- Include context in exception messages
- Catch and handle at appropriate layer

**Benefits**:
- Clear error semantics
- Full error traceback preserved
- Better debugging with complete context

**Pattern**:
```python
# CORRECT: Chain exceptions to preserve context
try:
    data = json.loads(line)
except json.JSONDecodeError as err:
    msg = f"The file {self.file_path} is not valid JSONL."
    raise InvalidInputError(msg) from err  # Preserves original error

# WRONG: Loses original error context
except json.JSONDecodeError:
    raise InvalidInputError("Invalid JSONL")  # Original error lost
```

---

## Anti-Patterns to Avoid

### 21. No Over-Engineering
**Principle**: Only implement what's immediately needed, not hypothetical future requirements.

**Avoid**:
- Multiple output formats before they're needed
- Complex configuration systems
- Premature abstractions
- Feature flags for single use case

**Do Instead**:
- Start with JSONL output
- Simple CLI arguments
- Direct implementations
- Add complexity when needed

---

### 22. No Global State
**Principle**: Avoid global variables, singletons, or mutable module-level state.

**Avoid**:
- Global model cache
- Module-level configuration
- Singleton pattern

**Do Instead**:
- Instance-level caching in classes
- Pass configuration through constructors
- Dependency injection

---

## Summary

These constitutional principles ensure:
1. **Maintainable**: Clear architecture, explicit design, comprehensive types
2. **Scalable**: Streaming, batching, async I/O
3. **Testable**: Dependency injection, ports/adapters, proper test levels
4. **Extensible**: Rich schemas, adapter pattern, clean interfaces
5. **Production-Ready**: Structured logging, error handling, L2-normalized embeddings

When making decisions, refer back to these principles. When principles conflict, favor simplicity and explicitness over cleverness.
