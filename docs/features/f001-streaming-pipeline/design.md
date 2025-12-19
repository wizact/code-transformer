# F001 Design: Streaming Pipeline

## Architecture Overview

The streaming pipeline follows **Hexagonal Architecture** (Ports & Adapters pattern) to separate business logic from I/O concerns:

```
┌─────────────────────────────────────────┐
│      Application Layer                  │
│      (EmbeddingPipeline)                │
│  - Orchestrates data flow                │
│  - Manages batching                      │
│  - Coordinates async/sync boundary       │
└────────┬──────────────┬─────────────────┘
         │              │
   ┌─────▼──────┐  ┌───▼────────────────┐
   │   Domain   │  │     Adapters       │
   │  Services  │  │  (Ports impl)      │
   │            │  │                     │
   │ Embedding  │  │  JSONLReader       │
   │  Service   │  │  EmbeddingWriter   │
   └────────────┘  └────────────────────┘
```

## Components

### EmbeddingPipeline (Application Layer)
- **Type**: Application / Orchestrator
- **Responsibility**: Coordinate streaming, batching, and embedding generation
- **Dependencies**: InputAdapter, OutputAdapter, CodeEmbeddingService
- **Interface**:
  ```python
  class EmbeddingPipeline:
      def __init__(
          self,
          input_adapter: InputAdapter,
          embedding_service: CodeEmbeddingService,
          output_adapter: OutputAdapter,
          batch_size: int = 16,
      ) -> None:
          ...

      async def process(self) -> None:
          """Process input through embedding pipeline."""
  ```

### CodeEmbeddingService (Domain Layer)
- **Type**: Domain / Core Business Logic
- **Responsibility**: Generate L2-normalized embeddings using transformers
- **Dependencies**: None (pure domain logic)
- **Interface**:
  ```python
  class CodeEmbeddingService:
      def __init__(
          self,
          model_name: str = "microsoft/codebert-base",
          max_length: int = 512,
          device: str = "auto",
      ) -> None:
          ...

      def embed_batch(self, snippets: list[CodeSnippet]) -> list[Embedding]:
          """Generate embeddings for batch (blocking/sync)."""
  ```

### JSONLReader (Input Adapter)
- **Type**: Adapter / Input Port Implementation
- **Responsibility**: Stream code snippets from JSONL file
- **Dependencies**: aiofiles
- **Interface**:
  ```python
  class JSONLReader:
      async def read(self) -> AsyncGenerator[CodeSnippet, None]:
          """Async generator yielding snippets one at a time."""
  ```

### EmbeddingWriter (Output Adapter)
- **Type**: Adapter / Output Port Implementation
- **Responsibility**: Write embeddings to JSONL file
- **Dependencies**: aiofiles
- **Interface**:
  ```python
  class EmbeddingWriter:
      async def write(self, embeddings: list[Embedding]) -> None:
          """Write batch of embeddings to file."""

      async def write_batch(
          self, batches: AsyncGenerator[list[Embedding], None]
      ) -> None:
          """Write from async generator of embedding batches."""
  ```

## Data Flow

```
JSONL File
    ↓
JSONLReader.read() [async generator]
    ↓
Pipeline._batch_snippets() [accumulate into batches]
    ↓
asyncio.to_thread(service.embed_batch) [sync → async boundary]
    ↓
EmbeddingWriter.write() [async file I/O]
    ↓
Output JSONL File
```

**Step-by-step:**
1. **Read**: JSONLReader yields CodeSnippet objects one at a time (async)
2. **Batch**: Pipeline accumulates snippets into batches of size N
3. **Embed**: When batch is full, call embed_batch() in thread pool (blocking operation)
4. **Write**: Write embeddings to output file (async)
5. **Repeat**: Continue until input exhausted, handle final partial batch

## Data Structures

### CodeSnippet
```python
@dataclass(frozen=True)
class CodeSnippet:
    id: str
    file_path: str
    language: str
    content: str
    version: str | None = None
    git_hash: str | None = None
    chunk_name: str | None = None
    chunk_type: str | None = None
    metadata: dict[str, Any] | None = None
```
**Purpose**: Immutable representation of input code snippet with all metadata

### Embedding
```python
@dataclass(frozen=True)
class Embedding:
    snippet: CodeSnippet
    vector: torch.Tensor
    model_name: str
    normalized: bool
```
**Purpose**: Pairs code snippet with its embedding vector and metadata

## Key Algorithms

### Batching Accumulator
**Purpose**: Collect snippets into fixed-size batches for efficient processing

**Approach**: Async generator that yields when batch is full or input exhausted

**Implementation**:
```python
async def _batch_snippets(self) -> AsyncGenerator[list[CodeSnippet], None]:
    batch = []
    async for snippet in self.input_adapter.read():
        batch.append(snippet)
        if len(batch) >= self.batch_size:
            yield batch
            batch = []
    if batch:  # Final partial batch
        yield batch
```

**Complexity**: Time O(n), Space O(batch_size)

### Mean Pooling
**Purpose**: Average token embeddings to create snippet-level embedding

**Approach**: Weighted average using attention mask to ignore padding

**Implementation**:
```python
def _mean_pooling(
    token_embeddings: torch.Tensor,  # [batch, seq_len, hidden_dim]
    attention_mask: torch.Tensor,     # [batch, seq_len]
) -> torch.Tensor:                    # [batch, hidden_dim]
    # Expand mask to match embedding dimensions
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # Sum embeddings weighted by mask
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)

    # Count non-padding tokens
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

    # Compute weighted average
    return sum_embeddings / sum_mask
```

**Complexity**: Time O(batch_size × seq_len × hidden_dim), Space O(batch_size × hidden_dim)

## Error Handling

| Error Condition | Handling Strategy | User Impact |
|----------------|-------------------|-------------|
| Invalid JSON in input | Raise InvalidInputError with line number | Clear error message, processing stops |
| Missing required field | Raise ValidationError with field name | Clear error message, processing stops |
| Model loading fails | Raise ModelLoadError with model name | Clear error message, processing stops |
| Embedding generation fails | Raise EmbeddingError with batch info | Clear error message, processing stops |
| File I/O error | Raise OSError with file path | System error message |

## Performance Considerations

- **Memory**:
  - Streaming input: O(1) memory per snippet read
  - Batching: O(batch_size) memory for accumulation
  - Model inference: O(batch_size × hidden_dim) for embeddings
  - **Total**: ~200-500MB depending on batch size (constant w.r.t. input size)

- **CPU**:
  - Tokenization: Fast (< 1ms per snippet)
  - Model inference: Slow (10-100ms per batch on CPU)
  - Bottleneck: Model inference (can be parallelized with GPU)

- **I/O**:
  - Async file operations prevent blocking
  - Streaming reads: one line at a time
  - Streaming writes: append mode

## Testing Strategy

### Unit Tests
- **CodeEmbeddingService**:
  - Test lazy loading
  - Test mean pooling math
  - Test L2 normalization
  - Mock AutoModel and AutoTokenizer
- **JSONLReader**:
  - Test valid input parsing
  - Test invalid JSON handling
  - Test empty file handling
- **EmbeddingWriter**:
  - Test single write
  - Test batch write
  - Test overwrite behavior

### Integration Tests
- **Full Pipeline**:
  - End-to-end with real files
  - Mock only model loading (keep inference path)
  - Test batching with 5 snippets, batch_size=2
  - Test metadata preservation
  - Test empty input handling

### Edge Cases
- Empty input file → empty output file
- Single snippet → batch of 1
- Batch size > input size → single batch
- Partial final batch → processes correctly

## Alternatives Considered

### Alternative 1: Fully Sync Pipeline
- **Pros**: Simpler code, no async complexity
- **Cons**: Blocks on I/O, poor throughput for large files
- **Decision**: Rejected - async I/O significantly improves performance

### Alternative 2: Load Entire File into Memory
- **Pros**: Simpler batching, all data available
- **Cons**: OOM for large files, doesn't scale
- **Decision**: Rejected - violates core requirement of constant memory usage

### Alternative 3: Use [CLS] Token Instead of Mean Pooling
- **Pros**: Simpler, single token
- **Cons**: Less robust, loses sequence information
- **Decision**: Rejected - mean pooling empirically better

### Alternative 4: Sync Model Inference Wrapped in fake async
- **Pros**: Everything appears async
- **Cons**: Blocking async is anti-pattern, blocks event loop
- **Decision**: Rejected - use asyncio.to_thread() for true async/sync boundary

## Implementation Notes

- **Device Resolution**: Auto-detect CUDA > MPS > CPU in that order
- **Model Caching**: Lazy load model on first use, reuse for all batches
- **Error Messages**: Include context (file path, line number, field name)
- **Logging**: Structured logs with batch size, count, duration
- **Type Safety**: Comprehensive type hints, validated by mypy in strict mode

## Rollout Plan

1. ✅ **Phase 1**: Domain layer (CodeEmbeddingService)
2. ✅ **Phase 2**: Adapters (JSONLReader, EmbeddingWriter)
3. ✅ **Phase 3**: Application layer (EmbeddingPipeline)
4. ✅ **Phase 4**: Integration tests
5. 🔮 **Phase 5**: CLI interface (future)
