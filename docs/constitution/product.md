# Product Vision & Scope

## Mission
Transform project artifacts into semantic embeddings for context engineering and AI-powered understanding.

## Core Value Proposition
Enable developers to:
- Create semantic embeddings from diverse project artifacts (code, documentation, commits, PRs)
- Build context-aware AI systems with comprehensive project understanding
- Support semantic search across all project dimensions
- Enable intelligent code and documentation recommendations
- Power AI-assisted development with rich contextual embeddings

## Target Users

### Primary
- **ML Engineers**: Building code search and recommendation systems
- **Developer Tool Creators**: Integrating code intelligence into IDEs and platforms
- **Research Scientists**: Studying code semantics and software engineering

### Secondary
- **Platform Teams**: Building internal code intelligence infrastructure
- **DevOps Engineers**: Analyzing codebases for patterns and anti-patterns
- **Security Teams**: Finding similar vulnerable code patterns

## Scope

### In Scope (v1.0)
- ✅ JSONL input/output format (chunks in → embeddings out)
- ✅ CodeBERT-based embeddings (microsoft/codebert-base)
- ✅ Mean pooling with L2 normalization
- ✅ Streaming batch processing (constant memory usage)
- ✅ Multi-language code support (Python, Go, JavaScript, Java, etc.)
- ✅ Configurable batch size for performance tuning
- ✅ Async I/O for non-blocking operations

### Planned (v2.0+)
- 🔮 Documentation embeddings (Markdown, reStructuredText, API docs)
- 🔮 Git commit message embeddings
- 🔮 Pull request description embeddings
- 🔮 Issue and ticket embeddings
- 🔮 Multiple embedding model support (beyond CodeBERT)
- 🔮 Custom model fine-tuning utilities
- 🔮 Embedding quality metrics and evaluation tools

### Out of Scope
- ❌ Vector database integration (use output embeddings with your preferred DB)
- ❌ Training new embedding models (use existing pre-trained models)
- ❌ Content parsing/chunking (assumes pre-chunked input)
- ❌ Search or retrieval UX (only embedding generation)
- ❌ Code execution or static analysis

## Success Metrics

### Performance
- Process 100K+ code snippets without running out of memory
- Generate embeddings in < 1 second per snippet on CPU
- Support codebases up to 10M lines of code
- Memory usage stays under 500MB regardless of input size

### Quality
- 95%+ test coverage (unit + integration)
- Zero crashes on valid input
- Clear error messages for invalid input
- Deterministic outputs (same input = same embedding)

### Adoption
- Used by at least 3 external teams within 6 months
- Positive feedback from early adopters
- Contributions from external developers
- Clear documentation that onboards users in < 30 minutes

## Principles

See [CLAUDE.md](../../CLAUDE.md) for technical principles and [tech.md](./tech.md) for technology decisions.

## Product Roadmap

### Phase 1: Code Embeddings (v0.1 - v1.0) ✅
- Streaming pipeline with JSONL I/O (chunks in → embeddings out)
- CodeBERT embeddings with mean pooling
- Multi-language code support
- Comprehensive testing

### Phase 2: Extended Artifacts (v1.1 - v2.0)
- Documentation embeddings (Markdown, RST, API docs)
- Git commit and PR description embeddings
- Issue and ticket embeddings
- Multiple embedding model support

### Phase 3: Advanced Features (v2.1+)
- Custom model fine-tuning utilities
- Embedding quality metrics
- Performance optimizations
- Additional artifact types based on user feedback

## Decision Log

### Why JSONL Format?
- Simple, line-oriented format
- Streaming-friendly (process one line at a time)
- Human-readable for debugging
- Easy to generate from code parsers

**Alternatives considered**: CSV (too rigid), JSON (not streaming-friendly), Parquet (overkill for v1.0)

### Why CodeBERT?
- State-of-the-art for code understanding
- Pre-trained on 6 programming languages
- Well-documented and maintained by Microsoft
- Good balance of performance and quality

**Alternatives considered**: UniXcoder (newer, less proven), GraphCodeBERT (graph structure overkill), Custom model (too much work for v1.0)

### Why Mean Pooling?
- More robust than [CLS] token alone
- Captures full sequence semantics
- Standard practice in sentence embeddings
- Works well empirically

**Alternatives considered**: [CLS] token only (less robust), Max pooling (loses information), Weighted pooling (added complexity)

### Why No Vector Database Integration?
**Decision**: This project focuses exclusively on transforming chunks into embeddings. Vector database integration is permanently out of scope.

**Rationale**:
- Single responsibility: chunks in → embeddings out
- Users have diverse vector DB preferences (Pinecone, Weaviate, Chroma, etc.)
- Each database has different APIs and requirements
- JSONL output format is universal - works with any database
- Keeps the project focused and maintainable
- Embedding generation and storage are separate concerns

**Usage pattern**: Generate embeddings with this tool, then load them into your preferred vector database using that database's SDK or tools.

## Contact

For product questions or feedback:
- Create an issue on GitHub
- Check existing feature discussions
- Read the documentation first

This document is maintained by the project maintainers and updated as product direction evolves.
