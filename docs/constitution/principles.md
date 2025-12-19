# Constitutional Principles

This document serves as a pointer to the main constitutional document.

For comprehensive architectural and development principles, see:
**[CLAUDE.md](../../CLAUDE.md)** at the repository root.

## Summary

CLAUDE.md defines 20 core principles organized into the following categories:

### Architecture Principles
1. **Hexagonal Architecture (Ports & Adapters)**: Separate business logic from external dependencies
2. **Dependency Injection**: Pass dependencies through constructors, never use global state
3. **Immutability by Default**: Domain models and configuration should be immutable

### Data Processing Principles
4. **Streaming and Batching**: Process data in streams and batches to minimize memory footprint
5. **Mean Pooling with L2 Normalization**: Generate embeddings using mean pooling over token embeddings with L2 normalization

### Concurrency Principles
6. **Async I/O with Sync Model Inference**: Use async for I/O operations, run CPU/GPU-bound model inference in thread pool

### Type Safety Principles
7. **Comprehensive Type Hints**: Use type hints on all functions and validate with static type checker
8. **Runtime Validation at Boundaries**: Validate data at system boundaries, trust internal types

### Code Quality Principles
9. **Single Responsibility**: Each component has one clear responsibility
10. **Explicit Over Implicit**: Prefer explicit, obvious code over clever implicit solutions

### Testing Principles
11. **Test at the Right Level**: Unit test domain logic, integration test the pipeline
12. **Async Test Support**: Test async code naturally with pytest-asyncio

### Development Workflow Principles
13. **Modern Python Tooling**: Use uv, ruff, mypy, pytest
14. **Package with pyproject.toml**: Use modern Python packaging standards (PEP 621)

### Schema Design Principles
15. **Rich, Extensible Schemas**: Design schemas to capture comprehensive metadata
16. **Metadata Preservation**: Preserve all input metadata in output for traceability

### Error Handling Principles
17. **Structured Logging**: Use structured logging (JSON) for production observability
18. **Custom Domain Exceptions**: Define domain-specific exceptions for clear error handling

### Anti-Patterns to Avoid
19. **No Over-Engineering**: Only implement what's immediately needed
20. **No Global State**: Avoid global variables, singletons, or mutable module-level state

## Why CLAUDE.md is at Root

Constitutional principles are:
1. Used by AI assistants during development
2. Referenced frequently during coding
3. Foundational to all work
4. Should be immediately visible

Keeping it at root makes it easily discoverable and reinforces its importance.

## When to Update

CLAUDE.md should be updated when:
- Making fundamental architectural decisions
- Changing core development patterns
- Adding new principles that affect all development
- Learning significant lessons that should guide future work

These principles rarely change - they're constitutional, not tactical.
