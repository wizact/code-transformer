# F### Design: [Feature Name]

## Architecture Overview

[High-level diagram or description of how this feature fits into the system]

```
[Component Diagram]
```

## Components

### [Component 1 Name]
- **Type**: Domain | Adapter | Application
- **Responsibility**: [What this component does]
- **Dependencies**: [What it depends on]
- **Interface**:
  ```python
  class ComponentName:
      def method_name(self, param: Type) -> ReturnType:
          """Description"""
  ```

### [Component 2 Name]
[Repeat structure...]

## Data Flow

```
Input → [Step 1] → [Step 2] → [Step 3] → Output
```

**Step-by-step:**
1. [Description of step 1]
2. [Description of step 2]
3. [Description of step 3]

## Data Structures

### [Structure Name]
```python
@dataclass(frozen=True)
class StructureName:
    field1: Type
    field2: Type
    # Purpose: [Explain why this structure exists]
```

## Key Algorithms

### [Algorithm Name]
**Purpose**: [Why this algorithm is needed]

**Approach**: [High-level explanation]

**Pseudocode**:
```
function algorithmName(input):
    step1
    step2
    return result
```

**Complexity**: Time O(...), Space O(...)

## Error Handling

| Error Condition | Handling Strategy | User Impact |
|----------------|-------------------|-------------|
| [Condition 1]  | [How we handle it] | [What user sees] |
| [Condition 2]  | [How we handle it] | [What user sees] |

## Performance Considerations

- **Memory**: [Expected usage and optimization strategies]
- **CPU**: [Computational complexity and bottlenecks]
- **I/O**: [File/network operations and async handling]

## Testing Strategy

### Unit Tests
- [What to test in isolation]

### Integration Tests
- [What to test end-to-end]

### Edge Cases
- [Specific scenarios to test]

## Alternatives Considered

### Alternative 1: [Name]
- **Pros**: [Benefits]
- **Cons**: [Drawbacks]
- **Decision**: Rejected because [reason]

### Alternative 2: [Name]
[Repeat structure...]

## Implementation Notes

- [Any gotchas, tricky parts, or important considerations]
- [Patterns to follow from existing code]
- [Security considerations]

## Rollout Plan

1. [Phase 1: What gets implemented first]
2. [Phase 2: What comes next]
3. [Phase 3: Final pieces]
