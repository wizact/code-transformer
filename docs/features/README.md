# Feature Specifications

This directory contains specifications for features in the code-transformer project.

## Structure

Each feature has its own folder with a **three-file pattern**:

```
f###-feature-name/
├── requirements.md   # What and why
├── design.md         # How (technical approach)
└── tasks.md          # Implementation breakdown
```

## Naming Convention

- **Prefix**: `f###` where ### is a zero-padded number (e.g., `f001`, `f042`)
- **Name**: Lowercase with hyphens, descriptive (e.g., `streaming-pipeline`, `cli-interface`)
- **Example**: `f001-streaming-pipeline/`

## Lifecycle

1. **Specification**: Create folder with requirements.md
2. **Design**: Add design.md with technical approach
3. **Implementation**: Add tasks.md, link to PR
4. **Completion**: Keep folder for context, mark complete in tasks.md

## Why Keep Completed Features?

- Build historical context
- Reference for similar features
- Onboarding new contributors
- Trace requirements to implementation

## Using Templates

The `TEMPLATES/` folder contains template files to help create consistent specifications:

### 1. requirements.md Template
Based on Kiro's EARS notation (GIVEN/WHEN/THEN) and SpecKit's requirement structure.

**When to use:**
- Starting a new feature
- Clarifying user needs
- Defining success criteria

**Key sections:**
- Overview (1-2 sentences)
- Functional Requirements (FR1, FR2, ...)
- Non-Functional Requirements (Performance, Reliability, Maintainability)
- Success Criteria (measurable outcomes)
- Out of Scope (what's NOT included)
- Dependencies

**How to use:**
```bash
# Copy template to new feature folder
cp TEMPLATES/requirements.md f002-new-feature/requirements.md

# Fill in the placeholders:
# - Replace F### with actual feature number (e.g., F002)
# - Replace [Feature Name] with descriptive name
# - Fill in each section based on the feature
# - Remove unused sections if not applicable
```

### 2. design.md Template
Based on SpecKit's design phase structure with hexagonal architecture focus.

**When to use:**
- After requirements are approved
- Before implementation starts
- When exploring technical approach

**Key sections:**
- Architecture Overview (how it fits in the system)
- Components (Domain, Adapter, Application layers)
- Data Flow (step-by-step processing)
- Data Structures (dataclasses, types)
- Key Algorithms (with complexity analysis)
- Error Handling (strategy table)
- Performance Considerations
- Testing Strategy
- Alternatives Considered (with rationale)

**How to use:**
```bash
# Copy template
cp TEMPLATES/design.md f002-new-feature/design.md

# Fill in:
# - Architecture diagram or description
# - List all components with responsibilities
# - Document data flow
# - Detail key algorithms
# - Consider alternatives and explain choices
```

### 3. tasks.md Template
Based on SpecKit's task breakdown approach with progress tracking.

**When to use:**
- After design is approved
- To plan implementation phases
- To track progress during development

**Key sections:**
- Status (In Progress / Completed / Blocked)
- Blocked By (dependencies)
- Implementation Breakdown (phased approach)
- Pull Requests (tracking)
- Testing Checklist
- Completion Criteria
- Lessons Learned (retrospective)

**How to use:**
```bash
# Copy template
cp TEMPLATES/tasks.md f002-new-feature/tasks.md

# Fill in:
# - Break work into logical phases
# - List specific tasks with file locations
# - Track dependencies between tasks
# - Update status as you progress
# - Record lessons learned when complete
```

## Example: f001-streaming-pipeline

See `f001-streaming-pipeline/` for a complete example of how to use the three-file pattern:

- **requirements.md**: Defines memory efficiency, batch processing, async I/O requirements
- **design.md**: Documents hexagonal architecture, components, data flow, testing strategy
- **tasks.md**: Shows phased implementation with completed status and lessons learned

## Best Practices

### Requirements
✅ Use GIVEN/WHEN/THEN for acceptance criteria
✅ Separate functional from non-functional requirements
✅ Define measurable success criteria
✅ Explicitly list what's out of scope
❌ Don't describe implementation details (save for design.md)
❌ Don't create requirements without user stories

### Design
✅ Start with architecture overview (big picture)
✅ Document all components with clear responsibilities
✅ Include data flow diagrams or descriptions
✅ Consider and document alternatives
✅ Plan for error handling upfront
❌ Don't skip testing strategy
❌ Don't over-design for hypothetical futures

### Tasks
✅ Break work into logical phases
✅ Include file locations for each task
✅ Track dependencies between tasks
✅ Update status as you go
✅ Record lessons learned for future reference
❌ Don't make tasks too granular (aim for 2-3 hours each)
❌ Don't forget testing and documentation tasks

## Template Customization

The templates are starting points, not rigid requirements:

- **Remove sections** that don't apply to your feature
- **Add sections** if your feature needs them
- **Adapt format** to fit your feature's complexity
- **Keep it practical** - templates serve you, not vice versa

## AI-Assisted Specification

These templates are designed to work well with AI assistants:

1. **Planning Phase**: AI can use templates to structure feature plans
2. **Review Phase**: Humans review structured, consistent specs
3. **Implementation Phase**: AI and humans follow documented design
4. **Retrospective Phase**: Record lessons learned for future features

## Current Features

- **f001-streaming-pipeline**: Async batch processing with embeddings (✅ Completed)

---

For questions about the specification process, see:
- [CLAUDE.md](../../CLAUDE.md) for architectural principles
- [docs/constitution/product.md](../constitution/product.md) for product vision
- [docs/constitution/tech.md](../constitution/tech.md) for technical standards
