---
name: code-review
description: >
  Use when reviewing code for quality, correctness, security, and
  performance. Activate after completing a coding task or when the user
  asks for a code review.
---

# Code Review Skill

Perform a structured code review on completed code. Follow the checklist
below and provide feedback in the specified format.

## When to Use This Skill

- After completing a coding task, before presenting results to the user
- When the user explicitly asks for a code review
- When refactoring or improving existing code
- Before finalizing any non-trivial code change

## Review Workflow

1. **Read the code** — Use read_file to load all relevant source files
2. **Understand the intent** — What is this code supposed to do?
3. **Run the checklist** — Go through each category below systematically
4. **Test if possible** — Execute the code and its tests to verify behavior
5. **Report findings** — Use the feedback format at the bottom

## Review Checklist

### Correctness
- [ ] Does the code do what it's supposed to do?
- [ ] Are edge cases handled (empty input, None, boundary values)?
- [ ] Are there any off-by-one errors?
- [ ] Are return values and error conditions checked?
- [ ] Does the code handle concurrent/async scenarios correctly?

### Security
- [ ] No hardcoded credentials, API keys, or secrets
- [ ] Input validation on all external data
- [ ] No injection vectors (SQL, command, path traversal)
- [ ] Proper error handling that doesn't leak internal details
- [ ] Sensitive data is not logged or printed

### Performance
- [ ] No unnecessary loops or redundant computations
- [ ] Appropriate data structures used (dict for lookups, set for membership)
- [ ] No memory leaks or unbounded growth (unbounded lists, caches without eviction)
- [ ] I/O operations are efficient (buffered reads, batch operations)
- [ ] No premature optimization that hurts readability

### Readability
- [ ] Clear, descriptive variable and function names
- [ ] Functions are focused and not too long (< 50 lines preferred)
- [ ] Complex logic has comments explaining *why*, not *what*
- [ ] Consistent code style throughout
- [ ] No dead code or commented-out blocks

### Testing
- [ ] Are there tests for the new code?
- [ ] Do tests cover happy path and edge cases?
- [ ] Are tests readable and maintainable?
- [ ] Do all tests pass?

### Python-Specific
- [ ] Type hints on function signatures
- [ ] Docstrings on public functions and classes
- [ ] Context managers used for resource management (files, connections)
- [ ] f-strings preferred over .format() or % formatting
- [ ] pathlib used for file path operations

## Feedback Format

When reporting review findings, use this structure:

```
## Code Review Summary
[1-2 sentence overview of the code quality]

## Issues Found

### [Critical/Major/Minor]: [Short Description]
- **File:** path/to/file.py
- **Line:** 42
- **Issue:** Description of the problem
- **Suggestion:** How to fix it
- **Example:**
  ```python
  # Before
  data = eval(user_input)
  # After
  data = json.loads(user_input)
  ```

## Positive Notes
- [Things done well — acknowledge good patterns]

## Recommendations
- [Optional: broader suggestions for improvement]
```

## Severity Levels

- **Critical**: Security vulnerabilities, data loss risk, crashes. Must fix.
- **Major**: Bugs, missing error handling, performance issues. Should fix.
- **Minor**: Style issues, naming, minor improvements. Nice to fix.
