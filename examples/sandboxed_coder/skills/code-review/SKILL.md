---
name: code-review
description: Code review guidelines and checklist for quality assurance
---

# Code Review Skill

## Review Checklist

### Correctness
- [ ] Does the code do what it's supposed to do?
- [ ] Are edge cases handled?
- [ ] Are there any off-by-one errors?
- [ ] Are return values checked?

### Security
- [ ] No hardcoded credentials or secrets
- [ ] Input validation on all external data
- [ ] No SQL injection, XSS, or command injection vectors
- [ ] Proper error handling that doesn't leak internals

### Performance
- [ ] No unnecessary loops or redundant computations
- [ ] Appropriate data structures used
- [ ] No memory leaks or unbounded growth
- [ ] Database queries are efficient (no N+1)

### Readability
- [ ] Clear variable and function names
- [ ] Functions are focused and not too long
- [ ] Complex logic has comments explaining why
- [ ] Consistent code style

### Testing
- [ ] Are there tests for the new code?
- [ ] Do tests cover edge cases?
- [ ] Are tests readable and maintainable?

## Review Format
When reviewing code, provide feedback in this format:

```
## Summary
[1-2 sentence overview]

## Issues Found
### [Critical/Major/Minor]: [Description]
- **File:** path/to/file.py
- **Line:** 42
- **Issue:** Description of the problem
- **Suggestion:** How to fix it

## Positive Notes
- [Things done well]
```
