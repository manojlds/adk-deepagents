# CLAUDE.md

## Project Overview

adk-deepagents is a Python library that re-implements deepagents using Google ADK primitives. It provides planning, filesystem operations, shell execution, sub-agent delegation, conversation summarization, memory management, and skills integration.

## Build & Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies (including dev group)
uv sync

# Run tests
uv run pytest

# Run a specific test file
uv run pytest tests/unit_tests/test_memory.py

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run ty check
```

## Code Style

- **Formatter/Linter:** Ruff (configured in pyproject.toml)
- **Type checker:** ty
- **Line length:** 100 characters
- **Target Python:** 3.11+
- **Import sorting:** isort via ruff, `adk_deepagents` is first-party

## Project Structure

- `adk_deepagents/` — Main package
  - `backends/` — Storage backends (filesystem, state, protocol)
  - `callbacks/` — Agent lifecycle hooks (before_agent, before_model, before_tool, after_tool)
  - `execution/` — Sandbox execution (local, bridge, heimdall)
  - `skills/` — Skills integration
  - `tools/` — Agent tools (filesystem, todos, task)
  - `graph.py` — Main `create_deep_agent` factory function
  - `types.py` — Type definitions
- `tests/` — Test suite (pytest + pytest-asyncio)
  - `unit_tests/` — Unit tests
  - `integration_tests/` — Integration tests
- `examples/` — Usage examples

## Testing

- Framework: pytest with pytest-asyncio (asyncio_mode = "auto")
- Tests go in `tests/unit_tests/` or `tests/integration_tests/`
- Shared fixtures are in `tests/conftest.py`

## Key APIs

The main entry point is `create_deep_agent` from `adk_deepagents`. It accepts configuration for skills, sub-agents, and summarization and returns a configured ADK agent.
