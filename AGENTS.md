# adk-deepagents

adk-deepagents is a Python library that re-implements deepagents using Google ADK primitives. It provides autonomous, tool-using agent infrastructure with filesystem, execution, delegation, summarization, and memory support.

## Commands

Install/update dependencies:

```bash
uv sync
```

After ANY code change, run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest -m "not llm"
```

Build the package:

```bash
uv build
```

## Quality Checks

- Do not consider a change complete until lint, type checks, and tests pass.
- Use the repo's default CI-style test command: `uv run pytest -m "not llm"`.

## Conventions

- Use `uv` for dependency management and command execution.
- Keep Python compatibility at 3.11+ and line length at 100 characters.
- Run Ruff format checks before Ruff lint checks.
- Keep unit tests in `tests/unit_tests/` and integration tests in `tests/integration_tests/`.
- Use pytest markers consistently (`integration`, `llm`, `browser`) where applicable.
- Preserve module boundaries under `adk_deepagents/` (`backends`, `callbacks`, `execution`, `skills`, `tools`, `browser`).

## Directory Structure

```text
adk_deepagents/      # Library source
tests/               # Unit and integration tests
examples/            # Usage examples
docs/                # Documentation
.github/workflows/   # CI and release workflows
```

## Testing Patterns

- Test runner: `pytest` with `pytest-asyncio` (`asyncio_mode = "auto"`).
- Reuse fixtures from `tests/conftest.py` and `tests/integration_tests/conftest.py`.
- Add new tests to existing domain folders (e.g. `tests/unit_tests/<area>/`).
