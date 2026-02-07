"""Shared test fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from adk_deepagents.backends.state import StateBackend
from adk_deepagents.backends.utils import create_file_data


@pytest.fixture
def empty_state() -> dict:
    """An empty session state dict."""
    return {"files": {}}


@pytest.fixture
def populated_state() -> dict:
    """A session state with some pre-existing files."""
    return {
        "files": {
            "/hello.txt": create_file_data("Hello, World!"),
            "/src/main.py": create_file_data("def main():\n    print('hello')\n"),
            "/src/utils.py": create_file_data("def add(a, b):\n    return a + b\n"),
            "/docs/readme.md": create_file_data("# My Project\n\nA description."),
        }
    }


@pytest.fixture
def state_backend(populated_state) -> StateBackend:
    """A StateBackend with pre-populated files."""
    return StateBackend(populated_state)


@pytest.fixture
def mock_tool_context(populated_state):
    """A mock ToolContext backed by a populated state."""
    ctx = MagicMock()
    ctx.state = populated_state
    ctx.state["_backend"] = StateBackend(populated_state)
    return ctx
