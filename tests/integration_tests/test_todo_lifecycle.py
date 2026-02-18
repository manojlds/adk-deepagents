"""Integration tests — todo tools end-to-end lifecycle.

Verifies write_todos / read_todos round-trip, overwrite, empty reads,
and status tracking via the mock ToolContext.
No API key or LLM required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from adk_deepagents.tools.todos import read_todos, write_todos


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx():
    """A mock ToolContext with a plain state dict."""
    mock = MagicMock()
    mock.state = {}
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWriteThenRead:
    def test_write_then_read(self, ctx):
        todos = [
            {"content": "Buy milk", "status": "pending"},
            {"content": "Write tests", "status": "in_progress"},
        ]
        write_result = write_todos(todos, ctx)
        assert write_result["status"] == "success"
        assert write_result["count"] == 2

        read_result = read_todos(ctx)
        assert len(read_result["todos"]) == 2
        assert read_result["todos"][0]["content"] == "Buy milk"
        assert read_result["todos"][1]["content"] == "Write tests"


class TestOverwriteTodos:
    def test_overwrite_todos(self, ctx):
        initial = [{"content": "Old task", "status": "pending"}]
        write_todos(initial, ctx)

        replacement = [
            {"content": "New task A", "status": "pending"},
            {"content": "New task B", "status": "completed"},
        ]
        write_todos(replacement, ctx)

        read_result = read_todos(ctx)
        assert len(read_result["todos"]) == 2
        assert read_result["todos"][0]["content"] == "New task A"
        assert read_result["todos"][1]["content"] == "New task B"


class TestEmptyRead:
    def test_empty_read(self, ctx):
        read_result = read_todos(ctx)
        assert read_result["todos"] == []


class TestTodoStatusTracking:
    def test_todo_status_tracking(self, ctx):
        todos = [
            {"content": "Task A", "status": "pending"},
            {"content": "Task B", "status": "in_progress"},
            {"content": "Task C", "status": "completed"},
        ]
        write_todos(todos, ctx)

        read_result = read_todos(ctx)
        statuses = [t["status"] for t in read_result["todos"]]
        assert statuses == ["pending", "in_progress", "completed"]
