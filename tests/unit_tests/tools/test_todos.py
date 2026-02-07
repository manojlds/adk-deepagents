"""Tests for todo tools."""

from unittest.mock import MagicMock

from adk_deepagents.tools.todos import read_todos, write_todos


def test_write_todos():
    ctx = MagicMock()
    ctx.state = {}
    todos = [
        {"content": "Task 1", "status": "pending"},
        {"content": "Task 2", "status": "in_progress"},
    ]
    result = write_todos(todos, ctx)
    assert result["status"] == "success"
    assert result["count"] == 2
    assert ctx.state["todos"] == todos


def test_read_todos_empty():
    ctx = MagicMock()
    ctx.state = {}
    result = read_todos(ctx)
    assert result["todos"] == []


def test_read_todos_with_data():
    ctx = MagicMock()
    ctx.state = {"todos": [{"content": "Task 1", "status": "completed"}]}
    result = read_todos(ctx)
    assert len(result["todos"]) == 1
    assert result["todos"][0]["content"] == "Task 1"


def test_write_then_read():
    ctx = MagicMock()
    ctx.state = {}
    todos = [{"content": "Do stuff", "status": "pending"}]
    write_todos(todos, ctx)
    result = read_todos(ctx)
    assert result["todos"] == todos
