"""Todo list tools.

Provides write_todos and read_todos backed by ADK session state.
Ported from deepagents.middleware.todolist.
"""

from __future__ import annotations

from google.adk.tools import ToolContext


def write_todos(todos: list[dict], tool_context: ToolContext) -> dict:
    """Write or update the todo list.

    Each todo item should be a dict with at least a ``content`` key and a
    ``status`` key (``"pending"``, ``"in_progress"``, or ``"completed"``).

    Args:
        todos: List of todo items to set as the current todo list.
    """
    tool_context.state["todos"] = todos
    return {"status": "success", "count": len(todos)}


def read_todos(tool_context: ToolContext) -> dict:
    """Read the current todo list.

    Returns the list of todo items previously set with write_todos.
    """
    return {"todos": tool_context.state.get("todos", [])}
