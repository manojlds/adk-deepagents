"""Tests for after_tool callback."""

from __future__ import annotations

from unittest.mock import MagicMock

from adk_deepagents.callbacks.after_tool import (
    TOOLS_EXCLUDED_FROM_EVICTION,
    make_after_tool_callback,
)


def _make_tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    return tool


def _make_tool_context(state: dict | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.state = state or {}
    return ctx


class TestAfterToolCallback:
    def test_returns_none_for_excluded_tools(self):
        cb = make_after_tool_callback()
        for tool_name in TOOLS_EXCLUDED_FROM_EVICTION:
            tool = _make_tool(tool_name)
            result = cb(tool, {}, _make_tool_context())
            assert result is None

    def test_returns_none_for_other_tools(self):
        cb = make_after_tool_callback()
        tool = _make_tool("custom_tool")
        result = cb(tool, {"arg": "value"}, _make_tool_context())
        assert result is None

    def test_returns_none_with_backend_factory(self):
        mock_factory = MagicMock()
        cb = make_after_tool_callback(backend_factory=mock_factory)
        tool = _make_tool("some_tool")
        result = cb(tool, {}, _make_tool_context())
        assert result is None

    def test_excluded_tools_set(self):
        assert "ls" in TOOLS_EXCLUDED_FROM_EVICTION
        assert "glob" in TOOLS_EXCLUDED_FROM_EVICTION
        assert "grep" in TOOLS_EXCLUDED_FROM_EVICTION
        assert "read_file" in TOOLS_EXCLUDED_FROM_EVICTION
