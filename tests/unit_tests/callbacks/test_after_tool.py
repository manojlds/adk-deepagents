"""Tests for after_tool callback."""

from __future__ import annotations

from unittest.mock import MagicMock

from adk_deepagents.backends.state import StateBackend
from adk_deepagents.callbacks.after_tool import (
    LAST_TOOL_RESULT_KEY,
    TOOLS_EXCLUDED_FROM_EVICTION,
    make_after_tool_callback,
)


def _make_tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    return tool


def _make_tool_context(state: dict | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.state = state if state is not None else {}
    ctx.function_call_id = "test_call_id"
    return ctx


class TestAfterToolCallback:
    def test_returns_none_for_excluded_tools(self):
        cb = make_after_tool_callback()
        for tool_name in TOOLS_EXCLUDED_FROM_EVICTION:
            tool = _make_tool(tool_name)
            result = cb(tool, {}, _make_tool_context())
            assert result is None

    def test_returns_none_for_other_tools_no_stored_result(self):
        cb = make_after_tool_callback()
        tool = _make_tool("custom_tool")
        result = cb(tool, {"arg": "value"}, _make_tool_context())
        assert result is None

    def test_excluded_tools_set(self):
        assert "ls" in TOOLS_EXCLUDED_FROM_EVICTION
        assert "glob" in TOOLS_EXCLUDED_FROM_EVICTION
        assert "grep" in TOOLS_EXCLUDED_FROM_EVICTION
        assert "read_file" in TOOLS_EXCLUDED_FROM_EVICTION

    def test_small_result_not_evicted(self):
        """Small results stored via cooperative key are not evicted."""
        state = {"files": {}, LAST_TOOL_RESULT_KEY: "small result"}
        backend = StateBackend(state)

        def factory(_s):
            return backend

        cb = make_after_tool_callback(backend_factory=factory)
        tool = _make_tool("custom_tool")
        ctx = _make_tool_context(state)
        result = cb(tool, {}, ctx)
        assert result is None
        # The key should be consumed (popped)
        assert LAST_TOOL_RESULT_KEY not in ctx.state

    def test_large_result_evicted_to_backend(self):
        """Large results stored via cooperative key are evicted."""
        large_content = "x" * 100_000  # ~25K tokens, exceeds 20K limit
        state = {"files": {}, LAST_TOOL_RESULT_KEY: large_content}
        backend = StateBackend(state)

        def factory(_s):
            return backend

        cb = make_after_tool_callback(backend_factory=factory)
        tool = _make_tool("custom_tool")
        ctx = _make_tool_context(state)
        result = cb(tool, {}, ctx)

        assert result is not None
        assert result["status"] == "result_too_large"
        assert "saved_to" in result
        assert "/large_tool_results/" in result["saved_to"]
        assert "message" in result

    def test_large_result_no_backend_returns_none(self):
        """Large results without a backend factory are not evicted."""
        large_content = "x" * 100_000
        state = {LAST_TOOL_RESULT_KEY: large_content}

        cb = make_after_tool_callback(backend_factory=None)
        tool = _make_tool("custom_tool")
        ctx = _make_tool_context(state)
        result = cb(tool, {}, ctx)
        assert result is None

    def test_excluded_tool_skips_even_with_large_stored_result(self):
        """Excluded tools are skipped even if they have a stored result."""
        state = {LAST_TOOL_RESULT_KEY: "x" * 100_000}
        cb = make_after_tool_callback()
        tool = _make_tool("read_file")
        ctx = _make_tool_context(state)
        result = cb(tool, {}, ctx)
        assert result is None

    def test_cooperative_key_consumed(self):
        """The _last_tool_result key is consumed (popped) after check."""
        state = {LAST_TOOL_RESULT_KEY: "some value"}
        cb = make_after_tool_callback()
        tool = _make_tool("my_tool")
        ctx = _make_tool_context(state)
        cb(tool, {}, ctx)
        assert LAST_TOOL_RESULT_KEY not in ctx.state
