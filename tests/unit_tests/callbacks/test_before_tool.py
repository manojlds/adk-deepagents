"""Tests for before_tool callback."""

from __future__ import annotations

from unittest.mock import MagicMock

from adk_deepagents.callbacks.before_tool import make_before_tool_callback


def _make_tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    return tool


def _make_tool_context(state: dict | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.state = state or {}
    return ctx


class TestBeforeToolCallback:
    def test_none_when_no_interrupt(self):
        result = make_before_tool_callback(interrupt_on=None)
        assert result is None

    def test_none_when_empty_interrupt(self):
        result = make_before_tool_callback(interrupt_on={})
        assert result is None

    def test_none_when_all_false(self):
        result = make_before_tool_callback(interrupt_on={"write_file": False})
        assert result is None

    def test_blocks_specified_tool(self):
        cb = make_before_tool_callback(interrupt_on={"write_file": True})
        assert cb is not None

        tool = _make_tool("write_file")
        ctx = _make_tool_context()
        result = cb(tool, {"file_path": "/test.txt"}, ctx)

        assert result is not None
        assert result["status"] == "awaiting_approval"
        assert result["tool"] == "write_file"
        assert "_pending_approval" in ctx.state

    def test_allows_unspecified_tool(self):
        cb = make_before_tool_callback(interrupt_on={"write_file": True})
        assert cb is not None

        tool = _make_tool("read_file")
        ctx = _make_tool_context()
        result = cb(tool, {"file_path": "/test.txt"}, ctx)

        assert result is None

    def test_multiple_tools(self):
        cb = make_before_tool_callback(
            interrupt_on={"write_file": True, "execute": True, "read_file": False}
        )
        assert cb is not None

        # write_file should be blocked
        assert cb(_make_tool("write_file"), {}, _make_tool_context()) is not None

        # execute should be blocked
        assert cb(_make_tool("execute"), {}, _make_tool_context()) is not None

        # read_file should NOT be blocked (False)
        assert cb(_make_tool("read_file"), {}, _make_tool_context()) is None

    def test_pending_approval_state(self):
        cb = make_before_tool_callback(interrupt_on={"execute": True})
        assert cb is not None

        ctx = _make_tool_context()
        args = {"command": "rm -rf /"}
        cb(_make_tool("execute"), args, ctx)

        pending = ctx.state["_pending_approval"]
        assert pending["tool"] == "execute"
        assert pending["args"] == args
