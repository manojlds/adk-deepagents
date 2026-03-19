"""Tests for tool error handler."""

from __future__ import annotations

import pytest

from adk_deepagents.tools.error_handler import (
    TOOLS_WITH_INTERNAL_ERROR_HANDLING,
    _format_error,
    wrap_tool_with_error_handler,
    wrap_tools_with_error_handler,
)


def _sync_tool(x: int) -> int:
    """A simple sync tool."""
    return x * 2


def _sync_tool_that_raises(x: int) -> int:
    """A sync tool that always raises."""
    raise ValueError(f"bad value: {x}")


async def _async_tool(x: int) -> int:
    """A simple async tool."""
    return x * 2


async def _async_tool_that_raises(x: int) -> int:
    """An async tool that always raises."""
    raise RuntimeError(f"async failure: {x}")


class TestFormatError:
    def test_basic_format(self):
        try:
            raise ValueError("test error")
        except ValueError as exc:
            result = _format_error(exc)
        assert result["status"] == "error"
        assert result["error_type"] == "ValueError"
        assert result["message"] == "test error"
        assert "traceback" in result
        assert "ValueError" in result["traceback"]

    def test_truncation(self):
        try:
            raise ValueError("x" * 500)
        except ValueError as exc:
            result = _format_error(exc, max_tb_lines=3)
        assert "truncated" in result["traceback"]


class TestWrapSyncTool:
    def test_successful_call_passes_through(self):
        wrapped = wrap_tool_with_error_handler(_sync_tool)
        assert wrapped(5) == 10

    def test_preserves_name_and_doc(self):
        wrapped = wrap_tool_with_error_handler(_sync_tool)
        assert wrapped.__name__ == "_sync_tool"
        assert wrapped.__doc__ == "A simple sync tool."

    def test_exception_returns_error_dict(self):
        wrapped = wrap_tool_with_error_handler(_sync_tool_that_raises)
        result = wrapped(42)
        assert result["status"] == "error"
        assert result["error_type"] == "ValueError"
        assert "bad value: 42" in result["message"]

    def test_skips_internal_error_handling_tools(self):
        for name in TOOLS_WITH_INTERNAL_ERROR_HANDLING:

            def dummy():
                pass

            dummy.__name__ = name
            wrapped = wrap_tool_with_error_handler(dummy)
            assert wrapped is dummy


class TestWrapAsyncTool:
    @pytest.mark.asyncio
    async def test_successful_call_passes_through(self):
        wrapped = wrap_tool_with_error_handler(_async_tool)
        assert await wrapped(5) == 10

    @pytest.mark.asyncio
    async def test_preserves_name_and_doc(self):
        wrapped = wrap_tool_with_error_handler(_async_tool)
        assert wrapped.__name__ == "_async_tool"
        assert wrapped.__doc__ == "A simple async tool."

    @pytest.mark.asyncio
    async def test_exception_returns_error_dict(self):
        wrapped = wrap_tool_with_error_handler(_async_tool_that_raises)
        result = await wrapped(7)
        assert result["status"] == "error"
        assert result["error_type"] == "RuntimeError"
        assert "async failure: 7" in result["message"]


class TestWrapToolsList:
    def test_wraps_all_tools(self):
        tools = [_sync_tool, _sync_tool_that_raises]
        wrapped = wrap_tools_with_error_handler(tools)
        assert len(wrapped) == 2
        assert wrapped[0](3) == 6
        result = wrapped[1](1)
        assert result["status"] == "error"

    def test_does_not_modify_original_list(self):
        original = [_sync_tool]
        wrapped = wrap_tools_with_error_handler(original)
        assert original[0] is _sync_tool
        assert wrapped[0] is not _sync_tool
