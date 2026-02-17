"""Tests for before_tool callback — HITL via ADK ToolConfirmation."""

from __future__ import annotations

from unittest.mock import MagicMock

from google.adk.tools.tool_confirmation import ToolConfirmation

from adk_deepagents.callbacks.before_tool import (
    make_before_tool_callback,
    resume_approval,
)


def _make_tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    return tool


def _make_tool_context(
    *,
    state: dict | None = None,
    tool_confirmation: ToolConfirmation | None = None,
    function_call_id: str = "call_123",
) -> MagicMock:
    ctx = MagicMock()
    ctx.state = state or {}
    ctx.tool_confirmation = tool_confirmation
    ctx.function_call_id = function_call_id
    ctx.actions = MagicMock()
    return ctx


class TestMakeBeforeToolCallback:
    def test_none_when_no_interrupt(self):
        result = make_before_tool_callback(interrupt_on=None)
        assert result is None

    def test_none_when_empty_interrupt(self):
        result = make_before_tool_callback(interrupt_on={})
        assert result is None

    def test_none_when_all_false(self):
        result = make_before_tool_callback(interrupt_on={"write_file": False})
        assert result is None

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

        # write_file should request confirmation
        ctx1 = _make_tool_context()
        assert cb(_make_tool("write_file"), {}, ctx1) is not None

        # execute should request confirmation
        ctx2 = _make_tool_context()
        assert cb(_make_tool("execute"), {}, ctx2) is not None

        # read_file should NOT be blocked (False)
        ctx3 = _make_tool_context()
        assert cb(_make_tool("read_file"), {}, ctx3) is None


class TestHITLPauseResume:
    """Tests for the true pause/resume mechanism using ADK ToolConfirmation."""

    def test_first_call_requests_confirmation(self):
        """First invocation pauses via request_confirmation()."""
        cb = make_before_tool_callback(interrupt_on={"write_file": True})
        assert cb is not None

        tool = _make_tool("write_file")
        ctx = _make_tool_context()
        args = {"file_path": "/test.txt", "content": "hello"}
        result = cb(tool, args, ctx)

        assert result is not None
        assert result["status"] == "awaiting_approval"
        assert result["approval_id"] == "call_123"
        assert result["tool"] == "write_file"

        # Verify request_confirmation was called
        ctx.request_confirmation.assert_called_once()
        call_kwargs = ctx.request_confirmation.call_args
        payload = call_kwargs.kwargs.get("payload") or call_kwargs[1].get("payload")
        assert payload["tool"] == "write_file"
        assert payload["args"] == args

        # skip_summarization should be set
        assert ctx.actions.skip_summarization is True

    def test_approved_proceeds(self):
        """When confirmation.confirmed=True, tool execution proceeds."""
        cb = make_before_tool_callback(interrupt_on={"write_file": True})
        assert cb is not None

        confirmation = ToolConfirmation(confirmed=True)
        tool = _make_tool("write_file")
        ctx = _make_tool_context(tool_confirmation=confirmation)
        result = cb(tool, {"file_path": "/test.txt"}, ctx)

        assert result is None  # None means proceed with tool execution

    def test_rejected_returns_rejection(self):
        """When confirmation.confirmed=False, tool is skipped."""
        cb = make_before_tool_callback(interrupt_on={"execute": True})
        assert cb is not None

        confirmation = ToolConfirmation(confirmed=False)
        tool = _make_tool("execute")
        ctx = _make_tool_context(tool_confirmation=confirmation)
        result = cb(tool, {"command": "rm -rf /"}, ctx)

        assert result is not None
        assert result["status"] == "rejected"
        assert result["tool"] == "execute"

    def test_approved_with_modified_args(self):
        """When approved with modified_args in payload, args are updated."""
        cb = make_before_tool_callback(interrupt_on={"write_file": True})
        assert cb is not None

        modified = {"content": "modified content"}
        confirmation = ToolConfirmation(
            confirmed=True,
            payload={"modified_args": modified},
        )
        tool = _make_tool("write_file")
        ctx = _make_tool_context(tool_confirmation=confirmation)
        args = {"file_path": "/test.txt", "content": "original"}
        result = cb(tool, args, ctx)

        assert result is None  # Proceed
        assert args["content"] == "modified content"
        assert args["file_path"] == "/test.txt"  # Unchanged keys preserved

    def test_approved_without_payload_proceeds(self):
        """Approved with no payload proceeds with original args."""
        cb = make_before_tool_callback(interrupt_on={"write_file": True})
        assert cb is not None

        confirmation = ToolConfirmation(confirmed=True, payload=None)
        tool = _make_tool("write_file")
        ctx = _make_tool_context(tool_confirmation=confirmation)
        original_args = {"file_path": "/test.txt", "content": "original"}
        result = cb(tool, original_args, ctx)

        assert result is None
        assert original_args["content"] == "original"


class TestResumeApproval:
    """Tests for the resume_approval helper function."""

    def test_approve(self):
        tc = resume_approval(approved=True)
        assert isinstance(tc, ToolConfirmation)
        assert tc.confirmed is True
        assert tc.payload is None

    def test_reject(self):
        tc = resume_approval(approved=False)
        assert isinstance(tc, ToolConfirmation)
        assert tc.confirmed is False
        assert tc.payload is None

    def test_approve_with_modified_args(self):
        tc = resume_approval(approved=True, modified_args={"command": "echo hello"})
        assert tc.confirmed is True
        assert tc.payload == {"modified_args": {"command": "echo hello"}}

    def test_reject_ignores_modified_args(self):
        tc = resume_approval(approved=False, modified_args={"command": "echo hello"})
        assert tc.confirmed is False
        assert tc.payload is None
