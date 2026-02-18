"""Integration tests — human-in-the-loop pipeline.

Verifies before_tool_callback approval flow: first invocation requests
confirmation, confirmed proceeds, rejected returns rejection, modified
args are applied.  No API key required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from google.adk.tools.tool_confirmation import ToolConfirmation

from adk_deepagents.callbacks.before_tool import make_before_tool_callback, resume_approval

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str):
    tool = MagicMock()
    tool.name = name
    return tool


def _make_tool_context(tool_confirmation=None):
    ctx = MagicMock()
    ctx.tool_confirmation = tool_confirmation
    ctx.function_call_id = "fc_test_001"
    ctx.actions = MagicMock()
    return ctx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBeforeToolNoInterrupt:
    def test_no_interrupt_on_returns_none(self):
        result = make_before_tool_callback(interrupt_on=None)
        assert result is None

    def test_interrupt_on_false_returns_none(self):
        result = make_before_tool_callback(interrupt_on={"read_file": False})
        assert result is None


class TestBeforeToolApprovalFlow:
    def test_first_invocation_requests_confirmation(self):
        cb = make_before_tool_callback(interrupt_on={"write_file": True})
        tool = _make_tool("write_file")
        tc = _make_tool_context(tool_confirmation=None)
        result = cb(tool, {"path": "/f.txt", "content": "data"}, tc)
        assert result is not None
        assert result["status"] == "awaiting_approval"
        tc.request_confirmation.assert_called_once()

    def test_non_interrupted_tool_passes_through(self):
        cb = make_before_tool_callback(interrupt_on={"write_file": True})
        tool = _make_tool("read_file")
        tc = _make_tool_context()
        result = cb(tool, {"path": "/f.txt"}, tc)
        assert result is None

    def test_confirmed_proceeds(self):
        cb = make_before_tool_callback(interrupt_on={"write_file": True})
        tool = _make_tool("write_file")
        confirmation = ToolConfirmation(confirmed=True, payload=None)
        tc = _make_tool_context(tool_confirmation=confirmation)
        result = cb(tool, {"path": "/f.txt"}, tc)
        assert result is None  # Proceed

    def test_rejected_returns_rejection(self):
        cb = make_before_tool_callback(interrupt_on={"write_file": True})
        tool = _make_tool("write_file")
        confirmation = ToolConfirmation(confirmed=False, payload=None)
        tc = _make_tool_context(tool_confirmation=confirmation)
        result = cb(tool, {"path": "/f.txt"}, tc)
        assert result is not None
        assert result["status"] == "rejected"
        assert "write_file" in result["tool"]

    def test_modified_args_applied(self):
        cb = make_before_tool_callback(interrupt_on={"write_file": True})
        tool = _make_tool("write_file")
        confirmation = ToolConfirmation(
            confirmed=True,
            payload={"modified_args": {"content": "new content"}},
        )
        tc = _make_tool_context(tool_confirmation=confirmation)
        args = {"path": "/f.txt", "content": "old content"}
        result = cb(tool, args, tc)
        assert result is None  # Proceed
        assert args["content"] == "new content"


class TestResumeApproval:
    def test_resume_approval_approved(self):
        conf = resume_approval(approved=True)
        assert isinstance(conf, ToolConfirmation)
        assert conf.confirmed is True
        assert conf.payload is None

    def test_resume_approval_rejected(self):
        conf = resume_approval(approved=False)
        assert isinstance(conf, ToolConfirmation)
        assert conf.confirmed is False

    def test_resume_approval_with_modified_args(self):
        conf = resume_approval(approved=True, modified_args={"content": "updated"})
        assert conf.confirmed is True
        assert conf.payload == {"modified_args": {"content": "updated"}}
