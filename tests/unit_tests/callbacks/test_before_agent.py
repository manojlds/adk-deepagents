"""Tests for before_agent callback."""

from __future__ import annotations

from unittest.mock import MagicMock

from adk_deepagents.backends.state import StateBackend
from adk_deepagents.backends.utils import create_file_data
from adk_deepagents.callbacks.before_agent import (
    _patch_dangling_tool_calls,
    make_before_agent_callback,
)


def _make_callback_context(state: dict | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.state = state or {}
    return ctx


class TestBeforeAgentCallback:
    def test_no_memory_sources(self):
        cb = make_before_agent_callback(memory_sources=None)
        ctx = _make_callback_context()
        result = cb(ctx)
        assert result is None
        assert "memory_contents" not in ctx.state

    def test_loads_memory_files(self):
        state = {
            "files": {
                "/AGENTS.md": create_file_data("# Agent Memory\nRemember this."),
            }
        }
        backend = StateBackend(state)

        def factory(_s):
            return backend

        cb = make_before_agent_callback(
            memory_sources=["/AGENTS.md"],
            backend_factory=factory,
        )
        ctx = _make_callback_context(state)
        result = cb(ctx)
        assert result is None
        assert "memory_contents" in ctx.state
        assert "/AGENTS.md" in ctx.state["memory_contents"]
        assert "Remember this" in ctx.state["memory_contents"]["/AGENTS.md"]

    def test_loads_only_once(self):
        state = {
            "files": {"/AGENTS.md": create_file_data("content")},
            "memory_contents": {"/AGENTS.md": "cached"},
        }
        backend = StateBackend(state)

        call_count = 0

        def factory(_s):
            nonlocal call_count
            call_count += 1
            return backend

        cb = make_before_agent_callback(
            memory_sources=["/AGENTS.md"],
            backend_factory=factory,
        )
        ctx = _make_callback_context(state)
        cb(ctx)
        # Factory should NOT be called because memory_contents already exists
        assert call_count == 0
        assert ctx.state["memory_contents"]["/AGENTS.md"] == "cached"

    def test_missing_memory_file(self):
        state = {"files": {}}
        backend = StateBackend(state)

        def factory(_s):
            return backend

        cb = make_before_agent_callback(
            memory_sources=["/missing.md"],
            backend_factory=factory,
        )
        ctx = _make_callback_context(state)
        cb(ctx)
        assert ctx.state["memory_contents"] == {}

    def test_no_backend_factory(self):
        cb = make_before_agent_callback(
            memory_sources=["/AGENTS.md"],
            backend_factory=None,
        )
        ctx = _make_callback_context()
        result = cb(ctx)
        assert result is None
        assert "memory_contents" not in ctx.state


class TestDanglingToolCallPatching:
    """Tests for _patch_dangling_tool_calls."""

    def test_no_session_returns_false(self):
        """When callback_context has no session, patching is skipped."""
        ctx = MagicMock()
        ctx.state = {}
        ctx.session = None
        result = _patch_dangling_tool_calls(ctx)
        assert result is False
        assert "_dangling_tool_calls" not in ctx.state

    def test_no_events_returns_false(self):
        """When session has no events, patching is skipped."""
        ctx = MagicMock()
        ctx.state = {}
        ctx.session = MagicMock()
        ctx.session.events = None
        result = _patch_dangling_tool_calls(ctx)
        assert result is False

    def test_no_dangling_calls(self):
        """When all tool calls have responses, nothing is patched."""
        call_part = MagicMock()
        call_part.function_call = MagicMock()
        call_part.function_call.id = "call_123"
        call_part.function_call.name = "read_file"
        call_part.function_response = None

        response_part = MagicMock()
        response_part.function_call = None
        response_part.function_response = MagicMock()
        response_part.function_response.id = "call_123"

        event1 = MagicMock()
        event1.content = MagicMock()
        event1.content.parts = [call_part]

        event2 = MagicMock()
        event2.content = MagicMock()
        event2.content.parts = [response_part]

        ctx = MagicMock()
        ctx.state = {}
        ctx.session = MagicMock()
        ctx.session.events = [event1, event2]

        result = _patch_dangling_tool_calls(ctx)
        assert result is False
        assert "_dangling_tool_calls" not in ctx.state

    def test_detects_dangling_call(self):
        """When a tool call has no response, it's detected as dangling."""
        call_part = MagicMock()
        call_part.function_call = MagicMock()
        call_part.function_call.id = "call_456"
        call_part.function_call.name = "write_file"
        call_part.function_response = None

        event = MagicMock()
        event.content = MagicMock()
        event.content.parts = [call_part]

        ctx = MagicMock()
        ctx.state = {}
        ctx.session = MagicMock()
        ctx.session.events = [event]

        result = _patch_dangling_tool_calls(ctx)
        assert result is True
        assert "_dangling_tool_calls" in ctx.state
        dangling = ctx.state["_dangling_tool_calls"]
        assert len(dangling) == 1
        assert dangling[0]["id"] == "call_456"
        assert dangling[0]["name"] == "write_file"

    def test_multiple_dangling_calls(self):
        """Multiple dangling calls across events are all detected."""
        call1 = MagicMock()
        call1.function_call = MagicMock()
        call1.function_call.id = "call_a"
        call1.function_call.name = "ls"
        call1.function_response = None

        call2 = MagicMock()
        call2.function_call = MagicMock()
        call2.function_call.id = "call_b"
        call2.function_call.name = "grep"
        call2.function_response = None

        event = MagicMock()
        event.content = MagicMock()
        event.content.parts = [call1, call2]

        ctx = MagicMock()
        ctx.state = {}
        ctx.session = MagicMock()
        ctx.session.events = [event]

        result = _patch_dangling_tool_calls(ctx)
        assert result is True
        dangling = ctx.state["_dangling_tool_calls"]
        assert len(dangling) == 2
        ids = {d["id"] for d in dangling}
        assert ids == {"call_a", "call_b"}

    def test_callback_calls_patching(self):
        """The full callback invokes dangling tool call patching."""
        call_part = MagicMock()
        call_part.function_call = MagicMock()
        call_part.function_call.id = "call_789"
        call_part.function_call.name = "execute"
        call_part.function_response = None

        event = MagicMock()
        event.content = MagicMock()
        event.content.parts = [call_part]

        ctx = MagicMock()
        ctx.state = {}
        ctx.session = MagicMock()
        ctx.session.events = [event]

        cb = make_before_agent_callback()
        cb(ctx)
        assert "_dangling_tool_calls" in ctx.state

    def test_empty_events_no_patch(self):
        """Empty events list produces no dangling calls."""
        ctx = MagicMock()
        ctx.state = {}
        ctx.session = MagicMock()
        ctx.session.events = []

        result = _patch_dangling_tool_calls(ctx)
        assert result is False
