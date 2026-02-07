"""Tests for before_agent callback."""

from __future__ import annotations

from unittest.mock import MagicMock

from adk_deepagents.backends.state import StateBackend
from adk_deepagents.backends.utils import create_file_data
from adk_deepagents.callbacks.before_agent import make_before_agent_callback


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
