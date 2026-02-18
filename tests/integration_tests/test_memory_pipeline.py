"""Integration tests — memory loading end-to-end with backends.

Verifies load_memory, format_memory, and their integration with
before_agent and before_model callbacks.  No API key required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.genai import types

from adk_deepagents.backends.state import StateBackend
from adk_deepagents.backends.utils import create_file_data
from adk_deepagents.callbacks.before_agent import make_before_agent_callback
from adk_deepagents.callbacks.before_model import make_before_model_callback
from adk_deepagents.memory import format_memory, load_memory
from adk_deepagents.prompts import MEMORY_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_callback_context(state=None):
    ctx = MagicMock(spec=CallbackContext)
    ctx.state = state if state is not None else {}
    return ctx


def _make_llm_request(system_instruction=None):
    req = MagicMock(spec=LlmRequest)
    req.config = types.GenerateContentConfig(system_instruction=system_instruction)
    req.contents = []
    return req


# ---------------------------------------------------------------------------
# load_memory
# ---------------------------------------------------------------------------


class TestLoadMemory:
    def test_load_memory_from_state_backend(self):
        state = {
            "files": {
                "/AGENTS.md": create_file_data("# Project\nThis is a coding project."),
            }
        }
        backend = StateBackend(state)
        contents = load_memory(backend, ["/AGENTS.md"])
        assert "/AGENTS.md" in contents
        assert "coding project" in contents["/AGENTS.md"]

    def test_load_memory_missing_file(self):
        state = {"files": {}}
        backend = StateBackend(state)
        contents = load_memory(backend, ["/nonexistent.md"])
        assert contents == {}

    def test_load_memory_multiple_files(self):
        state = {
            "files": {
                "/AGENTS.md": create_file_data("Root memory"),
                "/src/AGENTS.md": create_file_data("Src memory"),
            }
        }
        backend = StateBackend(state)
        contents = load_memory(backend, ["/AGENTS.md", "/src/AGENTS.md"])
        assert len(contents) == 2
        assert "Root memory" in contents["/AGENTS.md"]
        assert "Src memory" in contents["/src/AGENTS.md"]


# ---------------------------------------------------------------------------
# format_memory
# ---------------------------------------------------------------------------


class TestFormatMemory:
    def test_format_memory_with_contents(self):
        contents = {
            "/AGENTS.md": "I am a helpful assistant.",
            "/src/AGENTS.md": "Source-specific context.",
        }
        sources = ["/AGENTS.md", "/src/AGENTS.md"]
        result = format_memory(contents, sources)
        assert "I am a helpful assistant." in result
        assert "Source-specific context." in result
        assert "### /AGENTS.md" in result
        assert "### /src/AGENTS.md" in result
        assert "agent_memory" in result

    def test_format_memory_empty(self):
        result = format_memory({}, ["/AGENTS.md"])
        assert "(No memory loaded)" in result

    def test_memory_ordering(self):
        contents = {
            "/first.md": "First content",
            "/second.md": "Second content",
            "/third.md": "Third content",
        }
        sources = ["/first.md", "/second.md", "/third.md"]
        result = format_memory(contents, sources)
        # Verify order is preserved: first before second before third
        idx_first = result.index("First content")
        idx_second = result.index("Second content")
        idx_third = result.index("Third content")
        assert idx_first < idx_second < idx_third


# ---------------------------------------------------------------------------
# Memory in callbacks
# ---------------------------------------------------------------------------


class TestMemoryInCallbacks:
    def test_memory_in_before_agent_callback(self):
        state = {
            "files": {
                "/AGENTS.md": create_file_data("Project context here."),
            }
        }
        factory = lambda s: StateBackend(s)  # noqa: E731
        cb = make_before_agent_callback(
            memory_sources=["/AGENTS.md"],
            backend_factory=factory,
        )
        ctx = _make_callback_context(state=state)
        cb(ctx)
        assert "memory_contents" in ctx.state
        assert "/AGENTS.md" in ctx.state["memory_contents"]
        assert "Project context here." in ctx.state["memory_contents"]["/AGENTS.md"]

    def test_memory_in_before_model_callback(self):
        memory_contents = {"/AGENTS.md": "Remember to be concise."}
        cb = make_before_model_callback(memory_sources=["/AGENTS.md"])
        ctx = _make_callback_context(
            state={"memory_contents": memory_contents}
        )
        req = _make_llm_request()
        cb(ctx, req)
        si = req.config.system_instruction
        assert "Remember to be concise." in si
        assert "agent_memory" in si
