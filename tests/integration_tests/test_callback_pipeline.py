"""Integration tests — full callback pipeline.

Verifies before_agent → before_model → tool execution → after_tool
work together correctly.  No API key required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.genai import types

from adk_deepagents.backends.state import StateBackend
from adk_deepagents.backends.utils import create_file_data
from adk_deepagents.callbacks.after_tool import make_after_tool_callback
from adk_deepagents.callbacks.before_agent import make_before_agent_callback
from adk_deepagents.callbacks.before_model import make_before_model_callback
from adk_deepagents.prompts import (
    EXECUTION_SYSTEM_PROMPT,
    FILESYSTEM_SYSTEM_PROMPT,
    TASK_SYSTEM_PROMPT,
    TODO_SYSTEM_PROMPT,
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_callback_context(state=None):
    ctx = MagicMock(spec=CallbackContext)
    ctx.state = state if state is not None else {}
    return ctx


def make_llm_request(system_instruction=None, contents=None):
    req = MagicMock(spec=LlmRequest)
    req.config = types.GenerateContentConfig(system_instruction=system_instruction)
    req.contents = contents or []
    return req


# ---------------------------------------------------------------------------
# before_agent_callback tests
# ---------------------------------------------------------------------------


class TestBeforeAgentBackendFactory:
    def test_backend_factory_stored_in_state(self):
        factory = lambda state: StateBackend(state)  # noqa: E731
        cb = make_before_agent_callback(backend_factory=factory)
        ctx = make_callback_context(state={})
        cb(ctx)
        assert ctx.state["_backend_factory"] is factory

    def test_memory_loaded_on_first_invocation(self):
        state = {
            "files": {
                "/AGENTS.md": create_file_data("# Project\nRemember this."),
            }
        }
        factory = lambda s: StateBackend(s)  # noqa: E731
        cb = make_before_agent_callback(
            memory_sources=["/AGENTS.md"],
            backend_factory=factory,
        )
        ctx = make_callback_context(state=state)
        cb(ctx)
        assert "memory_contents" in ctx.state
        assert "/AGENTS.md" in ctx.state["memory_contents"]
        assert "Remember this." in ctx.state["memory_contents"]["/AGENTS.md"]

    def test_memory_loaded_only_once(self):
        load_count = 0

        def counting_factory(s):
            nonlocal load_count
            load_count += 1
            return StateBackend(s)

        state = {
            "files": {
                "/AGENTS.md": create_file_data("# Memory"),
            }
        }
        cb = make_before_agent_callback(
            memory_sources=["/AGENTS.md"],
            backend_factory=counting_factory,
        )
        ctx = make_callback_context(state=state)
        cb(ctx)
        assert load_count == 1
        # Second call — memory_contents already in state, should not reload
        cb(ctx)
        assert load_count == 1

    def test_dangling_tool_calls_detected(self):
        ctx = make_callback_context(state={})
        # Build a mock session with events containing a dangling function_call
        session = MagicMock()
        fc = types.FunctionCall(id="call_123", name="read_file", args={"path": "/x"})
        event = MagicMock()
        event.content = types.Content(
            role="model",
            parts=[types.Part(function_call=fc)],
        )
        session.events = [event]
        ctx.session = session

        cb = make_before_agent_callback()
        cb(ctx)
        assert "_dangling_tool_calls" in ctx.state
        dangling = ctx.state["_dangling_tool_calls"]
        assert len(dangling) == 1
        assert dangling[0]["id"] == "call_123"
        assert dangling[0]["name"] == "read_file"


# ---------------------------------------------------------------------------
# before_model_callback tests
# ---------------------------------------------------------------------------


class TestBeforeModelSystemPromptInjection:
    def test_system_prompt_injection_filesystem(self):
        cb = make_before_model_callback()
        ctx = make_callback_context()
        req = make_llm_request()
        cb(ctx, req)
        si = req.config.system_instruction
        assert FILESYSTEM_SYSTEM_PROMPT in si

    def test_system_prompt_injection_todo(self):
        cb = make_before_model_callback()
        ctx = make_callback_context()
        req = make_llm_request()
        cb(ctx, req)
        si = req.config.system_instruction
        assert TODO_SYSTEM_PROMPT in si

    def test_system_prompt_injection_execution(self):
        cb = make_before_model_callback(has_execution=True)
        ctx = make_callback_context()
        req = make_llm_request()
        cb(ctx, req)
        si = req.config.system_instruction
        assert EXECUTION_SYSTEM_PROMPT in si

    def test_system_prompt_injection_memory(self):
        cb = make_before_model_callback(memory_sources=["/AGENTS.md"])
        state = {"memory_contents": {"/AGENTS.md": "Be helpful and kind."}}
        ctx = make_callback_context(state=state)
        req = make_llm_request()
        cb(ctx, req)
        si = req.config.system_instruction
        assert "Be helpful and kind." in si
        assert "agent_memory" in si

    def test_system_prompt_injection_subagents(self):
        descs = [{"name": "researcher", "description": "Researches topics"}]
        cb = make_before_model_callback(subagent_descriptions=descs)
        ctx = make_callback_context()
        req = make_llm_request()
        cb(ctx, req)
        si = req.config.system_instruction
        assert TASK_SYSTEM_PROMPT in si
        assert "researcher" in si
        assert "Researches topics" in si

    def test_dangling_tool_calls_patched(self):
        cb = make_before_model_callback()
        dangling = [{"id": "call_abc", "name": "write_file"}]
        ctx = make_callback_context(state={"_dangling_tool_calls": dangling})
        # Build contents with the dangling function_call
        fc = types.FunctionCall(id="call_abc", name="write_file", args={})
        model_msg = types.Content(role="model", parts=[types.Part(function_call=fc)])
        req = make_llm_request(contents=[model_msg])
        cb(ctx, req)
        # Should have inserted a synthetic function_response
        assert len(req.contents) == 2
        patched = req.contents[1]
        assert patched.role == "user"
        fr = patched.parts[0].function_response
        assert fr.id == "call_abc"
        assert fr.name == "write_file"
        assert fr.response["status"] == "cancelled"
        # dangling should be cleared from state
        assert "_dangling_tool_calls" not in ctx.state

    def test_all_prompts_combined(self):
        descs = [{"name": "worker", "description": "Does work"}]
        cb = make_before_model_callback(
            memory_sources=["/AGENTS.md"],
            has_execution=True,
            subagent_descriptions=descs,
        )
        state = {"memory_contents": {"/AGENTS.md": "Context info."}}
        ctx = make_callback_context(state=state)
        req = make_llm_request()
        cb(ctx, req)
        si = req.config.system_instruction
        assert FILESYSTEM_SYSTEM_PROMPT in si
        assert TODO_SYSTEM_PROMPT in si
        assert EXECUTION_SYSTEM_PROMPT in si
        assert "Context info." in si
        assert TASK_SYSTEM_PROMPT in si
        assert "worker" in si


# ---------------------------------------------------------------------------
# after_tool_callback tests
# ---------------------------------------------------------------------------


class TestAfterToolCallback:
    def _make_tool(self, name: str):
        tool = MagicMock()
        tool.name = name
        return tool

    def _make_tool_context(self, state=None):
        ctx = MagicMock()
        ctx.state = state if state is not None else {}
        ctx.function_call_id = "fc_001"
        return ctx

    def test_excluded_tools_not_evicted(self):
        cb = make_after_tool_callback()
        tool = self._make_tool("read_file")
        tc = self._make_tool_context()
        result = cb(tool, {}, tc)
        assert result is None

    def test_no_raw_result_no_eviction(self):
        cb = make_after_tool_callback()
        tool = self._make_tool("custom_tool")
        tc = self._make_tool_context()
        result = cb(tool, {}, tc)
        assert result is None

    def test_small_result_no_eviction(self):
        cb = make_after_tool_callback()
        tool = self._make_tool("custom_tool")
        tc = self._make_tool_context(state={"_last_tool_result": "small result"})
        result = cb(tool, {}, tc)
        assert result is None

    def test_large_result_evicted(self):
        state = {"files": {}}
        factory = lambda s: StateBackend(s)  # noqa: E731
        cb = make_after_tool_callback(backend_factory=factory)
        tool = self._make_tool("custom_tool")
        large_content = "x" * 100_000
        tc = self._make_tool_context(state={**state, "_last_tool_result": large_content})
        result = cb(tool, {}, tc)
        assert result is not None
        assert result["status"] == "result_too_large"
        assert "saved_to" in result
        assert "Preview" in result["message"]
