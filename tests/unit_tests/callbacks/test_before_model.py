"""Tests for before_model callback."""

from unittest.mock import MagicMock

from google.genai import types

from adk_deepagents.callbacks.before_model import (
    MODEL_CONTEXT_WINDOWS,
    _resolve_context_window,
    make_before_model_callback,
)
from adk_deepagents.summarization import DEFAULT_CONTEXT_WINDOW
from adk_deepagents.types import SummarizationConfig


def _make_llm_request(system_instruction=None):
    """Create a mock LlmRequest."""
    request = MagicMock()
    if system_instruction:
        request.config = types.GenerateContentConfig(
            system_instruction=system_instruction,
        )
    else:
        request.config = types.GenerateContentConfig()
    return request


def test_basic_callback_injects_prompts():
    cb = make_before_model_callback()
    ctx = MagicMock()
    ctx.state = {}
    request = _make_llm_request()

    result = cb(ctx, request)
    assert result is None  # Should return None to proceed

    # System instruction should have been set
    si = request.config.system_instruction
    assert si is not None
    assert "Filesystem" in str(si) or "Todo" in str(si)


def test_memory_injection():
    cb = make_before_model_callback(memory_sources=["./AGENTS.md"])
    ctx = MagicMock()
    ctx.state = {"memory_contents": {"./AGENTS.md": "I am a helpful agent."}}
    request = _make_llm_request()

    cb(ctx, request)

    si = str(request.config.system_instruction)
    assert "agent_memory" in si or "memory" in si.lower()


def test_execution_prompt_injected():
    cb = make_before_model_callback(has_execution=True)
    ctx = MagicMock()
    ctx.state = {}
    request = _make_llm_request()

    cb(ctx, request)

    si = str(request.config.system_instruction)
    assert "execute" in si.lower()


def test_subagent_docs_injected():
    descs = [{"name": "researcher", "description": "Researches topics"}]
    cb = make_before_model_callback(subagent_descriptions=descs)
    ctx = MagicMock()
    ctx.state = {}
    request = _make_llm_request()

    cb(ctx, request)

    si = str(request.config.system_instruction)
    assert "researcher" in si


def test_dangling_tool_calls_patched():
    """Dangling tool calls in state are injected as synthetic responses."""
    cb = make_before_model_callback()
    ctx = MagicMock()
    ctx.state = {
        "_dangling_tool_calls": [
            {"id": "call_abc", "name": "write_file"},
        ]
    }

    # Create a request with a model message containing the dangling function_call
    fc_part = types.Part(
        function_call=types.FunctionCall(
            id="call_abc",
            name="write_file",
            args={"file_path": "/test.txt", "content": "hello"},
        )
    )
    model_msg = types.Content(role="model", parts=[fc_part])

    request = _make_llm_request()
    request.contents = [model_msg]

    result = cb(ctx, request)
    assert result is None

    # The dangling calls should be consumed from state
    assert "_dangling_tool_calls" not in ctx.state

    # A synthetic function_response should have been injected
    assert len(request.contents) == 2
    patched_content = request.contents[1]
    assert patched_content.parts[0].function_response is not None
    assert patched_content.parts[0].function_response.name == "write_file"
    assert patched_content.parts[0].function_response.id == "call_abc"


def test_dangling_calls_not_double_patched():
    """If a response already exists for a dangling call, it's not duplicated."""
    cb = make_before_model_callback()
    ctx = MagicMock()
    ctx.state = {
        "_dangling_tool_calls": [
            {"id": "call_xyz", "name": "ls"},
        ]
    }

    # Message with function_call AND matching function_response
    fc_part = types.Part(
        function_call=types.FunctionCall(
            id="call_xyz",
            name="ls",
            args={"path": "/"},
        )
    )
    fr_part = types.Part(
        function_response=types.FunctionResponse(
            id="call_xyz",
            name="ls",
            response={"status": "success"},
        )
    )
    model_msg = types.Content(role="model", parts=[fc_part])
    tool_msg = types.Content(role="user", parts=[fr_part])

    request = _make_llm_request()
    request.contents = [model_msg, tool_msg]

    cb(ctx, request)

    # No extra messages should be injected since the response already exists
    assert len(request.contents) == 2


def test_no_dangling_calls_no_patching():
    """When there are no dangling calls, contents are not modified."""
    cb = make_before_model_callback()
    ctx = MagicMock()
    ctx.state = {}

    msg = types.Content(role="user", parts=[types.Part(text="Hello")])
    request = _make_llm_request()
    request.contents = [msg]

    cb(ctx, request)

    assert len(request.contents) == 1


# ---------------------------------------------------------------------------
# Model-aware context window resolution
# ---------------------------------------------------------------------------


def test_resolve_context_window_explicit_config():
    """Explicit context_window in config takes priority over model lookup."""
    config = SummarizationConfig(model="gemini-2.5-flash", context_window=500_000)
    assert _resolve_context_window(config) == 500_000


def test_resolve_context_window_known_model():
    """Known models resolve to their lookup table values."""
    config = SummarizationConfig(model="gemini-2.5-flash")
    assert _resolve_context_window(config) == MODEL_CONTEXT_WINDOWS["gemini-2.5-flash"]

    config = SummarizationConfig(model="gpt-4o")
    assert _resolve_context_window(config) == 128_000


def test_resolve_context_window_unknown_model():
    """Unknown models fall back to DEFAULT_CONTEXT_WINDOW."""
    config = SummarizationConfig(model="some-unknown-model-v9")
    assert _resolve_context_window(config) == DEFAULT_CONTEXT_WINDOW


def test_resolve_context_window_explicit_overrides_model():
    """Explicit config.context_window wins even for a known model."""
    config = SummarizationConfig(model="gpt-4o", context_window=64_000)
    assert _resolve_context_window(config) == 64_000


def test_resolve_context_window_none_defaults():
    """Default SummarizationConfig (no context_window) uses model lookup."""
    config = SummarizationConfig()  # model defaults to "gemini-2.5-flash"
    assert _resolve_context_window(config) == MODEL_CONTEXT_WINDOWS["gemini-2.5-flash"]
