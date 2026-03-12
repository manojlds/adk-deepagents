"""Tests for before_model callback."""

from unittest.mock import AsyncMock, MagicMock, patch

from google.genai import types

from adk_deepagents.callbacks.before_model import (
    _resolve_context_window,
    make_before_model_callback,
)
from adk_deepagents.model_info import DEFAULT_CONTEXT_WINDOW, resolve_context_window
from adk_deepagents.tools.compact import COMPACT_CONVERSATION_REQUEST_KEY
from adk_deepagents.types import DynamicTaskConfig, SummarizationConfig


def _make_llm_request(system_instruction=None):
    """Create a mock LlmRequest."""
    request = MagicMock()
    if system_instruction:
        request.config = types.GenerateContentConfig(
            system_instruction=system_instruction,
        )
    else:
        request.config = types.GenerateContentConfig()
    request.contents = []
    return request


async def test_basic_callback_injects_prompts():
    cb = make_before_model_callback()
    ctx = MagicMock()
    ctx.state = {}
    request = _make_llm_request()

    result = await cb(ctx, request)
    assert result is None  # Should return None to proceed

    # System instruction should have been set
    si = request.config.system_instruction
    assert si is not None
    assert "Filesystem" in str(si) or "Todo" in str(si)


async def test_memory_injection():
    cb = make_before_model_callback(memory_sources=["./AGENTS.md"])
    ctx = MagicMock()
    ctx.state = {"memory_contents": {"./AGENTS.md": "I am a helpful agent."}}
    request = _make_llm_request()

    await cb(ctx, request)

    si = str(request.config.system_instruction)
    assert "agent_memory" in si or "memory" in si.lower()


async def test_execution_prompt_injected():
    cb = make_before_model_callback(has_execution=True)
    ctx = MagicMock()
    ctx.state = {}
    request = _make_llm_request()

    await cb(ctx, request)

    si = str(request.config.system_instruction)
    assert "execute" in si.lower()


async def test_subagent_docs_injected():
    descs = [{"name": "researcher", "description": "Researches topics"}]
    cb = make_before_model_callback(subagent_descriptions=descs)
    ctx = MagicMock()
    ctx.state = {}
    request = _make_llm_request()

    await cb(ctx, request)

    si = str(request.config.system_instruction)
    assert "researcher" in si


async def test_runtime_subagent_docs_injected_from_state():
    cb = make_before_model_callback()
    ctx = MagicMock()
    ctx.state = {
        "_dynamic_subagent_specs": {
            "summarizer": {
                "name": "summarizer",
                "description": "Summarizes groups of files by topic.",
            }
        }
    }
    request = _make_llm_request()

    await cb(ctx, request)

    si = str(request.config.system_instruction)
    assert "summarizer" in si
    assert "register_subagent" not in si


async def test_runtime_subagent_docs_include_dynamic_guidance_when_enabled():
    cb = make_before_model_callback(
        dynamic_task_config=DynamicTaskConfig(),
    )
    ctx = MagicMock()
    ctx.state = {
        "_dynamic_subagent_specs": {
            "summarizer": {
                "name": "summarizer",
                "description": "Summarizes groups of files by topic.",
            }
        }
    }
    request = _make_llm_request()

    await cb(ctx, request)

    si = str(request.config.system_instruction)
    assert "summarizer" in si
    assert "register_subagent" in si


async def test_dynamic_task_concurrency_limits_injected():
    cb = make_before_model_callback(
        subagent_descriptions=[{"name": "general_purpose", "description": "General task agent"}],
        dynamic_task_config=DynamicTaskConfig(
            max_parallel=3,
            concurrency_policy="wait",
            queue_timeout_seconds=12.5,
        ),
    )
    ctx = MagicMock()
    ctx.state = {}
    request = _make_llm_request()

    await cb(ctx, request)

    si = str(request.config.system_instruction)
    assert "Dynamic Task Concurrency Limits" in si
    assert "max_parallel=3" in si
    assert "concurrency_policy=wait" in si
    assert "queue_timeout_seconds=12.5" in si


async def test_summarization_injects_compaction_prompt():
    cb = make_before_model_callback(
        summarization_config=SummarizationConfig(use_llm_summary=False),
    )
    ctx = MagicMock()
    ctx.state = {}
    request = _make_llm_request()

    await cb(ctx, request)

    si = str(request.config.system_instruction)
    assert "compact_conversation" in si


async def test_compaction_request_forces_one_summarization_pass():
    cb = make_before_model_callback(
        summarization_config=SummarizationConfig(use_llm_summary=False),
    )
    ctx = MagicMock()
    ctx.state = {COMPACT_CONVERSATION_REQUEST_KEY: True}
    request = _make_llm_request()
    request.contents = [
        types.Content(role="user", parts=[types.Part(text="hello")]),
    ]

    with patch(
        "adk_deepagents.summarization.maybe_summarize",
        new_callable=AsyncMock,
        return_value=False,
    ) as patched:
        await cb(ctx, request)

    assert COMPACT_CONVERSATION_REQUEST_KEY not in ctx.state
    assert patched.call_args is not None
    assert patched.call_args.kwargs["force"] is True


async def test_dangling_tool_calls_patched():
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

    result = await cb(ctx, request)
    assert result is None

    # The dangling calls should be consumed from state
    assert "_dangling_tool_calls" not in ctx.state

    # A synthetic function_response should have been injected
    assert len(request.contents) == 2
    patched_content = request.contents[1]
    assert patched_content.parts[0].function_response is not None
    assert patched_content.parts[0].function_response.name == "write_file"
    assert patched_content.parts[0].function_response.id == "call_abc"


async def test_dangling_calls_not_double_patched():
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

    await cb(ctx, request)

    # No extra messages should be injected since the response already exists
    assert len(request.contents) == 2


async def test_no_dangling_calls_no_patching():
    """When there are no dangling calls, contents are not modified."""
    cb = make_before_model_callback()
    ctx = MagicMock()
    ctx.state = {}

    msg = types.Content(role="user", parts=[types.Part(text="Hello")])
    request = _make_llm_request()
    request.contents = [msg]

    await cb(ctx, request)

    assert len(request.contents) == 1


# ---------------------------------------------------------------------------
# Model-aware context window resolution
# ---------------------------------------------------------------------------


def test_resolve_context_window_explicit_config():
    """Explicit context_window in config takes priority over model lookup."""
    config = SummarizationConfig(model="gemini-2.5-flash", context_window=500_000)
    assert _resolve_context_window(config) == 500_000


def test_resolve_context_window_known_model():
    """Known models resolve via dynamic lookup (litellm / fallback table)."""
    config = SummarizationConfig(model="gemini-2.5-flash")
    assert _resolve_context_window(config) == resolve_context_window("gemini-2.5-flash")

    config = SummarizationConfig(model="gpt-4o")
    assert _resolve_context_window(config) == resolve_context_window("gpt-4o")


def test_resolve_context_window_unknown_model():
    """Unknown models fall back to DEFAULT_CONTEXT_WINDOW."""
    config = SummarizationConfig(model="some-unknown-model-v9")
    assert _resolve_context_window(config) == DEFAULT_CONTEXT_WINDOW


def test_resolve_context_window_explicit_overrides_model():
    """Explicit config.context_window wins even for a known model."""
    config = SummarizationConfig(model="gpt-4o", context_window=64_000)
    assert _resolve_context_window(config) == 64_000


def test_resolve_context_window_none_defaults():
    """Default SummarizationConfig (no context_window) uses dynamic model lookup."""
    config = SummarizationConfig()  # model defaults to "gemini-2.5-flash"
    assert _resolve_context_window(config) == resolve_context_window("gemini-2.5-flash")
