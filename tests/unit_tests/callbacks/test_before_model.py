"""Tests for before_model callback."""

from unittest.mock import MagicMock

from google.genai import types

from adk_deepagents.callbacks.before_model import make_before_model_callback


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
