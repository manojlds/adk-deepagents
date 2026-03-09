"""Tests for manual conversation compaction tool."""

from __future__ import annotations

from unittest.mock import MagicMock

from adk_deepagents.tools.compact import (
    COMPACT_CONVERSATION_REQUEST_KEY,
    create_compact_conversation_tool,
)
from adk_deepagents.types import SummarizationConfig


def _make_tool_context(state: dict | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.state = state if state is not None else {}
    return ctx


def test_compact_tool_sets_request_flag():
    tool = create_compact_conversation_tool(summarization_config=SummarizationConfig())
    ctx = _make_tool_context()

    result = tool(ctx)

    assert result["status"] == "queued"
    assert COMPACT_CONVERSATION_REQUEST_KEY in ctx.state
    assert ctx.state[COMPACT_CONVERSATION_REQUEST_KEY] is True
    assert "next model turn" in result["message"].lower()


def test_compact_tool_is_idempotent_when_already_queued():
    tool = create_compact_conversation_tool(summarization_config=SummarizationConfig())
    ctx = _make_tool_context({COMPACT_CONVERSATION_REQUEST_KEY: True})

    result = tool(ctx)

    assert result["status"] == "queued"
    assert "already queued" in result["message"].lower()
    assert ctx.state[COMPACT_CONVERSATION_REQUEST_KEY] is True
