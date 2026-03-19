"""Tests for message queue injection in before_model callback."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from google.adk.models import LlmRequest
from google.genai import types

from adk_deepagents.callbacks.before_model import (
    _inject_queued_messages,
    make_before_model_callback,
)


def _make_callback_context(state: dict[str, Any] | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.state = state if state is not None else {}
    return ctx


def _make_llm_request(contents: list[types.Content] | None = None) -> LlmRequest:
    req = LlmRequest()
    if contents is not None:
        req.contents = contents
    req.config = types.GenerateContentConfig()
    return req


def _make_llm_request_no_contents() -> LlmRequest:
    """Create an LlmRequest with contents explicitly set to None."""
    req = LlmRequest()
    req.contents = None  # type: ignore[assignment]
    req.config = types.GenerateContentConfig()
    return req


def _get_text(content: types.Content, part_index: int = 0) -> str:
    """Extract text from a content part, asserting it exists."""
    assert content.parts is not None
    part = content.parts[part_index]
    assert part.text is not None
    return part.text


class TestInjectQueuedMessages:
    def test_empty_queue_no_change(self):
        contents = [types.Content(role="user", parts=[types.Part(text="Hello")])]
        req = _make_llm_request(contents)
        _inject_queued_messages(req, [])
        assert req.contents is not None
        assert len(req.contents) == 1

    def test_single_message_injected(self):
        contents = [types.Content(role="user", parts=[types.Part(text="Hello")])]
        req = _make_llm_request(contents)
        _inject_queued_messages(req, [{"text": "New info"}])
        assert req.contents is not None
        assert len(req.contents) == 2
        assert "New info" in _get_text(req.contents[-1])

    def test_multiple_messages_merged(self):
        contents = [types.Content(role="user", parts=[types.Part(text="Hello")])]
        req = _make_llm_request(contents)
        _inject_queued_messages(req, [{"text": "First"}, {"text": "Second"}])
        assert req.contents is not None
        assert len(req.contents) == 2  # Merged into one content
        injected_text = _get_text(req.contents[-1])
        assert "First" in injected_text
        assert "Second" in injected_text

    def test_whitespace_only_messages_skipped(self):
        contents = [types.Content(role="user", parts=[types.Part(text="Hello")])]
        req = _make_llm_request(contents)
        _inject_queued_messages(req, [{"text": "  "}, {"text": "\n"}])
        assert req.contents is not None
        assert len(req.contents) == 1

    def test_non_dict_entries_skipped(self):
        contents = [types.Content(role="user", parts=[types.Part(text="Hello")])]
        req = _make_llm_request(contents)
        queued: list[Any] = ["not a dict", {"text": "Valid"}]
        _inject_queued_messages(req, queued)
        assert req.contents is not None
        assert len(req.contents) == 2
        assert "Valid" in _get_text(req.contents[-1])

    def test_no_contents_no_crash(self):
        req = _make_llm_request_no_contents()
        _inject_queued_messages(req, [{"text": "Test"}])
        # Should not crash, contents stays None

    def test_injected_message_has_user_role(self):
        contents = [types.Content(role="user", parts=[types.Part(text="Hello")])]
        req = _make_llm_request(contents)
        _inject_queued_messages(req, [{"text": "Injected"}])
        assert req.contents is not None
        assert req.contents[-1].role == "user"

    def test_injected_message_has_prefix(self):
        contents = [types.Content(role="user", parts=[types.Part(text="Hello")])]
        req = _make_llm_request(contents)
        _inject_queued_messages(req, [{"text": "Important update"}])
        assert req.contents is not None
        assert "[Injected message]" in _get_text(req.contents[-1])


class TestMessageQueueInCallback:
    async def test_queue_consumed_and_cleared(self):
        state: dict[str, Any] = {"_message_queue": [{"text": "External message"}]}
        cb = make_before_model_callback(message_queue=True)
        ctx = _make_callback_context(state)
        contents = [types.Content(role="user", parts=[types.Part(text="Hello")])]
        req = _make_llm_request(contents)
        await cb(ctx, req)
        # Queue should be consumed
        assert req.contents is not None
        assert len(req.contents) >= 2
        # Check message was injected
        all_text = " ".join(
            str(p.text) for c in req.contents for p in (c.parts or []) if getattr(p, "text", None)
        )
        assert "External message" in all_text

    async def test_empty_queue_no_injection(self):
        state: dict[str, Any] = {"_message_queue": []}
        cb = make_before_model_callback(message_queue=True)
        ctx = _make_callback_context(state)
        contents = [types.Content(role="user", parts=[types.Part(text="Hello")])]
        req = _make_llm_request(contents)
        await cb(ctx, req)
        # Only system prompt additions, no extra user content
        assert req.contents is not None
        user_contents = [c for c in req.contents if c.role == "user"]
        assert len(user_contents) == 1

    async def test_no_queue_flag_skips_check(self):
        state: dict[str, Any] = {"_message_queue": [{"text": "Should not appear"}]}
        cb = make_before_model_callback(message_queue=False)
        ctx = _make_callback_context(state)
        contents = [types.Content(role="user", parts=[types.Part(text="Hello")])]
        req = _make_llm_request(contents)
        await cb(ctx, req)
        # Message should NOT be injected
        assert req.contents is not None
        user_contents = [c for c in req.contents if c.role == "user"]
        assert len(user_contents) == 1
