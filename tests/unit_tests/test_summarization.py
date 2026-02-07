"""Tests for summarization module."""

from __future__ import annotations

from unittest.mock import MagicMock

from google.genai import types

from adk_deepagents.summarization import (
    count_content_tokens,
    count_messages_tokens,
    count_tokens_approximate,
    create_summary_content,
    format_messages_for_summary,
    maybe_summarize,
    partition_messages,
)

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def test_count_tokens_approximate_empty():
    assert count_tokens_approximate("") == 1  # min 1


def test_count_tokens_approximate_short():
    assert count_tokens_approximate("hello") == 1  # 5 chars / 4 = 1


def test_count_tokens_approximate_long():
    text = "x" * 400
    assert count_tokens_approximate(text) == 100  # 400 / 4


def test_count_content_tokens_text():
    content = types.Content(
        role="user",
        parts=[types.Part(text="Hello, how are you?")],
    )
    tokens = count_content_tokens(content)
    assert tokens > 0


def test_count_content_tokens_empty_parts():
    content = types.Content(role="user", parts=[])
    assert count_content_tokens(content) == 0


def test_count_content_tokens_none_parts():
    content = types.Content(role="user", parts=None)
    assert count_content_tokens(content) == 0


def test_count_messages_tokens():
    messages = [
        types.Content(role="user", parts=[types.Part(text="Hello")]),
        types.Content(role="model", parts=[types.Part(text="Hi there!")]),
    ]
    total = count_messages_tokens(messages)
    assert total > 0


def test_count_messages_tokens_empty():
    assert count_messages_tokens([]) == 0


# ---------------------------------------------------------------------------
# Message partitioning
# ---------------------------------------------------------------------------


def test_partition_messages_fewer_than_keep():
    messages = [
        types.Content(role="user", parts=[types.Part(text="msg1")]),
        types.Content(role="model", parts=[types.Part(text="msg2")]),
    ]
    to_summarize, to_keep = partition_messages(messages, keep_count=6)
    assert to_summarize == []
    assert to_keep == messages


def test_partition_messages_exact_keep():
    messages = [types.Content(role="user", parts=[types.Part(text=f"msg{i}")]) for i in range(6)]
    to_summarize, to_keep = partition_messages(messages, keep_count=6)
    assert to_summarize == []
    assert to_keep == messages


def test_partition_messages_split():
    messages = [types.Content(role="user", parts=[types.Part(text=f"msg{i}")]) for i in range(10)]
    to_summarize, to_keep = partition_messages(messages, keep_count=4)
    assert len(to_summarize) == 6
    assert len(to_keep) == 4
    assert to_keep[-1].parts[0].text == "msg9"


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------


def test_format_messages_for_summary_text():
    messages = [
        types.Content(role="user", parts=[types.Part(text="What is 2+2?")]),
        types.Content(role="model", parts=[types.Part(text="4")]),
    ]
    result = format_messages_for_summary(messages)
    assert "[user]: What is 2+2?" in result
    assert "[model]: 4" in result


def test_format_messages_for_summary_empty_parts():
    messages = [
        types.Content(role="user", parts=[]),
    ]
    result = format_messages_for_summary(messages)
    assert "(empty)" in result


def test_create_summary_content():
    summary = create_summary_content("This is a summary.")
    assert summary.role == "user"
    assert len(summary.parts) == 1
    assert "<conversation_summary>" in summary.parts[0].text
    assert "This is a summary." in summary.parts[0].text


# ---------------------------------------------------------------------------
# maybe_summarize integration
# ---------------------------------------------------------------------------


def _make_mock_context(state: dict | None = None):
    ctx = MagicMock()
    ctx.state = state or {}
    return ctx


def _make_mock_request(messages: list[types.Content] | None = None):
    req = MagicMock()
    req.contents = messages or []
    return req


def test_maybe_summarize_no_contents():
    ctx = _make_mock_context()
    req = _make_mock_request([])
    assert maybe_summarize(ctx, req) is False


def test_maybe_summarize_below_threshold():
    messages = [
        types.Content(role="user", parts=[types.Part(text="short message")]),
    ]
    ctx = _make_mock_context()
    req = _make_mock_request(messages)
    result = maybe_summarize(ctx, req, context_window=200_000)
    assert result is False


def test_maybe_summarize_triggers():
    # Create enough content to exceed threshold
    long_text = "x" * 100_000  # ~25k tokens
    messages = [types.Content(role="user", parts=[types.Part(text=long_text)]) for _ in range(10)]
    ctx = _make_mock_context()
    req = _make_mock_request(messages)

    # Set a small context window so summarization triggers
    result = maybe_summarize(ctx, req, context_window=1000, trigger_fraction=0.5, keep_messages=2)
    assert result is True

    # Should have replaced messages with summary + kept messages
    assert len(req.contents) == 3  # 1 summary + 2 kept

    # First message should be the summary
    summary_text = req.contents[0].parts[0].text
    assert "<conversation_summary>" in summary_text

    # State should be updated
    assert ctx.state["_summarization_state"]["summaries_performed"] == 1


def test_maybe_summarize_offloads_to_backend():
    long_text = "x" * 100_000
    messages = [types.Content(role="user", parts=[types.Part(text=long_text)]) for _ in range(10)]
    ctx = _make_mock_context()
    req = _make_mock_request(messages)

    mock_backend = MagicMock()
    mock_factory = MagicMock(return_value=mock_backend)

    result = maybe_summarize(
        ctx,
        req,
        context_window=1000,
        trigger_fraction=0.5,
        keep_messages=2,
        backend_factory=mock_factory,
    )
    assert result is True

    # Backend write should have been called for history offloading
    mock_backend.write.assert_called_once()
    call_args = mock_backend.write.call_args
    assert "/conversation_history/chunk_0000.txt" in call_args[0][0]


def test_maybe_summarize_not_enough_to_partition():
    # Only 2 messages with small context — keep_count=6 means no split
    messages = [
        types.Content(role="user", parts=[types.Part(text="x" * 100_000)]) for _ in range(2)
    ]
    ctx = _make_mock_context()
    req = _make_mock_request(messages)

    result = maybe_summarize(ctx, req, context_window=1000, trigger_fraction=0.5, keep_messages=6)
    # Tokens exceed threshold but not enough messages to split
    assert result is False
