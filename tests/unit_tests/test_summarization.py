"""Tests for summarization module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from google.genai import types

from adk_deepagents.summarization import (
    TRUNCATABLE_TOOLS,
    count_content_tokens,
    count_messages_tokens,
    count_tokens_approximate,
    create_summary_content,
    format_messages_for_summary,
    generate_llm_summary,
    maybe_summarize,
    partition_messages,
    truncate_tool_args,
)
from adk_deepagents.types import TruncateArgsConfig

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
    assert to_keep[-1].parts is not None
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
    assert summary.parts is not None
    assert len(summary.parts) == 1
    assert summary.parts[0].text is not None
    assert "<conversation_summary>" in summary.parts[0].text
    assert "This is a summary." in summary.parts[0].text


def test_create_summary_content_with_offload_path():
    summary = create_summary_content("Summary here.", offload_path="/history/session.md")
    assert summary.parts is not None
    text = summary.parts[0].text
    assert text is not None
    assert "/history/session.md" in text
    assert "<summary>" in text
    assert "Summary here." in text
    assert "saved to" in text


# ---------------------------------------------------------------------------
# Tool argument truncation
# ---------------------------------------------------------------------------


def test_truncatable_tools_set():
    assert "write_file" in TRUNCATABLE_TOOLS
    assert "edit_file" in TRUNCATABLE_TOOLS
    assert "read_file" not in TRUNCATABLE_TOOLS


def test_truncate_tool_args_no_trigger():
    config = TruncateArgsConfig(trigger=None)
    messages = [types.Content(role="user", parts=[types.Part(text="hello")])]
    result, modified = truncate_tool_args(messages, config)
    assert result == messages
    assert modified is False


def test_truncate_tool_args_below_threshold():
    config = TruncateArgsConfig(trigger=("messages", 100), keep=("messages", 5))
    messages = [types.Content(role="user", parts=[types.Part(text="hello")])]
    result, modified = truncate_tool_args(messages, config)
    assert modified is False


def test_truncate_tool_args_truncates_large_write_file():
    """Large write_file arguments in old messages should be truncated."""
    large_content = "x" * 5000  # Exceeds default max_length of 2000

    fc_part = types.Part(
        function_call=types.FunctionCall(
            name="write_file",
            args={"file_path": "/test.py", "content": large_content},
        )
    )
    old_msg = types.Content(role="model", parts=[fc_part])
    recent_msg = types.Content(role="user", parts=[types.Part(text="thanks")])

    messages = [old_msg, recent_msg]

    config = TruncateArgsConfig(
        trigger=("messages", 1),
        keep=("messages", 1),
        max_length=100,
        truncation_text="...(truncated)",
    )

    result, modified = truncate_tool_args(messages, config)
    assert modified is True
    assert len(result) == 2

    assert result[0].parts is not None
    truncated_fc = result[0].parts[0].function_call
    assert truncated_fc is not None
    assert truncated_fc.args is not None
    assert len(truncated_fc.args["content"]) < 100
    assert "...(truncated)" in truncated_fc.args["content"]
    assert truncated_fc.args["file_path"] == "/test.py"


def test_truncate_tool_args_preserves_recent_messages():
    """Messages within the keep window should not be truncated."""
    large_content = "x" * 5000
    fc_part = types.Part(
        function_call=types.FunctionCall(
            name="write_file",
            args={"file_path": "/test.py", "content": large_content},
        )
    )
    recent_msg = types.Content(role="model", parts=[fc_part])
    messages = [recent_msg]

    config = TruncateArgsConfig(
        trigger=("messages", 1),
        keep=("messages", 5),
    )

    result, modified = truncate_tool_args(messages, config)
    assert modified is False


def test_truncate_tool_args_skips_non_truncatable_tools():
    """Only write_file and edit_file arguments should be truncated."""
    large_content = "x" * 5000
    fc_part = types.Part(
        function_call=types.FunctionCall(
            name="read_file",
            args={"file_path": "/test.py", "content": large_content},
        )
    )
    old_msg = types.Content(role="model", parts=[fc_part])
    recent_msg = types.Content(role="user", parts=[types.Part(text="ok")])

    config = TruncateArgsConfig(
        trigger=("messages", 1),
        keep=("messages", 1),
        max_length=100,
    )

    result, modified = truncate_tool_args([old_msg, recent_msg], config)
    assert modified is False


# ---------------------------------------------------------------------------
# LLM summary generation
# ---------------------------------------------------------------------------


def test_generate_llm_summary_success():
    """LLM summary returns text when API call succeeds."""
    messages = [types.Content(role="user", parts=[types.Part(text="Hello")])]

    mock_response = MagicMock()
    mock_response.text = "## SESSION INTENT\nGreeting exchange."

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        result = generate_llm_summary(messages, model="gemini-2.5-flash")

    assert result is not None
    assert "SESSION INTENT" in result


def test_generate_llm_summary_failure_returns_none():
    """LLM summary returns None when API call fails."""
    messages = [types.Content(role="user", parts=[types.Part(text="Hello")])]

    with patch("google.genai.Client", side_effect=Exception("API error")):
        result = generate_llm_summary(messages)

    assert result is None


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
    result = maybe_summarize(ctx, req, context_window=200_000, use_llm_summary=False)
    assert result is False


def test_maybe_summarize_triggers():
    """Summarization triggers with inline mode (no LLM call)."""
    long_text = "x" * 100_000
    messages = [types.Content(role="user", parts=[types.Part(text=long_text)]) for _ in range(10)]
    ctx = _make_mock_context()
    req = _make_mock_request(messages)

    result = maybe_summarize(
        ctx, req, context_window=1000, trigger_fraction=0.5, keep_messages=2, use_llm_summary=False
    )
    assert result is True

    assert len(req.contents) == 3  # 1 summary + 2 kept

    summary_text = req.contents[0].parts[0].text
    assert "<conversation_summary>" in summary_text

    assert ctx.state["_summarization_state"]["summaries_performed"] == 1


def test_maybe_summarize_triggers_with_llm():
    """Summarization triggers with LLM-based summary."""
    long_text = "x" * 100_000
    messages = [types.Content(role="user", parts=[types.Part(text=long_text)]) for _ in range(10)]
    ctx = _make_mock_context()
    req = _make_mock_request(messages)

    mock_response = MagicMock()
    mock_response.text = "## SESSION INTENT\nTest task."

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        result = maybe_summarize(
            ctx,
            req,
            context_window=1000,
            trigger_fraction=0.5,
            keep_messages=2,
            use_llm_summary=True,
        )

    assert result is True
    summary_text = req.contents[0].parts[0].text
    assert "SESSION INTENT" in summary_text


def test_maybe_summarize_offloads_to_backend():
    long_text = "x" * 100_000
    messages = [types.Content(role="user", parts=[types.Part(text=long_text)]) for _ in range(10)]
    ctx = _make_mock_context()
    req = _make_mock_request(messages)

    mock_backend = MagicMock()
    mock_backend.download_files.return_value = [MagicMock(content=None)]
    mock_factory = MagicMock(return_value=mock_backend)

    result = maybe_summarize(
        ctx,
        req,
        context_window=1000,
        trigger_fraction=0.5,
        keep_messages=2,
        backend_factory=mock_factory,
        use_llm_summary=False,
    )
    assert result is True

    mock_backend.write.assert_called_once()
    call_args = mock_backend.write.call_args
    assert "/conversation_history/" in call_args[0][0]


def test_maybe_summarize_with_offload_path_in_summary():
    """When history is offloaded, summary includes the file path."""
    long_text = "x" * 100_000
    messages = [types.Content(role="user", parts=[types.Part(text=long_text)]) for _ in range(10)]
    ctx = _make_mock_context()
    req = _make_mock_request(messages)

    mock_backend = MagicMock()
    mock_backend.download_files.return_value = [MagicMock(content=None)]
    mock_factory = MagicMock(return_value=mock_backend)

    maybe_summarize(
        ctx,
        req,
        context_window=1000,
        trigger_fraction=0.5,
        keep_messages=2,
        backend_factory=mock_factory,
        use_llm_summary=False,
    )

    summary_text = req.contents[0].parts[0].text
    assert "saved to" in summary_text
    assert "/conversation_history/" in summary_text


def test_maybe_summarize_not_enough_to_partition():
    messages = [
        types.Content(role="user", parts=[types.Part(text="x" * 100_000)]) for _ in range(2)
    ]
    ctx = _make_mock_context()
    req = _make_mock_request(messages)

    result = maybe_summarize(
        ctx, req, context_window=1000, trigger_fraction=0.5, keep_messages=6, use_llm_summary=False
    )
    assert result is False


def test_maybe_summarize_with_arg_truncation():
    """Argument truncation runs before summarization check."""
    large_content = "x" * 5000
    fc_part = types.Part(
        function_call=types.FunctionCall(
            name="write_file",
            args={"file_path": "/test.py", "content": large_content},
        )
    )
    messages = [
        types.Content(role="model", parts=[fc_part]),
        types.Content(role="user", parts=[types.Part(text="ok")]),
        types.Content(role="model", parts=[types.Part(text="done")]),
    ]
    ctx = _make_mock_context()
    req = _make_mock_request(messages)

    config = TruncateArgsConfig(
        trigger=("messages", 2),
        keep=("messages", 2),
        max_length=100,
    )

    result = maybe_summarize(
        ctx,
        req,
        context_window=200_000,
        trigger_fraction=0.85,
        keep_messages=6,
        truncate_args_config=config,
        use_llm_summary=False,
    )

    # Args were truncated (returns True) but no full summarization
    assert result is True
    truncated = req.contents[0].parts[0].function_call.args["content"]
    assert len(truncated) < 200
