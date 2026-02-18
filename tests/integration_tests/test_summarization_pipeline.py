"""Integration tests — summarization pipeline.

Verifies token counting, message partitioning, formatting, offloading,
and the maybe_summarize trigger all work together correctly.
No API key required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.genai import types

from adk_deepagents.backends.filesystem import FilesystemBackend
from adk_deepagents.summarization import (
    NUM_CHARS_PER_TOKEN,
    count_content_tokens,
    count_tokens_approximate,
    create_summary_content,
    format_messages_for_summary,
    maybe_summarize,
    offload_messages_to_backend,
    partition_messages,
    truncate_tool_args,
)
from adk_deepagents.types import TruncateArgsConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_content(role: str, text: str) -> types.Content:
    return types.Content(role=role, parts=[types.Part(text=text)])


def _make_callback_context(state=None):
    ctx = MagicMock(spec=CallbackContext)
    ctx.state = state if state is not None else {}
    return ctx


def _make_llm_request(contents=None):
    req = MagicMock(spec=LlmRequest)
    req.config = types.GenerateContentConfig()
    req.contents = contents or []
    return req


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TestTokenCounting:
    def test_token_counting_consistency(self):
        text = "hello world"
        tokens = count_tokens_approximate(text)
        assert tokens == max(1, len(text) // NUM_CHARS_PER_TOKEN)

    def test_content_token_counting(self):
        # Text message
        msg = _make_text_content("user", "This is a test message.")
        tokens = count_content_tokens(msg)
        assert tokens > 0
        assert tokens == count_tokens_approximate("This is a test message.")

        # Function call
        fc_msg = types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(name="read_file", args={"path": "/hello.txt"})
                )
            ],
        )
        fc_tokens = count_content_tokens(fc_msg)
        assert fc_tokens > 0

        # Function response
        fr_msg = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="read_file", response={"content": "file data"}
                    )
                )
            ],
        )
        fr_tokens = count_content_tokens(fr_msg)
        assert fr_tokens > 0


# ---------------------------------------------------------------------------
# Partition messages
# ---------------------------------------------------------------------------


class TestPartitionMessages:
    def test_partition_messages_basic(self):
        messages = [_make_text_content("user", f"msg {i}") for i in range(10)]
        to_summarize, to_keep = partition_messages(messages, keep_count=4)
        assert len(to_summarize) == 6
        assert len(to_keep) == 4
        # to_keep should be the last 4
        assert to_keep[0].parts is not None
        assert to_keep[0].parts[0].text == "msg 6"

    def test_partition_messages_fewer_than_keep(self):
        messages = [_make_text_content("user", f"msg {i}") for i in range(3)]
        to_summarize, to_keep = partition_messages(messages, keep_count=6)
        assert len(to_summarize) == 0
        assert len(to_keep) == 3


# ---------------------------------------------------------------------------
# Format messages for summary
# ---------------------------------------------------------------------------


class TestFormatMessagesForSummary:
    def test_format_messages_for_summary(self):
        messages = [
            _make_text_content("user", "Hello"),
            types.Content(
                role="model",
                parts=[types.Part(function_call=types.FunctionCall(name="ls", args={"path": "/"}))],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="ls", response={"files": ["a.txt"]}
                        )
                    )
                ],
            ),
        ]
        formatted = format_messages_for_summary(messages)
        assert "[user]: Hello" in formatted
        assert "[Tool Call: ls" in formatted
        assert "[Tool Result: ls" in formatted


# ---------------------------------------------------------------------------
# Create summary content
# ---------------------------------------------------------------------------


class TestCreateSummaryContent:
    def test_create_summary_content_without_offload(self):
        content = create_summary_content("This is the summary.")
        assert content.role == "user"
        assert content.parts is not None
        text = content.parts[0].text
        assert text is not None
        assert "This is the summary." in text
        assert "conversation_summary" in text

    def test_create_summary_content_with_offload(self):
        content = create_summary_content("Summary text.", offload_path="/history/session.md")
        assert content.role == "user"
        assert content.parts is not None
        text = content.parts[0].text
        assert text is not None
        assert "Summary text." in text
        assert "/history/session.md" in text


# ---------------------------------------------------------------------------
# Offload to backend
# ---------------------------------------------------------------------------


class TestOffloadToBackend:
    def test_offload_to_backend(self, tmp_path):
        backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
        messages = [_make_text_content("user", "Hello")]
        path = offload_messages_to_backend(messages, backend)
        assert path == "/conversation_history/session_history.md"
        # Verify file was written to disk
        written = tmp_path / "conversation_history" / "session_history.md"
        assert written.exists()
        assert "Hello" in written.read_text()

    def test_offload_appends(self, tmp_path):
        backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
        messages1 = [_make_text_content("user", "First batch")]
        offload_messages_to_backend(messages1, backend, chunk_index=0)

        messages2 = [_make_text_content("user", "Second batch")]
        offload_messages_to_backend(messages2, backend, chunk_index=1)

        # Verify both sections are in the file
        written = tmp_path / "conversation_history" / "session_history.md"
        content = written.read_text()
        assert "First batch" in content
        assert "Second batch" in content


# ---------------------------------------------------------------------------
# maybe_summarize
# ---------------------------------------------------------------------------


class TestMaybeSummarize:
    def test_maybe_summarize_below_threshold(self):
        messages = [_make_text_content("user", "short")]
        ctx = _make_callback_context()
        req = _make_llm_request(contents=messages)
        result = maybe_summarize(
            ctx,
            req,
            context_window=1_000_000,
            trigger_fraction=0.85,
            use_llm_summary=False,
        )
        assert result is False
        # Messages should be unchanged
        assert len(req.contents) == 1

    def test_maybe_summarize_triggers(self):
        # Create enough content to exceed a small threshold
        big_text = "x" * 4000  # ~1000 tokens
        messages = [_make_text_content("user", big_text) for _ in range(10)]
        ctx = _make_callback_context()
        req = _make_llm_request(contents=messages)
        # Set a tiny context window so we exceed 85% easily
        result = maybe_summarize(
            ctx,
            req,
            context_window=100,
            trigger_fraction=0.85,
            keep_messages=4,
            use_llm_summary=False,
        )
        assert result is True
        # Should have a summary + 4 kept messages
        assert len(req.contents) == 5

    def test_maybe_summarize_updates_state(self):
        big_text = "x" * 4000
        messages = [_make_text_content("user", big_text) for _ in range(10)]
        ctx = _make_callback_context()
        req = _make_llm_request(contents=messages)
        maybe_summarize(
            ctx,
            req,
            context_window=100,
            trigger_fraction=0.85,
            keep_messages=4,
            use_llm_summary=False,
        )
        assert "_summarization_state" in ctx.state
        ss = ctx.state["_summarization_state"]
        assert ss["summaries_performed"] == 1
        assert ss["total_tokens_summarized"] > 0


# ---------------------------------------------------------------------------
# Truncate tool args
# ---------------------------------------------------------------------------


class TestTruncateToolArgs:
    def test_truncate_tool_args(self):
        large_content = "a" * 5000
        messages = [
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id="fc1",
                            name="write_file",
                            args={"path": "/f.txt", "content": large_content},
                        )
                    )
                ],
            ),
            _make_text_content("user", "recent message"),
        ]
        config = TruncateArgsConfig(
            trigger=("messages", 1),
            keep=("messages", 1),
            max_length=100,
        )
        result, modified = truncate_tool_args(messages, config)
        assert modified is True
        assert result[0].parts is not None
        fc = result[0].parts[0].function_call
        assert fc is not None
        assert fc.args is not None
        assert len(fc.args["content"]) < len(large_content)
        assert "truncated" in fc.args["content"]
