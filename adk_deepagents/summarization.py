"""Conversation summarization — context window management.

Implements token counting, message partitioning, and summary generation
to prevent context window overflow during long conversations. Integrates
with ``before_model_callback`` via ``maybe_summarize()``.

Ported from deepagents.middleware.summarization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from google.genai import types

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models import LlmRequest

    from adk_deepagents.backends.protocol import Backend, BackendFactory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CHARS_PER_TOKEN = 4
"""Rough heuristic: ~4 characters per token for English text."""

DEFAULT_CONTEXT_WINDOW = 200_000
"""Default context window size in tokens (Gemini 2.5 Flash)."""

SUMMARY_PROMPT = """\
You are a summarization assistant. Summarize the following conversation \
concisely, preserving all important context, decisions, tool results, \
file paths, and task progress. The summary will replace the original \
messages in the agent's context window.

Conversation to summarize:

{conversation}

Provide a concise summary that captures:
1. The user's goals and requests
2. Key decisions and actions taken
3. Important tool results (file contents, command outputs)
4. Current task state and any pending work
5. Any errors encountered and their resolution"""


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def count_tokens_approximate(text: str) -> int:
    """Approximate token count using a character-based heuristic.

    Uses ~4 characters per token, matching deepagents' approach.
    """
    return max(1, len(text) // NUM_CHARS_PER_TOKEN)


def count_content_tokens(content: types.Content) -> int:
    """Count approximate tokens in a ``Content`` message."""
    total = 0
    if content.parts:
        for part in content.parts:
            if hasattr(part, "text") and part.text:
                total += count_tokens_approximate(part.text)
            elif hasattr(part, "function_call") and part.function_call:
                # Estimate tokens for function call
                fc = part.function_call
                total += count_tokens_approximate(str(fc.name or ""))
                total += count_tokens_approximate(str(fc.args or {}))
            elif hasattr(part, "function_response") and part.function_response:
                fr = part.function_response
                total += count_tokens_approximate(str(fr.name or ""))
                total += count_tokens_approximate(str(fr.response or {}))
    return total


def count_messages_tokens(messages: list[types.Content]) -> int:
    """Count approximate tokens across all messages."""
    return sum(count_content_tokens(msg) for msg in messages)


# ---------------------------------------------------------------------------
# Message partitioning
# ---------------------------------------------------------------------------


def partition_messages(
    messages: list[types.Content],
    keep_count: int = 6,
) -> tuple[list[types.Content], list[types.Content]]:
    """Split messages into (to_summarize, to_keep).

    Always keeps the most recent ``keep_count`` messages intact.
    The rest are candidates for summarization.

    Parameters
    ----------
    messages:
        The full list of conversation messages.
    keep_count:
        Number of recent messages to keep verbatim.

    Returns
    -------
    tuple
        ``(to_summarize, to_keep)`` — both are lists of ``Content``.
    """
    if len(messages) <= keep_count:
        return [], messages

    split_point = len(messages) - keep_count
    return messages[:split_point], messages[split_point:]


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def format_messages_for_summary(messages: list[types.Content]) -> str:
    """Convert a list of ``Content`` messages to a readable string for summarization."""
    parts: list[str] = []
    for msg in messages:
        role = msg.role or "unknown"
        text_parts: list[str] = []
        if msg.parts:
            for part in msg.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    text_parts.append(f"[Tool Call: {fc.name}({fc.args})]")
                elif hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    resp_str = str(fr.response or {})
                    # Truncate very long tool responses in summary input
                    if len(resp_str) > 2000:
                        resp_str = resp_str[:1000] + "... (truncated) ..." + resp_str[-500:]
                    text_parts.append(f"[Tool Result: {fr.name} -> {resp_str}]")
        text = "\n".join(text_parts) if text_parts else "(empty)"
        parts.append(f"[{role}]: {text}")
    return "\n\n".join(parts)


def create_summary_content(summary_text: str) -> types.Content:
    """Wrap a summary string in a ``Content`` message with role ``user``."""
    return types.Content(
        role="user",
        parts=[
            types.Part(
                text=(
                    "<conversation_summary>\n"
                    "The following is a summary of the earlier conversation:\n\n"
                    f"{summary_text}\n"
                    "</conversation_summary>"
                )
            )
        ],
    )


# ---------------------------------------------------------------------------
# History offloading
# ---------------------------------------------------------------------------


def offload_messages_to_backend(
    messages: list[types.Content],
    backend: Backend,
    history_path_prefix: str = "/conversation_history",
    chunk_index: int = 0,
) -> str:
    """Save summarized messages to the backend for reference.

    Returns the path where messages were saved.
    """
    formatted = format_messages_for_summary(messages)
    path = f"{history_path_prefix}/chunk_{chunk_index:04d}.txt"
    try:
        backend.write(path, formatted)
    except Exception:
        logger.exception("Failed to offload conversation history to %s", path)
    return path


# ---------------------------------------------------------------------------
# Main integration point
# ---------------------------------------------------------------------------


@dataclass
class SummarizationState:
    """Tracks summarization state across callback invocations."""

    summaries_performed: int = 0
    total_tokens_summarized: int = 0
    last_summary: str = ""


def maybe_summarize(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    *,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    trigger_fraction: float = 0.85,
    keep_messages: int = 6,
    backend_factory: BackendFactory | None = None,
    history_path_prefix: str = "/conversation_history",
) -> bool:
    """Check if summarization is needed and perform it if so.

    This is called from ``before_model_callback``. It inspects the
    ``llm_request.contents`` list and, if the token count exceeds
    the trigger threshold, replaces old messages with a summary.

    Parameters
    ----------
    callback_context:
        The ADK callback context (provides state access).
    llm_request:
        The LLM request being prepared (contains ``contents``).
    context_window:
        Total context window size in tokens.
    trigger_fraction:
        Fraction of context window that triggers summarization (0.0-1.0).
    keep_messages:
        Number of recent messages to keep verbatim.
    backend_factory:
        Optional factory to create a backend for history offloading.
    history_path_prefix:
        Path prefix for offloaded conversation history files.

    Returns
    -------
    bool
        ``True`` if summarization was performed.
    """
    contents = llm_request.contents
    if not contents:
        return False

    # Count current tokens
    current_tokens = count_messages_tokens(contents)
    trigger_threshold = int(context_window * trigger_fraction)

    if current_tokens < trigger_threshold:
        return False

    logger.info(
        "Summarization triggered: %d tokens exceeds threshold %d (%.0f%% of %d)",
        current_tokens,
        trigger_threshold,
        trigger_fraction * 100,
        context_window,
    )

    # Partition messages
    to_summarize, to_keep = partition_messages(contents, keep_count=keep_messages)
    if not to_summarize:
        return False

    # Offload old messages to backend if available
    state = callback_context.state
    summ_state = state.get("_summarization_state")
    if summ_state is None:
        summ_state = {"summaries_performed": 0, "total_tokens_summarized": 0}
        state["_summarization_state"] = summ_state

    if backend_factory:
        try:
            backend = backend_factory(state)
            offload_messages_to_backend(
                to_summarize,
                backend,
                history_path_prefix=history_path_prefix,
                chunk_index=summ_state["summaries_performed"],
            )
        except Exception:
            logger.exception("Failed to offload messages to backend")

    # Generate summary (inline — no external model call, to avoid async complexity)
    # We create a condensed version of the old messages
    summary_text = format_messages_for_summary(to_summarize)

    # Truncate the summary itself if it's too long
    max_summary_tokens = int(context_window * 0.15)  # Use at most 15% for summary
    max_summary_chars = max_summary_tokens * NUM_CHARS_PER_TOKEN
    if len(summary_text) > max_summary_chars:
        summary_text = summary_text[:max_summary_chars] + "\n\n... (earlier context truncated)"

    # Replace old messages with summary
    summary_content = create_summary_content(summary_text)
    llm_request.contents = [summary_content] + list(to_keep)

    # Update state
    summarized_tokens = count_messages_tokens(to_summarize)
    summ_state["summaries_performed"] += 1
    summ_state["total_tokens_summarized"] += summarized_tokens
    summ_state["last_summary"] = summary_text[:500]  # Keep preview in state

    logger.info(
        "Summarized %d messages (%d tokens). Kept %d recent messages.",
        len(to_summarize),
        summarized_tokens,
        len(to_keep),
    )

    return True
