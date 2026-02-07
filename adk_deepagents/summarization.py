"""Conversation summarization — context window management.

Implements token counting, message partitioning, LLM-based summary
generation, argument truncation, and history offloading to prevent
context window overflow during long conversations.

Integrates with ``before_model_callback`` via ``maybe_summarize()``.

Ported from deepagents.middleware.summarization with full feature parity:
- LLM-based summary generation with structured prompt
- Tool argument truncation for older messages (TruncateArgsConfig)
- Append-based history offloading with timestamps
- Inline text fallback when LLM summary is disabled
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from google.genai import types

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models import LlmRequest

    from adk_deepagents.backends.protocol import Backend, BackendFactory
    from adk_deepagents.types import TruncateArgsConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CHARS_PER_TOKEN = 4
"""Rough heuristic: ~4 characters per token for English text."""

DEFAULT_CONTEXT_WINDOW = 200_000
"""Default context window size in tokens (Gemini 2.5 Flash)."""

# Tool names whose arguments are eligible for truncation
TRUNCATABLE_TOOLS = frozenset({"write_file", "edit_file"})

# ---------------------------------------------------------------------------
# Summary prompt (structured, matching deepagents)
# ---------------------------------------------------------------------------

LLM_SUMMARY_PROMPT = """\
<role>
Context Extraction Assistant
</role>

<primary_objective>
Your sole objective in this task is to extract the highest quality/most \
relevant context from the conversation history below.
</primary_objective>

<objective_information>
You're nearing the total number of input tokens you can accept, so you \
must extract the highest quality/most relevant pieces of information from \
your conversation history.
This context will then overwrite the conversation history presented below. \
Because of this, ensure the context you extract is only the most important \
information to continue working toward your overall goal.
</objective_information>

<instructions>
The conversation history below will be replaced with the context you \
extract in this step.
You want to ensure that you don't repeat any actions you've already \
completed, so the context you extract from the conversation history should \
be focused on the most important information to your overall goal.

You should structure your summary using the following sections. Each \
section acts as a checklist - you must populate it with relevant \
information or explicitly state "None" if there is nothing to report \
for that section:

## SESSION INTENT
What is the user's primary goal or request? What overall task are you \
trying to accomplish? This should be concise but complete enough to \
understand the purpose of the entire session.

## SUMMARY
Extract and record all of the most important context from the \
conversation history. Include important choices, conclusions, or \
strategies determined during this conversation. Include the reasoning \
behind key decisions. Document any rejected options and why they were \
not pursued.

## ARTIFACTS
What artifacts, files, or resources were created, modified, or accessed \
during this conversation? For file modifications, list specific file \
paths and briefly describe the changes made to each. This section \
prevents silent loss of artifact information.

## NEXT STEPS
What specific tasks remain to be completed to achieve the session intent? \
What should you do next?

</instructions>

With all of this in mind, please carefully read over the entire \
conversation history, and extract the most important and relevant \
context to replace it so that you can free up space in the conversation \
history.
Respond ONLY with the extracted context. Do not include any additional \
information, or text before or after the extracted context.

<messages>
Messages to summarize:
{messages}
</messages>"""

# Fallback prompt used when no LLM summary is generated
INLINE_SUMMARY_PROMPT = """\
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


def create_summary_content(
    summary_text: str,
    offload_path: str | None = None,
) -> types.Content:
    """Wrap a summary string in a ``Content`` message with role ``user``.

    Parameters
    ----------
    summary_text:
        The summary text.
    offload_path:
        If provided, includes a reference to where the full conversation
        history was saved.
    """
    if offload_path:
        text = (
            "You are in the middle of a conversation that has been summarized.\n\n"
            f"The full conversation history has been saved to {offload_path} "
            "should you need to refer back to it for details.\n\n"
            "A condensed summary follows:\n\n"
            "<summary>\n"
            f"{summary_text}\n"
            "</summary>"
        )
    else:
        text = (
            "<conversation_summary>\n"
            "The following is a summary of the earlier conversation:\n\n"
            f"{summary_text}\n"
            "</conversation_summary>"
        )
    return types.Content(
        role="user",
        parts=[types.Part(text=text)],
    )


def generate_llm_summary(
    messages: list[types.Content],
    model: str = "gemini-2.5-flash",
    max_input_tokens: int = 4000,
) -> str | None:
    """Generate a summary of messages using an LLM call.

    Calls the configured model with the structured summary prompt.
    Returns ``None`` on error (caller should fall back to inline summary).

    Parameters
    ----------
    messages:
        Messages to summarize.
    model:
        Model name to use for the summarization call.
    max_input_tokens:
        Maximum tokens of conversation text to include in the prompt.
    """
    try:
        from google import genai

        formatted = format_messages_for_summary(messages)

        # Trim to max_input_tokens to avoid exceeding the summary model's limit
        max_chars = max_input_tokens * NUM_CHARS_PER_TOKEN
        if len(formatted) > max_chars:
            # Keep last portion (most recent context is usually most important)
            formatted = formatted[-max_chars:]

        prompt = LLM_SUMMARY_PROMPT.format(messages=formatted)

        client = genai.Client()
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        if response.text:
            return response.text.strip()
        return None
    except Exception:
        logger.exception("LLM summary generation failed, falling back to inline")
        return None


# ---------------------------------------------------------------------------
# Tool argument truncation
# ---------------------------------------------------------------------------


def truncate_tool_args(
    messages: list[types.Content],
    config: TruncateArgsConfig,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> tuple[list[types.Content], bool]:
    """Truncate large tool arguments in older messages.

    Ported from deepagents ``SummarizationMiddleware._truncate_args()``.
    Only truncates arguments to ``write_file`` and ``edit_file`` calls
    in messages older than the ``keep`` threshold.

    Parameters
    ----------
    messages:
        All conversation messages.
    config:
        Truncation settings.
    context_window:
        Total context window for fraction-based triggers.

    Returns
    -------
    tuple[list[Content], bool]
        ``(possibly_modified_messages, was_modified)``
    """
    if config.trigger is None:
        return messages, False

    # Check trigger
    trigger_kind, trigger_value = config.trigger
    total_tokens = count_messages_tokens(messages)

    should_truncate = False
    if trigger_kind == "messages":
        should_truncate = len(messages) >= int(trigger_value)
    elif trigger_kind == "tokens":
        should_truncate = total_tokens >= int(trigger_value)
    elif trigger_kind == "fraction":
        threshold = int(context_window * float(trigger_value))
        should_truncate = total_tokens >= threshold

    if not should_truncate:
        return messages, False

    # Determine cutoff index (messages before this are candidates)
    keep_kind, keep_value = config.keep
    if keep_kind == "messages":
        keep_n = int(keep_value)
        if len(messages) <= keep_n:
            return messages, False
        cutoff = len(messages) - keep_n
    elif keep_kind == "fraction":
        target_tokens = int(context_window * float(keep_value))
        # Walk backwards to find cutoff
        tokens_kept = 0
        cutoff = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            msg_tokens = count_content_tokens(messages[i])
            if tokens_kept + msg_tokens > target_tokens:
                cutoff = i + 1
                break
            tokens_kept += msg_tokens
        else:
            cutoff = 0
    else:
        return messages, False

    if cutoff <= 0:
        return messages, False

    # Truncate tool arguments in older messages
    modified = False
    result: list[types.Content] = []
    max_len = config.max_length
    trunc_text = config.truncation_text

    for i, msg in enumerate(messages):
        if i >= cutoff or not msg.parts:
            result.append(msg)
            continue

        new_parts = []
        msg_modified = False
        for part in msg.parts:
            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "name", "") in TRUNCATABLE_TOOLS:
                args = fc.args or {}
                new_args = {}
                arg_modified = False
                for key, value in args.items():
                    if isinstance(value, str) and len(value) > max_len:
                        new_args[key] = value[:20] + trunc_text
                        arg_modified = True
                    else:
                        new_args[key] = value
                if arg_modified:
                    new_part = types.Part(
                        function_call=types.FunctionCall(
                            id=getattr(fc, "id", None),
                            name=fc.name,
                            args=new_args,
                        )
                    )
                    new_parts.append(new_part)
                    msg_modified = True
                else:
                    new_parts.append(part)
            else:
                new_parts.append(part)

        if msg_modified:
            modified = True
            result.append(types.Content(role=msg.role, parts=new_parts))
        else:
            result.append(msg)

    return result, modified


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

    Uses an append-based running log with timestamps (matching deepagents'
    approach). Each summarization event appends a new section to the file.

    Returns the path where messages were saved.
    """
    formatted = format_messages_for_summary(messages)
    timestamp = datetime.now(UTC).isoformat()
    new_section = f"## Summarized at {timestamp}\n\n{formatted}\n\n"

    path = f"{history_path_prefix}/session_history.md"

    # Try to read existing content and append
    existing_content = ""
    try:
        responses = backend.download_files([path])
        if responses and responses[0].content is not None:
            content = responses[0].content
            if isinstance(content, bytes):
                existing_content = content.decode("utf-8")
            elif isinstance(content, str):
                existing_content = content
    except Exception:
        logger.debug("No existing history at %s, creating new file", path)

    combined = existing_content + new_section

    try:
        if existing_content:
            result = backend.edit(path, existing_content, combined)
            if result.error:
                # edit failed, try writing fresh
                backend.write(path, combined)
        else:
            backend.write(path, combined)
    except Exception:
        # Last resort: write to a chunk-indexed file
        fallback_path = f"{history_path_prefix}/chunk_{chunk_index:04d}.txt"
        try:
            backend.write(fallback_path, new_section)
            return fallback_path
        except Exception:
            logger.exception("Failed to offload conversation history")

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
    use_llm_summary: bool = True,
    summary_model: str = "gemini-2.5-flash",
    truncate_args_config: TruncateArgsConfig | None = None,
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
    use_llm_summary:
        If True, use an LLM call to generate intelligent summaries.
    summary_model:
        Model to use for LLM-based summary generation.
    truncate_args_config:
        Optional config for truncating large tool arguments.

    Returns
    -------
    bool
        ``True`` if summarization was performed.
    """
    contents = llm_request.contents
    if not contents:
        return False

    state = callback_context.state

    # Step 0: Truncate tool arguments in older messages (if configured)
    args_were_truncated = False
    if truncate_args_config is not None:
        contents, args_were_truncated = truncate_tool_args(
            contents, truncate_args_config, context_window
        )
        if args_were_truncated:
            llm_request.contents = contents

    # Step 1: Count current tokens and check threshold
    current_tokens = count_messages_tokens(contents)
    trigger_threshold = int(context_window * trigger_fraction)

    if current_tokens < trigger_threshold:
        return args_were_truncated  # Args may have been truncated even if no summary

    logger.info(
        "Summarization triggered: %d tokens exceeds threshold %d (%.0f%% of %d)",
        current_tokens,
        trigger_threshold,
        trigger_fraction * 100,
        context_window,
    )

    # Step 2: Partition messages
    to_summarize, to_keep = partition_messages(contents, keep_count=keep_messages)
    if not to_summarize:
        return args_were_truncated

    # Step 3: Initialize summarization state
    summ_state = state.get("_summarization_state")
    if summ_state is None:
        summ_state = {"summaries_performed": 0, "total_tokens_summarized": 0}
        state["_summarization_state"] = summ_state

    # Step 4: Offload old messages to backend (for reference)
    offload_path: str | None = None
    if backend_factory:
        try:
            backend = backend_factory(state)
            offload_path = offload_messages_to_backend(
                to_summarize,
                backend,
                history_path_prefix=history_path_prefix,
                chunk_index=summ_state["summaries_performed"],
            )
        except Exception:
            logger.exception("Failed to offload messages to backend")

    # Step 5: Generate summary
    summary_text: str | None = None

    if use_llm_summary:
        summary_text = generate_llm_summary(
            to_summarize,
            model=summary_model,
            max_input_tokens=4000,
        )

    if summary_text is None:
        # Fallback: inline text summary (no LLM call)
        summary_text = format_messages_for_summary(to_summarize)
        # Truncate the summary itself if it's too long
        max_summary_chars = int(context_window * 0.15) * NUM_CHARS_PER_TOKEN
        if len(summary_text) > max_summary_chars:
            summary_text = summary_text[:max_summary_chars] + "\n\n... (earlier context truncated)"

    # Step 6: Replace old messages with summary
    summary_content = create_summary_content(summary_text, offload_path=offload_path)
    llm_request.contents = [summary_content] + list(to_keep)

    # Step 7: Update state
    summarized_tokens = count_messages_tokens(to_summarize)
    summ_state["summaries_performed"] += 1
    summ_state["total_tokens_summarized"] += summarized_tokens
    summ_state["last_summary"] = summary_text[:500]  # Keep preview in state

    logger.info(
        "Summarized %d messages (%d tokens) using %s. Kept %d recent messages.",
        len(to_summarize),
        summarized_tokens,
        "LLM" if use_llm_summary and summary_text else "inline",
        len(to_keep),
    )

    return True
