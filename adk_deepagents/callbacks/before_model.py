"""Before-model callback — dynamic system prompt injection.

Replaces deepagents' ``wrap_model_call`` from all middleware classes.
Injects memory, skills, filesystem docs, and sub-agent docs into the
system instruction before each LLM call. Optionally triggers
conversation summarization.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

from adk_deepagents.backends.protocol import BackendFactory
from adk_deepagents.prompts import (
    COMPACT_CONVERSATION_SYSTEM_PROMPT,
    EXECUTION_SYSTEM_PROMPT,
    FILESYSTEM_SYSTEM_PROMPT,
    MEMORY_SYSTEM_PROMPT,
    TASK_CONCURRENCY_SYSTEM_PROMPT,
    TASK_RUNTIME_SUBAGENT_PROMPT,
    TASK_SYSTEM_PROMPT,
    TODO_SYSTEM_PROMPT,
)
from adk_deepagents.tools.compact import COMPACT_CONVERSATION_REQUEST_KEY
from adk_deepagents.types import DynamicTaskConfig, SummarizationConfig


def _append_to_system_instruction(llm_request: LlmRequest, text: str) -> None:
    """Append *text* to the system instruction in *llm_request*."""
    config = llm_request.config
    if config is None:
        llm_request.config = types.GenerateContentConfig(
            system_instruction=text,
        )
        return

    existing = config.system_instruction
    if existing is None:
        config.system_instruction = text
    elif isinstance(existing, str):
        config.system_instruction = existing + "\n\n" + text
    elif isinstance(existing, types.Content):
        # Append a new text part
        new_part = types.Part(text="\n\n" + text)
        if existing.parts:
            existing.parts.append(new_part)
        else:
            existing.parts = [new_part]
    else:
        # Fallback: replace with combined string
        config.system_instruction = str(existing) + "\n\n" + text


def _format_memory(memory_contents: dict[str, str], sources: list[str]) -> str:
    """Format memory contents for system prompt injection."""
    if not memory_contents:
        return MEMORY_SYSTEM_PROMPT.format(agent_memory="(No memory loaded)")
    sections = []
    for path in sources:
        content = memory_contents.get(path)
        if content:
            sections.append(f"### {path}\n{content}")
    return MEMORY_SYSTEM_PROMPT.format(agent_memory="\n\n".join(sections) or "(No memory loaded)")


def _format_subagent_docs(subagent_descriptions: list[dict[str, str]]) -> str:
    """Format sub-agent documentation for system prompt injection."""
    if not subagent_descriptions:
        return ""
    lines = [TASK_SYSTEM_PROMPT, "\n**Available sub-agents:**\n"]
    for desc in subagent_descriptions:
        lines.append(f"- **{desc['name']}**: {desc['description']}")
    return "\n".join(lines)


def _runtime_subagent_descriptions(state: Any) -> list[dict[str, str]]:
    """Read runtime-defined sub-agent descriptions from session state."""
    raw_specs = state.get("_dynamic_subagent_specs", {})
    if not isinstance(raw_specs, dict):
        return []

    descriptions: list[dict[str, str]] = []
    for raw_key, raw_spec in raw_specs.items():
        if not isinstance(raw_spec, dict):
            continue

        raw_name = raw_spec.get("name", raw_key)
        raw_description = raw_spec.get("description")
        if not isinstance(raw_name, str) or not isinstance(raw_description, str):
            continue

        name = raw_name.strip()
        description = raw_description.strip()
        if not name or not description:
            continue

        descriptions.append({"name": name, "description": description})

    return descriptions


def _format_dynamic_task_limits(config: DynamicTaskConfig) -> str:
    """Format dynamic task concurrency limits for prompt injection."""
    return TASK_CONCURRENCY_SYSTEM_PROMPT.format(
        max_parallel=config.max_parallel,
        concurrency_policy=config.concurrency_policy,
        queue_timeout_seconds=config.queue_timeout_seconds,
    )


def _inject_dangling_tool_responses(
    llm_request: LlmRequest,
    dangling: list[dict],
) -> None:
    """Inject synthetic function_response parts for dangling tool calls.

    For each dangling tool call (a function_call with no matching
    function_response), we find it in the conversation contents and
    insert a synthetic response immediately after the message that
    contains the call.

    This mirrors deepagents' ``PatchToolCallsMiddleware`` which creates
    synthetic ``ToolMessage`` objects for orphaned tool calls.
    """
    if not llm_request.contents:
        return

    dangling_ids = {d["id"]: d["name"] for d in dangling}

    # Scan existing contents for function_responses to avoid double-patching
    existing_response_ids: set[str] = set()
    for content in llm_request.contents:
        if content.parts:
            for part in content.parts:
                fr = getattr(part, "function_response", None)
                if fr is not None and getattr(fr, "id", None):
                    existing_response_ids.add(fr.id)

    # Remove already-resolved from dangling
    still_dangling = {
        cid: name for cid, name in dangling_ids.items() if cid not in existing_response_ids
    }
    if not still_dangling:
        return

    # Build patched contents list: for each model message with dangling calls,
    # insert a synthetic tool response content immediately after it
    patched: list[types.Content] = []
    for content in llm_request.contents:
        patched.append(content)
        if not content.parts:
            continue

        # Collect dangling call IDs from this message
        msg_dangling_ids = []
        for part in content.parts:
            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "id", None) in still_dangling:
                msg_dangling_ids.append((fc.id, still_dangling[fc.id]))

        # Insert synthetic responses
        for call_id, call_name in msg_dangling_ids:
            cancel_msg = (
                f"Tool call {call_name} with id {call_id} was cancelled — "
                "another message came in before it could be completed."
            )
            response_content = types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=call_id,
                            name=call_name,
                            response={"status": "cancelled", "message": cancel_msg},
                        )
                    )
                ],
            )
            patched.append(response_content)
            # Remove from still_dangling to avoid duplicate patching
            still_dangling.pop(call_id, None)

    llm_request.contents = patched


def _clear_state_key(state: Any, key: str) -> None:
    """Best-effort key clearing for ADK state wrappers.

    Some ADK state objects support ``get``/``__setitem__`` but not
    ``__delitem__``.
    """
    with suppress(Exception):
        del state[key]
        return

    with suppress(Exception):
        state[key] = None


def make_before_model_callback(
    *,
    memory_sources: list[str] | None = None,
    has_execution: bool = False,
    subagent_descriptions: list[dict[str, str]] | None = None,
    dynamic_task_config: DynamicTaskConfig | None = None,
    summarization_config: SummarizationConfig | None = None,
    backend_factory: BackendFactory | None = None,
) -> Callable:
    """Create a ``before_model_callback`` that injects dynamic system prompts.

    Parameters
    ----------
    memory_sources:
        List of AGENTS.md paths (loaded by before_agent_callback into state).
    has_execution:
        Whether execution tools (Heimdall/local) are available.
    subagent_descriptions:
        List of ``{"name": ..., "description": ...}`` for sub-agents.
    dynamic_task_config:
        Optional dynamic task concurrency configuration injected into
        the system prompt as delegation guidance.
    summarization_config:
        Optional config for context window summarization.
    backend_factory:
        Optional factory for creating backends (used by summarization
        for history offloading).
    """

    def before_model_callback(
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        state = callback_context.state

        force_compaction = bool(state.get(COMPACT_CONVERSATION_REQUEST_KEY))
        if COMPACT_CONVERSATION_REQUEST_KEY in state:
            _clear_state_key(state, COMPACT_CONVERSATION_REQUEST_KEY)

        # 0. Patch dangling tool calls into the LLM request contents
        dangling = state.get("_dangling_tool_calls")
        if "_dangling_tool_calls" in state:
            _clear_state_key(state, "_dangling_tool_calls")
        if dangling and llm_request.contents:
            _inject_dangling_tool_responses(llm_request, dangling)

        additions: list[str] = []

        # Todo tools documentation
        additions.append(TODO_SYSTEM_PROMPT)

        # Filesystem tools documentation
        additions.append(FILESYSTEM_SYSTEM_PROMPT)

        # Execution tools documentation
        if has_execution:
            additions.append(EXECUTION_SYSTEM_PROMPT)

        # Memory injection
        if memory_sources:
            memory_contents = state.get("memory_contents", {})
            additions.append(_format_memory(memory_contents, memory_sources))

        # Sub-agent documentation (configured + runtime-defined)
        merged_subagent_descriptions: list[dict[str, str]] = list(subagent_descriptions or [])
        known_names = {desc["name"] for desc in merged_subagent_descriptions}
        for runtime_desc in _runtime_subagent_descriptions(state):
            if runtime_desc["name"] in known_names:
                continue
            merged_subagent_descriptions.append(runtime_desc)
            known_names.add(runtime_desc["name"])

        if merged_subagent_descriptions:
            additions.append(_format_subagent_docs(merged_subagent_descriptions))

        if dynamic_task_config is not None:
            additions.append(TASK_RUNTIME_SUBAGENT_PROMPT)
            additions.append(_format_dynamic_task_limits(dynamic_task_config))

        if summarization_config:
            additions.append(COMPACT_CONVERSATION_SYSTEM_PROMPT)

        # Inject all additions into system instruction
        combined = "\n\n".join(additions)
        _append_to_system_instruction(llm_request, combined)

        # Summarization check
        if summarization_config:
            from adk_deepagents.summarization import maybe_summarize

            maybe_summarize(
                callback_context,
                llm_request,
                context_window=_resolve_context_window(summarization_config),
                trigger_fraction=_resolve_trigger_fraction(summarization_config),
                keep_messages=_resolve_keep_messages(summarization_config),
                backend_factory=backend_factory,
                history_path_prefix=summarization_config.history_path_prefix,
                use_llm_summary=summarization_config.use_llm_summary,
                summary_model=summarization_config.model,
                truncate_args_config=summarization_config.truncate_args,
                force=force_compaction,
            )

        return None  # Proceed with LLM call

    return before_model_callback


MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    "gemini-1.5-flash": 1_048_576,
    "gemini-1.5-pro": 2_097_152,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-3.5-sonnet": 200_000,
}
"""Known model context window sizes in tokens."""


def _resolve_context_window(config: SummarizationConfig) -> int:
    """Resolve context window size from config.

    Resolution order:
    1. Explicit ``config.context_window`` (if set)
    2. Model-name lookup in ``MODEL_CONTEXT_WINDOWS``
    3. ``DEFAULT_CONTEXT_WINDOW`` fallback
    """
    from adk_deepagents.summarization import DEFAULT_CONTEXT_WINDOW

    if config.context_window is not None:
        return config.context_window

    return MODEL_CONTEXT_WINDOWS.get(config.model, DEFAULT_CONTEXT_WINDOW)


def _resolve_trigger_fraction(config: SummarizationConfig) -> float:
    """Resolve trigger fraction from config."""
    kind, value = config.trigger
    if kind == "fraction":
        return float(value)
    return 0.85  # default


def _resolve_keep_messages(config: SummarizationConfig) -> int:
    """Resolve keep messages count from config."""
    kind, value = config.keep
    if kind == "messages":
        return int(value)
    return 6  # default
