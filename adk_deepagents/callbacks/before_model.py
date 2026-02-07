"""Before-model callback — dynamic system prompt injection.

Replaces deepagents' ``wrap_model_call`` from all middleware classes.
Injects memory, skills, filesystem docs, and sub-agent docs into the
system instruction before each LLM call. Optionally triggers
conversation summarization.
"""

from __future__ import annotations

from collections.abc import Callable

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

from adk_deepagents.backends.protocol import BackendFactory
from adk_deepagents.prompts import (
    EXECUTION_SYSTEM_PROMPT,
    FILESYSTEM_SYSTEM_PROMPT,
    MEMORY_SYSTEM_PROMPT,
    TASK_SYSTEM_PROMPT,
    TODO_SYSTEM_PROMPT,
)
from adk_deepagents.types import SummarizationConfig


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


def make_before_model_callback(
    *,
    memory_sources: list[str] | None = None,
    has_execution: bool = False,
    subagent_descriptions: list[dict[str, str]] | None = None,
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

        # Sub-agent documentation
        if subagent_descriptions:
            additions.append(_format_subagent_docs(subagent_descriptions))

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
            )

        return None  # Proceed with LLM call

    return before_model_callback


def _resolve_context_window(config: SummarizationConfig) -> int:
    """Resolve context window size from config."""
    from adk_deepagents.summarization import DEFAULT_CONTEXT_WINDOW

    return DEFAULT_CONTEXT_WINDOW


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
