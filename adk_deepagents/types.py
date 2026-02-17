"""Shared type definitions for adk-deepagents."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypedDict


class SubAgentSpec(TypedDict, total=False):
    """Specification for a sub-agent.

    Mirrors deepagents' ``SubAgent`` TypedDict.
    """

    name: str  # required
    description: str  # required
    system_prompt: str
    tools: Sequence[Callable]
    model: str
    skills: list[str]
    interrupt_on: dict[str, bool]


SubAgentSpec.__required_keys__ = frozenset({"name", "description"})


@dataclass
class TruncateArgsConfig:
    """Settings for truncating large tool arguments in older messages.

    Ported from deepagents ``TruncateArgsSettings``.  Before summarization
    is triggered, arguments to ``write_file`` and ``edit_file`` calls in
    older messages are replaced with a truncated preview.  This frees
    context window space without losing the record of *which* tool was
    called.
    """

    trigger: tuple[str, float | int] | None = None
    """When to start truncating.  Same format as ``SummarizationConfig.trigger``."""

    keep: tuple[str, int | float] = ("messages", 20)
    """How many recent messages to leave untouched."""

    max_length: int = 2000
    """Maximum character length for a single tool argument before truncation."""

    truncation_text: str = "...(argument truncated)"
    """Replacement text appended after the 20-char prefix of a truncated arg."""


@dataclass
class SummarizationConfig:
    """Configuration for conversation summarization."""

    model: str = "gemini-2.5-flash"
    trigger: tuple[str, float] = ("fraction", 0.85)
    keep: tuple[str, int] = ("messages", 6)
    history_path_prefix: str = "/conversation_history"
    use_llm_summary: bool = True
    """If True (default), use the configured model to generate summaries.
    If False, fall back to inline text truncation (faster, no extra API call)."""
    truncate_args: TruncateArgsConfig | None = None
    """Optional settings for truncating large tool arguments in older messages."""
    context_window: int | None = None
    """Explicit context window size in tokens. If set, overrides the model-based
    lookup. If None, the context window is resolved from the model name."""


@dataclass
class SkillsConfig:
    """Configuration passed to adk-skills SkillsRegistry."""

    # Placeholder — populated when adk-skills integration is wired up.
    extra: dict[str, Any] = field(default_factory=dict)
