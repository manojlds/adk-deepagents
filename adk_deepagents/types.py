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


SubAgentSpec.__required_keys__ = frozenset({"name", "description"})


@dataclass
class SummarizationConfig:
    """Configuration for conversation summarization."""

    model: str = "gemini-2.5-flash"
    trigger: tuple[str, float] = ("fraction", 0.85)
    keep: tuple[str, int] = ("messages", 6)
    history_path_prefix: str = "/conversation_history"


@dataclass
class SkillsConfig:
    """Configuration passed to adk-skills SkillsRegistry."""

    # Placeholder — populated when adk-skills integration is wired up.
    extra: dict[str, Any] = field(default_factory=dict)
