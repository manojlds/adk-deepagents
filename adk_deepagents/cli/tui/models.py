"""Data models for the TUI — conversation records and agent profiles."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Conversation data model — tracks messages for export & future undo/redo
# ---------------------------------------------------------------------------

MessageRole = Literal[
    "user",
    "assistant",
    "system",
    "tool_call",
    "tool_result",
    "diff",
    "approval",
    "error",
    "queued",
]


@dataclass
class MessageRecord:
    """Immutable record of a single conversation message or event.

    ``MessageDisplay`` remains the rendering engine; this model backs it
    with serialisable data so we can export, search, and (in future phases)
    undo/redo.
    """

    role: MessageRole
    text: str
    timestamp: float = field(default_factory=time.time)
    tool_name: str | None = None
    # For assistant streaming: final accumulated text, not deltas.
    # For diffs: the full unified diff text.
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ConversationLog:
    """Ordered list of message records for the current session."""

    records: list[MessageRecord] = field(default_factory=list)

    def append(self, record: MessageRecord) -> None:
        self.records.append(record)

    def clear(self) -> None:
        self.records.clear()

    def to_markdown(self) -> str:
        """Serialise the conversation to Markdown suitable for export."""
        lines: list[str] = []
        for record in self.records:
            if record.role == "user":
                lines.append(f"**User:** {record.text}")
                lines.append("")
            elif record.role == "assistant":
                lines.append(f"**Assistant:** {record.text}")
                lines.append("")
            elif record.role == "system":
                lines.append(f"*System: {record.text}*")
                lines.append("")
            elif record.role == "error":
                lines.append(f"*Error: {record.text}*")
                lines.append("")
            elif record.role == "tool_call":
                name = record.tool_name or "tool"
                detail = f" — {record.text}" if record.text else ""
                lines.append(f"$ **{name}**{detail}")
                lines.append("")
            elif record.role == "tool_result":
                name = record.tool_name or "tool"
                detail = f" {record.text}" if record.text else ""
                lines.append(f"  -> {name}{detail}")
                lines.append("")
            elif record.role == "diff":
                lines.append("```diff")
                lines.append(record.text)
                lines.append("```")
                lines.append("")
            elif record.role == "queued":
                lines.append(f"*[queued] {record.text}*")
                lines.append("")
            elif record.role == "approval":
                lines.append(f"*Approval: {record.text}*")
                lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent profiles — config-based agent definitions
# ---------------------------------------------------------------------------

AgentMode = Literal["primary", "subagent"]


@dataclass
class AgentProfile:
    """A named agent configuration.

    Mirrors OpenCode's agent config: each profile has a name, model
    override, custom prompt/instruction, tool permissions, and a mode
    (primary agents can be cycled with Tab; subagents are invoked via
    ``@name``).
    """

    name: str
    description: str = ""
    mode: AgentMode = "primary"
    model: str | None = None
    prompt: str | None = None
    color: str | None = None
    # When True, the agent is excluded from pickers/cycling.
    hidden: bool = False


# Built-in agent profiles analogous to OpenCode's Build / Plan agents.
BUILTIN_AGENTS: list[AgentProfile] = [
    AgentProfile(
        name="build",
        description="Full tool access — build, edit, execute.",
        mode="primary",
        color="#a6e3a1",
    ),
    AgentProfile(
        name="plan",
        description="Read-only analysis and planning.",
        mode="primary",
        prompt=(
            "You are a planning assistant.  Analyse the codebase and create "
            "detailed plans, but do NOT make changes or run commands.  Focus on "
            "architecture, trade-offs, and step-by-step implementation outlines."
        ),
        color="#89b4fa",
    ),
]

DEFAULT_AGENT_NAME = "build"


@dataclass
class AgentRegistry:
    """Registry of available agent profiles."""

    profiles: list[AgentProfile] = field(default_factory=lambda: list(BUILTIN_AGENTS))
    _by_name: dict[str, AgentProfile] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._by_name = {p.name: p for p in self.profiles}

    def get(self, name: str) -> AgentProfile | None:
        return self._by_name.get(name)

    def primary_agents(self) -> list[AgentProfile]:
        """Return visible primary agents in order."""
        return [p for p in self.profiles if p.mode == "primary" and not p.hidden]

    def subagents(self) -> list[AgentProfile]:
        """Return visible subagent profiles in order."""
        return [p for p in self.profiles if p.mode == "subagent" and not p.hidden]

    def all_visible(self) -> list[AgentProfile]:
        """Return all visible agents regardless of mode."""
        return [p for p in self.profiles if not p.hidden]

    def add(self, profile: AgentProfile) -> None:
        """Add or replace a profile."""
        self._by_name[profile.name] = profile
        # Replace in-place if exists, otherwise append.
        for i, p in enumerate(self.profiles):
            if p.name == profile.name:
                self.profiles[i] = profile
                return
        self.profiles.append(profile)

    def cycle_next(self, current_name: str) -> AgentProfile | None:
        """Return the next primary agent after *current_name*, wrapping."""
        primaries = self.primary_agents()
        if len(primaries) <= 1:
            return None
        for i, p in enumerate(primaries):
            if p.name == current_name:
                return primaries[(i + 1) % len(primaries)]
        return primaries[0] if primaries else None

    def cycle_prev(self, current_name: str) -> AgentProfile | None:
        """Return the previous primary agent before *current_name*, wrapping."""
        primaries = self.primary_agents()
        if len(primaries) <= 1:
            return None
        for i, p in enumerate(primaries):
            if p.name == current_name:
                return primaries[(i - 1) % len(primaries)]
        return primaries[-1] if primaries else None
