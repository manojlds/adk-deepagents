"""Core data models for optimization trajectories and feedback."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

EventKind = Literal[
    "agent_turn",
    "model_call",
    "model_response",
    "tool_call",
    "tool_result",
    "delegation",
    "approval",
    "feedback",
    "unknown",
]


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


@dataclass(slots=True)
class TrajectoryEvent:
    """Normalized event derived from one OTEL span."""

    kind: EventKind
    timestamp: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    name: str | None = None
    agent_name: str | None = None
    session_id: str | None = None
    tool_name: str | None = None
    model: str | None = None
    status: str | None = None
    input_text: str | None = None
    output_text: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Trajectory:
    """A trace-level trajectory for optimization and evaluation."""

    trace_id: str
    started_at: str | None = None
    ended_at: str | None = None
    root_span_name: str | None = None
    session_id: str | None = None
    agent_name: str | None = None
    events: list[TrajectoryEvent] = field(default_factory=list)


@dataclass(slots=True)
class FeedbackRecord:
    """Human or automated feedback linked to a trajectory/span."""

    feedback_id: str
    timestamp: str = field(default_factory=utc_now_iso)
    source: Literal["human", "auto", "judge"] = "human"
    score: float | None = None
    label: str | None = None
    rationale: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
