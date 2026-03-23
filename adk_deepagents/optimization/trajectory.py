"""Trajectory data types for agent execution tracing and optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single tool invocation within a trajectory."""

    name: str
    args: dict[str, Any]
    response: Any
    duration_ms: float
    error: str | None = None


@dataclass
class ModelCall:
    """A single LLM call within a trajectory."""

    model: str
    input_tokens: int
    output_tokens: int
    duration_ms: float
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    finish_reason: str | None = None


@dataclass
class AgentStep:
    """One agent turn: a model call followed by zero or more tool calls."""

    agent_name: str
    model_call: ModelCall | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class FeedbackEntry:
    """User or automated feedback on a trajectory."""

    source: str  # "user", "auto", "evaluator"
    rating: float | None = None  # 0.0 to 1.0
    comment: str = ""
    timestamp_ns: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """A complete agent execution trajectory reconstructed from OTEL spans."""

    trace_id: str
    session_id: str | None = None
    agent_name: str | None = None
    steps: list[AgentStep] = field(default_factory=list)
    start_time_ns: int = 0
    end_time_ns: int = 0
    status: str = "unset"  # "unset", "ok", "error"

    # Optimization metadata
    score: float | None = None
    is_golden: bool = False
    feedback: list[FeedbackEntry] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end_time_ns and self.start_time_ns:
            return (self.end_time_ns - self.start_time_ns) / 1_000_000
        return 0.0

    @property
    def total_input_tokens(self) -> int:
        return sum(s.model_call.input_tokens for s in self.steps if s.model_call is not None)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.model_call.output_tokens for s in self.steps if s.model_call is not None)
