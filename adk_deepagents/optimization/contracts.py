"""Contract models for optimization datasets, evaluation, and loop state.

These models define a stable schema boundary between:
- trajectory ingestion,
- human/automatic feedback,
- golden session replay evaluation,
- and iterative optimizer bookkeeping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Split = Literal["train", "val", "test"]
FeedbackSource = Literal["human", "auto", "judge"]
OutcomeStatus = Literal["success", "partial", "failure"]
Decision = Literal["accepted", "rejected"]


@dataclass(slots=True)
class TrajectoryEvent:
    """One normalized event inside a trajectory bundle."""

    kind: str
    text: str | None = None
    tool: str | None = None
    ok: bool | None = None
    decision: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskOutcome:
    """Top-level outcome signals for a full trajectory bundle."""

    status: OutcomeStatus
    quality_score: float | None = None
    latency_ms: int | None = None
    token_input: int | None = None
    token_output: int | None = None
    cost_usd: float | None = None


@dataclass(slots=True)
class DerivedSignals:
    """Implicit behavioral signals extracted from events."""

    hitl_reject_count: int = 0
    tool_error_count: int = 0
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrajectoryBundle:
    """Optimization unit representing one thread/session execution story."""

    bundle_id: str
    session_id: str
    user_id: str | None = None
    agent_name: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    task_prompt: str | None = None
    traces: list[str] = field(default_factory=list)
    events: list[TrajectoryEvent] = field(default_factory=list)
    outcome: TaskOutcome | None = None
    derived_signals: DerivedSignals = field(default_factory=DerivedSignals)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OptimizationFeedback:
    """Explicit feedback linked to bundle/trace/span."""

    feedback_id: str
    timestamp: str
    source: FeedbackSource
    bundle_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    score: float | None = None
    label: str | None = None
    rationale: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GoldenTaskInput:
    """Task input and execution context for replay."""

    prompt: str
    workspace_snapshot_ref: str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HitlCheckpoint:
    """Scripted HITL checkpoint behavior during replay."""

    checkpoint: str
    decision: Literal["approve", "reject"]
    reason: str | None = None


@dataclass(slots=True)
class GoldenExpectations:
    """Hard and soft constraints for replay scoring."""

    must_pass_checks: list[str] = field(default_factory=list)
    must_not: list[str] = field(default_factory=list)
    soft_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ScorerConfig:
    """Scorer strategy and metric weights."""

    type: str
    weights: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class GoldenSessionSpec:
    """Replayable gold example for train/val/test optimization."""

    golden_id: str
    split: Split
    task_input: GoldenTaskInput
    hitl_script: list[HitlCheckpoint] = field(default_factory=list)
    expectations: GoldenExpectations = field(default_factory=GoldenExpectations)
    scorer: ScorerConfig = field(default_factory=lambda: ScorerConfig(type="rule_plus_llm_judge"))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvaluationResult:
    """Evaluation output for one candidate on one golden session."""

    candidate_id: str
    golden_id: str
    bundle_id: str
    metrics: dict[str, float]
    overall_score: float
    asi: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OptimizationIterationRecord:
    """One optimization-loop iteration and selection decision."""

    iteration: int
    base_candidate_id: str
    proposed_candidates: list[str]
    selected_candidate: str | None
    decision: Decision
    reason: str
    train_summary: dict[str, Any] = field(default_factory=dict)
    val_summary: dict[str, Any] = field(default_factory=dict)
    test_gate: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
