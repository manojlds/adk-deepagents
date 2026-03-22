"""Trajectory and feedback primitives for optimization workflows.

This package provides OTEL-first trajectory ingestion so ADK spans can be
collected externally (OTLP collector) and imported for offline optimization.
"""

from adk_deepagents.optimization.contracts import (
    DerivedSignals,
    EvaluationResult,
    GoldenExpectations,
    GoldenSessionSpec,
    GoldenTaskInput,
    HitlCheckpoint,
    OptimizationFeedback,
    OptimizationIterationRecord,
    ScorerConfig,
    TaskOutcome,
    TrajectoryBundle,
)
from adk_deepagents.optimization.contracts import (
    TrajectoryEvent as BundleEvent,
)
from adk_deepagents.optimization.feedback import (
    append_feedback_jsonl,
    load_feedback_jsonl,
)
from adk_deepagents.optimization.loop import (
    OptimizationReport,
    format_optimization_report,
    run_optimize_loop,
)
from adk_deepagents.optimization.models import FeedbackRecord, Trajectory, TrajectoryEvent
from adk_deepagents.optimization.otel import (
    import_otel_traces,
    load_otel_json,
    load_trajectories_jsonl,
    save_trajectories_jsonl,
)

__all__ = [
    "FeedbackRecord",
    "DerivedSignals",
    "EvaluationResult",
    "GoldenExpectations",
    "GoldenSessionSpec",
    "GoldenTaskInput",
    "HitlCheckpoint",
    "OptimizationReport",
    "OptimizationFeedback",
    "OptimizationIterationRecord",
    "ScorerConfig",
    "TaskOutcome",
    "Trajectory",
    "BundleEvent",
    "TrajectoryBundle",
    "TrajectoryEvent",
    "append_feedback_jsonl",
    "format_optimization_report",
    "import_otel_traces",
    "load_feedback_jsonl",
    "load_otel_json",
    "load_trajectories_jsonl",
    "run_optimize_loop",
    "save_trajectories_jsonl",
]
