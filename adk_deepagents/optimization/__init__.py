"""Optimization primitives for adk-deepagents."""

from __future__ import annotations

from adk_deepagents.optimization.evaluator import (
    EvaluationCriterion,
    EvaluationRubric,
    evaluate_trajectory,
)
from adk_deepagents.optimization.loop import (
    ImprovementSuggestion,
    OptimizationCandidate,
    OptimizationResult,
    run_optimization_loop,
)
from adk_deepagents.optimization.replay import (
    BuiltAgent,
    ReplayConfig,
    ReplayResult,
    replay_trajectory,
)
from adk_deepagents.optimization.store import TrajectoryStore
from adk_deepagents.optimization.trajectory import (
    AgentStep,
    FeedbackEntry,
    ModelCall,
    ToolCall,
    Trajectory,
)

__all__ = [
    "AgentStep",
    "BuiltAgent",
    "EvaluationCriterion",
    "EvaluationRubric",
    "FeedbackEntry",
    "ImprovementSuggestion",
    "ModelCall",
    "OptimizationCandidate",
    "OptimizationResult",
    "ReplayConfig",
    "ReplayResult",
    "ToolCall",
    "Trajectory",
    "TrajectoryStore",
    "evaluate_trajectory",
    "replay_trajectory",
    "run_optimization_loop",
]
