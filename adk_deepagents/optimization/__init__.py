"""Optimization primitives for adk-deepagents."""

from __future__ import annotations

from adk_deepagents.optimization.benchmark import (
    BenchmarkResult,
    BenchmarkRunner,
    LocalBenchmarkRunner,
    TaskResult,
    TaskSpec,
)
from adk_deepagents.optimization.evaluator import (
    EvaluationCriterion,
    EvaluationRubric,
    TrajectoryFilter,
    evaluate_trajectory,
    evaluate_trajectory_majority,
    filter_trajectories,
)
from adk_deepagents.optimization.gate import (
    GateConfig,
    GateResult,
    RegressionSuite,
    run_gate,
)
from adk_deepagents.optimization.history import (
    HistoryEntry,
    ScoreHistory,
)
from adk_deepagents.optimization.learnings import (
    LearningEntry,
    LearningsStore,
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
from adk_deepagents.optimization.tools import create_experience_tools
from adk_deepagents.optimization.trajectory import (
    AgentStep,
    FeedbackEntry,
    ModelCall,
    ToolCall,
    Trajectory,
)

__all__ = [
    "AgentStep",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BuiltAgent",
    "EvaluationCriterion",
    "EvaluationRubric",
    "FeedbackEntry",
    "GateConfig",
    "GateResult",
    "HistoryEntry",
    "ImprovementSuggestion",
    "LearningEntry",
    "LearningsStore",
    "LocalBenchmarkRunner",
    "ModelCall",
    "OptimizationCandidate",
    "OptimizationResult",
    "RegressionSuite",
    "ReplayConfig",
    "ReplayResult",
    "ScoreHistory",
    "TaskResult",
    "TaskSpec",
    "ToolCall",
    "Trajectory",
    "TrajectoryFilter",
    "TrajectoryStore",
    "create_experience_tools",
    "evaluate_trajectory",
    "evaluate_trajectory_majority",
    "filter_trajectories",
    "replay_trajectory",
    "run_gate",
    "run_optimization_loop",
]
