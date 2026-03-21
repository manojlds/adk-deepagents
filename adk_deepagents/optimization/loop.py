"""Simple optimization loop primitives over trajectories + feedback."""

from __future__ import annotations

from dataclasses import dataclass

from adk_deepagents.optimization.feedback import load_feedback_jsonl
from adk_deepagents.optimization.otel import load_trajectories_jsonl


@dataclass(slots=True)
class OptimizationReport:
    """Aggregate stats used to drive iterative optimizer improvements."""

    trajectories_total: int
    feedback_total: int
    linked_feedback: int
    avg_score: float | None


def run_optimize_loop(*, trajectories_path: str, feedback_path: str) -> OptimizationReport:
    """Compute a first-pass optimization report from stored artifacts."""
    trajectories = load_trajectories_jsonl(trajectories_path)
    feedback = load_feedback_jsonl(feedback_path)

    known_trace_ids = {trajectory.trace_id for trajectory in trajectories}
    linked = [
        item for item in feedback if item.trace_id is not None and item.trace_id in known_trace_ids
    ]

    scored = [item.score for item in feedback if item.score is not None]
    avg_score = (sum(scored) / len(scored)) if scored else None

    return OptimizationReport(
        trajectories_total=len(trajectories),
        feedback_total=len(feedback),
        linked_feedback=len(linked),
        avg_score=avg_score,
    )


def format_optimization_report(report: OptimizationReport) -> str:
    """Render a concise human-readable report."""
    score_text = f"{report.avg_score:.3f}" if report.avg_score is not None else "n/a"
    return (
        "Optimization loop report:\n"
        f"- trajectories: {report.trajectories_total}\n"
        f"- feedback records: {report.feedback_total}\n"
        f"- linked feedback: {report.linked_feedback}\n"
        f"- average score: {score_text}"
    )
