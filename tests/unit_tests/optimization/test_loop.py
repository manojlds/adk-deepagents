"""Unit tests for optimization loop report helpers."""

from __future__ import annotations

from adk_deepagents.optimization.loop import format_optimization_report, run_optimize_loop


def test_run_optimize_loop_counts_linked_feedback(tmp_path) -> None:
    trajectories_path = tmp_path / "trajectories.jsonl"
    feedback_path = tmp_path / "feedback.jsonl"

    trajectories_path.write_text(
        '{"trace_id":"t1","events":[]}\n{"trace_id":"t2","events":[]}\n',
        encoding="utf-8",
    )
    feedback_path.write_text(
        '{"feedback_id":"fb1","source":"human","score":1.0,"trace_id":"t1"}\n'
        '{"feedback_id":"fb2","source":"auto","score":0.0,"trace_id":"nope"}\n',
        encoding="utf-8",
    )

    report = run_optimize_loop(
        trajectories_path=str(trajectories_path),
        feedback_path=str(feedback_path),
    )
    assert report.trajectories_total == 2
    assert report.feedback_total == 2
    assert report.linked_feedback == 1
    assert report.avg_score == 0.5


def test_format_optimization_report_contains_key_fields() -> None:
    trajectories_path = "missing-trajectories.jsonl"
    feedback_path = "missing-feedback.jsonl"
    report = run_optimize_loop(
        trajectories_path=trajectories_path,
        feedback_path=feedback_path,
    )
    text = format_optimization_report(report)
    assert "Optimization loop report" in text
    assert "trajectories:" in text
    assert "feedback records:" in text
