"""Unit tests for optimization contract dataclasses."""

from __future__ import annotations

from adk_deepagents.optimization.contracts import (
    EvaluationResult,
    GoldenSessionSpec,
    GoldenTaskInput,
    HitlCheckpoint,
    OptimizationFeedback,
    OptimizationIterationRecord,
    ScorerConfig,
    TaskOutcome,
    TrajectoryBundle,
    TrajectoryEvent,
)


def test_trajectory_bundle_to_dict_round_trip_shape() -> None:
    bundle = TrajectoryBundle(
        bundle_id="b1",
        session_id="s1",
        agent_name="demo",
        task_prompt="hello",
        traces=["t1", "t2"],
        events=[TrajectoryEvent(kind="user_input", text="hi")],
        outcome=TaskOutcome(status="success", quality_score=0.8),
    )

    data = bundle.to_dict()
    assert data["bundle_id"] == "b1"
    assert data["events"][0]["kind"] == "user_input"
    assert data["outcome"]["status"] == "success"


def test_golden_session_spec_to_dict_contains_nested_fields() -> None:
    spec = GoldenSessionSpec(
        golden_id="g1",
        split="train",
        task_input=GoldenTaskInput(prompt="solve it", allowed_tools=["read_file"]),
        hitl_script=[HitlCheckpoint(checkpoint="danger", decision="reject")],
        scorer=ScorerConfig(type="rule_plus_llm_judge", weights={"task_success": 0.5}),
    )

    data = spec.to_dict()
    assert data["split"] == "train"
    assert data["task_input"]["allowed_tools"] == ["read_file"]
    assert data["hitl_script"][0]["decision"] == "reject"
    assert data["scorer"]["weights"]["task_success"] == 0.5


def test_feedback_and_iteration_records_to_dict() -> None:
    feedback = OptimizationFeedback(
        feedback_id="f1",
        timestamp="2026-03-22T00:00:00Z",
        source="human",
        trace_id="trace-1",
        score=0.2,
        label="bad",
    )
    result = EvaluationResult(
        candidate_id="c1",
        golden_id="g1",
        bundle_id="b1",
        metrics={"task_success": 1.0},
        overall_score=0.9,
    )
    iteration = OptimizationIterationRecord(
        iteration=1,
        base_candidate_id="c0",
        proposed_candidates=["c1"],
        selected_candidate="c1",
        decision="accepted",
        reason="better",
    )

    assert feedback.to_dict()["label"] == "bad"
    assert result.to_dict()["overall_score"] == 0.9
    assert iteration.to_dict()["decision"] == "accepted"
