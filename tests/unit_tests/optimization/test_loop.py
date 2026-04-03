"""Tests for the loop module (non-LLM helpers)."""

from __future__ import annotations

from adk_deepagents.optimization.loop import (
    ImprovementSuggestion,
    IterationResult,
    OptimizationCandidate,
    OptimizationResult,
    _apply_suggestion,
    _build_reflector_payload,
    _resolve_baseline_score,
)
from adk_deepagents.optimization.replay import ReplayResult
from adk_deepagents.optimization.trajectory import (
    AgentStep,
    FeedbackEntry,
    ModelCall,
    ToolCall,
    Trajectory,
)


def _sample_trajectory(
    trace_id: str = "loop-test-123",
    *,
    agent_name: str = "test_agent",
    status: str = "ok",
    score: float | None = None,
    is_golden: bool = False,
    feedback: list[FeedbackEntry] | None = None,
) -> Trajectory:
    traj = Trajectory(
        trace_id=trace_id,
        session_id="session_1",
        agent_name=agent_name,
        steps=[
            AgentStep(
                agent_name=agent_name,
                model_call=ModelCall(
                    model="gemini-2.5-flash",
                    input_tokens=100,
                    output_tokens=50,
                    duration_ms=500.0,
                    request={
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": "Write hello world"}],
                            }
                        ]
                    },
                    response={"candidates": [{"content": {"parts": [{"text": "Hello, World!"}]}}]},
                    finish_reason="stop",
                ),
                tool_calls=[
                    ToolCall(
                        name="write_file",
                        args={
                            "path": "/hello.py",
                            "content": "print('hello')",
                        },
                        response={"status": "ok"},
                        duration_ms=10.0,
                    ),
                ],
            ),
        ],
        start_time_ns=1_000_000_000,
        end_time_ns=2_000_000_000,
        status=status,
        score=score,
        is_golden=is_golden,
    )
    if feedback:
        traj.feedback.extend(feedback)
    return traj


# ---------------------------------------------------------------------------
# _resolve_baseline_score
# ---------------------------------------------------------------------------


class TestResolveBaselineScore:
    def test_uses_explicit_score(self):
        traj = _sample_trajectory(score=0.85)
        assert _resolve_baseline_score(traj) == 0.85

    def test_falls_back_to_evaluator_feedback_average(self):
        traj = _sample_trajectory(
            feedback=[
                FeedbackEntry(source="evaluator", rating=0.6),
                FeedbackEntry(source="evaluator", rating=0.8),
            ]
        )
        score = _resolve_baseline_score(traj)
        assert score is not None
        assert abs(score - 0.7) < 1e-9

    def test_ignores_non_evaluator_feedback(self):
        traj = _sample_trajectory(
            feedback=[
                FeedbackEntry(source="user", rating=0.9),
                FeedbackEntry(source="auto", rating=0.8),
            ]
        )
        assert _resolve_baseline_score(traj) is None

    def test_golden_returns_one(self):
        traj = _sample_trajectory(is_golden=True)
        assert _resolve_baseline_score(traj) == 1.0

    def test_returns_none_when_nothing(self):
        traj = _sample_trajectory()
        assert _resolve_baseline_score(traj) is None

    def test_score_takes_precedence_over_feedback(self):
        traj = _sample_trajectory(
            score=0.5,
            feedback=[FeedbackEntry(source="user", rating=0.9)],
        )
        assert _resolve_baseline_score(traj) == 0.5

    def test_evaluator_feedback_takes_precedence_over_golden(self):
        traj = _sample_trajectory(
            is_golden=True,
            feedback=[FeedbackEntry(source="evaluator", rating=0.6)],
        )
        score = _resolve_baseline_score(traj)
        assert score is not None
        assert abs(score - 0.6) < 1e-9


# ---------------------------------------------------------------------------
# _apply_suggestion
# ---------------------------------------------------------------------------


class TestApplySuggestion:
    def test_instruction_append_to_empty(self):
        cand = OptimizationCandidate(agent_kwargs={})
        sugg = ImprovementSuggestion(
            kind="instruction_append",
            target="instruction",
            proposal="Be concise.",
            rationale="too verbose",
        )
        new = _apply_suggestion(cand, sugg)
        assert new.agent_kwargs["instruction"] == "Be concise."
        assert new.iteration == 1

    def test_instruction_append_to_existing(self):
        cand = OptimizationCandidate(agent_kwargs={"instruction": "Be helpful."})
        sugg = ImprovementSuggestion(
            kind="instruction_append",
            target="instruction",
            proposal="Be concise.",
            rationale="too verbose",
        )
        new = _apply_suggestion(cand, sugg)
        assert new.agent_kwargs["instruction"] == ("Be helpful.\n\nBe concise.")

    def test_instruction_replace(self):
        cand = OptimizationCandidate(agent_kwargs={"instruction": "Old instruction"})
        sugg = ImprovementSuggestion(
            kind="instruction_replace",
            target="instruction",
            proposal="New instruction",
            rationale="complete rewrite needed",
        )
        new = _apply_suggestion(cand, sugg)
        assert new.agent_kwargs["instruction"] == "New instruction"

    def test_skill_add(self):
        cand = OptimizationCandidate(agent_kwargs={"skills": ["skill_a"]})
        sugg = ImprovementSuggestion(
            kind="skill_add",
            target="skill_b",
            proposal="add skill_b",
            rationale="needed",
        )
        new = _apply_suggestion(cand, sugg)
        assert "skill_b" in new.agent_kwargs["skills"]
        assert "skill_a" in new.agent_kwargs["skills"]

    def test_skill_add_no_duplicate(self):
        cand = OptimizationCandidate(agent_kwargs={"skills": ["skill_a"]})
        sugg = ImprovementSuggestion(
            kind="skill_add",
            target="skill_a",
            proposal="add skill_a",
            rationale="needed",
        )
        new = _apply_suggestion(cand, sugg)
        assert new.agent_kwargs["skills"].count("skill_a") == 1

    def test_skill_remove(self):
        cand = OptimizationCandidate(agent_kwargs={"skills": ["skill_a", "skill_b"]})
        sugg = ImprovementSuggestion(
            kind="skill_remove",
            target="skill_a",
            proposal="remove skill_a",
            rationale="not needed",
        )
        new = _apply_suggestion(cand, sugg)
        assert "skill_a" not in new.agent_kwargs["skills"]
        assert "skill_b" in new.agent_kwargs["skills"]

    def test_skill_remove_missing_is_safe(self):
        cand = OptimizationCandidate(agent_kwargs={"skills": ["skill_a"]})
        sugg = ImprovementSuggestion(
            kind="skill_remove",
            target="skill_x",
            proposal="remove skill_x",
            rationale="cleanup",
        )
        new = _apply_suggestion(cand, sugg)
        assert new.agent_kwargs["skills"] == ["skill_a"]

    def test_does_not_mutate_original(self):
        cand = OptimizationCandidate(agent_kwargs={"instruction": "Original"})
        sugg = ImprovementSuggestion(
            kind="instruction_replace",
            target="instruction",
            proposal="New",
            rationale="r",
        )
        _apply_suggestion(cand, sugg)
        assert cand.agent_kwargs["instruction"] == "Original"
        assert cand.iteration == 0


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------


class TestDataclassConstruction:
    def test_optimization_candidate_defaults(self):
        c = OptimizationCandidate()
        assert c.agent_kwargs == {}
        assert c.iteration == 0

    def test_improvement_suggestion(self):
        s = ImprovementSuggestion(
            kind="instruction_append",
            target="instruction",
            proposal="Be concise.",
            rationale="too verbose",
        )
        assert s.kind == "instruction_append"
        assert s.evidence_trace_ids == []
        assert s.auto_applicable is False

    def test_iteration_result_defaults(self):
        r = IterationResult(
            iteration=1,
            candidate=OptimizationCandidate(),
        )
        assert r.examples == []
        assert r.average_score is None
        assert r.regressions == 0

    def test_optimization_result_defaults(self):
        r = OptimizationResult(
            best_candidate=OptimizationCandidate(),
        )
        assert r.iterations == []
        assert r.stopped_reason == ""


# ---------------------------------------------------------------------------
# _build_reflector_payload
# ---------------------------------------------------------------------------


class TestBuildReflectorPayload:
    def _make_iteration(self) -> IterationResult:
        from adk_deepagents.optimization.loop import ExampleResult

        traj = _sample_trajectory()
        feedback = FeedbackEntry(
            source="evaluator",
            rating=0.7,
            comment="decent job",
            metadata={
                "criteria": [
                    {
                        "name": "task_completion",
                        "score": 0.8,
                        "reasoning": "ok",
                    }
                ],
                "issues": ["Minor inefficiency"],
            },
        )
        example = ExampleResult(
            source_trajectory=traj,
            replay=ReplayResult(
                source_trace_id=traj.trace_id,
                replay_session_id="sess-1",
                prompts=["Write hello world"],
                output_text="Hello!",
            ),
            feedback=feedback,
            baseline_score=0.6,
            delta=0.1,
        )
        return IterationResult(
            iteration=1,
            candidate=OptimizationCandidate(agent_kwargs={"instruction": "Be helpful."}),
            examples=[example],
            average_score=0.7,
            average_delta=0.1,
            regressions=0,
        )

    def test_contains_instruction(self):
        iteration = self._make_iteration()
        payload = _build_reflector_payload(iteration.candidate, iteration)
        assert "Be helpful." in payload

    def test_contains_iteration_number(self):
        iteration = self._make_iteration()
        payload = _build_reflector_payload(iteration.candidate, iteration)
        assert "Iteration 1" in payload

    def test_contains_average_score(self):
        iteration = self._make_iteration()
        payload = _build_reflector_payload(iteration.candidate, iteration)
        assert "0.700" in payload

    def test_contains_example_details(self):
        iteration = self._make_iteration()
        payload = _build_reflector_payload(iteration.candidate, iteration)
        assert "loop-test-1" in payload
        assert "decent job" in payload

    def test_contains_issues(self):
        iteration = self._make_iteration()
        payload = _build_reflector_payload(iteration.candidate, iteration)
        assert "Minor inefficiency" in payload


# ---------------------------------------------------------------------------
# _build_reflector_payload with hindsight hints
# ---------------------------------------------------------------------------


class TestReflectorPayloadHindsight:
    def test_includes_tool_outcomes(self):
        from adk_deepagents.optimization.loop import ExampleResult
        from adk_deepagents.optimization.replay import ReplayResult

        traj = _sample_trajectory()
        replay_traj = Trajectory(
            trace_id="replay-123",
            steps=[
                AgentStep(
                    agent_name="test_agent",
                    tool_calls=[
                        ToolCall(
                            name="write_file",
                            args={"path": "/out.txt"},
                            response={"status": "ok"},
                            duration_ms=5.0,
                        ),
                    ],
                ),
            ],
        )
        feedback = FeedbackEntry(
            source="evaluator",
            rating=0.8,
            comment="good",
            metadata={"criteria": [], "issues": []},
        )
        example = ExampleResult(
            source_trajectory=traj,
            replay=ReplayResult(
                source_trace_id=traj.trace_id,
                replay_session_id="sess-1",
                prompts=["Write hello world"],
                output_text="Hello!",
                replay_trajectory=replay_traj,
            ),
            feedback=feedback,
            baseline_score=0.6,
            delta=0.2,
        )
        iteration = IterationResult(
            iteration=1,
            candidate=OptimizationCandidate(agent_kwargs={"instruction": "Be helpful."}),
            examples=[example],
            average_score=0.8,
            average_delta=0.2,
            regressions=0,
        )

        payload = _build_reflector_payload(iteration.candidate, iteration)
        assert "Tool call outcomes" in payload
        assert "write_file" in payload
        assert "OK" in payload

    def test_includes_error_status(self):
        from adk_deepagents.optimization.loop import ExampleResult
        from adk_deepagents.optimization.replay import ReplayResult

        traj = _sample_trajectory()
        replay_traj = Trajectory(
            trace_id="replay-err",
            steps=[
                AgentStep(
                    agent_name="test_agent",
                    tool_calls=[
                        ToolCall(
                            name="bad_tool",
                            args={},
                            response=None,
                            duration_ms=5.0,
                            error="permission denied",
                        ),
                    ],
                ),
            ],
        )
        feedback = FeedbackEntry(
            source="evaluator",
            rating=0.3,
            comment="failed",
            metadata={"criteria": [], "issues": []},
        )
        example = ExampleResult(
            source_trajectory=traj,
            replay=ReplayResult(
                source_trace_id=traj.trace_id,
                replay_session_id="sess-1",
                prompts=["Write hello world"],
                output_text="",
                replay_trajectory=replay_traj,
            ),
            feedback=feedback,
            baseline_score=0.6,
            delta=-0.3,
        )
        iteration = IterationResult(
            iteration=1,
            candidate=OptimizationCandidate(agent_kwargs={}),
            examples=[example],
            average_score=0.3,
            average_delta=-0.3,
            regressions=1,
        )

        payload = _build_reflector_payload(iteration.candidate, iteration)
        assert "ERROR" in payload
        assert "bad_tool" in payload

    def test_no_tool_outcomes_without_replay_trajectory(self):
        from adk_deepagents.optimization.loop import ExampleResult
        from adk_deepagents.optimization.replay import ReplayResult

        traj = _sample_trajectory()
        feedback = FeedbackEntry(
            source="evaluator",
            rating=0.7,
            comment="ok",
            metadata={"criteria": [], "issues": []},
        )
        example = ExampleResult(
            source_trajectory=traj,
            replay=ReplayResult(
                source_trace_id=traj.trace_id,
                replay_session_id="sess-1",
                prompts=["Write hello world"],
                output_text="Hello!",
            ),
            feedback=feedback,
            baseline_score=0.6,
            delta=0.1,
        )
        iteration = IterationResult(
            iteration=1,
            candidate=OptimizationCandidate(agent_kwargs={}),
            examples=[example],
        )

        payload = _build_reflector_payload(iteration.candidate, iteration)
        assert "Tool call outcomes" not in payload
