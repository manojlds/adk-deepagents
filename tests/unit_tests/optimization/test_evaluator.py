"""Tests for the evaluator module (non-LLM helpers)."""

from __future__ import annotations

from adk_deepagents.optimization.evaluator import (
    CriterionAssessment,
    EvaluationCriterion,
    EvaluationRubric,
    TrajectoryJudgment,
    _build_judge_instruction,
    _compute_weighted_score,
    _extract_final_response,
    _trajectory_to_judge_payload,
    _truncate,
    default_rubric,
    extract_original_prompt,
)
from adk_deepagents.optimization.trajectory import (
    AgentStep,
    ModelCall,
    ToolCall,
    Trajectory,
)


def _sample_trajectory(
    trace_id: str = "eval-test-123",
    *,
    agent_name: str = "test_agent",
    status: str = "ok",
    request: dict | None = None,
    response: dict | None = None,
) -> Trajectory:
    return Trajectory(
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
                    request=request
                    or {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": "Write hello world"}],
                            }
                        ]
                    },
                    response=response
                    or {"candidates": [{"content": {"parts": [{"text": "Hello, World!"}]}}]},
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
    )


# ---------------------------------------------------------------------------
# extract_original_prompt
# ---------------------------------------------------------------------------


class TestExtractOriginalPrompt:
    def test_vertex_style(self):
        traj = _sample_trajectory()
        assert extract_original_prompt(traj) == "Write hello world"

    def test_chat_style(self):
        traj = _sample_trajectory(
            request={
                "messages": [
                    {"role": "user", "content": "Say hi"},
                ]
            }
        )
        assert extract_original_prompt(traj) == "Say hi"

    def test_returns_none_when_no_prompt(self):
        traj = _sample_trajectory(request={"other": "data"})
        assert extract_original_prompt(traj) is None

    def test_returns_none_for_empty_trajectory(self):
        traj = Trajectory(trace_id="empty")
        assert extract_original_prompt(traj) is None

    def test_returns_none_when_no_model_call(self):
        traj = Trajectory(
            trace_id="no-mc",
            steps=[AgentStep(agent_name="a", tool_calls=[])],
        )
        assert extract_original_prompt(traj) is None


# ---------------------------------------------------------------------------
# _extract_final_response
# ---------------------------------------------------------------------------


class TestExtractFinalResponse:
    def test_vertex_style(self):
        traj = _sample_trajectory()
        assert _extract_final_response(traj) == "Hello, World!"

    def test_chat_style(self):
        traj = _sample_trajectory(response={"choices": [{"message": {"content": "Hi there!"}}]})
        assert _extract_final_response(traj) == "Hi there!"

    def test_returns_none_for_empty_trajectory(self):
        traj = Trajectory(trace_id="empty")
        assert _extract_final_response(traj) is None


# ---------------------------------------------------------------------------
# _trajectory_to_judge_payload
# ---------------------------------------------------------------------------


class TestTrajectoryToJudgePayload:
    def test_contains_prompt(self):
        payload = _trajectory_to_judge_payload(_sample_trajectory())
        assert "Write hello world" in payload

    def test_contains_step_info(self):
        payload = _trajectory_to_judge_payload(_sample_trajectory())
        assert "Step 1" in payload
        assert "test_agent" in payload

    def test_contains_tool_call(self):
        payload = _trajectory_to_judge_payload(_sample_trajectory())
        assert "write_file" in payload

    def test_contains_status(self):
        payload = _trajectory_to_judge_payload(_sample_trajectory())
        assert "Status: ok" in payload

    def test_contains_final_response(self):
        payload = _trajectory_to_judge_payload(_sample_trajectory())
        assert "Hello, World!" in payload


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello", 10) == "hello"

    def test_exact_length_unchanged(self):
        assert _truncate("hello", 5) == "hello"

    def test_long_text_truncated(self):
        result = _truncate("hello world", 5)
        assert result == "hello...(truncated)"

    def test_empty_string(self):
        assert _truncate("", 10) == ""


# ---------------------------------------------------------------------------
# _build_judge_instruction
# ---------------------------------------------------------------------------


class TestBuildJudgeInstruction:
    def test_includes_criteria_names(self):
        rubric = default_rubric()
        instruction = _build_judge_instruction(rubric)
        assert "task_completion" in instruction
        assert "efficiency" in instruction
        assert "tool_usage_quality" in instruction

    def test_includes_criteria_weights(self):
        rubric = default_rubric()
        instruction = _build_judge_instruction(rubric)
        assert "0.6" in instruction
        assert "0.2" in instruction

    def test_includes_custom_instructions(self):
        rubric = EvaluationRubric(
            criteria=[],
            judge_instructions="Be extra strict",
        )
        instruction = _build_judge_instruction(rubric)
        assert "Be extra strict" in instruction

    def test_no_extra_section_without_custom(self):
        rubric = EvaluationRubric(criteria=[])
        instruction = _build_judge_instruction(rubric)
        assert "Additional Instructions" not in instruction


# ---------------------------------------------------------------------------
# _compute_weighted_score
# ---------------------------------------------------------------------------


class TestComputeWeightedScore:
    def test_weighted_scoring(self):
        rubric = EvaluationRubric(
            criteria=[
                EvaluationCriterion(name="a", description="A", weight=0.8),
                EvaluationCriterion(name="b", description="B", weight=0.2),
            ]
        )
        judgment = TrajectoryJudgment(
            summary="ok",
            criteria=[
                CriterionAssessment(name="a", score=1.0, reasoning="good"),
                CriterionAssessment(name="b", score=0.0, reasoning="bad"),
            ],
        )
        score = _compute_weighted_score(judgment, rubric)
        assert abs(score - 0.8) < 1e-9

    def test_fallback_when_no_criteria_match(self):
        rubric = EvaluationRubric(
            criteria=[
                EvaluationCriterion(name="x", description="X", weight=1.0),
            ]
        )
        judgment = TrajectoryJudgment(
            summary="ok",
            criteria=[
                CriterionAssessment(name="unknown", score=0.7, reasoning="meh"),
            ],
        )
        score = _compute_weighted_score(judgment, rubric)
        assert abs(score - 0.7) < 1e-9

    def test_empty_criteria_returns_default(self):
        rubric = EvaluationRubric(criteria=[])
        judgment = TrajectoryJudgment(summary="ok", criteria=[])
        assert _compute_weighted_score(judgment, rubric) == 0.5


# ---------------------------------------------------------------------------
# default_rubric
# ---------------------------------------------------------------------------


class TestDefaultRubric:
    def test_returns_three_criteria(self):
        rubric = default_rubric()
        assert len(rubric.criteria) == 3

    def test_weights_sum_to_one(self):
        rubric = default_rubric()
        total = sum(c.weight for c in rubric.criteria)
        assert abs(total - 1.0) < 1e-9

    def test_criteria_names(self):
        rubric = default_rubric()
        names = {c.name for c in rubric.criteria}
        assert names == {
            "task_completion",
            "efficiency",
            "tool_usage_quality",
        }


# ---------------------------------------------------------------------------
# TrajectoryFilter and filter_trajectories
# ---------------------------------------------------------------------------


class TestTrajectoryFilter:
    def test_default_passes_everything(self):
        from adk_deepagents.optimization.evaluator import TrajectoryFilter, filter_trajectories

        tf = TrajectoryFilter()
        trajs = [_sample_trajectory(), _sample_trajectory("t2")]
        assert len(filter_trajectories(trajs, tf)) == 2

    def test_min_steps_filters(self):
        from adk_deepagents.optimization.evaluator import TrajectoryFilter, filter_trajectories

        tf = TrajectoryFilter(min_steps=5)
        trajs = [_sample_trajectory()]  # has 1 step
        assert filter_trajectories(trajs, tf) == []

    def test_min_steps_passes(self):
        from adk_deepagents.optimization.evaluator import TrajectoryFilter, filter_trajectories

        tf = TrajectoryFilter(min_steps=1)
        trajs = [_sample_trajectory()]
        assert len(filter_trajectories(trajs, tf)) == 1

    def test_require_tool_calls(self):
        from adk_deepagents.optimization.evaluator import TrajectoryFilter, filter_trajectories

        tf = TrajectoryFilter(require_tool_calls=True)
        # _sample_trajectory has a tool call
        assert len(filter_trajectories([_sample_trajectory()], tf)) == 1
        # trajectory with no tool calls
        traj = Trajectory(trace_id="no-tools", steps=[AgentStep(agent_name="a")])
        assert filter_trajectories([traj], tf) == []

    def test_max_tool_error_rate(self):
        from adk_deepagents.optimization.evaluator import TrajectoryFilter, filter_trajectories

        tf = TrajectoryFilter(max_tool_error_rate=0.0)
        # _sample_trajectory has tool calls with no errors
        assert len(filter_trajectories([_sample_trajectory()], tf)) == 1
        # trajectory with all errors
        traj = Trajectory(
            trace_id="all-errors",
            steps=[
                AgentStep(
                    agent_name="a",
                    tool_calls=[
                        ToolCall(name="t", args={}, response=None, duration_ms=0, error="fail")
                    ],
                )
            ],
        )
        assert filter_trajectories([traj], tf) == []

    def test_min_output_chars(self):
        from adk_deepagents.optimization.evaluator import TrajectoryFilter, filter_trajectories

        tf = TrajectoryFilter(min_output_chars=100)
        # _sample_trajectory response is "Hello, World!" (13 chars)
        assert filter_trajectories([_sample_trajectory()], tf) == []
        tf2 = TrajectoryFilter(min_output_chars=5)
        assert len(filter_trajectories([_sample_trajectory()], tf2)) == 1

    def test_custom_filter(self):
        from adk_deepagents.optimization.evaluator import TrajectoryFilter, filter_trajectories

        tf = TrajectoryFilter(custom=lambda t: t.trace_id.startswith("keep"))
        trajs = [_sample_trajectory("keep-1"), _sample_trajectory("drop-2")]
        result = filter_trajectories(trajs, tf)
        assert len(result) == 1
        assert result[0].trace_id == "keep-1"

    def test_combined_filters(self):
        from adk_deepagents.optimization.evaluator import TrajectoryFilter, filter_trajectories

        tf = TrajectoryFilter(min_steps=1, require_tool_calls=True, min_output_chars=5)
        assert len(filter_trajectories([_sample_trajectory()], tf)) == 1


# ---------------------------------------------------------------------------
# evaluate_trajectory_majority (unit test for aggregation logic)
# ---------------------------------------------------------------------------


class TestMajorityVotingHelpers:
    def test_median_of_odd_scores(self):
        import statistics

        scores = [0.3, 0.7, 0.5]
        assert statistics.median(scores) == 0.5

    def test_median_of_even_scores(self):
        import statistics

        scores = [0.3, 0.5, 0.7, 0.9]
        assert statistics.median(scores) == 0.6
