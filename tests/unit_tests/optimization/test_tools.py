"""Tests for optimization/tools.py — experience query tools."""

from __future__ import annotations

from adk_deepagents.optimization.history import HistoryEntry, ScoreHistory
from adk_deepagents.optimization.learnings import LearningEntry, LearningsStore
from adk_deepagents.optimization.store import TrajectoryStore
from adk_deepagents.optimization.tools import create_experience_tools
from adk_deepagents.optimization.trajectory import (
    AgentStep,
    ToolCall,
    Trajectory,
)


def _make_test_trajectory(
    trace_id: str,
    score: float | None = None,
) -> Trajectory:
    return Trajectory(
        trace_id=trace_id,
        agent_name="test_agent",
        steps=[
            AgentStep(
                agent_name="test_agent",
                tool_calls=[
                    ToolCall(
                        name="read_file",
                        args={"path": "/test"},
                        response="ok",
                        duration_ms=10,
                    ),
                ],
            ),
        ],
        status="ok",
        score=score,
    )


def _populate_store(store: TrajectoryStore) -> None:
    store.save(_make_test_trajectory("trace_aaa", score=0.9))
    store.save(_make_test_trajectory("trace_bbb", score=0.4))
    store.save(_make_test_trajectory("trace_ccc", score=0.7))


# ---------------------------------------------------------------------------
# create_experience_tools count
# ---------------------------------------------------------------------------


class TestCreateExperienceTools:
    def test_without_learnings(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        history = ScoreHistory(tmp_path / "h.jsonl")
        tools = create_experience_tools(store, history)
        assert len(tools) == 4

    def test_with_learnings(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        history = ScoreHistory(tmp_path / "h.jsonl")
        learnings = LearningsStore(tmp_path / "l.jsonl")
        tools = create_experience_tools(store, history, learnings)
        assert len(tools) == 5


# ---------------------------------------------------------------------------
# list_trajectories tool
# ---------------------------------------------------------------------------


class TestListTrajectoriesTool:
    def test_returns_trajectory_info(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        _populate_store(store)
        history = ScoreHistory(tmp_path / "h.jsonl")
        tools = create_experience_tools(store, history)
        list_traj = tools[0]

        result = list_traj(sort_by="score", limit=20, status=None)
        assert "Trajectories" in result
        assert "trace_aaa" in result
        assert "trace_bbb" in result
        assert "trace_ccc" in result

    def test_empty_store(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        history = ScoreHistory(tmp_path / "h.jsonl")
        tools = create_experience_tools(store, history)
        list_traj = tools[0]

        result = list_traj(sort_by="score", limit=20, status=None)
        assert "No trajectories found" in result


# ---------------------------------------------------------------------------
# show_failures tool
# ---------------------------------------------------------------------------


class TestShowFailuresTool:
    def test_existing_trace(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        _populate_store(store)
        history = ScoreHistory(tmp_path / "h.jsonl")
        tools = create_experience_tools(store, history)
        show_failures = tools[1]

        result = show_failures(trace_id="trace_aaa")
        assert "trace_aaa" in result
        assert "Status: ok" in result
        assert "No tool errors" in result

    def test_nonexistent_trace(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        history = ScoreHistory(tmp_path / "h.jsonl")
        tools = create_experience_tools(store, history)
        show_failures = tools[1]

        result = show_failures(trace_id="nonexistent")
        assert "not found" in result

    def test_with_tool_error(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        traj = Trajectory(
            trace_id="err_trace",
            agent_name="agent",
            steps=[
                AgentStep(
                    agent_name="agent",
                    tool_calls=[
                        ToolCall(
                            name="write_file",
                            args={"path": "/x"},
                            response=None,
                            duration_ms=5,
                            error="permission denied",
                        ),
                    ],
                ),
            ],
            status="error",
        )
        store.save(traj)
        history = ScoreHistory(tmp_path / "h.jsonl")
        tools = create_experience_tools(store, history)
        show_failures = tools[1]

        result = show_failures(trace_id="err_trace")
        assert "Tool errors" in result
        assert "permission denied" in result


# ---------------------------------------------------------------------------
# show_score_history tool
# ---------------------------------------------------------------------------


class TestShowScoreHistoryTool:
    def test_returns_summary(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        history = ScoreHistory(tmp_path / "h.jsonl")
        history.append(HistoryEntry(iteration=1, val_score=0.5, pass_rate=0.4))
        history.append(HistoryEntry(iteration=2, val_score=0.7, pass_rate=0.6))
        tools = create_experience_tools(store, history)
        show_history = tools[3]

        result = show_history(last_n=20)
        assert "Score history" in result
        assert "iter 1" in result
        assert "iter 2" in result

    def test_empty_history(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        history = ScoreHistory(tmp_path / "h.jsonl")
        tools = create_experience_tools(store, history)
        show_history = tools[3]

        result = show_history(last_n=20)
        assert "No optimization history" in result


# ---------------------------------------------------------------------------
# show_learnings tool
# ---------------------------------------------------------------------------


class TestShowLearningsTool:
    def test_returns_text(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        history = ScoreHistory(tmp_path / "h.jsonl")
        learnings = LearningsStore(tmp_path / "l.jsonl")
        learnings.append(
            LearningEntry(
                iteration=1,
                category="failed_attempt",
                summary="bad approach",
            )
        )
        learnings.append(
            LearningEntry(
                iteration=2,
                category="successful_change",
                summary="good approach",
            )
        )
        tools = create_experience_tools(store, history, learnings)
        show_learnings = tools[4]

        result = show_learnings(category=None, last_n=10)
        assert "Learnings" in result
        assert "bad approach" in result
        assert "good approach" in result

    def test_filter_by_category(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        history = ScoreHistory(tmp_path / "h.jsonl")
        learnings = LearningsStore(tmp_path / "l.jsonl")
        learnings.append(
            LearningEntry(
                iteration=1,
                category="failed_attempt",
                summary="fail",
            )
        )
        learnings.append(
            LearningEntry(
                iteration=2,
                category="successful_change",
                summary="win",
            )
        )
        tools = create_experience_tools(store, history, learnings)
        show_learnings = tools[4]

        result = show_learnings(category="failed_attempt", last_n=10)
        assert "fail" in result
        assert "win" not in result

    def test_empty_learnings(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        history = ScoreHistory(tmp_path / "h.jsonl")
        learnings = LearningsStore(tmp_path / "l.jsonl")
        tools = create_experience_tools(store, history, learnings)
        show_learnings = tools[4]

        result = show_learnings(category=None, last_n=10)
        assert "No learnings" in result
