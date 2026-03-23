"""Tests for optimization/trajectory.py dataclasses and properties."""

from __future__ import annotations

from adk_deepagents.optimization.trajectory import (
    AgentStep,
    FeedbackEntry,
    ModelCall,
    ToolCall,
    Trajectory,
)


def test_tool_call_defaults():
    tc = ToolCall(name="read_file", args={"path": "/a.txt"}, response="ok", duration_ms=10.0)
    assert tc.name == "read_file"
    assert tc.args == {"path": "/a.txt"}
    assert tc.response == "ok"
    assert tc.duration_ms == 10.0
    assert tc.error is None


def test_tool_call_with_error():
    tc = ToolCall(
        name="write_file",
        args={},
        response=None,
        duration_ms=5.0,
        error="permission denied",
    )
    assert tc.error == "permission denied"


def test_model_call_defaults():
    mc = ModelCall(model="gemini-2.5-flash", input_tokens=100, output_tokens=50, duration_ms=200.0)
    assert mc.model == "gemini-2.5-flash"
    assert mc.request is None
    assert mc.response is None


def test_agent_step_defaults():
    step = AgentStep(agent_name="agent_a")
    assert step.model_call is None
    assert step.tool_calls == []


def test_agent_step_with_calls():
    mc = ModelCall(model="m", input_tokens=10, output_tokens=5, duration_ms=1.0)
    tc1 = ToolCall(name="t1", args={}, response=None, duration_ms=1.0)
    tc2 = ToolCall(name="t2", args={}, response=None, duration_ms=2.0)
    step = AgentStep(agent_name="agent_b", model_call=mc, tool_calls=[tc1, tc2])
    assert step.model_call is mc
    assert len(step.tool_calls) == 2
    assert step.tool_calls[0].name == "t1"
    assert step.tool_calls[1].name == "t2"


def test_feedback_entry_defaults():
    fe = FeedbackEntry(source="user")
    assert fe.source == "user"
    assert fe.rating is None
    assert fe.comment == ""
    assert fe.timestamp_ns == 0
    assert fe.metadata == {}


def test_trajectory_defaults():
    t = Trajectory(trace_id="abc123")
    assert t.trace_id == "abc123"
    assert t.session_id is None
    assert t.agent_name is None
    assert t.steps == []
    assert t.start_time_ns == 0
    assert t.end_time_ns == 0
    assert t.status == "unset"
    assert t.score is None
    assert t.is_golden is False
    assert t.feedback == []
    assert t.tags == {}


def test_trajectory_duration_ms():
    t = Trajectory(
        trace_id="t1",
        start_time_ns=1_000_000_000,
        end_time_ns=2_000_000_000,
    )
    assert t.duration_ms == 1000.0


def test_trajectory_duration_ms_zero():
    t = Trajectory(trace_id="t2")
    assert t.duration_ms == 0.0


def test_trajectory_total_tokens():
    mc1 = ModelCall(model="m", input_tokens=100, output_tokens=50, duration_ms=1.0)
    mc2 = ModelCall(model="m", input_tokens=200, output_tokens=80, duration_ms=1.0)
    t = Trajectory(
        trace_id="t3",
        steps=[
            AgentStep(agent_name="a", model_call=mc1),
            AgentStep(agent_name="a", model_call=mc2),
        ],
    )
    assert t.total_input_tokens == 300
    assert t.total_output_tokens == 130


def test_trajectory_total_tokens_no_model_call():
    t = Trajectory(
        trace_id="t4",
        steps=[AgentStep(agent_name="a")],
    )
    assert t.total_input_tokens == 0
    assert t.total_output_tokens == 0


def test_trajectory_golden_and_score():
    t = Trajectory(trace_id="t5", is_golden=True, score=0.95)
    assert t.is_golden is True
    assert t.score == 0.95


def test_trajectory_feedback_list():
    fe1 = FeedbackEntry(source="user", rating=1.0, comment="great")
    fe2 = FeedbackEntry(source="auto", rating=0.5)
    t = Trajectory(trace_id="t6", feedback=[fe1, fe2])
    assert len(t.feedback) == 2
    assert t.feedback[0].comment == "great"
    assert t.feedback[1].source == "auto"
