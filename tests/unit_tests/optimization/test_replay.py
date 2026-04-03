"""Tests for the replay module (non-LLM helpers)."""

from __future__ import annotations

import pytest

from adk_deepagents.optimization.replay import (
    BuiltAgent,
    ReplayConfig,
    ReplayResult,
    _original_tool_names_from_trajectory,
    _should_approve_tool,
    extract_all_user_prompts,
    extract_original_prompt,
)
from adk_deepagents.optimization.trajectory import (
    AgentStep,
    ModelCall,
    ToolCall,
    Trajectory,
)


def _sample_trajectory(
    trace_id: str = "replay-test-123",
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

    def test_raises_on_empty_trajectory(self):
        traj = Trajectory(trace_id="empty")
        with pytest.raises(ValueError, match="Could not extract"):
            extract_original_prompt(traj)

    def test_raises_when_no_user_message(self):
        traj = _sample_trajectory(request={"other": "data"})
        with pytest.raises(ValueError, match="Could not extract"):
            extract_original_prompt(traj)


# ---------------------------------------------------------------------------
# BuiltAgent dataclass
# ---------------------------------------------------------------------------


class TestBuiltAgent:
    def test_construction_defaults(self):
        # BuiltAgent requires an LlmAgent, but we can test with a
        # mock object since we're only checking dataclass behaviour.
        class _FakeAgent:
            name = "fake"

        built = BuiltAgent(agent=_FakeAgent())  # type: ignore[arg-type]
        assert built.agent.name == "fake"
        assert built.cleanup is None

    def test_construction_with_cleanup(self):
        class _FakeAgent:
            name = "fake"

        async def _noop() -> None:
            pass

        built = BuiltAgent(
            agent=_FakeAgent(),  # type: ignore[arg-type]
            cleanup=_noop,
        )
        assert built.cleanup is not None


# ---------------------------------------------------------------------------
# ReplayResult dataclass
# ---------------------------------------------------------------------------


class TestReplayResult:
    def test_construction_defaults(self):
        result = ReplayResult(
            source_trace_id="src-1",
            replay_session_id="sess-1",
            prompts=["hello"],
            output_text="world",
        )
        assert result.source_trace_id == "src-1"
        assert result.replay_session_id == "sess-1"
        assert result.prompts == ["hello"]
        assert result.output_text == "world"
        assert result.events == []
        assert result.replay_trajectory is None
        assert result.per_turn_outputs == []

    def test_construction_with_events(self):
        result = ReplayResult(
            source_trace_id="src-1",
            replay_session_id="sess-1",
            prompts=["hello"],
            output_text="world",
            events=[{"type": "function_call", "name": "read_file"}],
        )
        assert len(result.events) == 1
        assert result.events[0]["name"] == "read_file"


# ---------------------------------------------------------------------------
# extract_all_user_prompts
# ---------------------------------------------------------------------------


class TestExtractAllUserPrompts:
    def test_single_turn(self):
        traj = _sample_trajectory()
        prompts = extract_all_user_prompts(traj)
        assert prompts == ["Write hello world"]

    def test_multi_turn_deduplication(self):
        """Later steps may repeat earlier user messages in context."""
        traj = Trajectory(
            trace_id="multi-turn",
            steps=[
                AgentStep(
                    agent_name="a",
                    model_call=ModelCall(
                        model="m",
                        input_tokens=0,
                        output_tokens=0,
                        duration_ms=0,
                        request={
                            "contents": [
                                {"role": "user", "parts": [{"text": "Turn 1"}]},
                            ]
                        },
                    ),
                ),
                AgentStep(
                    agent_name="a",
                    model_call=ModelCall(
                        model="m",
                        input_tokens=0,
                        output_tokens=0,
                        duration_ms=0,
                        request={
                            "contents": [
                                {"role": "user", "parts": [{"text": "Turn 1"}]},
                                {"role": "user", "parts": [{"text": "Turn 2"}]},
                            ]
                        },
                    ),
                ),
            ],
        )
        prompts = extract_all_user_prompts(traj)
        assert prompts == ["Turn 1", "Turn 2"]

    def test_empty_trajectory(self):
        traj = Trajectory(trace_id="empty")
        assert extract_all_user_prompts(traj) == []

    def test_repeated_user_message_preserved(self):
        """Legitimate repeated messages like 'yes' should not be dropped."""
        traj = Trajectory(
            trace_id="repeated",
            steps=[
                AgentStep(
                    agent_name="a",
                    model_call=ModelCall(
                        model="m",
                        input_tokens=0,
                        output_tokens=0,
                        duration_ms=0,
                        request={
                            "contents": [
                                {"role": "user", "parts": [{"text": "Do something"}]},
                            ]
                        },
                    ),
                ),
                AgentStep(
                    agent_name="a",
                    model_call=ModelCall(
                        model="m",
                        input_tokens=0,
                        output_tokens=0,
                        duration_ms=0,
                        request={
                            "contents": [
                                {"role": "user", "parts": [{"text": "Do something"}]},
                                {"role": "user", "parts": [{"text": "yes"}]},
                            ]
                        },
                    ),
                ),
                AgentStep(
                    agent_name="a",
                    model_call=ModelCall(
                        model="m",
                        input_tokens=0,
                        output_tokens=0,
                        duration_ms=0,
                        request={
                            "contents": [
                                {"role": "user", "parts": [{"text": "Do something"}]},
                                {"role": "user", "parts": [{"text": "yes"}]},
                                {"role": "user", "parts": [{"text": "yes"}]},
                            ]
                        },
                    ),
                ),
            ],
        )
        prompts = extract_all_user_prompts(traj)
        assert prompts == ["Do something", "yes", "yes"]

    def test_chat_style_multi_turn(self):
        traj = Trajectory(
            trace_id="chat-multi",
            steps=[
                AgentStep(
                    agent_name="a",
                    model_call=ModelCall(
                        model="m",
                        input_tokens=0,
                        output_tokens=0,
                        duration_ms=0,
                        request={
                            "messages": [
                                {"role": "user", "content": "First"},
                                {"role": "assistant", "content": "Reply"},
                                {"role": "user", "content": "Second"},
                            ]
                        },
                    ),
                ),
            ],
        )
        prompts = extract_all_user_prompts(traj)
        assert prompts == ["First", "Second"]


# ---------------------------------------------------------------------------
# ReplayConfig
# ---------------------------------------------------------------------------


class TestReplayConfig:
    def test_defaults(self):
        config = ReplayConfig()
        assert config.tool_approval == "auto_approve"
        assert config.user_simulator is None
        assert config.max_approval_rounds == 10

    def test_custom_values(self):
        config = ReplayConfig(
            tool_approval="auto_reject",
            max_approval_rounds=5,
        )
        assert config.tool_approval == "auto_reject"
        assert config.max_approval_rounds == 5

    def test_with_user_simulator(self):
        def my_sim(task: str, history: list[str], output: str) -> str:
            return "follow up"

        config = ReplayConfig(user_simulator=my_sim)
        assert config.user_simulator is not None


# ---------------------------------------------------------------------------
# _should_approve_tool
# ---------------------------------------------------------------------------


class TestShouldApproveTool:
    def test_auto_approve_always_true(self):
        assert _should_approve_tool(
            "write_file",
            policy="auto_approve",
            original_tool_names=set(),
        )

    def test_auto_reject_always_false(self):
        assert not _should_approve_tool(
            "write_file",
            policy="auto_reject",
            original_tool_names={"write_file"},
        )

    def test_original_approves_known_tool(self):
        assert _should_approve_tool(
            "write_file",
            policy="original",
            original_tool_names={"write_file", "read_file"},
        )

    def test_original_rejects_unknown_tool(self):
        assert not _should_approve_tool(
            "execute",
            policy="original",
            original_tool_names={"write_file", "read_file"},
        )


# ---------------------------------------------------------------------------
# _original_tool_names_from_trajectory
# ---------------------------------------------------------------------------


class TestOriginalToolNames:
    def test_extracts_tool_names(self):
        traj = _sample_trajectory()
        names = _original_tool_names_from_trajectory(traj)
        assert names == {"write_file"}

    def test_skips_errored_tools(self):
        traj = Trajectory(
            trace_id="t1",
            steps=[
                AgentStep(
                    agent_name="a",
                    tool_calls=[
                        ToolCall(
                            name="good_tool",
                            args={},
                            response={"ok": True},
                            duration_ms=0,
                        ),
                        ToolCall(
                            name="bad_tool",
                            args={},
                            response=None,
                            duration_ms=0,
                            error="failed",
                        ),
                    ],
                ),
            ],
        )
        names = _original_tool_names_from_trajectory(traj)
        assert names == {"good_tool"}

    def test_empty_trajectory(self):
        traj = Trajectory(trace_id="empty")
        assert _original_tool_names_from_trajectory(traj) == set()


# ---------------------------------------------------------------------------
# ReplayConfig.ephemeral_instruction
# ---------------------------------------------------------------------------


class TestEphemeralInstruction:
    def test_default_is_none(self):
        config = ReplayConfig()
        assert config.ephemeral_instruction is None

    def test_can_set_ephemeral_instruction(self):
        config = ReplayConfig(ephemeral_instruction="Think step by step")
        assert config.ephemeral_instruction == "Think step by step"
