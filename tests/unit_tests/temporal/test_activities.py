"""Tests for Temporal activity payload models."""

from __future__ import annotations

from adk_deepagents.temporal.activities import TaskResult, TaskSnapshot


def test_task_snapshot_defaults():
    snapshot = TaskSnapshot(subagent_type="general_purpose", prompt="Do work")
    assert snapshot.subagent_type == "general_purpose"
    assert snapshot.prompt == "Do work"
    assert snapshot.depth == 1
    assert snapshot.files == {}
    assert snapshot.todos == []
    assert snapshot.history == []
    assert snapshot.model_override is None
    assert snapshot.subagent_spec is None
    assert snapshot.subagent_spec_hash is None
    assert snapshot.timeout_seconds == 120.0


def test_task_snapshot_roundtrip():
    snapshot = TaskSnapshot(
        subagent_type="researcher",
        prompt="Find references",
        depth=2,
        files={"/notes.txt": {"content": ["hello"]}},
        todos=[{"text": "done", "done": True}],
        history=[{"prompt": "first", "result": "second"}],
        model_override="openai/gpt-4o-mini",
        subagent_spec={
            "name": "runtime_specialist",
            "description": "Runtime specialist",
            "system_prompt": "Be precise",
            "tool_names": ["read_file"],
        },
        subagent_spec_hash="abc123",
        timeout_seconds=55.0,
    )

    restored = TaskSnapshot.from_dict(snapshot.to_dict())
    assert restored == snapshot


def test_task_result_defaults():
    result = TaskResult()
    assert result.result == ""
    assert result.function_calls == []
    assert result.files == {}
    assert result.todos == []
    assert result.timed_out is False
    assert result.error is None


def test_task_result_roundtrip():
    result = TaskResult(
        result="ok",
        function_calls=["read_file"],
        files={"/out.txt": {"content": ["done"]}},
        todos=[{"text": "ship", "done": True}],
        timed_out=False,
        error=None,
    )

    restored = TaskResult.from_dict(result.to_dict())
    assert restored == result
