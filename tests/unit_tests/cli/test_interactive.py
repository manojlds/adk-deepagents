"""Unit tests for interactive CLI REPL helpers."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, cast

import adk_deepagents.cli.interactive as repl
from adk_deepagents.cli.session_store import ThreadRecord
from adk_deepagents.optimization.store import TrajectoryStore
from adk_deepagents.optimization.trajectory import (
    AgentStep,
    FeedbackEntry,
    ModelCall,
    ToolCall,
    Trajectory,
)
from adk_deepagents.types import DynamicTaskConfig


class _FakeStdin:
    def __init__(self, *, isatty_value: bool):
        self._isatty_value = isatty_value

    def isatty(self) -> bool:
        return self._isatty_value


class _FakeFunctionCall:
    def __init__(
        self,
        name: str,
        *,
        call_id: str | None = None,
        args: dict[str, object] | None = None,
    ):
        self.name = name
        self.id = call_id
        self.args = args if args is not None else {}


class _FakeFunctionResponse:
    def __init__(
        self,
        name: str,
        response: dict[str, object],
        *,
        call_id: str | None = None,
    ):
        self.name = name
        self.response = response
        self.id = call_id


class _FakePart:
    def __init__(
        self,
        text: str | None = None,
        *,
        function_call: _FakeFunctionCall | None = None,
        function_response: _FakeFunctionResponse | None = None,
    ):
        self.text = text
        self.function_call = function_call
        self.function_response = function_response


class _FakeContent:
    def __init__(self, parts: list[_FakePart]):
        self.parts = parts


class _FakeEvent:
    def __init__(self, author: str, parts: list[_FakePart] | None):
        self.author = author
        self.content = _FakeContent(parts) if parts is not None else None


class _TurnFakeRunner:
    async def run_async(self, *, user_id, session_id, new_message):
        del user_id, session_id, new_message
        yield _FakeEvent("user", [_FakePart("ignore me")])
        yield _FakeEvent("assistant", [_FakePart("hello"), _FakePart(" world")])
        yield _FakeEvent("assistant", [_FakePart(function_call=_FakeFunctionCall("ls"))])
        yield _FakeEvent(
            "assistant",
            [
                _FakePart(
                    function_response=_FakeFunctionResponse(
                        "ls",
                        {"error": "permission denied"},
                    )
                )
            ],
        )


class _QueuedTaskRunner:
    async def run_async(self, *, user_id, session_id, new_message):
        del user_id, session_id, new_message
        yield _FakeEvent("assistant", [_FakePart(function_call=_FakeFunctionCall("task"))])
        yield _FakeEvent(
            "assistant",
            [
                _FakePart(
                    function_response=_FakeFunctionResponse(
                        "task",
                        {
                            "status": "completed",
                            "queued": True,
                            "queue_wait_seconds": 0.125,
                        },
                    )
                )
            ],
        )


def _confirmation_request_args(
    *,
    tool_name: str = "write_file",
    tool_args: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "originalFunctionCall": {
            "id": "fc-tool-1",
            "name": tool_name,
            "args": tool_args if tool_args is not None else {"file_path": "README.md"},
        },
        "toolConfirmation": {
            "hint": f"Tool '{tool_name}' requires approval.",
            "confirmed": False,
        },
    }


class _ApprovalFlowRunner:
    def __init__(self):
        self.messages: list[Any] = []

    async def run_async(self, *, user_id, session_id, new_message):
        del user_id, session_id
        self.messages.append(new_message)

        first_part = new_message.parts[0]
        function_response = getattr(first_part, "function_response", None)

        if function_response is None:
            yield _FakeEvent(
                "assistant",
                [
                    _FakePart(
                        function_call=_FakeFunctionCall(
                            repl.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                            call_id="req-1",
                            args=_confirmation_request_args(),
                        )
                    )
                ],
            )
            return

        decision = bool(function_response.response.get("confirmed"))
        if decision:
            yield _FakeEvent("assistant", [_FakePart("approved and resumed")])
            return

        yield _FakeEvent(
            "assistant",
            [
                _FakePart(
                    function_response=_FakeFunctionResponse(
                        "write_file",
                        {
                            "status": "rejected",
                            "message": "Tool 'write_file' was rejected.",
                        },
                    )
                )
            ],
        )


def _thread_record(
    session_id: str,
    *,
    user_id: str = "u1",
    agent_name: str = "demo",
    model: str | None = "gemini-2.5-flash",
    created_at: float | None = 1.0,
    updated_at: float = 1.0,
) -> ThreadRecord:
    return ThreadRecord(
        session_id=session_id,
        user_id=user_id,
        agent_name=agent_name,
        model=model,
        created_at=created_at,
        updated_at=updated_at,
    )


def test_handle_slash_command_help_prints_usage() -> None:
    out = io.StringIO()
    err = io.StringIO()

    result = repl.handle_slash_command("/help", stdout=out, stderr=err)

    assert result == "handled"
    assert "Interactive commands" in out.getvalue()
    assert "/threads" in out.getvalue()
    assert "/clear" in out.getvalue()
    assert "/model" in out.getvalue()
    assert "/trajectories" in out.getvalue()
    assert "/trajectories show" in out.getvalue()
    assert "/trajectories unmark" in out.getvalue()
    assert "/trajectories feedback" in out.getvalue()
    assert "/trajectories tag" in out.getvalue()
    assert "/trajectories untag" in out.getvalue()
    assert "/optimize gepa" in out.getvalue()
    assert "--detail" in out.getvalue()
    assert err.getvalue() == ""


def test_handle_slash_command_quit_variants_exit_loop() -> None:
    out = io.StringIO()
    err = io.StringIO()

    assert repl.handle_slash_command("/quit", stdout=out, stderr=err) == "exit"
    assert repl.handle_slash_command("/q", stdout=out, stderr=err) == "exit"


def test_handle_slash_command_unknown_reports_error() -> None:
    out = io.StringIO()
    err = io.StringIO()

    result = repl.handle_slash_command("/unknown", stdout=out, stderr=err)

    assert result == "handled"
    assert out.getvalue() == ""
    assert "Unknown slash command" in err.getvalue()


def test_handle_slash_command_threads_lists_and_switches_by_index(monkeypatch) -> None:
    thread_a = _thread_record("thread-a", created_at=1.0, updated_at=1.0)
    thread_b = _thread_record("thread-b", created_at=2.0, updated_at=2.0)

    monkeypatch.setattr(repl, "list_threads", lambda **_: [thread_b, thread_a])

    thread_context = repl._ThreadCommandContext(
        db_path=Path("/tmp/sessions.db"),
        user_id="u1",
        agent_name="demo",
        model="gemini-2.5-flash",
        active_session_id="thread-b",
    )
    out = io.StringIO()
    err = io.StringIO()

    list_result = repl.handle_slash_command(
        "/threads",
        stdout=out,
        stderr=err,
        thread_context=thread_context,
    )

    assert list_result == "handled"
    rendered = out.getvalue()
    assert "Threads for profile 'demo'" in rendered
    assert "*\t1\tthread-b" in rendered
    assert "-\t2\tthread-a" in rendered
    assert "Use /threads <index|thread_id|latest> to switch active thread." in rendered

    switch_result = repl.handle_slash_command(
        "/threads 2",
        stdout=out,
        stderr=err,
        thread_context=thread_context,
    )

    assert switch_result == "handled"
    assert thread_context.active_session_id == "thread-a"
    assert "switched active thread" in err.getvalue()


def test_handle_slash_command_clear_creates_new_thread(monkeypatch) -> None:
    monkeypatch.setattr(repl, "create_thread", lambda **_: _thread_record("thread-new"))

    thread_context = repl._ThreadCommandContext(
        db_path=Path("/tmp/sessions.db"),
        user_id="u1",
        agent_name="demo",
        model="gemini-2.5-flash",
        active_session_id="thread-a",
    )
    out = io.StringIO()
    err = io.StringIO()

    result = repl.handle_slash_command(
        "/clear",
        stdout=out,
        stderr=err,
        thread_context=thread_context,
    )

    assert result == "handled"
    assert thread_context.active_session_id == "thread-new"
    assert "started a new thread" in err.getvalue()


def test_handle_slash_command_model_queries_and_switches() -> None:
    switched_models: list[str | None] = []

    thread_context = repl._ThreadCommandContext(
        db_path=Path("/tmp/sessions.db"),
        user_id="u1",
        agent_name="demo",
        model="gemini-2.5-flash",
        active_session_id="thread-a",
    )
    model_context = repl._ModelCommandContext(
        model="gemini-2.5-flash",
        switch_model=lambda model: switched_models.append(model),
    )

    out = io.StringIO()
    err = io.StringIO()

    query_result = repl.handle_slash_command(
        "/model",
        stdout=out,
        stderr=err,
        thread_context=thread_context,
        model_context=model_context,
    )

    assert query_result == "handled"
    assert "Active model: gemini-2.5-flash" in out.getvalue()

    switch_result = repl.handle_slash_command(
        "/model gemini-2.5-pro",
        stdout=out,
        stderr=err,
        thread_context=thread_context,
        model_context=model_context,
    )

    assert switch_result == "handled"
    assert switched_models == ["gemini-2.5-pro"]
    assert model_context.model == "gemini-2.5-pro"
    assert thread_context.model == "gemini-2.5-pro"
    assert "switched active model" in err.getvalue()

    reset_result = repl.handle_slash_command(
        "/model default",
        stdout=out,
        stderr=err,
        thread_context=thread_context,
        model_context=model_context,
    )

    assert reset_result == "handled"
    assert switched_models == ["gemini-2.5-pro", None]
    assert model_context.model is None
    assert thread_context.model is None


def test_merge_ingested_trajectory_updates_core_and_preserves_annotations() -> None:
    existing = Trajectory(
        trace_id="t1",
        steps=[],
        status="unset",
        score=0.9,
        is_golden=True,
        tags={"env": "test"},
    )
    incoming = Trajectory(
        trace_id="t1",
        agent_name="demo_cli",
        session_id="session-1",
        steps=[AgentStep(agent_name="demo_cli")],
        status="unset",
    )

    merged, changed = repl._merge_ingested_trajectory(existing, incoming)

    assert changed is True
    assert merged.agent_name == "demo_cli"
    assert merged.session_id == "session-1"
    assert len(merged.steps) == 1
    assert merged.score == 0.9
    assert merged.is_golden is True
    assert merged.tags == {"env": "test"}


def test_merge_ingested_trajectory_noop_when_incoming_has_no_new_core_data() -> None:
    existing = Trajectory(
        trace_id="t1",
        agent_name="demo_cli",
        session_id="session-1",
        steps=[AgentStep(agent_name="demo_cli")],
        status="ok",
        score=0.7,
        is_golden=True,
    )
    incoming = Trajectory(trace_id="t1")

    merged, changed = repl._merge_ingested_trajectory(existing, incoming)

    assert changed is False
    assert merged.agent_name == "demo_cli"
    assert merged.session_id == "session-1"
    assert len(merged.steps) == 1
    assert merged.status == "ok"
    assert merged.score == 0.7
    assert merged.is_golden is True


def test_handle_trajectory_show_renders_full_details(tmp_path: Path) -> None:
    store_dir = tmp_path / "trajectories"
    store = TrajectoryStore(store_dir)
    trajectory = Trajectory(
        trace_id="abc123def456",
        session_id="session-1",
        agent_name="demo_cli",
        steps=[
            AgentStep(
                agent_name="demo_cli",
                model_call=ModelCall(
                    model="gemini-2.5-flash",
                    input_tokens=10,
                    output_tokens=5,
                    duration_ms=123.45,
                    request={
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": "hi"}],
                            }
                        ]
                    },
                    response={
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "I should greet the user.", "thought": True},
                                {"text": "Hello there!"},
                            ],
                        }
                    },
                    finish_reason="stop",
                ),
                tool_calls=[
                    ToolCall(
                        name="read_file",
                        args={"path": "README.md"},
                        response={"content": "..."},
                        duration_ms=12.0,
                    )
                ],
            )
        ],
        start_time_ns=1_000_000,
        end_time_ns=6_000_000,
        status="ok",
        score=0.8,
        is_golden=True,
        feedback=[
            FeedbackEntry(
                source="user",
                rating=0.9,
                comment="great",
                timestamp_ns=123,
                metadata={"reviewer": "qa"},
            )
        ],
        tags={"env": "test"},
    )
    store.save(trajectory)

    out = io.StringIO()
    err = io.StringIO()

    result = repl._handle_trajectory_slash_command(
        "/trajectories show abc123",
        trajectories_dir=store_dir,
        otel_traces_path=None,
        stdout=out,
        stderr=err,
    )

    assert result == "handled"
    rendered = out.getvalue()
    assert "Trajectory abc123def456:" in rendered
    assert "status=ok agent=demo_cli session=session-1" in rendered
    assert "score=0.80 golden=yes" in rendered
    assert "tags: env=test" in rendered
    assert "feedback (1):" in rendered
    assert "metadata:" in rendered
    assert "steps (1):" in rendered
    assert "model=gemini-2.5-flash input_tokens=10 output_tokens=5" in rendered
    assert "flow:" in rendered
    assert "user (1):" in rendered
    assert "hi" in rendered
    assert "thinking (1):" in rendered
    assert "I should greet the user." in rendered
    assert "assistant (1):" in rendered
    assert "Hello there!" in rendered
    assert "tool_calls (1):" in rendered
    assert "name=read_file duration_ms=12.00 error=-" in rendered
    assert "raw_payloads: hidden" in rendered
    assert "payloads: hidden" in rendered
    assert "      request:" not in rendered
    assert "      response:" not in rendered
    assert err.getvalue() == ""


def test_handle_trajectory_show_detail_includes_raw_payloads(tmp_path: Path) -> None:
    store_dir = tmp_path / "trajectories"
    store = TrajectoryStore(store_dir)
    trajectory = Trajectory(
        trace_id="abc123def456",
        agent_name="demo_cli",
        steps=[
            AgentStep(
                agent_name="demo_cli",
                model_call=ModelCall(
                    model="gemini-2.5-flash",
                    input_tokens=10,
                    output_tokens=5,
                    duration_ms=123.45,
                    request={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
                    response={
                        "content": {
                            "role": "model",
                            "parts": [{"text": "hello"}],
                        }
                    },
                ),
                tool_calls=[
                    ToolCall(
                        name="read_file",
                        args={"path": "README.md"},
                        response={"content": "..."},
                        duration_ms=12.0,
                    )
                ],
            )
        ],
    )
    store.save(trajectory)

    out = io.StringIO()
    err = io.StringIO()

    result = repl._handle_trajectory_slash_command(
        "/trajectories show abc123 --detail",
        trajectories_dir=store_dir,
        otel_traces_path=None,
        stdout=out,
        stderr=err,
    )

    assert result == "handled"
    rendered = out.getvalue()
    assert "request:" in rendered
    assert "response:" in rendered
    assert "args:" in rendered
    assert "raw_payloads: hidden" not in rendered
    assert "payloads: hidden" not in rendered
    assert err.getvalue() == ""


def test_handle_trajectory_show_usage_and_missing_prefix(tmp_path: Path) -> None:
    store_dir = tmp_path / "trajectories"
    store = TrajectoryStore(store_dir)
    store.save(Trajectory(trace_id="abc123", feedback=[FeedbackEntry(source="user")]))

    out = io.StringIO()
    err = io.StringIO()
    usage_result = repl._handle_trajectory_slash_command(
        "/trajectories show",
        trajectories_dir=store_dir,
        otel_traces_path=None,
        stdout=out,
        stderr=err,
    )
    missing_result = repl._handle_trajectory_slash_command(
        "/trajectories show does-not-exist",
        trajectories_dir=store_dir,
        otel_traces_path=None,
        stdout=out,
        stderr=err,
    )
    invalid_flag_result = repl._handle_trajectory_slash_command(
        "/trajectories show abc123 --verbose",
        trajectories_dir=store_dir,
        otel_traces_path=None,
        stdout=out,
        stderr=err,
    )

    assert usage_result == "handled"
    assert missing_result == "handled"
    assert invalid_flag_result == "handled"
    rendered_err = err.getvalue()
    assert "Usage: /trajectories show <trace_id_prefix> [--detail]" in rendered_err
    assert "No trajectory matching prefix 'does-not-exist'" in rendered_err


def test_handle_trajectory_export_summary_and_tip(tmp_path: Path) -> None:
    store_dir = tmp_path / "trajectories"
    store = TrajectoryStore(store_dir)
    store.save(Trajectory(trace_id="abc123", score=0.8, is_golden=True))

    out = io.StringIO()
    err = io.StringIO()

    result = repl._handle_trajectory_slash_command(
        "/trajectories export",
        trajectories_dir=store_dir,
        otel_traces_path=None,
        stdout=out,
        stderr=err,
    )

    assert result == "handled"
    rendered = out.getvalue()
    assert "Exported 1 trajectory(ies): 1 golden, 1 scored." in rendered
    assert "Tip: run '/trajectories export <path>' to write JSONL." in rendered
    assert err.getvalue() == ""


def test_handle_trajectory_export_writes_jsonl(tmp_path: Path) -> None:
    store_dir = tmp_path / "trajectories"
    store = TrajectoryStore(store_dir)
    store.save(Trajectory(trace_id="abc123", score=0.8, is_golden=True))

    output_path = tmp_path / "exports" / "dataset.jsonl"
    out = io.StringIO()
    err = io.StringIO()

    result = repl._handle_trajectory_slash_command(
        f"/trajectories export {output_path}",
        trajectories_dir=store_dir,
        otel_traces_path=None,
        stdout=out,
        stderr=err,
    )

    assert result == "handled"
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["trace_id"] == "abc123"
    assert "Wrote 1 trajectory(ies) to" in out.getvalue()
    assert err.getvalue() == ""


def test_handle_trajectory_export_usage_error(tmp_path: Path) -> None:
    store_dir = tmp_path / "trajectories"
    TrajectoryStore(store_dir)
    out = io.StringIO()
    err = io.StringIO()

    result = repl._handle_trajectory_slash_command(
        "/trajectories export one two",
        trajectories_dir=store_dir,
        otel_traces_path=None,
        stdout=out,
        stderr=err,
    )

    assert result == "handled"
    assert "Usage: /trajectories export [path]" in err.getvalue()


def test_handle_trajectory_feedback_adds_entry(tmp_path: Path) -> None:
    store_dir = tmp_path / "trajectories"
    store = TrajectoryStore(store_dir)
    store.save(Trajectory(trace_id="abc123"))

    out = io.StringIO()
    err = io.StringIO()
    result = repl._handle_trajectory_slash_command(
        "/trajectories feedback abc123 0.75 very helpful response",
        trajectories_dir=store_dir,
        otel_traces_path=None,
        stdout=out,
        stderr=err,
    )

    assert result == "handled"
    assert "Added feedback rating=0.75 on abc123" in out.getvalue()
    assert err.getvalue() == ""

    loaded = store.load("abc123")
    assert loaded is not None
    assert len(loaded.feedback) == 1
    assert loaded.feedback[0].source == "user"
    assert loaded.feedback[0].rating == 0.75
    assert loaded.feedback[0].comment == "very helpful response"


def test_handle_trajectory_feedback_usage_and_validation(tmp_path: Path) -> None:
    store_dir = tmp_path / "trajectories"
    store = TrajectoryStore(store_dir)
    store.save(Trajectory(trace_id="abc123"))

    out = io.StringIO()
    err = io.StringIO()

    usage_result = repl._handle_trajectory_slash_command(
        "/trajectories feedback",
        trajectories_dir=store_dir,
        otel_traces_path=None,
        stdout=out,
        stderr=err,
    )
    non_number_result = repl._handle_trajectory_slash_command(
        "/trajectories feedback abc123 not-a-number",
        trajectories_dir=store_dir,
        otel_traces_path=None,
        stdout=out,
        stderr=err,
    )
    out_of_range_result = repl._handle_trajectory_slash_command(
        "/trajectories feedback abc123 1.2",
        trajectories_dir=store_dir,
        otel_traces_path=None,
        stdout=out,
        stderr=err,
    )

    assert usage_result == "handled"
    assert non_number_result == "handled"
    assert out_of_range_result == "handled"
    rendered_err = err.getvalue()
    assert "Usage: /trajectories feedback <trace_id_prefix> <0-1> [comment]" in rendered_err
    assert "Feedback rating must be a number between 0 and 1" in rendered_err
    assert "Feedback rating must be between 0 and 1" in rendered_err


def test_handle_trajectory_tag_and_untag(tmp_path: Path) -> None:
    store_dir = tmp_path / "trajectories"
    store = TrajectoryStore(store_dir)
    store.save(Trajectory(trace_id="abc123"))

    out = io.StringIO()
    err = io.StringIO()

    tag_result = repl._handle_trajectory_slash_command(
        "/trajectories tag abc123 env staging",
        trajectories_dir=store_dir,
        otel_traces_path=None,
        stdout=out,
        stderr=err,
    )
    assert tag_result == "handled"
    assert "Set tag env=staging on abc123" in out.getvalue()

    loaded = store.load("abc123")
    assert loaded is not None
    assert loaded.tags == {"env": "staging"}

    untag_result = repl._handle_trajectory_slash_command(
        "/trajectories untag abc123 env",
        trajectories_dir=store_dir,
        otel_traces_path=None,
        stdout=out,
        stderr=err,
    )
    assert untag_result == "handled"
    assert "Removed tag env from abc123" in out.getvalue()

    loaded = store.load("abc123")
    assert loaded is not None
    assert loaded.tags == {}
    assert err.getvalue() == ""


def test_handle_trajectory_tag_and_untag_usage_errors(tmp_path: Path) -> None:
    store_dir = tmp_path / "trajectories"
    store = TrajectoryStore(store_dir)
    store.save(Trajectory(trace_id="abc123"))

    out = io.StringIO()
    err = io.StringIO()

    tag_usage = repl._handle_trajectory_slash_command(
        "/trajectories tag abc123",
        trajectories_dir=store_dir,
        otel_traces_path=None,
        stdout=out,
        stderr=err,
    )
    untag_usage = repl._handle_trajectory_slash_command(
        "/trajectories untag abc123",
        trajectories_dir=store_dir,
        otel_traces_path=None,
        stdout=out,
        stderr=err,
    )
    missing_tag_result = repl._handle_trajectory_slash_command(
        "/trajectories untag abc123 env",
        trajectories_dir=store_dir,
        otel_traces_path=None,
        stdout=out,
        stderr=err,
    )

    assert tag_usage == "handled"
    assert untag_usage == "handled"
    assert missing_tag_result == "handled"
    rendered_err = err.getvalue()
    assert "Usage: /trajectories tag <trace_id_prefix> <key> <value>" in rendered_err
    assert "Usage: /trajectories untag <trace_id_prefix> <key>" in rendered_err
    assert "Tag 'env' not found on abc123" in rendered_err


def test_handle_trajectory_unmark_clears_golden(tmp_path: Path) -> None:
    store_dir = tmp_path / "trajectories"
    store = TrajectoryStore(store_dir)
    store.save(Trajectory(trace_id="abc123", is_golden=True))

    out = io.StringIO()
    err = io.StringIO()
    result = repl._handle_trajectory_slash_command(
        "/trajectories unmark abc123",
        trajectories_dir=store_dir,
        otel_traces_path=None,
        stdout=out,
        stderr=err,
    )

    assert result == "handled"
    assert "Unmarked abc123 as golden." in out.getvalue()
    assert err.getvalue() == ""

    loaded = store.load("abc123")
    assert loaded is not None
    assert loaded.is_golden is False


def test_handle_trajectory_unmark_usage_error(tmp_path: Path) -> None:
    store_dir = tmp_path / "trajectories"
    TrajectoryStore(store_dir)
    out = io.StringIO()
    err = io.StringIO()

    result = repl._handle_trajectory_slash_command(
        "/trajectories unmark",
        trajectories_dir=store_dir,
        otel_traces_path=None,
        stdout=out,
        stderr=err,
    )

    assert result == "handled"
    assert "Usage: /trajectories unmark <trace_id_prefix>" in err.getvalue()


def test_handle_optimize_gepa_exports_dataset_and_prints_command(tmp_path: Path) -> None:
    store_dir = tmp_path / "trajectories"
    store = TrajectoryStore(store_dir)
    store.save(Trajectory(trace_id="abc123", score=0.8, is_golden=True))

    out = io.StringIO()
    err = io.StringIO()
    result = repl.handle_slash_command(
        "/optimize gepa",
        stdout=out,
        stderr=err,
        trajectories_dir=store_dir,
    )

    assert result == "handled"
    dataset_path = store_dir / "exports" / "gepa_dataset.jsonl"
    assert dataset_path.exists()
    lines = dataset_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["trace_id"] == "abc123"

    rendered_out = out.getvalue()
    assert "Wrote 1 GEPA row(s)" in rendered_out
    assert "GEPA command:" in rendered_out
    assert "Tip: rerun with --run to execute GEPA now." in rendered_out
    assert err.getvalue() == ""


def test_handle_optimize_gepa_supports_filters_and_custom_paths(tmp_path: Path) -> None:
    store_dir = tmp_path / "trajectories"
    store = TrajectoryStore(store_dir)
    store.save(Trajectory(trace_id="golden", score=0.9, is_golden=True))
    store.save(Trajectory(trace_id="other", score=0.2, is_golden=False))

    dataset_path = tmp_path / "dataset" / "filtered.jsonl"
    output_path = tmp_path / "dataset" / "optimized.txt"

    out = io.StringIO()
    err = io.StringIO()
    result = repl.handle_slash_command(
        f"/optimize gepa {dataset_path} --golden-only --min-score 0.8 --out {output_path}",
        stdout=out,
        stderr=err,
        trajectories_dir=store_dir,
    )

    assert result == "handled"
    assert dataset_path.exists()
    lines = dataset_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["trace_id"] == "golden"
    assert str(output_path) in out.getvalue()
    assert err.getvalue() == ""


def test_handle_optimize_gepa_run_executes_command(tmp_path: Path, monkeypatch) -> None:
    store_dir = tmp_path / "trajectories"
    store = TrajectoryStore(store_dir)
    store.save(Trajectory(trace_id="abc123", score=0.7, is_golden=True))

    monkeypatch.setenv(
        repl.GEPA_COMMAND_TEMPLATE_ENV_VAR,
        "gepa optimize --dataset {dataset} --out {output}",
    )

    captured: dict[str, Any] = {}

    class _Result:
        returncode = 0
        stdout = "optimization complete"
        stderr = ""

    def _fake_run(args: list[str], **kwargs: Any) -> _Result:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _Result()

    monkeypatch.setattr(repl.subprocess, "run", _fake_run)

    out = io.StringIO()
    err = io.StringIO()
    result = repl.handle_slash_command(
        "/optimize gepa --run",
        stdout=out,
        stderr=err,
        trajectories_dir=store_dir,
    )

    assert result == "handled"
    command_args = cast(list[str], captured["args"])
    assert command_args[0] == "gepa"
    assert "--dataset" in command_args
    assert "--out" in command_args
    assert cast(dict[str, Any], captured["kwargs"])["capture_output"] is True

    rendered_out = out.getvalue()
    assert "Running GEPA optimization..." in rendered_out
    assert "GEPA stdout:" in rendered_out
    assert "GEPA completed successfully." in rendered_out
    assert err.getvalue() == ""


def test_handle_optimize_gepa_run_missing_executable(tmp_path: Path, monkeypatch) -> None:
    store_dir = tmp_path / "trajectories"
    store = TrajectoryStore(store_dir)
    store.save(Trajectory(trace_id="abc123", score=0.7, is_golden=True))

    def _raise(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise FileNotFoundError("missing")

    monkeypatch.setattr(repl.subprocess, "run", _raise)

    out = io.StringIO()
    err = io.StringIO()
    result = repl.handle_slash_command(
        "/optimize gepa --run",
        stdout=out,
        stderr=err,
        trajectories_dir=store_dir,
    )

    assert result == "handled"
    assert "Failed to execute GEPA command" in err.getvalue()


def test_build_cli_agent_enables_hitl_interrupts(monkeypatch) -> None:
    monkeypatch.delenv("ADK_DYNAMIC_TASK_MAX_PARALLEL", raising=False)
    monkeypatch.delenv("ADK_DYNAMIC_TASK_CONCURRENCY_POLICY", raising=False)
    monkeypatch.delenv("ADK_DYNAMIC_TASK_QUEUE_TIMEOUT_SECONDS", raising=False)

    captured: dict[str, object] = {}

    def _fake_create_deep_agent(**kwargs: object):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(repl, "create_deep_agent", _fake_create_deep_agent)

    repl._build_cli_agent(agent_name="demo", model=None, cwd=Path("/tmp/workspace"))

    assert captured["name"] == "demo_cli"
    assert captured["execution"] == "local"
    assert captured["delegation_mode"] == "dynamic"
    dynamic_config = cast(DynamicTaskConfig, captured["dynamic_task_config"])
    assert dynamic_config.concurrency_policy == "wait"
    assert captured["interrupt_on"] == repl.INTERACTIVE_INTERRUPT_ON


def test_build_cli_agent_missing_skills_dependency_is_actionable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def _raise(**kwargs: object):
        del kwargs
        raise ImportError(
            "adk-skills-agent is required for skills support. "
            "Install it with: pip install adk-skills-agent"
        )

    monkeypatch.setattr(repl, "create_deep_agent", _raise)
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    try:
        repl._build_cli_agent(
            agent_name="demo",
            model=None,
            cwd=tmp_path,
            skills_dirs=[str(skills_dir)],
        )
    except RuntimeError as exc:
        assert "Install optional support" in str(exc)
    else:
        raise AssertionError("Expected _build_cli_agent to raise an actionable skills error")


def test_run_interactive_turn_streams_text_and_tool_events() -> None:
    out = io.StringIO()
    err = io.StringIO()

    repl.asyncio.run(
        repl._run_interactive_turn(
            runner=_TurnFakeRunner(),
            prompt="hello",
            user_id="u1",
            session_id="s1",
            stdout=out,
            stderr=err,
        )
    )

    assert out.getvalue() == "assistant> hello world\n[tool] ls\n"
    assert "[error] ls: permission denied" in err.getvalue()


def test_run_interactive_turn_renders_task_queue_notice() -> None:
    out = io.StringIO()
    err = io.StringIO()

    repl.asyncio.run(
        repl._run_interactive_turn(
            runner=_QueuedTaskRunner(),
            prompt="hello",
            user_id="u1",
            session_id="s1",
            stdout=out,
            stderr=err,
        )
    )

    rendered = out.getvalue()
    assert "[tool] task" in rendered
    assert "[tool] task -> queued (0.125s)" in rendered
    assert err.getvalue() == ""


def test_run_interactive_turn_hitl_approve_path() -> None:
    runner = _ApprovalFlowRunner()

    out = io.StringIO()
    err = io.StringIO()

    responses = iter(["approve"])

    def _input_reader(_prompt: str) -> str:
        return next(responses)

    repl.asyncio.run(
        repl._run_interactive_turn(
            runner=runner,
            prompt="please update README",
            user_id="u1",
            session_id="s1",
            stdout=out,
            stderr=err,
            input_reader=_input_reader,
            approval_context=repl._InteractiveApprovalContext(auto_approve=False),
        )
    )

    assert len(runner.messages) == 2
    first_message = runner.messages[0]
    assert first_message.parts[0].text == "please update README"

    confirmation_message = runner.messages[1]
    function_response = confirmation_message.parts[0].function_response
    assert function_response is not None
    assert function_response.name == repl.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
    assert function_response.id == "req-1"
    assert function_response.response["confirmed"] is True

    assert "approved and resumed" in out.getvalue()
    assert "requested confirmation" in err.getvalue()
    assert "approved tool 'write_file'" in err.getvalue()


def test_run_interactive_turn_hitl_reject_path() -> None:
    runner = _ApprovalFlowRunner()

    out = io.StringIO()
    err = io.StringIO()

    responses = iter(["reject"])

    def _input_reader(_prompt: str) -> str:
        return next(responses)

    repl.asyncio.run(
        repl._run_interactive_turn(
            runner=runner,
            prompt="please update README",
            user_id="u1",
            session_id="s1",
            stdout=out,
            stderr=err,
            input_reader=_input_reader,
            approval_context=repl._InteractiveApprovalContext(auto_approve=False),
        )
    )

    assert len(runner.messages) == 2
    confirmation_message = runner.messages[1]
    function_response = confirmation_message.parts[0].function_response
    assert function_response is not None
    assert function_response.response["confirmed"] is False

    assert "rejected tool 'write_file'" in err.getvalue()


def test_run_interactive_turn_hitl_auto_approve_path() -> None:
    runner = _ApprovalFlowRunner()

    out = io.StringIO()
    err = io.StringIO()

    def _input_reader(_prompt: str) -> str:
        raise AssertionError("input reader should not be called when auto-approve is enabled")

    repl.asyncio.run(
        repl._run_interactive_turn(
            runner=runner,
            prompt="please update README",
            user_id="u1",
            session_id="s1",
            stdout=out,
            stderr=err,
            input_reader=_input_reader,
            approval_context=repl._InteractiveApprovalContext(auto_approve=True),
        )
    )

    assert len(runner.messages) == 2
    confirmation_message = runner.messages[1]
    function_response = confirmation_message.parts[0].function_response
    assert function_response is not None
    assert function_response.response["confirmed"] is True

    assert "auto-approved tool 'write_file'" in err.getvalue()


def test_run_interactive_async_auto_submits_first_message(monkeypatch) -> None:
    monkeypatch.setattr(repl, "_build_cli_agent", lambda **_: object())
    monkeypatch.setattr(repl, "SqliteSessionService", lambda *_: object())

    prompts: list[str] = []

    class _EchoRunner:
        def __init__(self, *, app_name, agent, session_service):
            del app_name, agent, session_service

        async def run_async(self, *, user_id, session_id, new_message):
            del user_id, session_id
            prompt = new_message.parts[0].text
            prompts.append(prompt)
            yield _FakeEvent("assistant", [_FakePart(f"echo:{prompt}")])

    monkeypatch.setattr(repl, "Runner", _EchoRunner)

    user_inputs = iter(["/q"])

    def _input_reader(_prompt: str) -> str:
        return next(user_inputs)

    out = io.StringIO()
    err = io.StringIO()

    exit_code = repl.asyncio.run(
        repl._run_interactive_async(
            first_prompt="hello",
            model="gemini-2.5-flash",
            agent_name="demo",
            user_id="u1",
            session_id="s1",
            db_path=Path("/tmp/sessions.db"),
            auto_approve=False,
            input_reader=_input_reader,
            stdin=_FakeStdin(isatty_value=True),
            stdout=out,
            stderr=err,
        )
    )

    assert exit_code == 0
    assert prompts == ["hello"]
    assert "assistant> echo:hello" in out.getvalue()
    assert "interactive mode" in err.getvalue()


def test_run_interactive_async_thread_commands_switch_session_context(monkeypatch) -> None:
    monkeypatch.setattr(repl, "_build_cli_agent", lambda **_: object())
    monkeypatch.setattr(repl, "SqliteSessionService", lambda *_: object())

    turn_calls: list[tuple[str, str]] = []

    class _EchoRunner:
        def __init__(self, *, app_name, agent, session_service):
            del app_name, agent, session_service

        async def run_async(self, *, user_id, session_id, new_message):
            del user_id
            prompt = new_message.parts[0].text
            turn_calls.append((session_id, prompt))
            yield _FakeEvent("assistant", [_FakePart(f"echo:{session_id}:{prompt}")])

    monkeypatch.setattr(repl, "Runner", _EchoRunner)

    threads: dict[str, ThreadRecord] = {
        "thread-a": _thread_record("thread-a", created_at=1.0, updated_at=1.0),
    }

    def _create_thread(*, db_path, user_id, agent_name, model):
        del db_path, model
        thread = _thread_record(
            "thread-b",
            user_id=user_id,
            agent_name=agent_name,
            created_at=2.0,
            updated_at=2.0,
        )
        threads[thread.session_id] = thread
        return thread

    def _list_threads(*, db_path, user_id, agent_name, limit):
        del db_path, user_id
        sorted_threads = sorted(
            (thread for thread in threads.values() if thread.agent_name == agent_name),
            key=lambda thread: thread.updated_at,
            reverse=True,
        )
        return sorted_threads[:limit]

    def _get_thread(*, db_path, user_id, session_id):
        del db_path, user_id
        return threads.get(session_id)

    monkeypatch.setattr(repl, "create_thread", _create_thread)
    monkeypatch.setattr(repl, "list_threads", _list_threads)
    monkeypatch.setattr(repl, "get_thread", _get_thread)

    user_inputs = iter(["/clear", "turn one", "/threads thread-a", "turn two", "/q"])

    def _input_reader(_prompt: str) -> str:
        return next(user_inputs)

    out = io.StringIO()
    err = io.StringIO()

    exit_code = repl.asyncio.run(
        repl._run_interactive_async(
            first_prompt=None,
            model="gemini-2.5-flash",
            agent_name="demo",
            user_id="u1",
            session_id="thread-a",
            db_path=Path("/tmp/sessions.db"),
            auto_approve=False,
            input_reader=_input_reader,
            stdin=_FakeStdin(isatty_value=True),
            stdout=out,
            stderr=err,
        )
    )

    assert exit_code == 0
    assert turn_calls == [("thread-b", "turn one"), ("thread-a", "turn two")]
    assert "[thread thread-b] started a new thread." in err.getvalue()
    assert "[thread thread-a] switched active thread." in err.getvalue()


def test_run_interactive_async_model_command_switches_model_and_preserves_thread(
    monkeypatch,
) -> None:
    class _FakeAgent:
        def __init__(self, model: str | None):
            self.model = model

    def _fake_build_cli_agent(*, agent_name, model, cwd, **kwargs):
        del agent_name, cwd, kwargs
        return _FakeAgent(model)

    monkeypatch.setattr(repl, "_build_cli_agent", _fake_build_cli_agent)
    monkeypatch.setattr(repl, "SqliteSessionService", lambda *_: object())

    turn_calls: list[tuple[str, str, str | None]] = []

    class _EchoRunner:
        def __init__(self, *, app_name, agent, session_service):
            del app_name, session_service
            self.model = agent.model

        async def run_async(self, *, user_id, session_id, new_message):
            del user_id
            prompt = new_message.parts[0].text
            turn_calls.append((session_id, prompt, self.model))
            model_label = self.model or "default"
            yield _FakeEvent("assistant", [_FakePart(f"echo:{model_label}:{prompt}")])

    monkeypatch.setattr(repl, "Runner", _EchoRunner)

    user_inputs = iter(
        [
            "/model gemini-2.5-pro",
            "turn one",
            "/model default",
            "turn two",
            "/q",
        ]
    )

    def _input_reader(_prompt: str) -> str:
        return next(user_inputs)

    out = io.StringIO()
    err = io.StringIO()

    exit_code = repl.asyncio.run(
        repl._run_interactive_async(
            first_prompt=None,
            model="gemini-2.5-flash",
            agent_name="demo",
            user_id="u1",
            session_id="thread-a",
            db_path=Path("/tmp/sessions.db"),
            auto_approve=False,
            input_reader=_input_reader,
            stdin=_FakeStdin(isatty_value=True),
            stdout=out,
            stderr=err,
        )
    )

    assert exit_code == 0
    assert turn_calls == [
        ("thread-a", "turn one", "gemini-2.5-pro"),
        ("thread-a", "turn two", None),
    ]
    assert "[model gemini-2.5-pro] switched active model." in err.getvalue()
    assert "[model default] switched active model." in err.getvalue()


def test_run_interactive_async_returns_zero_when_no_tty_and_no_first_prompt(monkeypatch) -> None:
    monkeypatch.setattr(
        repl, "_build_cli_agent", lambda **_: (_ for _ in ()).throw(AssertionError())
    )

    exit_code = repl.asyncio.run(
        repl._run_interactive_async(
            first_prompt=None,
            model="gemini-2.5-flash",
            agent_name="demo",
            user_id="u1",
            session_id="s1",
            db_path=Path("/tmp/sessions.db"),
            auto_approve=False,
            stdin=_FakeStdin(isatty_value=False),
        )
    )

    assert exit_code == 0


def test_handle_slash_command_compact_returns_compact() -> None:
    out = io.StringIO()
    err = io.StringIO()

    result = repl.handle_slash_command("/compact", stdout=out, stderr=err)

    assert result == "compact"
    assert "compact and summarize" in out.getvalue()
    assert err.getvalue() == ""


def test_handle_slash_command_help_mentions_compact() -> None:
    out = io.StringIO()
    err = io.StringIO()

    repl.handle_slash_command("/help", stdout=out, stderr=err)

    assert "/compact" in out.getvalue()


def test_build_cli_agent_passes_summarization_config(monkeypatch) -> None:
    monkeypatch.delenv("ADK_DYNAMIC_TASK_MAX_PARALLEL", raising=False)
    monkeypatch.delenv("ADK_DYNAMIC_TASK_CONCURRENCY_POLICY", raising=False)
    monkeypatch.delenv("ADK_DYNAMIC_TASK_QUEUE_TIMEOUT_SECONDS", raising=False)

    from adk_deepagents.types import SummarizationConfig

    captured: dict[str, object] = {}

    def _fake_create_deep_agent(**kwargs: object):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(repl, "create_deep_agent", _fake_create_deep_agent)

    config = SummarizationConfig()
    repl._build_cli_agent(
        agent_name="demo",
        model=None,
        cwd=Path("/tmp/workspace"),
        summarization=config,
    )

    assert captured["summarization"] is config


def test_build_cli_agent_without_summarization_omits_key(monkeypatch) -> None:
    monkeypatch.delenv("ADK_DYNAMIC_TASK_MAX_PARALLEL", raising=False)
    monkeypatch.delenv("ADK_DYNAMIC_TASK_CONCURRENCY_POLICY", raising=False)
    monkeypatch.delenv("ADK_DYNAMIC_TASK_QUEUE_TIMEOUT_SECONDS", raising=False)

    captured: dict[str, object] = {}

    def _fake_create_deep_agent(**kwargs: object):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(repl, "create_deep_agent", _fake_create_deep_agent)

    repl._build_cli_agent(
        agent_name="demo",
        model=None,
        cwd=Path("/tmp/workspace"),
    )

    assert "summarization" not in captured


def test_run_interactive_returns_error_code_on_exception(monkeypatch, capsys) -> None:
    def _raise(coro):
        coro.close()
        raise RuntimeError("boom")

    monkeypatch.setattr(repl.asyncio, "run", _raise)

    exit_code = repl.run_interactive(
        first_prompt="hello",
        model="gemini-2.5-flash",
        agent_name="demo",
        user_id="u1",
        session_id="s1",
        db_path=Path("/tmp/sessions.db"),
        auto_approve=False,
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error: boom" in captured.err
