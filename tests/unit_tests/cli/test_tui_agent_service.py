"""Unit tests for TUI agent service tool detail rendering."""

from __future__ import annotations

import asyncio
from pathlib import Path

from adk_deepagents.cli.tui.agent_service import (
    AgentService,
    _activity_label_for_phase,
    _chunk_stream_text,
    _coerce_payload_dict,
    _format_tool_call_detail,
    _format_tool_response_detail,
)


class _FakeFunctionCall:
    def __init__(self, name: str, args: object) -> None:
        self.name = name
        self.args = args


class _FakeFunctionResponse:
    def __init__(self, name: str, response: object) -> None:
        self.name = name
        self.response = response


class _FakePart:
    def __init__(
        self,
        *,
        text: str | None = None,
        thought: bool = False,
        function_call: _FakeFunctionCall | None = None,
        function_response: _FakeFunctionResponse | None = None,
    ) -> None:
        self.function_call = function_call
        self.function_response = function_response
        self.text = text
        self.thought = thought


class _FakeContent:
    def __init__(self, parts: list[_FakePart]) -> None:
        self.parts = parts


class _FakeEvent:
    def __init__(self, parts: list[_FakePart]) -> None:
        self.content = _FakeContent(parts)
        self.error_message = None


def _service() -> AgentService:
    return AgentService(
        agent_name="demo",
        user_id="u1",
        model=None,
        db_path=Path("/tmp/demo.db"),
        auto_approve=False,
        session_id="s1",
    )


def test_coerce_payload_dict_parses_json_string() -> None:
    payload = _coerce_payload_dict('{"pattern":"**/*.py","path":"/"}')
    assert payload == {"pattern": "**/*.py", "path": "/"}


def test_format_tool_call_detail_for_task_includes_subagent_and_prompt() -> None:
    detail = _format_tool_call_detail(
        "task",
        {
            "subagent_type": "summarizer",
            "description": "Summarize Python files by module.",
            "task_id": "task_3",
        },
    )

    assert detail is not None
    assert "subagent=summarizer" in detail
    assert "task_id=task_3" in detail
    assert "description=Summarize Python files by module." in detail


def test_format_tool_response_detail_for_glob_includes_entry_count() -> None:
    detail = _format_tool_response_detail(
        "glob",
        {
            "status": "success",
            "entries": [{"path": "/a.py"}, {"path": "/b.py"}, {"path": "/c.py"}],
        },
    )
    assert detail == "status=success, entries=3"


def test_emit_event_updates_includes_tool_call_and_result_details() -> None:
    service = _service()
    event = _FakeEvent(
        [
            _FakePart(
                function_call=_FakeFunctionCall(
                    "glob",
                    {"pattern": "**/*.py", "path": "/"},
                )
            ),
            _FakePart(
                function_response=_FakeFunctionResponse(
                    "glob",
                    {"status": "success", "entries": [{"path": "/a.py"}]},
                )
            ),
        ]
    )

    asyncio.run(service._emit_event_updates(event))

    call_update = service.updates.get_nowait()
    result_update = service.updates.get_nowait()

    assert call_update.kind == "tool_call"
    assert call_update.tool_name == "glob"
    assert call_update.tool_detail == "pattern=**/*.py, path=/"

    assert result_update.kind == "tool_result"
    assert result_update.tool_name == "glob"
    assert result_update.tool_detail == "status=success, entries=1"


def test_format_tool_response_detail_for_task_includes_queue_metadata() -> None:
    detail = _format_tool_response_detail(
        "task",
        {
            "status": "completed",
            "subagent_type": "summarizer",
            "task_id": "task_2",
            "queued": True,
            "queue_wait_seconds": 0.125,
        },
    )

    assert detail is not None
    assert "status=completed" in detail
    assert "subagent=summarizer" in detail
    assert "task_id=task_2" in detail
    assert "queued=True" in detail
    assert "queue_wait=0.125s" in detail


def test_chunk_stream_text_splits_long_text_preserving_content() -> None:
    text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    chunks = _chunk_stream_text(text, chunk_size=16)

    assert len(chunks) > 1
    assert "".join(chunks) == text


def test_emit_event_updates_streams_assistant_text_in_chunks() -> None:
    service = _service()
    text = "This response should arrive in several chunks for better UI streaming visibility."
    event = _FakeEvent([_FakePart(text=text)])

    asyncio.run(service._emit_event_updates(event))

    updates = []
    while not service.updates.empty():
        updates.append(service.updates.get_nowait())

    assert len(updates) > 1
    assert all(update.kind == "assistant_delta" for update in updates)
    assert "".join(update.text or "" for update in updates) == text


def test_activity_labels_cover_all_phases() -> None:
    assert _activity_label_for_phase("working") == "Working"
    assert _activity_label_for_phase("thinking") == "Thinking"
    assert _activity_label_for_phase("tool") == "Running tools"
    assert _activity_label_for_phase("responding") == "Responding"
    assert _activity_label_for_phase("approval") == "Awaiting approval"
