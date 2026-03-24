"""Unit tests for TUI agent service tool detail rendering."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path

from adk_deepagents.cli.interactive import (
    _InteractiveApprovalContext,
    _ThreadCommandContext,
)
from adk_deepagents.cli.tui.agent_service import (
    AgentService,
    TrajectorySummary,
    UiUpdate,
    _activity_label_for_phase,
    _chunk_stream_text,
    _coerce_payload_dict,
    _expand_file_references,
    _extract_diff_content,
    _extract_tool_output,
    _format_tool_call_detail,
    _format_tool_response_detail,
    _generate_unified_diff,
    _guess_language_from_path,
    _SharedMessageQueue,
)
from adk_deepagents.optimization.store import TrajectoryStore
from adk_deepagents.optimization.trajectory import AgentStep, Trajectory


class _FakeFunctionCall:
    def __init__(self, name: str, args: object, id: str | None = None) -> None:
        self.name = name
        self.args = args
        self.id = id


class _FakeFunctionResponse:
    def __init__(self, name: str, response: object, id: str | None = None) -> None:
        self.name = name
        self.response = response
        self.id = id


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


def test_list_trajectory_summaries_empty_when_store_unavailable() -> None:
    service = _service()
    summaries = asyncio.run(service.list_trajectory_summaries(sync_from_otel=False))
    assert summaries == []


def test_list_trajectory_summaries_reads_and_sorts_entries(tmp_path: Path) -> None:
    store_dir = tmp_path / "trajectories"
    store = TrajectoryStore(store_dir)
    store.save(
        Trajectory(
            trace_id="older",
            agent_name="demo",
            steps=[AgentStep(agent_name="demo")],
            start_time_ns=100,
            status="ok",
            score=0.4,
        )
    )
    store.save(
        Trajectory(
            trace_id="newer",
            agent_name="demo",
            steps=[AgentStep(agent_name="demo")],
            start_time_ns=200,
            status="error",
            score=0.9,
            is_golden=True,
        )
    )

    service = AgentService(
        agent_name="demo",
        user_id="u1",
        model=None,
        db_path=tmp_path / "demo.db",
        auto_approve=False,
        session_id="s1",
        trajectories_dir=store_dir,
    )
    summaries = asyncio.run(service.list_trajectory_summaries(sync_from_otel=False))

    assert len(summaries) == 2
    assert isinstance(summaries[0], TrajectorySummary)
    assert [entry.trace_id for entry in summaries] == ["newer", "older"]
    assert summaries[0].status == "error"
    assert summaries[0].score == 0.9
    assert summaries[0].is_golden is True


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


class TestCancelTurn:
    """Test the cancel_turn() method."""

    def test_cancel_returns_false_when_not_busy(self) -> None:
        service = _service()
        assert service.cancel_turn() is False

    def test_cancel_returns_false_when_no_task(self) -> None:
        service = _service()
        service._busy = True
        service._turn_task = None
        assert service.cancel_turn() is False

    def test_cancel_returns_true_and_cancels_task(self) -> None:
        service = _service()
        service._busy = True

        async def _noop() -> None:
            await asyncio.sleep(10)

        loop = asyncio.new_event_loop()
        try:
            task = loop.create_task(_noop())
            service._turn_task = task
            result = service.cancel_turn()
            assert result is True
            assert task.cancelled() or task.cancel()
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# _extract_diff_content tests
# ---------------------------------------------------------------------------


class TestExtractDiffContent:
    """Test the _extract_diff_content helper function."""

    def test_returns_none_for_empty_response(self) -> None:
        assert _extract_diff_content("edit_file", {}) is None

    def test_returns_none_for_non_diff_content(self) -> None:
        assert _extract_diff_content("edit_file", {"status": "success"}) is None

    def test_generates_diff_from_call_args_on_success(self) -> None:
        result = _extract_diff_content(
            "edit_file",
            {"status": "success", "path": "readme.md"},
            call_args={
                "file_path": "readme.md",
                "old_string": "Hello world",
                "new_string": "Hello universe",
            },
        )
        assert result is not None
        assert "--- a/readme.md" in result
        assert "+++ b/readme.md" in result
        assert "-Hello world" in result
        assert "+Hello universe" in result

    def test_no_diff_from_call_args_on_error_status(self) -> None:
        result = _extract_diff_content(
            "edit_file",
            {"status": "error", "message": "not found"},
            call_args={
                "file_path": "readme.md",
                "old_string": "Hello",
                "new_string": "World",
            },
        )
        assert result is None

    def test_no_diff_from_call_args_when_strings_identical(self) -> None:
        result = _extract_diff_content(
            "edit_file",
            {"status": "success", "path": "f.py"},
            call_args={
                "file_path": "f.py",
                "old_string": "same",
                "new_string": "same",
            },
        )
        assert result is None

    def test_no_diff_from_call_args_when_missing_old_string(self) -> None:
        result = _extract_diff_content(
            "edit_file",
            {"status": "success", "path": "f.py"},
            call_args={"file_path": "f.py", "new_string": "new"},
        )
        assert result is None

    def test_call_args_uses_response_path_as_fallback(self) -> None:
        result = _extract_diff_content(
            "edit_file",
            {"status": "success", "path": "fallback.py"},
            call_args={"old_string": "a", "new_string": "b"},
        )
        assert result is not None
        assert "fallback.py" in result

    def test_extracts_diff_from_diff_key_starting_with_triple_dash(self) -> None:
        diff_text = "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n-old\n+new\n context"
        result = _extract_diff_content("edit_file", {"diff": diff_text})
        assert result is not None
        assert "--- a/file.py" in result

    def test_extracts_diff_from_diff_key_starting_with_hunk(self) -> None:
        diff_text = "@@ -1,3 +1,3 @@\n-old\n+new\n context"
        result = _extract_diff_content("edit_file", {"diff": diff_text})
        assert result is not None
        assert "@@ -1,3 +1,3 @@" in result

    def test_extracts_diff_from_diff_key_starting_with_diff_git(self) -> None:
        diff_text = "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py"
        result = _extract_diff_content("edit_file", {"diff": diff_text})
        assert result is not None
        assert "diff --git" in result

    def test_extracts_diff_from_diff_key_with_embedded_hunk(self) -> None:
        diff_text = "some header\n@@ -1,3 +1,3 @@\n-old\n+new"
        result = _extract_diff_content("edit_file", {"diff": diff_text})
        assert result is not None

    def test_extracts_diff_from_diff_key_with_embedded_triple_dash(self) -> None:
        diff_text = "some header\n--- a/file.py\n+++ b/file.py"
        result = _extract_diff_content("edit_file", {"diff": diff_text})
        assert result is not None

    def test_ignores_non_diff_string_in_diff_key(self) -> None:
        result = _extract_diff_content("edit_file", {"diff": "just some random text"})
        assert result is None

    def test_extracts_diff_from_execute_output(self) -> None:
        output = "diff --git a/file.py b/file.py\nindex 1234..5678\n--- a/file.py\n+++ b/file.py"
        result = _extract_diff_content("execute", {"output": output})
        assert result is not None
        assert "diff --git" in result

    def test_extracts_diff_from_execute_output_triple_dash(self) -> None:
        output = "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"
        result = _extract_diff_content("execute", {"output": output})
        assert result is not None

    def test_ignores_non_diff_execute_output(self) -> None:
        result = _extract_diff_content("execute", {"output": "Build succeeded."})
        assert result is None

    def test_ignores_empty_diff_key(self) -> None:
        assert _extract_diff_content("edit_file", {"diff": ""}) is None
        assert _extract_diff_content("edit_file", {"diff": "   "}) is None

    def test_strips_whitespace(self) -> None:
        diff_text = "  \n--- a/file.py\n+++ b/file.py\n  "
        result = _extract_diff_content("edit_file", {"diff": diff_text})
        assert result is not None
        assert result.startswith("---")

    def test_non_execute_tool_ignores_output_key(self) -> None:
        """Only the 'execute' tool should check the 'output' key for diffs."""
        result = _extract_diff_content("glob", {"output": "diff --git a/file.py b/file.py"})
        assert result is None

    def test_multiline_diff_from_call_args(self) -> None:
        result = _extract_diff_content(
            "edit_file",
            {"status": "success", "path": "app.py"},
            call_args={
                "file_path": "app.py",
                "old_string": "line1\nline2\nline3",
                "new_string": "line1\nmodified\nline3",
            },
        )
        assert result is not None
        assert "-line2" in result
        assert "+modified" in result
        assert " line1" in result  # context line


class TestGenerateUnifiedDiff:
    """Test the _generate_unified_diff helper."""

    def test_basic_diff(self) -> None:
        result = _generate_unified_diff("old line\n", "new line\n", file_path="test.py")
        assert result is not None
        assert "--- a/test.py" in result
        assert "+++ b/test.py" in result
        assert "-old line" in result
        assert "+new line" in result

    def test_identical_text_returns_none(self) -> None:
        assert _generate_unified_diff("same", "same") is None

    def test_empty_to_content(self) -> None:
        result = _generate_unified_diff("", "new content\n", file_path="f.py")
        assert result is not None
        assert "+new content" in result

    def test_content_to_empty(self) -> None:
        result = _generate_unified_diff("old content\n", "", file_path="f.py")
        assert result is not None
        assert "-old content" in result

    def test_default_file_path(self) -> None:
        result = _generate_unified_diff("a\n", "b\n")
        assert result is not None
        assert "a/file" in result
        assert "b/file" in result

    def test_context_lines(self) -> None:
        old = "line1\nline2\nline3\nline4\nline5\n"
        new = "line1\nline2\nCHANGED\nline4\nline5\n"
        result = _generate_unified_diff(old, new, context_lines=1)
        assert result is not None
        assert " line2" in result
        assert "-line3" in result
        assert "+CHANGED" in result
        assert " line4" in result


class TestDiffContentEmission:
    """Test that diff_content UiUpdates are emitted from _emit_event_updates."""

    def test_diff_content_emitted_for_edit_file_with_call_args(self) -> None:
        service = _service()
        event = _FakeEvent(
            [
                _FakePart(
                    function_call=_FakeFunctionCall(
                        "edit_file",
                        {
                            "file_path": "readme.md",
                            "old_string": "Hello",
                            "new_string": "Goodbye",
                        },
                    )
                ),
                _FakePart(
                    function_response=_FakeFunctionResponse(
                        "edit_file",
                        {"status": "success", "path": "readme.md", "occurrences": 1},
                    )
                ),
            ]
        )

        asyncio.run(service._emit_event_updates(event))

        updates = []
        while not service.updates.empty():
            updates.append(service.updates.get_nowait())

        diff_updates = [u for u in updates if u.kind == "diff_content"]
        assert len(diff_updates) == 1
        assert "-Hello" in (diff_updates[0].text or "")
        assert "+Goodbye" in (diff_updates[0].text or "")

    def test_diff_content_emitted_for_edit_file_with_diff_key(self) -> None:
        service = _service()
        diff_text = "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"
        event = _FakeEvent(
            [
                _FakePart(
                    function_response=_FakeFunctionResponse(
                        "edit_file",
                        {"status": "success", "diff": diff_text},
                    )
                ),
            ]
        )

        asyncio.run(service._emit_event_updates(event))

        updates = []
        while not service.updates.empty():
            updates.append(service.updates.get_nowait())

        diff_updates = [u for u in updates if u.kind == "diff_content"]
        assert len(diff_updates) == 1
        assert "--- a/file.py" in (diff_updates[0].text or "")

    def test_no_diff_content_for_response_without_diff(self) -> None:
        service = _service()
        event = _FakeEvent(
            [
                _FakePart(
                    function_response=_FakeFunctionResponse(
                        "glob",
                        {"status": "success", "entries": [{"path": "/a.py"}]},
                    )
                ),
            ]
        )

        asyncio.run(service._emit_event_updates(event))

        updates = []
        while not service.updates.empty():
            updates.append(service.updates.get_nowait())

        diff_updates = [u for u in updates if u.kind == "diff_content"]
        assert len(diff_updates) == 0


class TestCrossEventDiffRendering:
    """Test that diffs are emitted when call and response arrive in separate events."""

    def test_diff_emitted_when_call_and_response_in_separate_events(self) -> None:
        """Simulates the real-world ADK flow: function_call and function_response
        arrive in separate events."""
        service = _service()

        call_event = _FakeEvent(
            [
                _FakePart(
                    function_call=_FakeFunctionCall(
                        "edit_file",
                        {
                            "file_path": "readme.md",
                            "old_string": "Hello",
                            "new_string": "Goodbye",
                        },
                        id="call_abc",
                    )
                ),
            ]
        )
        response_event = _FakeEvent(
            [
                _FakePart(
                    function_response=_FakeFunctionResponse(
                        "edit_file",
                        {"status": "success", "path": "readme.md", "occurrences": 1},
                        id="call_abc",
                    )
                ),
            ]
        )

        asyncio.run(service._emit_event_updates(call_event))
        asyncio.run(service._emit_event_updates(response_event))

        updates = []
        while not service.updates.empty():
            updates.append(service.updates.get_nowait())

        diff_updates = [u for u in updates if u.kind == "diff_content"]
        assert len(diff_updates) == 1
        assert "-Hello" in (diff_updates[0].text or "")
        assert "+Goodbye" in (diff_updates[0].text or "")

    def test_diff_emitted_with_approval_event_between_call_and_response(self) -> None:
        """Simulates the HITL approval flow: function_call, then approval event,
        then function_response in separate events."""
        service = _service()

        call_event = _FakeEvent(
            [
                _FakePart(
                    function_call=_FakeFunctionCall(
                        "edit_file",
                        {
                            "file_path": "app.py",
                            "old_string": "debug = True",
                            "new_string": "debug = False",
                        },
                        id="call_71f2",
                    )
                ),
            ]
        )
        # Intermediate approval event (function_response with awaiting_approval).
        approval_event = _FakeEvent(
            [
                _FakePart(
                    function_response=_FakeFunctionResponse(
                        "edit_file",
                        {"status": "awaiting_approval", "path": "app.py"},
                        id="call_71f2",
                    )
                ),
            ]
        )
        # adk_request_confirmation call (should be skipped).
        confirmation_event = _FakeEvent(
            [
                _FakePart(
                    function_call=_FakeFunctionCall(
                        "adk_request_confirmation",
                        {"tool_name": "edit_file"},
                        id="call_conf",
                    )
                ),
            ]
        )
        # Final success response.
        success_event = _FakeEvent(
            [
                _FakePart(
                    function_response=_FakeFunctionResponse(
                        "edit_file",
                        {"status": "success", "path": "app.py", "occurrences": 1},
                        id="call_71f2",
                    )
                ),
            ]
        )

        asyncio.run(service._emit_event_updates(call_event))
        asyncio.run(service._emit_event_updates(approval_event))
        asyncio.run(service._emit_event_updates(confirmation_event))
        asyncio.run(service._emit_event_updates(success_event))

        updates = []
        while not service.updates.empty():
            updates.append(service.updates.get_nowait())

        diff_updates = [u for u in updates if u.kind == "diff_content"]
        assert len(diff_updates) == 1
        assert "-debug = True" in (diff_updates[0].text or "")
        assert "+debug = False" in (diff_updates[0].text or "")

    def test_diff_emitted_with_fallback_when_ids_dont_match(self) -> None:
        """When function_response has no id or a different id, the fallback
        logic should still match against the single pending entry."""
        service = _service()

        call_event = _FakeEvent(
            [
                _FakePart(
                    function_call=_FakeFunctionCall(
                        "edit_file",
                        {
                            "file_path": "f.py",
                            "old_string": "old",
                            "new_string": "new",
                        },
                        id="call_x",
                    )
                ),
            ]
        )
        response_event = _FakeEvent(
            [
                _FakePart(
                    function_response=_FakeFunctionResponse(
                        "edit_file",
                        {"status": "success", "path": "f.py", "occurrences": 1},
                        # No id — fallback should kick in.
                    )
                ),
            ]
        )

        asyncio.run(service._emit_event_updates(call_event))
        asyncio.run(service._emit_event_updates(response_event))

        updates = []
        while not service.updates.empty():
            updates.append(service.updates.get_nowait())

        diff_updates = [u for u in updates if u.kind == "diff_content"]
        assert len(diff_updates) == 1
        assert "-old" in (diff_updates[0].text or "")
        assert "+new" in (diff_updates[0].text or "")

    def test_pending_edit_args_cleared_after_consumption(self) -> None:
        """After the diff is emitted, the pending args should be removed."""
        service = _service()

        call_event = _FakeEvent(
            [
                _FakePart(
                    function_call=_FakeFunctionCall(
                        "edit_file",
                        {
                            "file_path": "f.py",
                            "old_string": "a",
                            "new_string": "b",
                        },
                        id="call_1",
                    )
                ),
            ]
        )
        response_event = _FakeEvent(
            [
                _FakePart(
                    function_response=_FakeFunctionResponse(
                        "edit_file",
                        {"status": "success", "path": "f.py", "occurrences": 1},
                        id="call_1",
                    )
                ),
            ]
        )

        asyncio.run(service._emit_event_updates(call_event))
        asyncio.run(service._emit_event_updates(response_event))

        assert len(service._pending_edit_args) == 0

    def test_same_event_call_and_response_still_works(self) -> None:
        """Existing same-event pattern should still work with the new code."""
        service = _service()
        event = _FakeEvent(
            [
                _FakePart(
                    function_call=_FakeFunctionCall(
                        "edit_file",
                        {
                            "file_path": "readme.md",
                            "old_string": "Hello",
                            "new_string": "Goodbye",
                        },
                        id="call_same",
                    )
                ),
                _FakePart(
                    function_response=_FakeFunctionResponse(
                        "edit_file",
                        {"status": "success", "path": "readme.md", "occurrences": 1},
                        id="call_same",
                    )
                ),
            ]
        )

        asyncio.run(service._emit_event_updates(event))

        updates = []
        while not service.updates.empty():
            updates.append(service.updates.get_nowait())

        diff_updates = [u for u in updates if u.kind == "diff_content"]
        assert len(diff_updates) == 1
        assert "-Hello" in (diff_updates[0].text or "")
        assert "+Goodbye" in (diff_updates[0].text or "")


# ---------------------------------------------------------------------------
# Message queuing tests
# ---------------------------------------------------------------------------


class TestQueueMessage:
    """Test the queue_message() method."""

    def test_queue_message_buffers_text(self) -> None:
        service = _service()
        asyncio.run(service.queue_message("hello"))
        assert service._queued_messages == ["hello"]

    def test_queue_message_emits_queued_message_update(self) -> None:
        service = _service()
        asyncio.run(service.queue_message("ping"))
        update = service.updates.get_nowait()
        assert update.kind == "queued_message"
        assert update.text == "ping"

    def test_queue_message_buffers_multiple(self) -> None:
        service = _service()
        asyncio.run(service.queue_message("first"))
        asyncio.run(service.queue_message("second"))
        assert service._queued_messages == ["first", "second"]

        updates: list[UiUpdate] = []
        while not service.updates.empty():
            updates.append(service.updates.get_nowait())
        assert len(updates) == 2
        assert all(u.kind == "queued_message" for u in updates)
        assert updates[0].text == "first"
        assert updates[1].text == "second"

    def test_queue_message_pushes_to_shared_queue(self) -> None:
        service = _service()
        asyncio.run(service.queue_message("shared"))
        drained = service._shared_queue.drain()
        assert len(drained) == 1
        assert drained[0] == {"text": "shared"}

    def test_queue_message_shared_queue_multiple(self) -> None:
        service = _service()
        asyncio.run(service.queue_message("a"))
        asyncio.run(service.queue_message("b"))
        drained = service._shared_queue.drain()
        assert len(drained) == 2
        assert drained[0] == {"text": "a"}
        assert drained[1] == {"text": "b"}
        # Drain again should be empty.
        assert service._shared_queue.drain() == []


class TestHandleInputWhenBusy:
    """Test that handle_input() queues messages when a turn is running."""

    def test_queues_message_when_busy(self) -> None:
        service = _service()
        service._busy = True
        asyncio.run(service.handle_input("steer this way"))
        assert service._queued_messages == ["steer this way"]

        update = service.updates.get_nowait()
        assert update.kind == "queued_message"
        assert update.text == "steer this way"

    def test_does_not_queue_empty_input(self) -> None:
        service = _service()
        service._busy = True
        asyncio.run(service.handle_input("  "))
        assert service._queued_messages == []
        assert service.updates.empty()

    def test_does_not_queue_slash_commands(self) -> None:
        """Slash commands should still be handled normally even when busy."""
        service = _service()
        service._busy = True
        # We can't fully test slash commands without initialization, but
        # we verify it does NOT go to queue_message by checking the buffer.
        # The slash command handler will fail (no runner), but the point is
        # the message should not be in _queued_messages.
        with suppress(Exception):
            asyncio.run(service.handle_input("/help"))
        assert service._queued_messages == []


class TestPostTurnQueueDrain:
    """Test that _run_turn auto-starts a follow-up turn for queued messages."""

    def test_queued_messages_trigger_follow_up_turn(self) -> None:
        """Messages left in _shared_queue after a turn should start a new turn."""
        service = _service()
        service._thread_context = _ThreadCommandContext(
            db_path=service.db_path,
            user_id=service.user_id,
            agent_name=service.agent_name,
            model=service.model,
            active_session_id=service.session_id,
        )
        service._approval_context = _InteractiveApprovalContext(auto_approve=True)

        # Simulate a runner that completes immediately (no events).
        async def _empty_run(**_kwargs: object) -> None:  # type: ignore[override]
            # Yield nothing — turn ends immediately.
            return
            yield  # make it an async generator  # noqa: RET504

        class _FakeRunner:
            def run_async(self, **kwargs: object) -> object:
                return _empty_run(**kwargs)

        service._runner = _FakeRunner()

        # Push a message into the shared queue *before* the turn runs.
        # This simulates a message queued during the turn that wasn't
        # drained by before_model_callback.
        service._shared_queue.push("follow-up instruction")

        asyncio.run(service._run_turn("initial prompt"))

        # Collect all updates.
        updates: list[UiUpdate] = []
        while not service.updates.empty():
            updates.append(service.updates.get_nowait())

        # Should have two turn_started / turn_finished pairs:
        # one for the initial turn, one for the follow-up.
        turn_started = [u for u in updates if u.kind == "turn_started"]
        turn_finished = [u for u in updates if u.kind == "turn_finished"]
        assert len(turn_started) >= 2, f"Expected 2+ turn_started, got {len(turn_started)}"
        assert len(turn_finished) >= 2, f"Expected 2+ turn_finished, got {len(turn_finished)}"

        # The follow-up prompt should appear as a user_message.
        user_messages = [u for u in updates if u.kind == "user_message"]
        assert any("follow-up instruction" in (u.text or "") for u in user_messages)

    def test_no_follow_up_when_queue_empty(self) -> None:
        """No extra turn should start when the shared queue is empty."""
        service = _service()
        service._thread_context = _ThreadCommandContext(
            db_path=service.db_path,
            user_id=service.user_id,
            agent_name=service.agent_name,
            model=service.model,
            active_session_id=service.session_id,
        )
        service._approval_context = _InteractiveApprovalContext(auto_approve=True)

        async def _empty_run(**_kwargs: object) -> None:  # type: ignore[override]
            return
            yield  # noqa: RET504

        class _FakeRunner:
            def run_async(self, **kwargs: object) -> object:
                return _empty_run(**kwargs)

        service._runner = _FakeRunner()

        asyncio.run(service._run_turn("hello"))

        updates: list[UiUpdate] = []
        while not service.updates.empty():
            updates.append(service.updates.get_nowait())

        # Only one turn cycle.
        turn_started = [u for u in updates if u.kind == "turn_started"]
        turn_finished = [u for u in updates if u.kind == "turn_finished"]
        assert len(turn_started) == 1
        assert len(turn_finished) == 1


class TestSharedMessageQueue:
    """Test the _SharedMessageQueue thread-safe buffer."""

    def test_push_and_drain(self) -> None:
        q = _SharedMessageQueue()
        q.push("hello")
        result = q.drain()
        assert result == [{"text": "hello"}]

    def test_drain_clears(self) -> None:
        q = _SharedMessageQueue()
        q.push("a")
        q.drain()
        assert q.drain() == []

    def test_multiple_push(self) -> None:
        q = _SharedMessageQueue()
        q.push("x")
        q.push("y")
        q.push("z")
        result = q.drain()
        assert len(result) == 3
        assert result[0] == {"text": "x"}
        assert result[1] == {"text": "y"}
        assert result[2] == {"text": "z"}

    def test_drain_empty(self) -> None:
        q = _SharedMessageQueue()
        assert q.drain() == []

    def test_thread_safety(self) -> None:
        """Push from multiple threads, drain should get all messages."""
        import threading

        q = _SharedMessageQueue()
        barrier = threading.Barrier(4)

        def pusher(prefix: str) -> None:
            barrier.wait()
            for i in range(10):
                q.push(f"{prefix}-{i}")

        threads = [threading.Thread(target=pusher, args=(f"t{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        result = q.drain()
        assert len(result) == 40
        assert q.drain() == []


# ---------------------------------------------------------------------------
# File reference expansion tests
# ---------------------------------------------------------------------------


class TestExpandFileReferences:
    """Test the _expand_file_references helper function."""

    def test_no_references(self) -> None:
        text, refs = _expand_file_references("hello world")
        assert text == "hello world"
        assert refs == []

    def test_nonexistent_file_skipped(self) -> None:
        text, refs = _expand_file_references("check @nonexistent_xyz_file.txt please")
        assert refs == []

    def test_existing_file_expanded(self, tmp_path: Path) -> None:
        f = tmp_path / "sample.txt"
        f.write_text("file content here")
        text, refs = _expand_file_references(f"look at @{f}")
        assert len(refs) == 1
        assert refs[0][1] == "file content here"

    def test_multiple_references(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A")
        f2.write_text("content B")
        text, refs = _expand_file_references(f"compare @{f1} and @{f2}")
        assert len(refs) == 2

    def test_duplicate_reference_deduplicated(self, tmp_path: Path) -> None:
        f = tmp_path / "dup.txt"
        f.write_text("only once")
        text, refs = _expand_file_references(f"@{f} and @{f}")
        assert len(refs) == 1

    def test_email_not_matched(self) -> None:
        """Email addresses should not be treated as file references."""
        text, refs = _expand_file_references("contact user@example.com")
        assert refs == []

    def test_relative_path_expanded(self, tmp_path: Path, monkeypatch: object) -> None:
        f = tmp_path / "rel.txt"
        f.write_text("relative content")

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)  # type: ignore[attr-defined]
        text, refs = _expand_file_references("check @rel.txt")
        assert len(refs) == 1
        assert refs[0][1] == "relative content"


# ---------------------------------------------------------------------------
# Bash shortcut tests
# ---------------------------------------------------------------------------


class TestHandleBashShortcut:
    """Test the _handle_bash_shortcut method."""

    def test_empty_command_shows_error(self) -> None:
        service = _service()
        asyncio.run(service._handle_bash_shortcut("!"))
        update = service.updates.get_nowait()
        assert update.kind == "error"
        assert "Usage" in (update.text or "")

    def test_whitespace_only_command_shows_error(self) -> None:
        service = _service()
        asyncio.run(service._handle_bash_shortcut("!  "))
        update = service.updates.get_nowait()
        assert update.kind == "error"

    def test_successful_command_produces_output(self) -> None:
        service = _service()
        asyncio.run(service._handle_bash_shortcut("!echo hello"))

        updates: list[UiUpdate] = []
        while not service.updates.empty():
            updates.append(service.updates.get_nowait())

        kinds = [u.kind for u in updates]
        assert "user_message" in kinds
        assert "system" in kinds
        # Should have the command echo and the output.
        system_texts = [u.text for u in updates if u.kind == "system"]
        assert any("$ echo hello" in (t or "") for t in system_texts)
        assert any("hello" in (t or "") for t in system_texts)

    def test_failed_command_shows_exit_code(self) -> None:
        service = _service()
        asyncio.run(service._handle_bash_shortcut("!false"))

        updates: list[UiUpdate] = []
        while not service.updates.empty():
            updates.append(service.updates.get_nowait())

        system_texts = [u.text for u in updates if u.kind == "system"]
        assert any("Exit code" in (t or "") for t in system_texts)

    def test_bash_shortcut_not_routed_to_queue(self) -> None:
        """Bash shortcuts should work even when the agent is busy."""
        service = _service()
        service._busy = True
        asyncio.run(service.handle_input("!echo test"))
        # Should NOT be queued.
        assert service._queued_messages == []
        # Should have produced output updates.
        assert not service.updates.empty()


# ---------------------------------------------------------------------------
# Concurrent handle_input during active turn
# ---------------------------------------------------------------------------


class TestConcurrentHandleInputDuringTurn:
    """Test that handle_input correctly queues messages while a turn runs."""

    def test_message_queued_during_slow_turn(self) -> None:
        """Simulate: first prompt starts a slow turn; second prompt while busy
        should be queued and produce a queued_message UI update."""
        service = _service()
        service._thread_context = _ThreadCommandContext(
            db_path=service.db_path,
            user_id=service.user_id,
            agent_name=service.agent_name,
            model=service.model,
            active_session_id=service.session_id,
        )
        service._approval_context = _InteractiveApprovalContext(auto_approve=True)

        # A runner that blocks until we tell it to finish.
        turn_event = asyncio.Event()

        async def _slow_run(**_kwargs: object):
            await turn_event.wait()
            return
            yield  # make it an async generator  # noqa: RET504

        class _SlowRunner:
            def run_async(self, **kwargs: object) -> object:
                return _slow_run(**kwargs)

        service._runner = _SlowRunner()

        async def _scenario() -> list[UiUpdate]:
            # First message — starts a slow turn
            await service.handle_input("first prompt")
            assert service._busy is True

            # Give the event loop a chance to start _run_turn
            await asyncio.sleep(0.01)

            # Second message — should be queued
            await service.handle_input("second prompt while busy")

            # Collect updates so far
            updates: list[UiUpdate] = []
            while not service.updates.empty():
                updates.append(service.updates.get_nowait())

            # Let the turn finish
            turn_event.set()
            # Allow the turn task to complete
            if service._turn_task:
                with suppress(Exception):
                    await asyncio.wait_for(service._turn_task, timeout=2.0)

            # Collect remaining updates
            while not service.updates.empty():
                updates.append(service.updates.get_nowait())

            return updates

        updates = asyncio.run(_scenario())
        kinds = [u.kind for u in updates]

        # The first prompt should produce a user_message
        user_messages = [u for u in updates if u.kind == "user_message"]
        assert any("first prompt" in (u.text or "") for u in user_messages)

        # The second prompt should produce a queued_message
        queued = [u for u in updates if u.kind == "queued_message"]
        assert len(queued) >= 1, f"Expected queued_message, got kinds: {kinds}"
        assert "second prompt while busy" in (queued[0].text or "")


# ---------------------------------------------------------------------------
# Open editor tests
# ---------------------------------------------------------------------------


class TestOpenEditor:
    """Test the open_editor method."""

    def test_no_editor_env_shows_error(self, monkeypatch: object) -> None:
        monkeypatch.setenv("EDITOR", "")  # type: ignore[attr-defined]
        monkeypatch.delenv("VISUAL", raising=False)  # type: ignore[attr-defined]
        service = _service()
        result = asyncio.run(service.open_editor())
        assert result is None
        update = service.updates.get_nowait()
        assert update.kind == "error"
        assert "EDITOR" in (update.text or "")


# ---------------------------------------------------------------------------
# Phase 4: Agent registry & switching
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    """Test the agent registry on AgentService."""

    def test_service_has_registry(self) -> None:
        service = _service()
        assert service.agent_registry is not None

    def test_default_active_agent_is_build(self) -> None:
        service = _service()
        # The active agent name matches the agent_name passed at construction.
        assert service.active_agent_name == "demo"

    def test_active_agent_profile_auto_registered(self) -> None:
        service = _service()
        # "demo" isn't a builtin but is auto-registered as a primary profile.
        profile = service.active_agent_profile
        assert profile is not None
        assert profile.name == "demo"
        assert profile.mode == "primary"

    def test_registry_has_builtins(self) -> None:
        service = _service()
        assert service.agent_registry.get("build") is not None
        assert service.agent_registry.get("plan") is not None


class TestAgentSwitching:
    """Test agent switching via the service."""

    def test_switch_same_agent_shows_message(self) -> None:
        service = _service()
        service._active_agent_name = "build"
        profile = service.agent_registry.get("build")
        assert profile is not None

        asyncio.run(service.switch_agent(profile))

        update = service.updates.get_nowait()
        assert update.kind == "system"
        assert "Already using" in (update.text or "")

    def test_switch_while_busy_shows_error(self) -> None:
        service = _service()
        service._busy = True
        profile = service.agent_registry.get("plan")
        assert profile is not None

        asyncio.run(service.switch_agent(profile))

        update = service.updates.get_nowait()
        assert update.kind == "error"
        assert "Cannot switch" in (update.text or "")

    def test_switch_updates_active_agent(self) -> None:
        service = _service()
        service._active_agent_name = "build"
        profile = service.agent_registry.get("plan")
        assert profile is not None

        asyncio.run(service.switch_agent(profile))

        assert service.active_agent_name == "plan"

        updates: list[UiUpdate] = []
        while not service.updates.empty():
            updates.append(service.updates.get_nowait())

        system_texts = [u.text for u in updates if u.kind == "system"]
        assert any("plan" in (t or "").lower() for t in system_texts)


# ---------------------------------------------------------------------------
# Phase 4: Conversation log & export
# ---------------------------------------------------------------------------


class TestConversationLog:
    """Test conversation logging on AgentService."""

    def test_service_has_conversation_log(self) -> None:
        service = _service()
        assert service.conversation_log is not None
        assert len(service.conversation_log.records) == 0

    def test_log_record_appends(self) -> None:
        from adk_deepagents.cli.tui.models import MessageRecord

        service = _service()
        service._log_record(MessageRecord(role="user", text="hello"))
        assert len(service.conversation_log.records) == 1
        assert service.conversation_log.records[0].text == "hello"


class TestExportConversation:
    """Test the export_conversation method."""

    def test_empty_export_shows_message(self) -> None:
        service = _service()
        result = asyncio.run(service.export_conversation())
        assert result is None
        update = service.updates.get_nowait()
        assert update.kind == "system"
        assert "Nothing to export" in (update.text or "")

    def test_export_without_editor_emits_markdown(self, monkeypatch: object) -> None:
        from adk_deepagents.cli.tui.models import MessageRecord

        monkeypatch.setenv("EDITOR", "")  # type: ignore[attr-defined]
        monkeypatch.delenv("VISUAL", raising=False)  # type: ignore[attr-defined]

        service = _service()
        service._log_record(MessageRecord(role="user", text="test question"))
        service._log_record(MessageRecord(role="assistant", text="test answer"))

        result = asyncio.run(service.export_conversation())
        assert result is not None
        assert "**User:** test question" in result
        assert "**Assistant:** test answer" in result

        # Should have emitted system updates with the markdown.
        updates: list[UiUpdate] = []
        while not service.updates.empty():
            updates.append(service.updates.get_nowait())
        texts = " ".join(u.text or "" for u in updates)
        assert "test question" in texts


# ---------------------------------------------------------------------------
# Phase 5: _guess_language_from_path tests
# ---------------------------------------------------------------------------


class TestGuessLanguageFromPath:
    """Test the _guess_language_from_path helper."""

    def test_python_extension(self) -> None:
        assert _guess_language_from_path("src/app.py") == "python"

    def test_typescript_extension(self) -> None:
        assert _guess_language_from_path("src/index.ts") == "typescript"

    def test_tsx_extension(self) -> None:
        assert _guess_language_from_path("App.tsx") == "typescript"

    def test_javascript_extension(self) -> None:
        assert _guess_language_from_path("util.js") == "javascript"

    def test_json_extension(self) -> None:
        assert _guess_language_from_path("config.json") == "json"

    def test_yaml_extension(self) -> None:
        assert _guess_language_from_path("docker-compose.yml") == "yaml"

    def test_unknown_extension(self) -> None:
        assert _guess_language_from_path("data.xyz") == ""

    def test_no_extension(self) -> None:
        assert _guess_language_from_path("Makefile") == ""

    def test_none_path(self) -> None:
        assert _guess_language_from_path(None) == ""

    def test_empty_string(self) -> None:
        assert _guess_language_from_path("") == ""

    def test_case_insensitive(self) -> None:
        assert _guess_language_from_path("README.MD") == "markdown"

    def test_deeply_nested_path(self) -> None:
        assert _guess_language_from_path("/home/user/project/src/lib/utils.rs") == "rust"


# ---------------------------------------------------------------------------
# Phase 5: _extract_tool_output tests
# ---------------------------------------------------------------------------


class TestExtractToolOutput:
    """Test the _extract_tool_output helper."""

    def test_read_file_with_content(self) -> None:
        content = "line1\nline2\nline3\nline4"
        result = _extract_tool_output("read_file", {"content": content, "path": "app.py"})
        assert result is not None
        assert "```python" in result
        assert "line1" in result

    def test_read_file_with_path_language_hint(self) -> None:
        content = "fn main() {\n    println!();\n    return;\n}"
        result = _extract_tool_output("read_file", {"content": content, "path": "main.rs"})
        assert result is not None
        assert "```rust" in result

    def test_read_file_too_short(self) -> None:
        """Content with fewer than _TOOL_OUTPUT_MIN_LINES should return None."""
        result = _extract_tool_output("read_file", {"content": "short", "path": "f.py"})
        assert result is None

    def test_read_file_empty_content(self) -> None:
        result = _extract_tool_output("read_file", {"content": "", "path": "f.py"})
        assert result is None

    def test_read_file_no_content_key(self) -> None:
        result = _extract_tool_output("read_file", {"path": "f.py"})
        assert result is None

    def test_execute_with_output(self) -> None:
        output = "BUILD SUCCESSFUL\nCompiled 3 files\nTests passed\nDone"
        result = _extract_tool_output("execute", {"output": output})
        assert result is not None
        assert "```\n" in result
        assert "BUILD SUCCESSFUL" in result

    def test_execute_too_short(self) -> None:
        result = _extract_tool_output("execute", {"output": "ok"})
        assert result is None

    def test_grep_with_result(self) -> None:
        grep_result = "file1.py:10: match1\nfile2.py:20: match2\nfile3.py:30: match3"
        result = _extract_tool_output("grep", {"result": grep_result})
        assert result is not None
        assert "```\n" in result

    def test_grep_too_short(self) -> None:
        result = _extract_tool_output("grep", {"result": "one match"})
        assert result is None

    def test_glob_with_entries(self) -> None:
        entries = [{"path": "/a.py"}, {"path": "/b.py"}, {"path": "/c.py"}]
        result = _extract_tool_output("glob", {"entries": entries})
        assert result is not None
        assert "```\n" in result

    def test_glob_too_few_entries(self) -> None:
        entries = [{"path": "/a.py"}]
        result = _extract_tool_output("glob", {"entries": entries})
        assert result is None

    def test_unknown_tool(self) -> None:
        result = _extract_tool_output("unknown_tool", {"data": "anything"})
        assert result is None


# ---------------------------------------------------------------------------
# Phase 5: tool_output emission tests
# ---------------------------------------------------------------------------


class TestToolOutputEmission:
    """Test that tool_output is included in UiUpdate for tool results."""

    def test_tool_output_emitted_for_read_file(self) -> None:
        service = _service()
        content = "line1\nline2\nline3\nline4"
        event = _FakeEvent(
            [
                _FakePart(
                    function_response=_FakeFunctionResponse(
                        "read_file",
                        {"status": "success", "content": content, "path": "app.py"},
                    )
                ),
            ]
        )

        asyncio.run(service._emit_event_updates(event))

        updates = []
        while not service.updates.empty():
            updates.append(service.updates.get_nowait())

        result_updates = [u for u in updates if u.kind == "tool_result"]
        assert len(result_updates) == 1
        assert result_updates[0].tool_output is not None
        assert "```python" in result_updates[0].tool_output

    def test_tool_output_none_for_short_content(self) -> None:
        service = _service()
        event = _FakeEvent(
            [
                _FakePart(
                    function_response=_FakeFunctionResponse(
                        "read_file",
                        {"status": "success", "content": "short", "path": "f.py"},
                    )
                ),
            ]
        )

        asyncio.run(service._emit_event_updates(event))

        updates = []
        while not service.updates.empty():
            updates.append(service.updates.get_nowait())

        result_updates = [u for u in updates if u.kind == "tool_result"]
        assert len(result_updates) == 1
        assert result_updates[0].tool_output is None

    def test_tool_output_none_for_unknown_tool(self) -> None:
        service = _service()
        event = _FakeEvent(
            [
                _FakePart(
                    function_response=_FakeFunctionResponse(
                        "unknown_tool",
                        {"status": "success", "data": "whatever"},
                    )
                ),
            ]
        )

        asyncio.run(service._emit_event_updates(event))

        updates = []
        while not service.updates.empty():
            updates.append(service.updates.get_nowait())

        result_updates = [u for u in updates if u.kind == "tool_result"]
        assert len(result_updates) == 1
        assert result_updates[0].tool_output is None
