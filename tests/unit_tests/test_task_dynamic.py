"""Unit tests for dynamic task delegation tool."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any, cast

from adk_deepagents.backends.filesystem import FilesystemBackend
from adk_deepagents.backends.runtime import clear_session_backend, register_backend_factory
from adk_deepagents.backends.state import StateBackend
from adk_deepagents.tools import task_dynamic
from adk_deepagents.tools.task_dynamic import (
    create_dynamic_task_tool,
    create_register_subagent_tool,
)
from adk_deepagents.types import A2ATaskConfig, DynamicTaskConfig, TemporalTaskConfig


@dataclass
class _DummyToolContext:
    state: dict
    session: Any | None = None


def _cleanup_runtime_registry(context: _DummyToolContext) -> None:
    logical_parent = context.state.get("_dynamic_parent_session_id")
    if not isinstance(logical_parent, str):
        return

    for key in list(task_dynamic._RUNTIME_REGISTRY):
        if key.startswith(f"{logical_parent}:"):
            task_dynamic._RUNTIME_REGISTRY.pop(key, None)


class TestDynamicTaskToolGuards:
    async def test_depth_limit_blocks_new_task_before_spawn(self):
        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=None,
            config=DynamicTaskConfig(max_depth=1),
        )

        context = _DummyToolContext(state={"_dynamic_delegation_depth": 1})
        result = await task_tool(
            description="delegate",
            prompt="delegate this",
            subagent_type="general",
            tool_context=cast(Any, context),
        )

        assert result["status"] == "error"
        assert "depth limit" in result["error"].lower()

    async def test_unknown_task_id_creates_new_task(self, monkeypatch):
        async def fake_run_dynamic_task(*args, **kwargs):
            del args, kwargs
            return {
                "result": "created",
                "function_calls": [],
                "files": {},
                "todos": [],
                "timed_out": False,
                "error": None,
            }

        monkeypatch.setattr(task_dynamic, "_run_dynamic_task", fake_run_dynamic_task)

        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=None,
            config=DynamicTaskConfig(),
        )

        context = _DummyToolContext(state={"_dynamic_tasks": {}})
        result = await task_tool(
            description="resume",
            prompt="resume",
            task_id="task_999",
            tool_context=cast(Any, context),
        )

        assert result["status"] == "completed"
        assert result["task_id"] == "task_999"
        assert context.state["_dynamic_tasks"]["task_999"]["subagent_type"] == "general_purpose"

    async def test_parallel_limit_blocks_execution(self):
        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=None,
            config=DynamicTaskConfig(max_parallel=1, concurrency_policy="error"),
        )

        context = _DummyToolContext(
            state={
                "_dynamic_running_tasks": ["parent:task_existing"],
            }
        )
        result = await task_tool(
            description="delegate",
            prompt="delegate this",
            subagent_type="general",
            tool_context=cast(Any, context),
        )

        assert result["status"] == "error"
        assert "concurrency limit" in result["error"].lower()
        assert result["queued"] is False
        task_store = context.state.get("_dynamic_tasks", {})
        created = task_store.get(result.get("task_id", ""), {})
        assert isinstance(created, dict)

    async def test_parallel_limit_wait_policy_queues_until_slot_frees(self, monkeypatch):
        async def fake_run_dynamic_task(*args, **kwargs):
            del args, kwargs
            return {
                "result": "done",
                "function_calls": [],
                "files": {},
                "todos": [],
                "timed_out": False,
                "error": None,
            }

        monkeypatch.setattr(task_dynamic, "_run_dynamic_task", fake_run_dynamic_task)

        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=None,
            config=DynamicTaskConfig(
                max_parallel=1,
                concurrency_policy="wait",
                queue_timeout_seconds=0.5,
            ),
        )

        key = "parent_wait:task_1"
        task_dynamic._RUNTIME_REGISTRY[key] = task_dynamic._TaskRuntime(
            runner=None,  # type: ignore[arg-type]
            session_id="s1",
            user_id="u1",
            subagent_type="general_purpose",
        )

        context = _DummyToolContext(
            state={
                "_dynamic_tasks": {"task_1": {"subagent_type": "general_purpose"}},
                "_dynamic_parent_session_id": "parent_wait",
                "_dynamic_running_tasks": ["parent_wait:task_existing"],
            }
        )

        async def _release_slot() -> None:
            await asyncio.sleep(0.05)
            running = context.state.get("_dynamic_running_tasks")
            if isinstance(running, list) and "parent_wait:task_existing" in running:
                running.remove("parent_wait:task_existing")

        release_task = asyncio.create_task(_release_slot())

        try:
            result = await task_tool(
                description="resume",
                prompt="resume",
                task_id="task_1",
                tool_context=cast(Any, context),
            )
        finally:
            release_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await release_task
            task_dynamic._RUNTIME_REGISTRY.pop(key, None)

        assert result["status"] == "completed"
        assert result["queued"] is True
        assert result["queue_wait_seconds"] > 0

    async def test_parallel_limit_wait_policy_times_out_when_queue_is_full(self):
        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=None,
            config=DynamicTaskConfig(
                max_parallel=1,
                concurrency_policy="wait",
                queue_timeout_seconds=0.01,
            ),
        )

        key = "parent_timeout:task_1"
        task_dynamic._RUNTIME_REGISTRY[key] = task_dynamic._TaskRuntime(
            runner=None,  # type: ignore[arg-type]
            session_id="s1",
            user_id="u1",
            subagent_type="general_purpose",
        )

        context = _DummyToolContext(
            state={
                "_dynamic_tasks": {"task_1": {"subagent_type": "general_purpose"}},
                "_dynamic_parent_session_id": "parent_timeout",
                "_dynamic_running_tasks": ["parent_timeout:task_existing"],
            }
        )

        try:
            result = await task_tool(
                description="resume",
                prompt="resume",
                task_id="task_1",
                tool_context=cast(Any, context),
            )
        finally:
            task_dynamic._RUNTIME_REGISTRY.pop(key, None)

        assert result["status"] == "error"
        assert "queue timeout" in result["error"].lower()
        assert result["queued"] is True
        assert result["queue_wait_seconds"] > 0

    async def test_timeout_is_returned_as_tool_error(self, monkeypatch):
        async def fake_run_dynamic_task(*args, **kwargs):
            return {
                "result": "partial output",
                "function_calls": ["web_search"],
                "files": {},
                "todos": [],
                "timed_out": True,
                "error": None,
            }

        monkeypatch.setattr(task_dynamic, "_run_dynamic_task", fake_run_dynamic_task)

        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=None,
            config=DynamicTaskConfig(timeout_seconds=5),
        )

        key = "parent_1:task_1"
        task_dynamic._RUNTIME_REGISTRY[key] = task_dynamic._TaskRuntime(
            runner=None,  # type: ignore[arg-type]
            session_id="s1",
            user_id="u1",
            subagent_type="general_purpose",
        )

        context = _DummyToolContext(
            state={
                "_dynamic_tasks": {"task_1": {"subagent_type": "general_purpose"}},
                "_dynamic_parent_session_id": "parent_1",
            }
        )

        try:
            result = await task_tool(
                description="resume",
                prompt="resume",
                task_id="task_1",
                tool_context=cast(Any, context),
            )
        finally:
            task_dynamic._RUNTIME_REGISTRY.pop(key, None)

        assert result["status"] == "error"
        assert "timed out" in result["error"].lower()
        assert result["result"] == "partial output"

    async def test_temporal_mode_skips_inprocess_runtime_for_new_tasks(self, monkeypatch):
        async def fake_temporal_run(*args, **kwargs):
            del args, kwargs
            return {
                "result": "done",
                "function_calls": [],
                "files": {},
                "todos": [],
                "timed_out": False,
                "error": None,
            }

        class _ForbiddenRunner:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                raise AssertionError("InMemoryRunner should not be created in Temporal mode")

        monkeypatch.setattr(task_dynamic, "_run_dynamic_task_temporal", fake_temporal_run)
        monkeypatch.setattr(task_dynamic, "InMemoryRunner", _ForbiddenRunner)

        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=None,
            config=DynamicTaskConfig(temporal=TemporalTaskConfig()),
        )

        context = _DummyToolContext(state={})
        result = await task_tool(
            description="delegate",
            prompt="delegate this",
            subagent_type="general",
            tool_context=cast(Any, context),
        )

        assert result["status"] == "completed"
        logical_parent = context.state.get("_dynamic_parent_session_id")
        assert isinstance(logical_parent, str)
        assert not any(
            key.startswith(f"{logical_parent}:") for key in task_dynamic._RUNTIME_REGISTRY
        )

    async def test_temporal_mode_rejects_model_override_when_disabled(self, monkeypatch):
        async def fake_temporal_run(*args, **kwargs):
            del args, kwargs
            raise AssertionError("Temporal dispatch should not run when model override is invalid")

        class _ForbiddenRunner:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                raise AssertionError("InMemoryRunner should not be created in Temporal mode")

        monkeypatch.setattr(task_dynamic, "_run_dynamic_task_temporal", fake_temporal_run)
        monkeypatch.setattr(task_dynamic, "InMemoryRunner", _ForbiddenRunner)

        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=None,
            config=DynamicTaskConfig(temporal=TemporalTaskConfig()),
        )

        result = await task_tool(
            description="delegate",
            prompt="delegate this",
            subagent_type="general",
            model="openai/gpt-4o-mini",
            tool_context=cast(Any, _DummyToolContext(state={})),
        )

        assert result["status"] == "error"
        assert "model override" in result["error"].lower()

    async def test_temporal_mode_resumes_without_runtime_registry(self, monkeypatch):
        async def fake_temporal_run(*args, **kwargs):
            del args, kwargs
            return {
                "result": "ORBIT",
                "function_calls": [],
                "files": {},
                "todos": [],
                "timed_out": False,
                "error": None,
            }

        class _ForbiddenRunner:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                raise AssertionError("InMemoryRunner should not be created in Temporal mode")

        monkeypatch.setattr(task_dynamic, "_run_dynamic_task_temporal", fake_temporal_run)
        monkeypatch.setattr(task_dynamic, "InMemoryRunner", _ForbiddenRunner)

        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=None,
            config=DynamicTaskConfig(temporal=TemporalTaskConfig()),
        )

        context = _DummyToolContext(
            state={
                "_dynamic_parent_session_id": "parent_temporal",
                "_dynamic_tasks": {
                    "task_1": {
                        "subagent_type": "general_purpose",
                        "depth": 1,
                        "files": {},
                        "todos": [],
                        "history": [],
                    }
                },
            }
        )

        result = await task_tool(
            description="resume",
            prompt="resume",
            task_id="task_1",
            tool_context=cast(Any, context),
        )

        assert result["status"] == "completed"
        assert result["task_id"] == "task_1"
        assert result["recovered_runtime"] is False
        assert not any(key.startswith("parent_temporal:") for key in task_dynamic._RUNTIME_REGISTRY)

    async def test_a2a_mode_skips_inprocess_runtime_for_new_tasks(self, monkeypatch):
        async def fake_a2a_run(*args, **kwargs):
            del args, kwargs
            return {
                "result": "done",
                "function_calls": [],
                "files": {},
                "todos": [],
                "timed_out": False,
                "error": None,
            }

        class _ForbiddenRunner:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                raise AssertionError("InMemoryRunner should not be created in A2A mode")

        monkeypatch.setattr(task_dynamic, "_run_dynamic_task_a2a", fake_a2a_run)
        monkeypatch.setattr(task_dynamic, "InMemoryRunner", _ForbiddenRunner)

        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=None,
            config=DynamicTaskConfig(a2a=A2ATaskConfig()),
        )

        context = _DummyToolContext(state={})
        result = await task_tool(
            description="delegate",
            prompt="delegate this",
            subagent_type="general",
            tool_context=cast(Any, context),
        )

        assert result["status"] == "completed"
        logical_parent = context.state.get("_dynamic_parent_session_id")
        assert isinstance(logical_parent, str)
        assert not any(
            key.startswith(f"{logical_parent}:") for key in task_dynamic._RUNTIME_REGISTRY
        )

    async def test_a2a_mode_resumes_without_runtime_registry(self, monkeypatch):
        async def fake_a2a_run(*args, **kwargs):
            del args, kwargs
            return {
                "result": "ORBIT",
                "function_calls": [],
                "files": {},
                "todos": [],
                "timed_out": False,
                "error": None,
            }

        class _ForbiddenRunner:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                raise AssertionError("InMemoryRunner should not be created in A2A mode")

        monkeypatch.setattr(task_dynamic, "_run_dynamic_task_a2a", fake_a2a_run)
        monkeypatch.setattr(task_dynamic, "InMemoryRunner", _ForbiddenRunner)

        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=None,
            config=DynamicTaskConfig(a2a=A2ATaskConfig()),
        )

        context = _DummyToolContext(
            state={
                "_dynamic_parent_session_id": "parent_a2a",
                "_dynamic_tasks": {
                    "task_1": {
                        "subagent_type": "general_purpose",
                        "depth": 1,
                        "files": {},
                        "todos": [],
                        "history": [],
                    }
                },
            }
        )

        result = await task_tool(
            description="resume",
            prompt="resume",
            task_id="task_1",
            tool_context=cast(Any, context),
        )

        assert result["status"] == "completed"
        assert result["task_id"] == "task_1"
        assert result["recovered_runtime"] is False
        assert not any(key.startswith("parent_a2a:") for key in task_dynamic._RUNTIME_REGISTRY)

    async def test_temporal_mode_forwards_runtime_subagent_spec_payload(self, monkeypatch):
        observed_snapshot: dict[str, Any] = {}

        async def fake_temporal_run(*, snapshot_data, **kwargs):
            del kwargs
            observed_snapshot.update(snapshot_data)
            return {
                "result": "done",
                "function_calls": [],
                "files": {},
                "todos": [],
                "timed_out": False,
                "error": None,
            }

        class _ForbiddenRunner:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                raise AssertionError("InMemoryRunner should not be created in Temporal mode")

        def dummy_tool() -> str:
            return "ok"

        monkeypatch.setattr(task_dynamic, "_run_dynamic_task_temporal", fake_temporal_run)
        monkeypatch.setattr(task_dynamic, "InMemoryRunner", _ForbiddenRunner)

        register_tool = create_register_subagent_tool(
            default_model="gemini-2.5-flash",
            default_tools=[dummy_tool],
            config=DynamicTaskConfig(temporal=TemporalTaskConfig()),
        )
        context = _DummyToolContext(state={})

        register_result = await register_tool(
            name="runtime_specialist",
            description="Runtime specialist",
            system_prompt="Use concise answers.",
            tool_names=["dummy_tool"],
            tool_context=cast(Any, context),
        )
        assert register_result["status"] == "registered"

        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[dummy_tool],
            subagents=None,
            config=DynamicTaskConfig(temporal=TemporalTaskConfig()),
        )

        result = await task_tool(
            description="delegate",
            prompt="delegate",
            subagent_type="runtime_specialist",
            tool_context=cast(Any, context),
        )

        assert result["status"] == "completed"
        assert observed_snapshot["subagent_spec"]["name"] == "runtime_specialist"
        assert observed_snapshot["subagent_spec"]["description"] == "Runtime specialist"
        assert observed_snapshot["subagent_spec"]["tool_names"] == ["dummy_tool"]

    async def test_temporal_mode_forwards_parent_backend_context(self, monkeypatch, tmp_path):
        observed_snapshot: dict[str, Any] = {}

        async def fake_temporal_run(*, snapshot_data, **kwargs):
            del kwargs
            observed_snapshot.update(snapshot_data)
            return {
                "result": "done",
                "function_calls": [],
                "files": {},
                "todos": [],
                "timed_out": False,
                "error": None,
            }

        class _ForbiddenRunner:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                raise AssertionError("InMemoryRunner should not be created in Temporal mode")

        monkeypatch.setattr(task_dynamic, "_run_dynamic_task_temporal", fake_temporal_run)
        monkeypatch.setattr(task_dynamic, "InMemoryRunner", _ForbiddenRunner)

        parent_session_id = "parent_session_temporal"
        root_dir = tmp_path / "workspace"
        root_dir.mkdir()

        register_backend_factory(
            parent_session_id,
            lambda _state: FilesystemBackend(root_dir=root_dir, virtual_mode=True),
        )

        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=None,
            config=DynamicTaskConfig(temporal=TemporalTaskConfig()),
        )

        context = _DummyToolContext(state={})
        context.session = type("_Session", (), {"id": parent_session_id})()

        try:
            result = await task_tool(
                description="delegate",
                prompt="delegate",
                subagent_type="general",
                tool_context=cast(Any, context),
            )
        finally:
            clear_session_backend(parent_session_id)

        assert result["status"] == "completed"
        assert observed_snapshot["backend_context"] == {
            "kind": "filesystem",
            "root_dir": str(root_dir.resolve()),
            "virtual_mode": True,
        }

    async def test_temporal_mode_serializes_registered_state_backend(self, monkeypatch):
        observed_snapshot: dict[str, Any] = {}

        async def fake_temporal_run(*, snapshot_data, **kwargs):
            del kwargs
            observed_snapshot.update(snapshot_data)
            return {
                "result": "done",
                "function_calls": [],
                "files": {},
                "todos": [],
                "timed_out": False,
                "error": None,
            }

        class _ForbiddenRunner:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                raise AssertionError("InMemoryRunner should not be created in Temporal mode")

        monkeypatch.setattr(task_dynamic, "_run_dynamic_task_temporal", fake_temporal_run)
        monkeypatch.setattr(task_dynamic, "InMemoryRunner", _ForbiddenRunner)

        parent_session_id = "parent_session_state_temporal"
        register_backend_factory(parent_session_id, lambda state: StateBackend(state))

        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=None,
            config=DynamicTaskConfig(temporal=TemporalTaskConfig()),
        )

        context = _DummyToolContext(state={})
        context.session = type("_Session", (), {"id": parent_session_id})()

        try:
            result = await task_tool(
                description="delegate",
                prompt="delegate",
                subagent_type="general",
                tool_context=cast(Any, context),
            )
        finally:
            clear_session_backend(parent_session_id)

        assert result["status"] == "completed"
        assert observed_snapshot["backend_context"] == {"kind": "state"}

    async def test_temporal_mode_defaults_to_state_backend_context(self, monkeypatch):
        observed_snapshot: dict[str, Any] = {}

        async def fake_temporal_run(*, snapshot_data, **kwargs):
            del kwargs
            observed_snapshot.update(snapshot_data)
            return {
                "result": "done",
                "function_calls": [],
                "files": {},
                "todos": [],
                "timed_out": False,
                "error": None,
            }

        class _ForbiddenRunner:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                raise AssertionError("InMemoryRunner should not be created in Temporal mode")

        monkeypatch.setattr(task_dynamic, "_run_dynamic_task_temporal", fake_temporal_run)
        monkeypatch.setattr(task_dynamic, "InMemoryRunner", _ForbiddenRunner)

        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=None,
            config=DynamicTaskConfig(temporal=TemporalTaskConfig()),
        )

        context = _DummyToolContext(state={})
        result = await task_tool(
            description="delegate",
            prompt="delegate",
            subagent_type="general",
            tool_context=cast(Any, context),
        )

        assert result["status"] == "completed"
        assert observed_snapshot["backend_context"] == {"kind": "state"}

    async def test_resume_recovers_runtime_from_persisted_task_state(self, monkeypatch):
        observed: dict[str, str] = {}

        async def fake_run_dynamic_task(runtime, *, prompt, timeout_seconds):
            del runtime, timeout_seconds
            observed["prompt"] = prompt
            return {
                "result": "ORBIT",
                "function_calls": [],
                "files": {},
                "todos": [],
                "timed_out": False,
                "error": None,
            }

        monkeypatch.setattr(task_dynamic, "_run_dynamic_task", fake_run_dynamic_task)

        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=None,
            config=DynamicTaskConfig(),
        )

        context = _DummyToolContext(
            state={
                "_dynamic_parent_session_id": "parent_recover",
                "_dynamic_tasks": {
                    "task_1": {
                        "subagent_type": "general_purpose",
                        "depth": 1,
                        "files": {},
                        "todos": [],
                        "history": [
                            {
                                "prompt": "Remember this codeword exactly: ORBIT",
                                "result": "Understood. I will remember ORBIT.",
                            }
                        ],
                    }
                },
            }
        )

        try:
            result = await task_tool(
                description="resume",
                prompt="What codeword did I ask you to remember?",
                task_id="task_1",
                tool_context=cast(Any, context),
            )
        finally:
            _cleanup_runtime_registry(context)

        assert result["status"] == "completed"
        assert result["recovered_runtime"] is True
        assert "Previous delegated turns" in observed["prompt"]
        assert "Remember this codeword exactly: ORBIT" in observed["prompt"]

    async def test_stale_running_task_entries_are_pruned_before_acquire(self, monkeypatch):
        async def fake_run_dynamic_task(*args, **kwargs):
            del args, kwargs
            return {
                "result": "done",
                "function_calls": [],
                "files": {},
                "todos": [],
                "timed_out": False,
                "error": None,
            }

        monkeypatch.setattr(task_dynamic, "_run_dynamic_task", fake_run_dynamic_task)

        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=None,
            config=DynamicTaskConfig(max_parallel=1, concurrency_policy="error"),
        )

        context = _DummyToolContext(
            state={
                "_dynamic_parent_session_id": "parent_stale",
                "_dynamic_running_tasks": ["parent_stale:task_ghost"],
            }
        )
        try:
            result = await task_tool(
                description="delegate",
                prompt="delegate",
                subagent_type="general",
                tool_context=cast(Any, context),
            )
        finally:
            _cleanup_runtime_registry(context)

        assert result["status"] == "completed"
        assert result["queue_wait_seconds"] < 0.05
        assert context.state.get("_dynamic_running_tasks") == []


class TestDynamicRuntimeSubagents:
    async def test_register_subagent_persists_runtime_spec(self):
        register_tool = create_register_subagent_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            config=DynamicTaskConfig(),
        )

        context = _DummyToolContext(state={})
        result = await register_tool(
            name="code_researcher",
            description="Searches and summarizes repository files.",
            system_prompt="Focus on repository exploration and concise summaries.",
            tool_context=cast(Any, context),
        )

        assert result["status"] == "registered"
        assert result["subagent_type"] == "code_researcher"

        store = context.state.get("_dynamic_subagent_specs", {})
        assert isinstance(store, dict)
        assert "code_researcher" in store

    async def test_unknown_subagent_type_auto_creates_runtime_specialist(self, monkeypatch):
        async def fake_run_dynamic_task(*args, **kwargs):
            del args, kwargs
            return {
                "result": "done",
                "function_calls": [],
                "files": {},
                "todos": [],
                "timed_out": False,
                "error": None,
            }

        monkeypatch.setattr(task_dynamic, "_run_dynamic_task", fake_run_dynamic_task)

        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=None,
            config=DynamicTaskConfig(),
        )

        context = _DummyToolContext(state={})
        try:
            result = await task_tool(
                description="Summarize Python modules by responsibility.",
                prompt="Summarize Python modules by responsibility.",
                subagent_type="summarizer",
                tool_context=cast(Any, context),
            )
        finally:
            _cleanup_runtime_registry(context)

        assert result["status"] == "completed"
        assert result["subagent_type"] == "summarizer"
        assert result["created_subagent"] is True

        store = context.state.get("_dynamic_subagent_specs", {})
        assert isinstance(store, dict)
        assert "summarizer" in store

    async def test_registered_runtime_subagent_is_reused(self, monkeypatch):
        async def fake_run_dynamic_task(*args, **kwargs):
            del args, kwargs
            return {
                "result": "done",
                "function_calls": [],
                "files": {},
                "todos": [],
                "timed_out": False,
                "error": None,
            }

        monkeypatch.setattr(task_dynamic, "_run_dynamic_task", fake_run_dynamic_task)

        register_tool = create_register_subagent_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            config=DynamicTaskConfig(),
        )
        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=None,
            config=DynamicTaskConfig(),
        )

        context = _DummyToolContext(state={})
        try:
            registration = await register_tool(
                name="summarizer",
                description="Summarizes groups of files by area.",
                tool_context=cast(Any, context),
            )
            result = await task_tool(
                description="Summarize repository files.",
                prompt="Summarize repository files.",
                subagent_type="summarizer",
                tool_context=cast(Any, context),
            )
        finally:
            _cleanup_runtime_registry(context)

        assert registration["status"] == "registered"
        assert result["status"] == "completed"
        assert result["subagent_type"] == "summarizer"
        assert result["created_subagent"] is False

    async def test_dynamic_spec_agent_receives_parent_callback_stack(self, monkeypatch):
        observed: dict[str, bool] = {}

        async def fake_run_dynamic_task(runtime, *, prompt, timeout_seconds):
            del prompt, timeout_seconds
            child_agent = runtime.runner.agent
            observed["before_agent"] = child_agent.before_agent_callback is not None
            observed["before_model"] = child_agent.before_model_callback is not None
            observed["after_tool"] = child_agent.after_tool_callback is not None
            observed["before_tool"] = child_agent.before_tool_callback is not None
            return {
                "result": "done",
                "function_calls": [],
                "files": {},
                "todos": [],
                "timed_out": False,
                "error": None,
            }

        monkeypatch.setattr(task_dynamic, "_run_dynamic_task", fake_run_dynamic_task)

        def before_agent_cb(callback_context):
            del callback_context
            return None

        def before_model_cb(callback_context, llm_request):
            del callback_context, llm_request
            return None

        def after_tool_cb(tool, args, tool_context, **kwargs):
            del tool, args, tool_context, kwargs
            return None

        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=[
                {
                    "name": "researcher",
                    "description": "Research specialist",
                }
            ],
            config=DynamicTaskConfig(),
            before_agent_callback=before_agent_cb,
            before_model_callback=before_model_cb,
            after_tool_callback=after_tool_cb,
            default_interrupt_on={"write_file": True},
        )

        context = _DummyToolContext(state={})
        try:
            result = await task_tool(
                description="Research this repository",
                prompt="Research this repository",
                subagent_type="researcher",
                tool_context=cast(Any, context),
            )
        finally:
            _cleanup_runtime_registry(context)

        assert result["status"] == "completed"
        assert observed == {
            "before_agent": True,
            "before_model": True,
            "after_tool": True,
            "before_tool": True,
        }
