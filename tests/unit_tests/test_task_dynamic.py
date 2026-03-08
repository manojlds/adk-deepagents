"""Unit tests for dynamic task delegation tool."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any, cast

from adk_deepagents.tools import task_dynamic
from adk_deepagents.tools.task_dynamic import (
    create_dynamic_task_tool,
    create_register_subagent_tool,
)
from adk_deepagents.types import DynamicTaskConfig


@dataclass
class _DummyToolContext:
    state: dict


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

    async def test_unknown_task_id_returns_error(self):
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

        assert result["status"] == "error"
        assert "unknown task_id" in result["error"].lower()

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
