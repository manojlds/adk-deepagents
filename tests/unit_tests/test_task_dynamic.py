"""Unit tests for dynamic task delegation tool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from adk_deepagents.tools import task_dynamic
from adk_deepagents.tools.task_dynamic import create_dynamic_task_tool
from adk_deepagents.types import DynamicTaskConfig


@dataclass
class _DummyToolContext:
    state: dict


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
            config=DynamicTaskConfig(max_parallel=1),
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
        task_store = context.state.get("_dynamic_tasks", {})
        created = task_store.get(result.get("task_id", ""), {})
        assert isinstance(created, dict)

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
        task_dynamic._RUNTIME_REGISTRY[key] = task_dynamic._TaskRuntime(  # type: ignore[reportPrivateUsage]
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
            task_dynamic._RUNTIME_REGISTRY.pop(key, None)  # type: ignore[reportPrivateUsage]

        assert result["status"] == "error"
        assert "timed out" in result["error"].lower()
        assert result["result"] == "partial output"
