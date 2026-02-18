"""Unit tests for dynamic task delegation tool."""

from __future__ import annotations

from dataclasses import dataclass

from adk_deepagents.tools.task_dynamic import create_dynamic_task_tool
from adk_deepagents.types import DynamicTaskConfig


@dataclass
class _DummyToolContext:
    state: dict


class TestDynamicTaskToolGuards:
    def test_depth_limit_blocks_new_task_before_spawn(self):
        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=None,
            config=DynamicTaskConfig(max_depth=1),
        )

        context = _DummyToolContext(state={"_dynamic_delegation_depth": 1})
        result = task_tool(
            description="delegate",
            prompt="delegate this",
            subagent_type="general",
            tool_context=context,
        )

        assert result["status"] == "error"
        assert "depth limit" in result["error"].lower()

    def test_unknown_task_id_returns_error(self):
        task_tool = create_dynamic_task_tool(
            default_model="gemini-2.5-flash",
            default_tools=[],
            subagents=None,
            config=DynamicTaskConfig(),
        )

        context = _DummyToolContext(state={"_dynamic_tasks": {}})
        result = task_tool(
            description="resume",
            prompt="resume",
            task_id="task_999",
            tool_context=context,
        )

        assert result["status"] == "error"
        assert "unknown task_id" in result["error"].lower()

    def test_parallel_limit_blocks_execution(self):
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
        result = task_tool(
            description="delegate",
            prompt="delegate this",
            subagent_type="general",
            tool_context=context,
        )

        assert result["status"] == "error"
        assert "concurrency limit" in result["error"].lower()
        task_store = context.state.get("_dynamic_tasks", {})
        created = task_store.get(result.get("task_id", ""), {})
        assert isinstance(created, dict)
