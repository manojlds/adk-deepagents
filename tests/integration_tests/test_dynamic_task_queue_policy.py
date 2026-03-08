"""Integration tests for dynamic task queue/concurrency policy.

No LLM calls: these tests exercise the runtime behavior of the dynamic task
tool via ``create_deep_agent`` wiring.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from google.genai import types

from adk_deepagents import create_deep_agent
from adk_deepagents.tools import task_dynamic
from adk_deepagents.types import DynamicTaskConfig

pytestmark = pytest.mark.integration


@dataclass
class _DummyToolContext:
    state: dict[str, Any]


def _task_tool_from_agent(agent: Any) -> Any:
    for tool in agent.tools:
        if getattr(tool, "__name__", None) == "task":
            return tool
    raise AssertionError("Dynamic task tool was not added to the agent")


def _make_llm_request() -> Any:
    request = MagicMock()
    request.config = types.GenerateContentConfig()
    return request


def _cleanup_dynamic_runtime(context: _DummyToolContext) -> None:
    logical_parent = context.state.get("_dynamic_parent_session_id")
    if not isinstance(logical_parent, str):
        return

    for key in list(task_dynamic._RUNTIME_REGISTRY):
        if key.startswith(f"{logical_parent}:"):
            task_dynamic._RUNTIME_REGISTRY.pop(key, None)

    task_dynamic._CONCURRENCY_LOCKS.pop(logical_parent, None)


@pytest.mark.timeout(30)
async def test_dynamic_wait_policy_queues_parallel_task_calls(monkeypatch):
    async def fake_run_dynamic_task(*args, **kwargs):
        del args, kwargs
        await asyncio.sleep(0.05)
        return {
            "result": "ok",
            "function_calls": [],
            "files": {},
            "todos": [],
            "timed_out": False,
            "error": None,
        }

    monkeypatch.setattr(task_dynamic, "_run_dynamic_task", fake_run_dynamic_task)

    agent = create_deep_agent(
        delegation_mode="dynamic",
        dynamic_task_config=DynamicTaskConfig(
            max_parallel=1,
            concurrency_policy="wait",
            queue_timeout_seconds=1.0,
        ),
    )
    task_tool = _task_tool_from_agent(agent)

    context = _DummyToolContext(state={"files": {}, "todos": []})
    first: asyncio.Task[Any] | None = None
    second: asyncio.Task[Any] | None = None
    try:
        first = asyncio.create_task(
            task_tool(
                description="first",
                prompt="first",
                subagent_type="general",
                tool_context=cast(Any, context),
            )
        )
        await asyncio.sleep(0.01)
        second = asyncio.create_task(
            task_tool(
                description="second",
                prompt="second",
                subagent_type="general",
                tool_context=cast(Any, context),
            )
        )

        first_result, second_result = await asyncio.gather(first, second)
    finally:
        _cleanup_dynamic_runtime(context)

    assert first_result["status"] == "completed"
    assert second_result["status"] == "completed"
    assert any(result["queued"] is True for result in (first_result, second_result))
    assert (
        max(
            first_result["queue_wait_seconds"],
            second_result["queue_wait_seconds"],
        )
        >= 0.02
    )


@pytest.mark.timeout(30)
async def test_dynamic_error_policy_rejects_overflow_parallel_calls(monkeypatch):
    async def fake_run_dynamic_task(*args, **kwargs):
        del args, kwargs
        await asyncio.sleep(0.05)
        return {
            "result": "ok",
            "function_calls": [],
            "files": {},
            "todos": [],
            "timed_out": False,
            "error": None,
        }

    monkeypatch.setattr(task_dynamic, "_run_dynamic_task", fake_run_dynamic_task)

    agent = create_deep_agent(
        delegation_mode="dynamic",
        dynamic_task_config=DynamicTaskConfig(
            max_parallel=1,
            concurrency_policy="error",
        ),
    )
    task_tool = _task_tool_from_agent(agent)

    context = _DummyToolContext(state={"files": {}, "todos": []})
    first: asyncio.Task[Any] | None = None
    second: asyncio.Task[Any] | None = None
    try:
        first = asyncio.create_task(
            task_tool(
                description="first",
                prompt="first",
                subagent_type="general",
                tool_context=cast(Any, context),
            )
        )
        await asyncio.sleep(0.01)
        second = asyncio.create_task(
            task_tool(
                description="second",
                prompt="second",
                subagent_type="general",
                tool_context=cast(Any, context),
            )
        )

        first_result, second_result = await asyncio.gather(first, second)
    finally:
        for pending in (first, second):
            if pending is None or pending.done():
                continue
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pending
        _cleanup_dynamic_runtime(context)

    assert first_result["status"] == "completed"
    assert second_result["status"] == "error"
    assert "concurrency limit" in second_result["error"].lower()
    assert second_result["queued"] is False


def test_dynamic_task_tool_doc_exposes_concurrency_limits() -> None:
    agent = create_deep_agent(
        delegation_mode="dynamic",
        dynamic_task_config=DynamicTaskConfig(
            max_parallel=3,
            concurrency_policy="wait",
            queue_timeout_seconds=12.5,
        ),
    )
    task_tool = _task_tool_from_agent(agent)

    doc = task_tool.__doc__ or ""
    assert "max_parallel=3" in doc
    assert "concurrency_policy=wait" in doc
    assert "queue_timeout_seconds=12.5" in doc
    assert "waves" in doc.lower()


def test_dynamic_task_limits_are_injected_into_system_instruction() -> None:
    agent = create_deep_agent(
        delegation_mode="dynamic",
        dynamic_task_config=DynamicTaskConfig(
            max_parallel=3,
            concurrency_policy="wait",
            queue_timeout_seconds=12.5,
        ),
    )

    callback = cast(Any, agent.before_model_callback)
    assert callback is not None

    context = MagicMock()
    context.state = {}
    request = _make_llm_request()

    callback(context, request)

    system_instruction = str(request.config.system_instruction)
    assert "Dynamic Task Concurrency Limits" in system_instruction
    assert "max_parallel=3" in system_instruction
    assert "concurrency_policy=wait" in system_instruction
    assert "queue_timeout_seconds=12.5" in system_instruction
