"""LLM integration tests for dynamic task delegation via Temporal.

Run with:
    uv run pytest -m llm tests/integration_tests/llm/test_temporal_dynamic_task_delegation.py

Requires:
- valid LLM credentials for ``make_litellm_model()``
- Temporal server reachable **or** `devenv` available to auto-start
  `devenv up temporal-server` for the duration of the test
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from typing import Any

import pytest

from adk_deepagents import create_deep_agent
from adk_deepagents.temporal.worker import create_temporal_worker
from adk_deepagents.types import DynamicTaskConfig, SubAgentSpec, TemporalTaskConfig
from tests.integration_tests.conftest import (
    make_litellm_model,
    run_agent_with_events,
    send_followup_with_events,
)

pytestmark = [pytest.mark.integration, pytest.mark.llm]
pytest.importorskip("temporalio")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_temporal_config(base: TemporalTaskConfig) -> TemporalTaskConfig:
    """Return a TemporalTaskConfig with a unique task queue for test isolation."""
    return TemporalTaskConfig(
        target_host=base.target_host,
        namespace=base.namespace,
        task_queue=f"adk-deepagents-llm-{uuid.uuid4().hex[:8]}",
        workflow_id_prefix="llm-temporal-it",
        idle_timeout_seconds=30.0,
    )


@contextlib.asynccontextmanager
async def _temporal_worker(
    model: Any,
    temporal_config: TemporalTaskConfig,
    dynamic_task_config: DynamicTaskConfig,
    *,
    subagents: list[SubAgentSpec] | None = None,
    default_tools: list | None = None,
):
    """Context manager that starts a Temporal worker and tears it down."""
    worker = await create_temporal_worker(
        default_model=model,
        dynamic_task_config=dynamic_task_config,
        subagents=subagents,
        default_tools=default_tools,
    )
    worker_task = asyncio.create_task(worker.run())
    await asyncio.sleep(0.4)
    try:
        yield worker
    finally:
        await worker.shutdown()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(240)
async def test_temporal_single_turn_delegation(
    ensure_temporal_server: TemporalTaskConfig,
):
    """A single-turn dynamic task dispatched through Temporal returns a result."""
    model = make_litellm_model()
    temporal_config = _make_temporal_config(ensure_temporal_server)
    dynamic_task_config = DynamicTaskConfig(timeout_seconds=90, temporal=temporal_config)

    async with _temporal_worker(model, temporal_config, dynamic_task_config):
        agent = create_deep_agent(
            model=model,
            name="temporal_single_turn_test",
            instruction=("Always delegate using the task tool. Never answer directly."),
            delegation_mode="dynamic",
            dynamic_task_config=dynamic_task_config,
        )

        texts, calls, responses, _runner, _session = await run_agent_with_events(
            agent,
            "Use task to solve: what is 256 divided by 16?",
        )

        response = " ".join(texts)
        assert "task" in calls, f"Expected task call, got: {calls}"
        assert "task" in responses, f"Expected task response, got: {responses}"
        assert "16" in response, f"Expected result 16, got: {response}"


@pytest.mark.timeout(240)
async def test_temporal_multi_turn_resume_state(
    ensure_temporal_server: TemporalTaskConfig,
):
    """Temporal workflows preserve state across multiple turns (task_id reuse)."""
    model = make_litellm_model()
    temporal_config = _make_temporal_config(ensure_temporal_server)
    dynamic_task_config = DynamicTaskConfig(timeout_seconds=90, temporal=temporal_config)

    async with _temporal_worker(model, temporal_config, dynamic_task_config):
        agent = create_deep_agent(
            model=model,
            name="temporal_multi_turn_test",
            instruction=(
                "Always delegate using the task tool. "
                "If the user gives a task_id, pass that same task_id to task."
            ),
            delegation_mode="dynamic",
            dynamic_task_config=dynamic_task_config,
        )

        texts1, calls1, responses1, runner, session = await run_agent_with_events(
            agent,
            "Use task to remember this exact codeword: ORBIT. Acknowledge when done.",
        )

        response1 = " ".join(texts1).lower()
        assert "task" in calls1, f"Expected task call on first turn, got: {calls1}"
        assert "task" in responses1, f"Expected task response on first turn, got: {responses1}"
        assert any(word in response1 for word in ("orbit", "remember", "done", "acknowledged")), (
            f"Expected acknowledgement, got: {response1}"
        )

        texts2, calls2, responses2 = await send_followup_with_events(
            runner,
            session,
            "Call task again with task_id task_1 and ask: what codeword did I ask "
            "you to remember? Return only the exact codeword.",
        )
        response2 = " ".join(texts2).lower()

        assert "task" in calls2, f"Expected task call on second turn, got: {calls2}"
        assert "task" in responses2, f"Expected task response on second turn, got: {responses2}"
        assert "orbit" in response2, f"Expected resumed memory ORBIT, got: {response2}"


@pytest.mark.timeout(240)
async def test_temporal_named_subagent_routing(
    ensure_temporal_server: TemporalTaskConfig,
):
    """Temporal worker routes tasks to the correct named sub-agent."""
    model = make_litellm_model()
    temporal_config = _make_temporal_config(ensure_temporal_server)
    dynamic_task_config = DynamicTaskConfig(timeout_seconds=90, temporal=temporal_config)

    math_subagent = SubAgentSpec(
        name="math_expert",
        description="Solves arithmetic problems. Always show your work.",
    )

    async with _temporal_worker(
        model,
        temporal_config,
        dynamic_task_config,
        subagents=[math_subagent],
    ):
        agent = create_deep_agent(
            model=model,
            name="temporal_named_subagent_test",
            instruction=(
                "Always delegate using the task tool. "
                "For arithmetic, use subagent_type='math_expert'."
            ),
            subagents=[math_subagent],
            delegation_mode="dynamic",
            dynamic_task_config=dynamic_task_config,
        )

        texts, calls, responses, _runner, _session = await run_agent_with_events(
            agent,
            "Use task with subagent_type math_expert to solve: what is 225 divided by 15?",
        )

        response = " ".join(texts)
        assert "task" in calls, f"Expected task call, got: {calls}"
        assert "task" in responses, f"Expected task response, got: {responses}"
        assert "15" in response, f"Expected result 15, got: {response}"


@pytest.mark.timeout(240)
async def test_temporal_multiple_independent_tasks(
    ensure_temporal_server: TemporalTaskConfig,
):
    """Separate task_ids create independent Temporal workflows with correct results."""
    model = make_litellm_model()
    temporal_config = _make_temporal_config(ensure_temporal_server)
    dynamic_task_config = DynamicTaskConfig(timeout_seconds=90, temporal=temporal_config)

    async with _temporal_worker(model, temporal_config, dynamic_task_config):
        agent = create_deep_agent(
            model=model,
            name="temporal_independent_tasks_test",
            instruction=(
                "Always delegate using the task tool. Never answer directly. "
                "When asked multiple questions, call task once for each question."
            ),
            delegation_mode="dynamic",
            dynamic_task_config=dynamic_task_config,
        )

        texts, calls, responses, _runner, _session = await run_agent_with_events(
            agent,
            "Use task to answer: what is the capital of France? "
            "Then use task again to answer: what is 7 times 8?",
        )

        response = " ".join(texts).lower()
        assert calls.count("task") >= 2, f"Expected at least 2 task calls, got: {calls}"
        assert "paris" in response, f"Expected Paris in response, got: {response}"
        assert "56" in response, f"Expected 56 in response, got: {response}"


@pytest.mark.timeout(240)
async def test_temporal_task_returns_function_calls_metadata(
    ensure_temporal_server: TemporalTaskConfig,
):
    """The task result from a Temporal worker includes function_calls metadata."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    model = make_litellm_model()
    temporal_config = _make_temporal_config(ensure_temporal_server)
    dynamic_task_config = DynamicTaskConfig(timeout_seconds=90, temporal=temporal_config)

    async with _temporal_worker(model, temporal_config, dynamic_task_config):
        agent = create_deep_agent(
            model=model,
            name="temporal_metadata_test",
            instruction=(
                "Always delegate using the task tool. "
                "Return the result from the task tool verbatim."
            ),
            delegation_mode="dynamic",
            dynamic_task_config=dynamic_task_config,
        )

        runner = InMemoryRunner(agent=agent, app_name="integration_test")
        session = await runner.session_service.create_session(
            app_name="integration_test",
            user_id="test_user",
            state={"files": {}},
        )

        content = types.Content(
            role="user", parts=[types.Part(text="Use task to solve: what is 99 + 1?")]
        )
        task_payloads: list[dict] = []

        async for event in runner.run_async(
            session_id=session.id,
            user_id="test_user",
            new_message=content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    fn_resp = getattr(part, "function_response", None)
                    if fn_resp is None:
                        continue
                    if getattr(fn_resp, "name", None) != "task":
                        continue
                    payload = getattr(fn_resp, "response", None)
                    if isinstance(payload, dict):
                        task_payloads.append(payload)

        assert task_payloads, "Expected at least one task function-response payload"
        payload = task_payloads[-1]
        assert payload.get("status") == "completed", f"Expected completed status, got: {payload}"
        assert "task_id" in payload, f"Expected task_id in payload, got: {payload}"
        assert "result" in payload, f"Expected result in payload, got: {payload}"


@pytest.mark.timeout(240)
async def test_temporal_workflow_idle_timeout_completes_workflow(
    ensure_temporal_server: TemporalTaskConfig,
):
    """A Temporal workflow auto-completes after the idle timeout expires."""
    from temporalio.client import Client, WorkflowExecutionStatus
    from temporalio.worker import UnsandboxedWorkflowRunner, Worker

    from adk_deepagents.temporal.activities import TaskSnapshot
    from adk_deepagents.temporal.client import _CLIENT_CACHE, run_task_via_temporal
    from adk_deepagents.temporal.workflows import DynamicTaskWorkflow, configure_workflow

    _CLIENT_CACHE.clear()

    temporal_config = _make_temporal_config(ensure_temporal_server)

    client = await Client.connect(
        temporal_config.target_host,
        namespace=temporal_config.namespace,
    )

    configure_workflow(
        activity_timeout_seconds=30.0,
        retry_max_attempts=1,
        idle_timeout_seconds=3.0,
    )

    from temporalio import activity

    @activity.defn(name="run_dynamic_task")
    async def stub_activity(snapshot_dict: dict[str, Any]) -> dict[str, Any]:
        return {
            "result": "stub",
            "function_calls": [],
            "files": {},
            "todos": [],
            "timed_out": False,
            "error": None,
        }

    worker = Worker(
        client,
        task_queue=temporal_config.task_queue,
        workflows=[DynamicTaskWorkflow],
        activities=[stub_activity],
        workflow_runner=UnsandboxedWorkflowRunner(),
    )
    worker_task = asyncio.create_task(worker.run())
    await asyncio.sleep(0.3)

    task_config = DynamicTaskConfig(temporal=temporal_config)
    _CLIENT_CACHE[(temporal_config.target_host, temporal_config.namespace)] = client

    try:
        result = await run_task_via_temporal(
            snapshot=TaskSnapshot(
                subagent_type="general_purpose",
                prompt="trigger workflow",
            ),
            logical_parent_id="idle-test",
            task_id="task_idle",
            task_config=task_config,
        )
        assert result["error"] is None
        assert result["result"] == "stub"

        wf_id = f"{temporal_config.workflow_id_prefix}:idle-test:task_idle"

        await asyncio.sleep(5.0)

        handle = client.get_workflow_handle(wf_id)
        desc = await handle.describe()
        assert desc.status == WorkflowExecutionStatus.COMPLETED, (
            f"Expected workflow to auto-complete after idle timeout, got: {desc.status}"
        )
    finally:
        await worker.shutdown()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task
        _CLIENT_CACHE.clear()
        configure_workflow()
