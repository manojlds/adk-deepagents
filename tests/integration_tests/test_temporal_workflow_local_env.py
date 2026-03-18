"""Integration tests for Temporal dynamic-task plumbing using local test environment."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from typing import Any

import pytest

pytest.importorskip("temporalio")


@pytest.mark.integration
@pytest.mark.timeout(180)
async def test_temporal_client_roundtrip_with_local_environment():
    from temporalio import activity
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import UnsandboxedWorkflowRunner, Worker

    from adk_deepagents.temporal.activities import TaskSnapshot
    from adk_deepagents.temporal.client import _CLIENT_CACHE, run_task_via_temporal
    from adk_deepagents.temporal.workflows import DynamicTaskWorkflow, configure_workflow
    from adk_deepagents.types import DynamicTaskConfig, TemporalTaskConfig

    _CLIENT_CACHE.clear()

    env = await WorkflowEnvironment.start_local(ui=False, dev_server_log_level="error")
    task_queue = f"adk-deepagents-it-{uuid.uuid4().hex[:8]}"
    worker: Worker | None = None
    worker_task: asyncio.Task | None = None

    try:

        @activity.defn(name="run_dynamic_task")
        async def run_dynamic_task(snapshot_dict: dict[str, Any]) -> dict[str, Any]:
            prompt = str(snapshot_dict.get("prompt", ""))
            files = dict(snapshot_dict.get("files", {}))
            todos = list(snapshot_dict.get("todos", []))

            files["/last_prompt.txt"] = {"content": [prompt]}
            todos.append({"text": f"handled:{prompt}", "done": True})

            return {
                "result": f"handled:{prompt}",
                "function_calls": ["stub_tool"],
                "files": files,
                "todos": todos,
                "timed_out": False,
                "error": None,
            }

        configure_workflow(activity_timeout_seconds=30.0, retry_max_attempts=1)
        worker = Worker(
            env.client,
            task_queue=task_queue,
            workflows=[DynamicTaskWorkflow],
            activities=[run_dynamic_task],
            workflow_runner=UnsandboxedWorkflowRunner(),
        )
        worker_task = asyncio.create_task(worker.run())
        await asyncio.sleep(0.2)

        config = env.client.config()
        service_config = config["service_client"].config
        temporal_config = TemporalTaskConfig(
            target_host=service_config.target_host,
            namespace=config["namespace"],
            task_queue=task_queue,
            workflow_id_prefix=f"it-{uuid.uuid4().hex[:8]}",
        )
        task_config = DynamicTaskConfig(temporal=temporal_config)

        first = await run_task_via_temporal(
            snapshot=TaskSnapshot(
                subagent_type="general_purpose",
                prompt="first turn",
                files={},
                todos=[],
                history=[],
            ),
            logical_parent_id="parent-it",
            task_id="task_1",
            task_config=task_config,
        )

        assert first["error"] is None
        assert first["result"] == "handled:first turn"
        assert first["function_calls"] == ["stub_tool"]
        assert first["files"]["/last_prompt.txt"]["content"] == ["first turn"]

        second = await run_task_via_temporal(
            snapshot=TaskSnapshot(
                subagent_type="general_purpose",
                prompt="second turn",
                files=first["files"],
                todos=first["todos"],
                history=[{"prompt": "first turn", "result": "handled:first turn"}],
            ),
            logical_parent_id="parent-it",
            task_id="task_1",
            task_config=task_config,
        )

        assert second["error"] is None
        assert second["result"] == "handled:second turn"
        assert second["files"]["/last_prompt.txt"]["content"] == ["second turn"]
        assert second["todos"][-1]["done"] is True
    finally:
        if worker is not None:
            await worker.shutdown()
        if worker_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await worker_task
        await env.shutdown()
        _CLIENT_CACHE.clear()
