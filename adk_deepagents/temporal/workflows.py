"""Temporal workflow definition for dynamic task turns."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

RUN_TASK_ACTIVITY_NAME = "run_dynamic_task"

_activity_timeout_seconds: float = 120.0
_retry_max_attempts: int = 1
_idle_timeout_seconds: float = 600.0


def configure_workflow(
    *,
    activity_timeout_seconds: float = 120.0,
    retry_max_attempts: int = 1,
    idle_timeout_seconds: float = 600.0,
) -> None:
    """Set module-level workflow runtime settings used by ``DynamicTaskWorkflow``."""
    global _activity_timeout_seconds, _retry_max_attempts, _idle_timeout_seconds  # noqa: PLW0603
    _activity_timeout_seconds = activity_timeout_seconds
    _retry_max_attempts = retry_max_attempts
    _idle_timeout_seconds = idle_timeout_seconds


@workflow.defn(name="DynamicTaskWorkflow")
class DynamicTaskWorkflow:
    """One workflow per logical ``task_id`` with update-based turn execution."""

    def __init__(self) -> None:
        self._snapshot: dict[str, Any] = {}
        self._completed = False
        self._last_activity_time = 0.0

    @workflow.run
    async def run(self, initial_snapshot: dict[str, Any]) -> dict[str, Any]:
        self._snapshot = dict(initial_snapshot)
        self._last_activity_time = workflow.time()

        while not self._completed:
            elapsed = workflow.time() - self._last_activity_time
            remaining = _idle_timeout_seconds - elapsed
            if remaining <= 0:
                break
            try:
                await workflow.wait_condition(
                    lambda: self._completed,
                    timeout=timedelta(seconds=remaining),
                )
            except TimeoutError:
                break

        return self._snapshot

    @workflow.update(name="run_turn")
    async def run_turn(self, turn_input: dict[str, Any]) -> dict[str, Any]:
        merged = dict(self._snapshot)
        merged["prompt"] = turn_input.get("prompt", merged.get("prompt", ""))
        if "history" in turn_input:
            merged["history"] = turn_input["history"]
        if "files" in turn_input:
            merged["files"] = turn_input["files"]
        if "todos" in turn_input:
            merged["todos"] = turn_input["todos"]
        if "subagent_spec" in turn_input:
            merged["subagent_spec"] = turn_input["subagent_spec"]
        if "subagent_spec_hash" in turn_input:
            merged["subagent_spec_hash"] = turn_input["subagent_spec_hash"]

        result = await workflow.execute_activity(
            RUN_TASK_ACTIVITY_NAME,
            arg=merged,
            start_to_close_timeout=timedelta(seconds=_activity_timeout_seconds),
            retry_policy=RetryPolicy(maximum_attempts=_retry_max_attempts),
        )

        if isinstance(result, dict):
            if "files" in result:
                self._snapshot["files"] = result["files"]
            if "todos" in result:
                self._snapshot["todos"] = result["todos"]

        self._last_activity_time = workflow.time()
        return result

    @workflow.signal(name="complete")
    async def complete(self) -> None:
        self._completed = True
