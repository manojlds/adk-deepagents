"""Temporal client helpers for dynamic task dispatch."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from adk_deepagents.temporal.activities import TaskResult, TaskSnapshot
from adk_deepagents.types import DynamicTaskConfig, TemporalTaskConfig

logger = logging.getLogger(__name__)

_CLIENT_CACHE: dict[tuple[str, str], Any] = {}
_SPEC_HASH_CACHE: dict[str, str] = {}
_SPEC_HASH_CACHE_MAX_ENTRIES = 2048


def _workflow_id(config: TemporalTaskConfig, logical_parent_id: str, task_id: str) -> str:
    return f"{config.workflow_id_prefix}:{logical_parent_id}:{task_id}"


async def _get_or_create_client(config: TemporalTaskConfig) -> Any:
    from temporalio.client import Client

    cache_key = (config.target_host, config.namespace)
    existing = _CLIENT_CACHE.get(cache_key)
    if existing is not None:
        return existing

    client = await Client.connect(config.target_host, namespace=config.namespace)
    _CLIENT_CACHE[cache_key] = client
    return client


def _is_not_found_error(exc: Exception) -> bool:
    try:
        from temporalio.service import RPCError, RPCStatusCode
    except ImportError:  # pragma: no cover - optional dependency guard
        return False

    if isinstance(exc, RPCError) and exc.status == RPCStatusCode.NOT_FOUND:
        return True

    cause = getattr(exc, "cause", None)
    return isinstance(cause, RPCError) and cause.status == RPCStatusCode.NOT_FOUND


def _subagent_spec_hash(spec: dict[str, Any]) -> str:
    serialized = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _remember_spec_hash(workflow_id: str, spec_hash: str) -> None:
    # Refresh insertion order for existing entries.
    if workflow_id in _SPEC_HASH_CACHE:
        _SPEC_HASH_CACHE.pop(workflow_id, None)

    _SPEC_HASH_CACHE[workflow_id] = spec_hash

    overflow = len(_SPEC_HASH_CACHE) - _SPEC_HASH_CACHE_MAX_ENTRIES
    for _ in range(max(0, overflow)):
        oldest = next(iter(_SPEC_HASH_CACHE), None)
        if oldest is None:
            break
        _SPEC_HASH_CACHE.pop(oldest, None)


async def run_task_via_temporal(
    *,
    snapshot: TaskSnapshot,
    logical_parent_id: str,
    task_id: str,
    task_config: DynamicTaskConfig,
) -> dict[str, Any]:
    """Dispatch a dynamic task turn to Temporal and return the activity payload."""
    temporal_config = task_config.temporal
    if temporal_config is None:
        return TaskResult(error="Temporal config is None").to_dict()

    from temporalio.client import WorkflowExecutionStatus
    from temporalio.common import WorkflowIDConflictPolicy

    try:
        client = await _get_or_create_client(temporal_config)
    except Exception as exc:
        logger.exception("Failed to connect to Temporal")
        return TaskResult(error=f"Temporal connection failed: {exc}").to_dict()

    workflow_id = _workflow_id(temporal_config, logical_parent_id, task_id)
    snapshot_dict = snapshot.to_dict()

    handle = None
    try:
        existing = client.get_workflow_handle(workflow_id)
        description = await existing.describe()
        unspecified = getattr(WorkflowExecutionStatus, "UNSPECIFIED", None)
        if description.status in {WorkflowExecutionStatus.RUNNING, unspecified, None}:
            handle = existing
    except Exception as exc:
        if not _is_not_found_error(exc):
            logger.debug("Unable to describe workflow %s", workflow_id, exc_info=True)

    if handle is None:
        try:
            handle = await client.start_workflow(
                "DynamicTaskWorkflow",
                snapshot_dict,
                id=workflow_id,
                task_queue=temporal_config.task_queue,
                id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
            )
        except Exception as exc:
            logger.exception("Failed to start Temporal workflow %s", workflow_id)
            return TaskResult(error=f"Temporal workflow start failed: {exc}").to_dict()

    turn_input = {
        "subagent_type": snapshot.subagent_type,
        "prompt": snapshot.prompt,
        "depth": snapshot.depth,
        "history": snapshot.history,
        "files": snapshot.files,
        "todos": snapshot.todos,
        "model_override": snapshot.model_override,
        "timeout_seconds": snapshot.timeout_seconds,
    }

    if snapshot.backend_context is not None:
        turn_input["backend_context"] = snapshot.backend_context

    spec_hash: str | None = None
    if snapshot.subagent_spec is not None:
        spec_hash = _subagent_spec_hash(snapshot.subagent_spec)
        turn_input["subagent_spec_hash"] = spec_hash

        previous_hash = _SPEC_HASH_CACHE.get(workflow_id)
        if previous_hash != spec_hash:
            turn_input["subagent_spec"] = snapshot.subagent_spec

    try:
        result = await handle.execute_update("run_turn", arg=turn_input)
    except Exception as exc:
        logger.exception("Temporal workflow update failed for %s", workflow_id)
        return TaskResult(error=f"Temporal task execution failed: {exc}").to_dict()

    if spec_hash is not None:
        _remember_spec_hash(workflow_id, spec_hash)

    if isinstance(result, dict):
        return result

    return TaskResult(error=f"Unexpected Temporal result type: {type(result).__name__}").to_dict()
