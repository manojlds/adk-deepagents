"""Runtime registry, concurrency slots, and sub-agent spec helpers for dynamic tasks."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner

from adk_deepagents.tools.task import (
    GENERAL_PURPOSE_SUBAGENT,
    _sanitize_agent_name,
)
from adk_deepagents.types import DynamicTaskConfig, SubAgentSpec

logger = logging.getLogger(__name__)

_TASK_STORE_KEY = "_dynamic_tasks"
_TASK_COUNTER_KEY = "_dynamic_task_counter"
_TASK_PARENT_ID_KEY = "_dynamic_parent_session_id"
_TASK_DEPTH_KEY = "_dynamic_delegation_depth"
_RUNNING_TASKS_KEY = "_dynamic_running_tasks"
_RUNTIME_SUBAGENT_STORE_KEY = "_dynamic_subagent_specs"


@dataclass
class _TaskRuntime:
    runner: InMemoryRunner
    session_id: str
    user_id: str
    subagent_type: str


_RUNTIME_REGISTRY: dict[str, _TaskRuntime] = {}
_CONCURRENCY_LOCKS: dict[str, asyncio.Lock] = {}


def _get_concurrency_lock(logical_parent_id: str) -> asyncio.Lock:
    lock = _CONCURRENCY_LOCKS.get(logical_parent_id)
    if lock is None:
        lock = asyncio.Lock()
        _CONCURRENCY_LOCKS[logical_parent_id] = lock
    return lock


async def _acquire_concurrency_slot(
    *,
    running_tasks: list[str],
    logical_parent_id: str,
    run_key: str,
    task_id: str,
    config: DynamicTaskConfig,
) -> tuple[bool, str | None, float]:
    policy = config.concurrency_policy
    if policy not in {"error", "wait"}:
        return False, f"Invalid dynamic task concurrency policy: {policy!r}", 0.0

    if config.max_parallel < 1:
        return False, "Dynamic task max_parallel must be >= 1", 0.0

    loop = asyncio.get_running_loop()
    started = loop.time()
    queue_timeout = max(0.0, config.queue_timeout_seconds)

    while True:
        lock = _get_concurrency_lock(logical_parent_id)
        async with lock:
            currently_running = len(running_tasks)
            task_already_running = run_key in running_tasks

            if not task_already_running and currently_running < config.max_parallel:
                running_tasks.append(run_key)
                return True, None, loop.time() - started

        if policy == "error":
            if task_already_running:
                return (
                    False,
                    f"Dynamic task is already running: task_id={task_id}",
                    0.0,
                )
            return (
                False,
                (
                    "Dynamic task concurrency limit exceeded: "
                    f"running={currently_running}, max_parallel={config.max_parallel}"
                ),
                0.0,
            )

        elapsed = loop.time() - started
        if elapsed >= queue_timeout:
            return (
                False,
                (
                    "Dynamic task queue timeout after "
                    f"{queue_timeout:.1f}s waiting for a concurrency slot "
                    f"(running={currently_running}, max_parallel={config.max_parallel})"
                ),
                elapsed,
            )

        await asyncio.sleep(min(0.05, queue_timeout - elapsed))


async def _release_concurrency_slot(
    *,
    running_tasks: list[str],
    logical_parent_id: str,
    run_key: str,
) -> None:
    lock = _get_concurrency_lock(logical_parent_id)
    async with lock:
        if run_key in running_tasks:
            running_tasks.remove(run_key)

        if not running_tasks:
            _CONCURRENCY_LOCKS.pop(logical_parent_id, None)


def _queue_wait_metadata(wait_seconds: float) -> dict[str, Any]:
    if wait_seconds <= 0:
        return {"queued": False, "queue_wait_seconds": 0.0}

    return {
        "queued": True,
        "queue_wait_seconds": round(wait_seconds, 3),
    }


def _prune_stale_running_tasks(*, running_tasks: list[str], logical_parent_id: str) -> None:
    """Drop stale in-flight task markers left behind after process restarts."""
    prefix = f"{logical_parent_id}:"
    if any(run_key.startswith(prefix) for run_key in _RUNTIME_REGISTRY):
        return

    running_tasks[:] = [
        run_key
        for run_key in running_tasks
        if not (isinstance(run_key, str) and run_key.startswith(prefix))
    ]


def _tool_name(tool: Any) -> str | None:
    raw_name = getattr(tool, "__name__", getattr(tool, "name", None))
    if isinstance(raw_name, str):
        normalized = raw_name.strip()
        if normalized:
            return normalized
    return None


def _build_tool_index(default_tools: list[Any]) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for tool in default_tools:
        name = _tool_name(tool)
        if name is not None and name not in index:
            index[name] = tool
    return index


def _normalize_subagent_type(subagent_type: str) -> str:
    normalized_input = subagent_type.strip()
    if not normalized_input:
        return "general_purpose"

    normalized = _sanitize_agent_name(normalized_input)
    if normalized in {"general", "generalpurpose"}:
        return "general_purpose"
    return normalized


def _get_runtime_subagent_store(state: Any) -> dict[str, Any]:
    raw_store = state.setdefault(_RUNTIME_SUBAGENT_STORE_KEY, {})
    if isinstance(raw_store, dict):
        return raw_store

    store: dict[str, Any] = {}
    state[_RUNTIME_SUBAGENT_STORE_KEY] = store
    return store


def _persist_runtime_subagent_spec(
    *,
    state: Any,
    name: str,
    description: str,
    system_prompt: str | None,
    model: str | None,
    tool_names: list[str] | None,
) -> None:
    store = _get_runtime_subagent_store(state)
    payload: dict[str, Any] = {
        "name": name,
        "description": description,
    }
    if system_prompt:
        payload["system_prompt"] = system_prompt
    if model:
        payload["model"] = model
    if tool_names is not None:
        payload["tool_names"] = tool_names

    store[name] = payload


def _resolve_runtime_tool_names(
    *,
    tool_names: list[str] | None,
    tool_index: dict[str, Any],
) -> tuple[list[str] | None, str | None]:
    if tool_names is None:
        return None, None

    normalized_tool_names: list[str] = []
    seen: set[str] = set()
    for raw_name in tool_names:
        if not isinstance(raw_name, str):
            continue

        name = raw_name.strip()
        if not name:
            continue

        if name.lower() == "all":
            return None, None

        if name in seen:
            continue
        seen.add(name)
        normalized_tool_names.append(name)

    unknown = [name for name in normalized_tool_names if name not in tool_index]
    if unknown:
        available = ", ".join(sorted(tool_index))
        return None, (
            f"Unknown tool_names: {', '.join(sorted(unknown))}. Available tools: {available}"
        )

    return normalized_tool_names, None


def _load_runtime_subagent_specs(
    *,
    state: Any,
    tool_index: dict[str, Any],
) -> dict[str, SubAgentSpec]:
    raw_store = state.get(_RUNTIME_SUBAGENT_STORE_KEY, {})
    if not isinstance(raw_store, dict):
        return {}

    runtime_registry: dict[str, SubAgentSpec] = {}
    for raw_key, raw_spec in raw_store.items():
        if not isinstance(raw_spec, dict):
            continue

        raw_name = raw_spec.get("name", raw_key)
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        normalized_name = _normalize_subagent_type(raw_name)

        raw_description = raw_spec.get("description")
        if not isinstance(raw_description, str):
            continue
        description = raw_description.strip()
        if not description:
            continue

        spec: SubAgentSpec = SubAgentSpec(
            name=normalized_name,
            description=description,
        )

        raw_system_prompt = raw_spec.get("system_prompt")
        if isinstance(raw_system_prompt, str) and raw_system_prompt.strip():
            spec["system_prompt"] = raw_system_prompt.strip()

        raw_model = raw_spec.get("model")
        if isinstance(raw_model, str) and raw_model.strip():
            spec["model"] = raw_model.strip()

        raw_tool_names = raw_spec.get("tool_names")
        if isinstance(raw_tool_names, list):
            spec["tools"] = [
                tool_index[name]
                for name in raw_tool_names
                if isinstance(name, str) and name in tool_index
            ]

        runtime_registry[normalized_name] = spec

    return runtime_registry


def _coerce_subagent_spec_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    raw_name = value.get("name")
    raw_description = value.get("description")
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None
    if not isinstance(raw_description, str) or not raw_description.strip():
        return None

    payload: dict[str, Any] = {
        "name": _normalize_subagent_type(raw_name),
        "description": raw_description.strip(),
    }

    raw_system_prompt = value.get("system_prompt")
    if isinstance(raw_system_prompt, str) and raw_system_prompt.strip():
        payload["system_prompt"] = raw_system_prompt.strip()

    raw_model = value.get("model")
    if isinstance(raw_model, str) and raw_model.strip():
        payload["model"] = raw_model.strip()

    raw_tool_names = value.get("tool_names")
    if isinstance(raw_tool_names, list):
        normalized_tool_names: list[str] = []
        seen: set[str] = set()
        for entry in raw_tool_names:
            if not isinstance(entry, str):
                continue
            name = entry.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            normalized_tool_names.append(name)
        payload["tool_names"] = normalized_tool_names

    return payload


def _runtime_subagent_spec_payload(*, state: Any, subagent_type: str) -> dict[str, Any] | None:
    raw_store = state.get(_RUNTIME_SUBAGENT_STORE_KEY, {})
    if not isinstance(raw_store, dict):
        return None

    direct = _coerce_subagent_spec_payload(raw_store.get(subagent_type))
    if direct is not None:
        return direct

    for raw_spec in raw_store.values():
        payload = _coerce_subagent_spec_payload(raw_spec)
        if payload is None:
            continue
        if _normalize_subagent_type(payload["name"]) == subagent_type:
            return payload

    return None


def _build_dynamic_registry(
    subagents: list[SubAgentSpec | LlmAgent] | None,
) -> dict[str, SubAgentSpec | LlmAgent]:
    """Build lookup map for dynamic sub-agent profile resolution."""
    registry: dict[str, SubAgentSpec | LlmAgent] = {
        "general_purpose": GENERAL_PURPOSE_SUBAGENT,
    }
    if not subagents:
        return registry

    for item in subagents:
        if isinstance(item, LlmAgent):
            registry[_sanitize_agent_name(item.name)] = item
        else:
            name = item.get("name")
            if isinstance(name, str) and name:
                registry[_sanitize_agent_name(name)] = item
    return registry
