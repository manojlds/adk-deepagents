"""Dynamic ``task`` tool for runtime sub-agent delegation.

Unlike static ``AgentTool`` wiring, this tool spawns or resumes sub-agent
sessions at runtime using ``task_id``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any, cast

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from adk_deepagents.backends.protocol import BackendFactory
from adk_deepagents.backends.runtime import get_registered_backend_factory, register_backend_factory
from adk_deepagents.callbacks.before_tool import make_before_tool_callback
from adk_deepagents.prompts import DEFAULT_SUBAGENT_PROMPT
from adk_deepagents.tools.task import (
    GENERAL_PURPOSE_SUBAGENT,
    _resolve_skills_tools,
    _sanitize_agent_name,
)
from adk_deepagents.types import DynamicTaskConfig, SkillsConfig, SubAgentSpec

logger = logging.getLogger(__name__)

_TASK_STORE_KEY = "_dynamic_tasks"
_TASK_COUNTER_KEY = "_dynamic_task_counter"
_TASK_PARENT_ID_KEY = "_dynamic_parent_session_id"
_TASK_DEPTH_KEY = "_dynamic_delegation_depth"
_RUNNING_TASKS_KEY = "_dynamic_running_tasks"
_RUNTIME_SUBAGENT_STORE_KEY = "_dynamic_subagent_specs"
_RUNTIME_REGISTRY: dict[str, _TaskRuntime] = {}
_CONCURRENCY_LOCKS: dict[str, asyncio.Lock] = {}


@dataclass
class _TaskRuntime:
    runner: InMemoryRunner
    session_id: str
    user_id: str
    subagent_type: str


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


def _coerce_backend_factory(value: Any) -> BackendFactory | None:
    if callable(value):
        return cast(BackendFactory, value)
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


def create_register_subagent_tool(
    *,
    default_model: str | Any,
    default_tools: list,
    config: DynamicTaskConfig | None = None,
):
    """Create a ``register_subagent`` tool for runtime specialization."""
    task_config = config or DynamicTaskConfig()
    tool_index = _build_tool_index(default_tools)

    async def register_subagent(
        name: str,
        description: str,
        system_prompt: str | None = None,
        model: str | None = None,
        tool_names: list[str] | None = None,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Register or update a runtime sub-agent profile."""
        if tool_context is None:
            return {"status": "error", "error": "tool_context is required"}

        raw_name = name.strip()
        if not raw_name:
            return {"status": "error", "error": "Sub-agent name cannot be empty"}
        normalized_name = _normalize_subagent_type(raw_name)

        normalized_description = description.strip()
        if not normalized_description:
            return {"status": "error", "error": "Sub-agent description cannot be empty"}

        normalized_system_prompt: str | None = None
        if isinstance(system_prompt, str):
            stripped_prompt = system_prompt.strip()
            if stripped_prompt:
                normalized_system_prompt = stripped_prompt

        normalized_model: str | None = None
        if isinstance(model, str):
            stripped_model = model.strip()
            if stripped_model:
                normalized_model = stripped_model

        if normalized_model and not task_config.allow_model_override:
            return {
                "status": "error",
                "error": "Model override is disabled for dynamic task delegation",
            }

        resolved_tool_names, tool_error = _resolve_runtime_tool_names(
            tool_names=tool_names,
            tool_index=tool_index,
        )
        if tool_error:
            return {"status": "error", "error": tool_error}

        _persist_runtime_subagent_spec(
            state=tool_context.state,
            name=normalized_name,
            description=normalized_description,
            system_prompt=normalized_system_prompt,
            model=normalized_model,
            tool_names=resolved_tool_names,
        )

        effective_model: str | None = normalized_model
        if effective_model is None and isinstance(default_model, str):
            effective_model = default_model

        selected_tools = (
            resolved_tool_names if resolved_tool_names is not None else sorted(tool_index)
        )

        return {
            "status": "registered",
            "subagent_type": normalized_name,
            "description": normalized_description,
            "model": effective_model,
            "tool_names": selected_tools,
        }

    register_subagent.__name__ = "register_subagent"
    return register_subagent


def _build_spec_agent(
    spec: SubAgentSpec,
    *,
    default_model: str | Any,
    default_tools: list,
    skills_config: SkillsConfig | None,
    model_override: str | None,
    config: DynamicTaskConfig,
) -> LlmAgent:
    spec_name = spec.get("name")
    spec_description = spec.get("description")
    if not isinstance(spec_name, str) or not spec_name:
        raise ValueError("Dynamic sub-agent spec is missing required field: name")
    if not isinstance(spec_description, str) or not spec_description:
        raise ValueError("Dynamic sub-agent spec is missing required field: description")

    sub_tools: list[Any] = list(spec.get("tools", default_tools))

    sub_skills = spec.get("skills")
    if sub_skills:
        sub_tools.extend(_resolve_skills_tools(sub_skills, skills_config))

    before_tool_cb = make_before_tool_callback(interrupt_on=spec.get("interrupt_on"))

    if model_override and not config.allow_model_override:
        raise ValueError("Model override is disabled for dynamic task delegation")
    resolved_model = model_override or spec.get("model", default_model)

    return LlmAgent(
        name=_sanitize_agent_name(spec_name),
        model=resolved_model,
        instruction=spec.get("system_prompt", DEFAULT_SUBAGENT_PROMPT),
        description=spec_description,
        tools=sub_tools,
        before_tool_callback=before_tool_cb,
    )


async def _run_dynamic_task(
    runtime: _TaskRuntime,
    *,
    prompt: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    texts: list[str] = []
    function_calls: list[str] = []

    async def _collect() -> None:
        async for event in runtime.runner.run_async(
            session_id=runtime.session_id,
            user_id=runtime.user_id,
            new_message=content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        texts.append(part.text)
                    if hasattr(part, "function_call") and part.function_call:
                        name = part.function_call.name
                        if isinstance(name, str) and name:
                            function_calls.append(name)

    timed_out = False
    error: str | None = None

    try:
        await asyncio.wait_for(_collect(), timeout=timeout_seconds)
    except TimeoutError:
        timed_out = True
    except Exception as exc:  # pragma: no cover - defensive path
        logger.exception("Dynamic task run failed")
        error = f"{type(exc).__name__}: {exc}"

    session_state: dict[str, Any] = {}
    try:
        session = await runtime.runner.session_service.get_session(
            app_name="dynamic_task",
            user_id=runtime.user_id,
            session_id=runtime.session_id,
        )
        if session is not None and isinstance(session.state, dict):
            session_state = session.state
    except Exception:  # pragma: no cover - defensive path
        logger.debug("Unable to fetch dynamic task session state", exc_info=True)

    return {
        "result": "\n".join(texts).strip(),
        "function_calls": function_calls,
        "files": session_state.get("files", {}),
        "todos": session_state.get("todos", []),
        "timed_out": timed_out,
        "error": error,
    }


def create_dynamic_task_tool(
    *,
    default_model: str | Any,
    default_tools: list,
    subagents: list[SubAgentSpec | LlmAgent] | None,
    skills_config: SkillsConfig | None = None,
    config: DynamicTaskConfig | None = None,
):
    """Create a ``task`` tool that dynamically spawns/resumes sub-agent sessions."""
    task_config = config or DynamicTaskConfig()
    registry = _build_dynamic_registry(subagents)
    tool_index = _build_tool_index(default_tools)

    async def task(
        description: str,
        prompt: str,
        subagent_type: str = "general",
        task_id: str | None = None,
        model: str | None = None,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Run a task in a dynamic sub-agent.

        Parameters mirror deepagents' task tool shape, with ``task_id`` support
        for session reuse across calls.
        """
        if tool_context is None:
            return {"status": "error", "error": "tool_context is required"}

        resolved_prompt = prompt.strip() if prompt.strip() else description.strip()
        if not resolved_prompt:
            return {"status": "error", "error": "Either prompt or description must be provided"}

        store = tool_context.state.setdefault(_TASK_STORE_KEY, {})
        if not isinstance(store, dict):
            return {"status": "error", "error": "Invalid dynamic task store in session state"}

        runtime: _TaskRuntime | None = None
        created_subagent = False
        normalized_type = _normalize_subagent_type(subagent_type)
        logical_parent_id = tool_context.state.get(_TASK_PARENT_ID_KEY)
        if not isinstance(logical_parent_id, str) or not logical_parent_id:
            logical_parent_id = uuid.uuid4().hex
            tool_context.state[_TASK_PARENT_ID_KEY] = logical_parent_id

        adk_parent_session_id = getattr(getattr(tool_context, "session", None), "id", None)
        runtime_backend_factory: BackendFactory | None = (
            get_registered_backend_factory(adk_parent_session_id)
            if isinstance(adk_parent_session_id, str) and adk_parent_session_id
            else None
        )

        if runtime_backend_factory is None:
            runtime_backend_factory = _coerce_backend_factory(
                tool_context.state.get("_backend_factory")
            )

        current_depth_raw = tool_context.state.get(_TASK_DEPTH_KEY, 0)
        current_depth = current_depth_raw if isinstance(current_depth_raw, int) else 0

        running_tasks = tool_context.state.setdefault(_RUNNING_TASKS_KEY, [])
        if not isinstance(running_tasks, list):
            return {"status": "error", "error": "Invalid dynamic running task tracker in state"}

        if task_id:
            existing = store.get(task_id)
            if not isinstance(existing, dict):
                return {"status": "error", "error": f"Unknown task_id: {task_id}"}
            runtime = _RUNTIME_REGISTRY.get(f"{logical_parent_id}:{task_id}")
            if runtime is None:
                return {
                    "status": "error",
                    "error": (
                        f"Task runtime for task_id {task_id} is unavailable in this process. "
                        "Start a new task."
                    ),
                }
            normalized_type = runtime.subagent_type
        else:
            if current_depth + 1 > task_config.max_depth:
                return {
                    "status": "error",
                    "error": (
                        f"Dynamic delegation depth limit exceeded: current depth {current_depth}, "
                        f"max_depth={task_config.max_depth}"
                    ),
                }
            counter = int(tool_context.state.get(_TASK_COUNTER_KEY, 0)) + 1
            tool_context.state[_TASK_COUNTER_KEY] = counter
            task_id = f"task_{counter}"

            runtime_registry = _load_runtime_subagent_specs(
                state=tool_context.state,
                tool_index=tool_index,
            )
            selected_registry: dict[str, SubAgentSpec | LlmAgent] = dict(registry)
            selected_registry.update(runtime_registry)

            selected = selected_registry.get(normalized_type)
            if selected is None:
                created_subagent = True
                auto_description = (
                    description.strip()
                    or f"Runtime-defined specialist for '{normalized_type}' tasks."
                )
                selected = SubAgentSpec(
                    name=normalized_type,
                    description=auto_description,
                    system_prompt=DEFAULT_SUBAGENT_PROMPT,
                )
                _persist_runtime_subagent_spec(
                    state=tool_context.state,
                    name=normalized_type,
                    description=auto_description,
                    system_prompt=DEFAULT_SUBAGENT_PROMPT,
                    model=None,
                    tool_names=None,
                )

            normalized_type = (
                _sanitize_agent_name(selected.name)
                if isinstance(selected, LlmAgent)
                else _sanitize_agent_name(selected.get("name", "general_purpose"))
            )

            if isinstance(selected, LlmAgent):
                if model and not task_config.allow_model_override:
                    return {
                        "status": "error",
                        "error": "Model override is disabled for dynamic task delegation",
                        "task_id": task_id,
                    }
                child_agent = selected
            else:
                try:
                    child_agent = _build_spec_agent(
                        selected,
                        default_model=default_model,
                        default_tools=default_tools,
                        skills_config=skills_config,
                        model_override=model,
                        config=task_config,
                    )
                except ValueError as exc:
                    return {
                        "status": "error",
                        "error": str(exc),
                        "task_id": task_id,
                        "created_subagent": created_subagent,
                    }

            runner = InMemoryRunner(agent=child_agent, app_name="dynamic_task")
            session = await runner.session_service.create_session(
                app_name="dynamic_task",
                user_id="dynamic_task_user",
                state={
                    "files": tool_context.state.get("files", {}),
                    "todos": tool_context.state.get("todos", []),
                    _TASK_DEPTH_KEY: current_depth + 1,
                },
            )
            runtime = _TaskRuntime(
                runner=runner,
                session_id=session.id,
                user_id="dynamic_task_user",
                subagent_type=normalized_type,
            )
            store[task_id] = {
                "subagent_type": normalized_type,
                "depth": current_depth + 1,
            }
            _RUNTIME_REGISTRY[f"{logical_parent_id}:{task_id}"] = runtime

            if runtime_backend_factory is not None:
                register_backend_factory(runtime.session_id, runtime_backend_factory)

        if runtime is None:
            return {"status": "error", "error": "Failed to initialize dynamic task runtime"}

        run_key = f"{logical_parent_id}:{task_id}"
        acquired, acquire_error, queue_wait_seconds = await _acquire_concurrency_slot(
            running_tasks=running_tasks,
            logical_parent_id=logical_parent_id,
            run_key=run_key,
            task_id=task_id,
            config=task_config,
        )
        if not acquired:
            return {
                "status": "error",
                "task_id": task_id,
                "subagent_type": normalized_type,
                "error": acquire_error or "Failed to acquire dynamic task concurrency slot",
                "created_subagent": created_subagent,
                **_queue_wait_metadata(queue_wait_seconds),
            }

        try:
            result = await _run_dynamic_task(
                runtime,
                prompt=resolved_prompt,
                timeout_seconds=task_config.timeout_seconds,
            )
        finally:
            await _release_concurrency_slot(
                running_tasks=running_tasks,
                logical_parent_id=logical_parent_id,
                run_key=run_key,
            )

        tool_context.state["files"] = result["files"]
        tool_context.state["todos"] = result["todos"]

        if result.get("timed_out"):
            return {
                "status": "error",
                "task_id": task_id,
                "subagent_type": normalized_type,
                "error": (f"Dynamic task timed out after {task_config.timeout_seconds} seconds"),
                "result": result["result"],
                "function_calls": result["function_calls"],
                "created_subagent": created_subagent,
                **_queue_wait_metadata(queue_wait_seconds),
            }

        error = result.get("error")
        if isinstance(error, str) and error:
            return {
                "status": "error",
                "task_id": task_id,
                "subagent_type": normalized_type,
                "error": f"Dynamic task failed: {error}",
                "result": result["result"],
                "function_calls": result["function_calls"],
                "created_subagent": created_subagent,
                **_queue_wait_metadata(queue_wait_seconds),
            }

        return {
            "status": "completed",
            "task_id": task_id,
            "subagent_type": normalized_type,
            "result": result["result"],
            "function_calls": result["function_calls"],
            "created_subagent": created_subagent,
            **_queue_wait_metadata(queue_wait_seconds),
        }

    task.__name__ = "task"
    return task
