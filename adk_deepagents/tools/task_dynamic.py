"""Dynamic ``task`` tool for runtime sub-agent delegation.

Unlike static ``AgentTool`` wiring, this tool spawns or resumes sub-agent
sessions at runtime using ``task_id``.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import Any, cast

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.tool_context import ToolContext

from adk_deepagents.backends.protocol import BackendFactory
from adk_deepagents.backends.runtime import (
    get_registered_backend_factory,
    register_backend_factory,
)
from adk_deepagents.prompts import DEFAULT_SUBAGENT_PROMPT
from adk_deepagents.tools.task import _sanitize_agent_name
from adk_deepagents.tools.task_dynamic_execution import (
    _build_spec_agent,
    _run_dynamic_task,
    _run_dynamic_task_a2a,
    _run_dynamic_task_temporal,
)
from adk_deepagents.tools.task_dynamic_history import (
    _append_task_history_entry,
    _build_resume_prompt,
    _dynamic_task_tool_doc,
    _normalized_task_history,
)
from adk_deepagents.tools.task_dynamic_runtime import (
    _CONCURRENCY_LOCKS,
    _RUNNING_TASKS_KEY,
    _RUNTIME_REGISTRY,
    _TASK_COUNTER_KEY,
    _TASK_DEPTH_KEY,
    _TASK_PARENT_ID_KEY,
    _TASK_STORE_KEY,
    _acquire_concurrency_slot,
    _build_dynamic_registry,
    _build_tool_index,
    _coerce_subagent_spec_payload,
    _load_runtime_subagent_specs,
    _normalize_subagent_type,
    _persist_runtime_subagent_spec,
    _prune_stale_running_tasks,
    _queue_wait_metadata,
    _release_concurrency_slot,
    _resolve_runtime_tool_names,
    _runtime_subagent_spec_payload,
    _TaskRuntime,
)
from adk_deepagents.tools.task_dynamic_state import (
    _coerce_backend_factory,
    _coerce_files_state,
    _coerce_positive_int,
    _coerce_todos_state,
    _extract_temporal_backend_context,
)
from adk_deepagents.types import DynamicTaskConfig, SkillsConfig, SubAgentSpec

logger = logging.getLogger(__name__)

# Re-exports for monkeypatch compatibility and external consumers.
__all__ = [
    "create_dynamic_task_tool",
    "create_register_subagent_tool",
    "_build_resume_prompt",
    "_run_dynamic_task",
    "_run_dynamic_task_a2a",
    "_run_dynamic_task_temporal",
    "_RUNTIME_REGISTRY",
    "_CONCURRENCY_LOCKS",
    "_TaskRuntime",
]


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


def create_dynamic_task_tool(
    *,
    default_model: str | Any,
    default_tools: list,
    subagents: list[SubAgentSpec | LlmAgent] | None,
    skills_config: SkillsConfig | None = None,
    config: DynamicTaskConfig | None = None,
    before_agent_callback: Callable | None = None,
    before_model_callback: Callable | None = None,
    after_tool_callback: Callable | None = None,
    default_interrupt_on: dict[str, bool] | None = None,
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
        task_state: dict[str, Any] | None = None
        temporal_enabled = task_config.temporal is not None
        a2a_enabled = task_config.a2a is not None
        external_backend_enabled = temporal_enabled or a2a_enabled
        created_subagent = False
        recovered_runtime = False
        resume_with_history = False
        subagent_spec_payload: dict[str, Any] | None = None
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

        temporal_backend_context = _extract_temporal_backend_context(
            tool_context=tool_context,
            adk_parent_session_id=adk_parent_session_id,
            runtime_backend_factory=runtime_backend_factory,
        )

        current_depth_raw = tool_context.state.get(_TASK_DEPTH_KEY, 0)
        current_depth = current_depth_raw if isinstance(current_depth_raw, int) else 0

        running_tasks = tool_context.state.setdefault(_RUNNING_TASKS_KEY, [])
        if not isinstance(running_tasks, list):
            return {"status": "error", "error": "Invalid dynamic running task tracker in state"}

        _prune_stale_running_tasks(
            running_tasks=running_tasks,
            logical_parent_id=logical_parent_id,
        )

        runtime_registry = _load_runtime_subagent_specs(
            state=tool_context.state,
            tool_index=tool_index,
        )
        selected_registry: dict[str, SubAgentSpec | LlmAgent] = dict(registry)
        selected_registry.update(runtime_registry)

        existing: dict[str, Any] | None = None
        resume_existing_task = False
        if isinstance(task_id, str):
            normalized_task_id = task_id.strip()
            task_id = normalized_task_id or None

        if task_id is not None:
            candidate_existing = store.get(task_id)
            if isinstance(candidate_existing, dict):
                existing = candidate_existing
                resume_existing_task = True

        if resume_existing_task:
            assert existing is not None
            task_state = existing
            stored_type = existing.get("subagent_type")
            if isinstance(stored_type, str) and stored_type.strip():
                normalized_type = _normalize_subagent_type(stored_type)

            task_depth = _coerce_positive_int(existing.get("depth"), current_depth + 1)
            run_key = f"{logical_parent_id}:{task_id}"
            runtime = None if external_backend_enabled else _RUNTIME_REGISTRY.get(run_key)
            if runtime is None:
                selected = selected_registry.get(normalized_type)
                subagent_spec_payload = _coerce_subagent_spec_payload(existing.get("subagent_spec"))
                if subagent_spec_payload is None:
                    subagent_spec_payload = _runtime_subagent_spec_payload(
                        state=tool_context.state,
                        subagent_type=normalized_type,
                    )

                if selected is None and subagent_spec_payload is None:
                    return {
                        "status": "error",
                        "task_id": task_id,
                        "subagent_type": normalized_type,
                        "error": (
                            f"Cannot recover dynamic task {task_id}: unknown sub-agent type "
                            f"{normalized_type!r}."
                        ),
                    }

                model_override_raw = existing.get("model_override")
                model_override = (
                    model_override_raw.strip()
                    if isinstance(model_override_raw, str) and model_override_raw.strip()
                    else None
                )

                if "files" in existing and isinstance(existing.get("files"), dict):
                    task_files = cast(dict[str, Any], existing["files"])
                else:
                    task_files = _coerce_files_state(tool_context.state.get("files", {}))

                if "todos" in existing and isinstance(existing.get("todos"), list):
                    task_todos = cast(list[Any], existing["todos"])
                else:
                    task_todos = _coerce_todos_state(tool_context.state.get("todos", []))

                if model_override and not task_config.allow_model_override:
                    return {
                        "status": "error",
                        "error": "Model override is disabled for dynamic task delegation",
                        "task_id": task_id,
                    }

                if selected is not None:
                    normalized_type = (
                        _sanitize_agent_name(selected.name)
                        if isinstance(selected, LlmAgent)
                        else _sanitize_agent_name(selected.get("name", "general_purpose"))
                    )
                elif subagent_spec_payload is not None:
                    normalized_type = _normalize_subagent_type(subagent_spec_payload["name"])

                if not external_backend_enabled:
                    assert selected is not None
                    if isinstance(selected, LlmAgent):
                        child_agent = selected
                    else:
                        try:
                            child_agent = _build_spec_agent(
                                selected,
                                default_model=default_model,
                                default_tools=default_tools,
                                skills_config=skills_config,
                                model_override=model_override,
                                config=task_config,
                                before_agent_callback=before_agent_callback,
                                before_model_callback=before_model_callback,
                                after_tool_callback=after_tool_callback,
                                default_interrupt_on=default_interrupt_on,
                            )
                        except ValueError as exc:
                            return {
                                "status": "error",
                                "error": str(exc),
                                "task_id": task_id,
                                "subagent_type": normalized_type,
                            }

                    runner = InMemoryRunner(agent=child_agent, app_name="dynamic_task")
                    session = await runner.session_service.create_session(
                        app_name="dynamic_task",
                        user_id="dynamic_task_user",
                        state={
                            "files": task_files,
                            "todos": task_todos,
                            _TASK_DEPTH_KEY: task_depth,
                        },
                    )
                    runtime = _TaskRuntime(
                        runner=runner,
                        session_id=session.id,
                        user_id="dynamic_task_user",
                        subagent_type=normalized_type,
                    )
                    _RUNTIME_REGISTRY[run_key] = runtime

                    if runtime_backend_factory is not None:
                        register_backend_factory(runtime.session_id, runtime_backend_factory)

                    recovered_runtime = True

                existing["subagent_type"] = normalized_type
                existing["depth"] = task_depth
                existing["files"] = task_files
                existing["todos"] = task_todos
                if model_override is not None:
                    existing["model_override"] = model_override
                if subagent_spec_payload is not None:
                    existing["subagent_spec"] = subagent_spec_payload

                history = _normalized_task_history(existing.get("history"))
                existing["history"] = history
                resume_with_history = bool(history)

            if runtime is not None:
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

            if task_id is None:
                counter = int(tool_context.state.get(_TASK_COUNTER_KEY, 0)) + 1
                tool_context.state[_TASK_COUNTER_KEY] = counter
                task_id = f"task_{counter}"

            task_depth = current_depth + 1

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

            subagent_spec_payload = _runtime_subagent_spec_payload(
                state=tool_context.state,
                subagent_type=normalized_type,
            )

            model_override = model.strip() if isinstance(model, str) and model.strip() else None
            if model_override and not task_config.allow_model_override:
                return {
                    "status": "error",
                    "error": "Model override is disabled for dynamic task delegation",
                    "task_id": task_id,
                }

            if not external_backend_enabled:
                if isinstance(selected, LlmAgent):
                    child_agent = selected
                else:
                    try:
                        child_agent = _build_spec_agent(
                            selected,
                            default_model=default_model,
                            default_tools=default_tools,
                            skills_config=skills_config,
                            model_override=model_override,
                            config=task_config,
                            before_agent_callback=before_agent_callback,
                            before_model_callback=before_model_callback,
                            after_tool_callback=after_tool_callback,
                            default_interrupt_on=default_interrupt_on,
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
                        "files": _coerce_files_state(tool_context.state.get("files", {})),
                        "todos": _coerce_todos_state(tool_context.state.get("todos", [])),
                        _TASK_DEPTH_KEY: task_depth,
                    },
                )
                runtime = _TaskRuntime(
                    runner=runner,
                    session_id=session.id,
                    user_id="dynamic_task_user",
                    subagent_type=normalized_type,
                )
                _RUNTIME_REGISTRY[f"{logical_parent_id}:{task_id}"] = runtime

                if runtime_backend_factory is not None:
                    register_backend_factory(runtime.session_id, runtime_backend_factory)

            task_state = {
                "subagent_type": normalized_type,
                "depth": task_depth,
                "files": _coerce_files_state(tool_context.state.get("files", {})),
                "todos": _coerce_todos_state(tool_context.state.get("todos", [])),
                "history": [],
            }
            if model_override is not None:
                task_state["model_override"] = model_override
            if subagent_spec_payload is not None:
                task_state["subagent_spec"] = subagent_spec_payload
            store[task_id] = task_state

        if not external_backend_enabled and runtime is None:
            return {"status": "error", "error": "Failed to initialize dynamic task runtime"}

        if task_id is None:
            return {"status": "error", "error": "Failed to initialize dynamic task id"}

        if task_state is None:
            existing_state = store.get(task_id)
            if isinstance(existing_state, dict):
                task_state = existing_state
            else:
                task_state = {
                    "subagent_type": normalized_type,
                    "depth": _coerce_positive_int(current_depth + 1, 1),
                    "history": [],
                }
                store[task_id] = task_state

        subagent_spec_payload = _coerce_subagent_spec_payload(task_state.get("subagent_spec"))
        if subagent_spec_payload is None:
            subagent_spec_payload = _runtime_subagent_spec_payload(
                state=tool_context.state,
                subagent_type=normalized_type,
            )
            if subagent_spec_payload is not None:
                task_state["subagent_spec"] = subagent_spec_payload

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

        task_prompt = resolved_prompt
        if resume_with_history:
            history = _normalized_task_history(task_state.get("history"))
            task_prompt = _build_resume_prompt(history=history, prompt=resolved_prompt)

        try:
            if temporal_enabled:
                result = await _run_dynamic_task_temporal(
                    snapshot_data={
                        "subagent_type": normalized_type,
                        "prompt": task_prompt,
                        "depth": _coerce_positive_int(task_state.get("depth"), current_depth + 1),
                        "files": _coerce_files_state(task_state.get("files")),
                        "todos": _coerce_todos_state(task_state.get("todos")),
                        "history": _normalized_task_history(task_state.get("history")),
                        "model_override": task_state.get("model_override"),
                        "subagent_spec": subagent_spec_payload,
                        "timeout_seconds": task_config.timeout_seconds,
                        "backend_context": temporal_backend_context,
                    },
                    logical_parent_id=logical_parent_id,
                    task_id=task_id,
                    task_config=task_config,
                )
            elif a2a_enabled:
                result = await _run_dynamic_task_a2a(
                    prompt=task_prompt,
                    task_id=task_id,
                    subagent_type=normalized_type,
                    task_config=task_config,
                )
            else:
                assert runtime is not None
                result = await _run_dynamic_task(
                    runtime,
                    prompt=task_prompt,
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
        task_state["files"] = _coerce_files_state(result.get("files"))
        task_state["todos"] = _coerce_todos_state(result.get("todos"))
        task_state["subagent_type"] = normalized_type
        task_state["depth"] = _coerce_positive_int(task_state.get("depth"), current_depth + 1)

        result_text = result.get("result")
        _append_task_history_entry(
            task_state=task_state,
            prompt=resolved_prompt,
            result=result_text if isinstance(result_text, str) else "",
        )

        if result.get("timed_out"):
            return {
                "status": "error",
                "task_id": task_id,
                "subagent_type": normalized_type,
                "error": (f"Dynamic task timed out after {task_config.timeout_seconds} seconds"),
                "result": result["result"],
                "function_calls": result["function_calls"],
                "created_subagent": created_subagent,
                "recovered_runtime": recovered_runtime,
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
                "recovered_runtime": recovered_runtime,
                **_queue_wait_metadata(queue_wait_seconds),
            }

        return {
            "status": "completed",
            "task_id": task_id,
            "subagent_type": normalized_type,
            "result": result["result"],
            "function_calls": result["function_calls"],
            "created_subagent": created_subagent,
            "recovered_runtime": recovered_runtime,
            **_queue_wait_metadata(queue_wait_seconds),
        }

    task.__name__ = "task"
    task.__doc__ = _dynamic_task_tool_doc(task_config)
    return task
