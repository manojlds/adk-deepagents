"""Dynamic ``task`` tool for runtime sub-agent delegation.

Unlike static ``AgentTool`` wiring, this tool spawns or resumes sub-agent
sessions at runtime using ``task_id``.
"""

from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import ToolContext
from google.genai import types

from adk_deepagents.callbacks.before_tool import make_before_tool_callback
from adk_deepagents.prompts import DEFAULT_SUBAGENT_PROMPT
from adk_deepagents.tools.task import (
    GENERAL_PURPOSE_SUBAGENT,
    _resolve_skills_tools,
    _sanitize_agent_name,
)
from adk_deepagents.types import DynamicTaskConfig, SkillsConfig, SubAgentSpec

_TASK_STORE_KEY = "_dynamic_tasks"
_TASK_COUNTER_KEY = "_dynamic_task_counter"
_TASK_PARENT_ID_KEY = "_dynamic_parent_session_id"
_RUNTIME_REGISTRY: dict[str, _TaskRuntime] = {}


@dataclass
class _TaskRuntime:
    runner: InMemoryRunner
    session_id: str
    user_id: str
    subagent_type: str


def _run_coro_sync(coro: Any) -> Any:
    """Run a coroutine from sync code in both loop/no-loop contexts."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


def _normalize_subagent_type(subagent_type: str) -> str:
    normalized = _sanitize_agent_name(subagent_type)
    if normalized in {"general", "generalpurpose"}:
        return "general_purpose"
    return normalized


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

    await asyncio.wait_for(_collect(), timeout=timeout_seconds)

    session = await runtime.runner.session_service.get_session(
        app_name="dynamic_task",
        user_id=runtime.user_id,
        session_id=runtime.session_id,
    )

    session_state = session.state if session is not None else {}

    return {
        "result": "\n".join(texts).strip(),
        "function_calls": function_calls,
        "files": session_state.get("files", {}),
        "todos": session_state.get("todos", []),
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

    def task(
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
        normalized_type = _normalize_subagent_type(subagent_type)
        parent_session_id = tool_context.state.get(_TASK_PARENT_ID_KEY)
        if not isinstance(parent_session_id, str) or not parent_session_id:
            parent_session_id = uuid.uuid4().hex
            tool_context.state[_TASK_PARENT_ID_KEY] = parent_session_id

        if task_id:
            existing = store.get(task_id)
            if not isinstance(existing, dict):
                return {"status": "error", "error": f"Unknown task_id: {task_id}"}
            runtime = _RUNTIME_REGISTRY.get(f"{parent_session_id}:{task_id}")
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
            counter = int(tool_context.state.get(_TASK_COUNTER_KEY, 0)) + 1
            tool_context.state[_TASK_COUNTER_KEY] = counter
            task_id = f"task_{counter}"

            selected = registry.get(normalized_type) or registry["general_purpose"]
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
                    return {"status": "error", "error": str(exc), "task_id": task_id}

            runner = InMemoryRunner(agent=child_agent, app_name="dynamic_task")
            session = _run_coro_sync(
                runner.session_service.create_session(
                    app_name="dynamic_task",
                    user_id="dynamic_task_user",
                    state={
                        "files": tool_context.state.get("files", {}),
                        "todos": tool_context.state.get("todos", []),
                        "_backend_factory": tool_context.state.get("_backend_factory"),
                    },
                )
            )
            runtime = _TaskRuntime(
                runner=runner,
                session_id=session.id,
                user_id="dynamic_task_user",
                subagent_type=normalized_type,
            )
            store[task_id] = {"subagent_type": normalized_type}
            _RUNTIME_REGISTRY[f"{parent_session_id}:{task_id}"] = runtime

        if runtime is None:
            return {"status": "error", "error": "Failed to initialize dynamic task runtime"}

        result = _run_coro_sync(
            _run_dynamic_task(
                runtime,
                prompt=resolved_prompt,
                timeout_seconds=task_config.timeout_seconds,
            )
        )

        tool_context.state["files"] = result["files"]
        tool_context.state["todos"] = result["todos"]

        return {
            "status": "completed",
            "task_id": task_id,
            "subagent_type": normalized_type,
            "result": result["result"],
            "function_calls": result["function_calls"],
        }

    task.__name__ = "task"
    return task
