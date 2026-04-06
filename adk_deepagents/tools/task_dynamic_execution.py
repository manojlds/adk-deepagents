"""Agent building and task execution for dynamic tasks."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from google.adk.agents import LlmAgent
from google.genai import types

from adk_deepagents.callbacks.before_tool import make_before_tool_callback
from adk_deepagents.prompts import DEFAULT_SUBAGENT_PROMPT
from adk_deepagents.tools.task import _resolve_skills_tools, _sanitize_agent_name
from adk_deepagents.tools.task_dynamic_runtime import _TaskRuntime
from adk_deepagents.types import DynamicTaskConfig, SkillsConfig, SubAgentSpec

logger = logging.getLogger(__name__)


def _build_spec_agent(
    spec: SubAgentSpec,
    *,
    default_model: str | Any,
    default_tools: list,
    skills_config: SkillsConfig | None,
    model_override: str | None,
    config: DynamicTaskConfig,
    before_agent_callback: Callable | None,
    before_model_callback: Callable | None,
    after_tool_callback: Callable | None,
    default_interrupt_on: dict[str, bool] | None,
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

    before_tool_cb = make_before_tool_callback(
        interrupt_on=spec.get("interrupt_on", default_interrupt_on)
    )

    if model_override and not config.allow_model_override:
        raise ValueError("Model override is disabled for dynamic task delegation")
    resolved_model = model_override or spec.get("model", default_model)

    return LlmAgent(
        name=_sanitize_agent_name(spec_name),
        model=resolved_model,
        instruction=spec.get("system_prompt", DEFAULT_SUBAGENT_PROMPT),
        description=spec_description,
        tools=sub_tools,
        before_agent_callback=before_agent_callback,
        before_model_callback=before_model_callback,
        after_tool_callback=after_tool_callback,
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


async def _run_dynamic_task_temporal(
    snapshot_data: dict[str, Any],
    *,
    logical_parent_id: str,
    task_id: str,
    task_config: DynamicTaskConfig,
) -> dict[str, Any]:
    """Dispatch a dynamic task turn to Temporal."""
    try:
        from adk_deepagents.temporal.activities import TaskSnapshot
        from adk_deepagents.temporal.client import run_task_via_temporal
    except ImportError:
        raise ImportError(
            "Temporal support requires the 'temporalio' package. "
            "Install it with: pip install adk-deepagents[temporal]"
        ) from None

    snapshot = TaskSnapshot.from_dict(snapshot_data)
    return await run_task_via_temporal(
        snapshot=snapshot,
        logical_parent_id=logical_parent_id,
        task_id=task_id,
        task_config=task_config,
    )


async def _run_dynamic_task_a2a(
    *,
    prompt: str,
    task_id: str,
    subagent_type: str,
    task_config: DynamicTaskConfig,
) -> dict[str, Any]:
    """Dispatch a dynamic task turn to an external A2A agent."""
    a2a_config = task_config.a2a
    if a2a_config is None:
        return {
            "result": "",
            "function_calls": [],
            "files": {},
            "todos": [],
            "timed_out": False,
            "error": "A2A config is None",
        }

    try:
        from a2a.client.client import ClientConfig
        from a2a.client.client_factory import ClientFactory
        from a2a.types.a2a_pb2 import Message, Part, Role
    except ImportError:
        raise ImportError(
            "A2A support requires the 'a2a-sdk' package. "
            "Install it with: pip install adk-deepagents[a2a]"
        ) from None

    message_id = f"{task_id}:{subagent_type}"
    request = Message(
        role=Role.ROLE_USER,
        parts=[Part(text=prompt)],
        message_id=message_id,
        context_id=task_id,
    )

    client = await ClientFactory.connect(
        a2a_config.agent_url,
        client_config=ClientConfig(streaming=True),
    )

    latest_task: Any | None = None
    response_text_parts: list[str] = []
    timed_out = False
    error: str | None = None

    def _collect_task_text(task_obj: Any) -> None:
        artifacts = getattr(task_obj, "artifacts", None)
        if not isinstance(artifacts, list):
            return
        for artifact in artifacts:
            parts = getattr(artifact, "parts", None)
            if not isinstance(parts, list):
                continue
            for part in parts:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    response_text_parts.append(text)

    async def _collect() -> None:
        nonlocal latest_task
        async for event in client.send_message(request):
            if isinstance(event, tuple) and len(event) == 2:
                task_obj, _update = event
                latest_task = task_obj
                _collect_task_text(task_obj)
            else:
                text = getattr(event, "text", None)
                if isinstance(text, str) and text:
                    response_text_parts.append(text)

    try:
        await asyncio.wait_for(_collect(), timeout=a2a_config.timeout_seconds)
    except TimeoutError:
        timed_out = True
    except Exception as exc:  # pragma: no cover - defensive path
        logger.exception("Dynamic task A2A run failed")
        error = f"{type(exc).__name__}: {exc}"
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            await close()

    result_text = "\n".join(response_text_parts).strip()
    if not result_text and latest_task is not None:
        status = getattr(latest_task, "status", None)
        message = getattr(status, "message", None) if status is not None else None
        parts = getattr(message, "parts", None) if message is not None else None
        if isinstance(parts, list):
            fallback_parts = [
                part.text
                for part in parts
                if isinstance(getattr(part, "text", None), str) and part.text
            ]
            result_text = "\n".join(fallback_parts).strip()

    return {
        "result": result_text,
        "function_calls": [],
        "files": {},
        "todos": [],
        "timed_out": timed_out,
        "error": error,
    }
