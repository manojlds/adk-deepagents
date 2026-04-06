"""Agent building and task execution for dynamic tasks."""

from __future__ import annotations

import asyncio
import json
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

_A2A_DYNAMIC_TASK_RESULT_SCHEMA_PREFIX = "adk_deepagents.dynamic_task_result."


def _a2a_is_dynamic_task_schema(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().startswith(_A2A_DYNAMIC_TASK_RESULT_SCHEMA_PREFIX)


def _normalize_function_call_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    names: list[str] = []
    for item in value:
        if isinstance(item, str) and item and item not in names:
            names.append(item)
    return names


def _parse_dynamic_task_result_payload(value: Any) -> dict[str, Any] | None:
    parsed: dict[str, Any] | None = None

    if isinstance(value, dict):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text.startswith("{"):
            return None
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(loaded, dict):
            return None
        parsed = loaded

    if parsed is None:
        return None

    schema = parsed.get("schema")
    payload = parsed.get("payload")

    candidate = payload if isinstance(payload, dict) else parsed
    if not isinstance(candidate, dict):
        return None

    if not _a2a_is_dynamic_task_schema(schema) and not any(
        key in candidate for key in ("result", "function_calls", "files", "todos", "error")
    ):
        return None

    normalized: dict[str, Any] = {}

    if "result" in candidate:
        result_raw = candidate.get("result")
        normalized["result"] = result_raw if isinstance(result_raw, str) else ""

    if "function_calls" in candidate:
        normalized["function_calls"] = _normalize_function_call_names(
            candidate.get("function_calls")
        )

    if "files" in candidate:
        files_raw = candidate.get("files")
        normalized["files"] = files_raw if isinstance(files_raw, dict) else {}

    if "todos" in candidate:
        todos_raw = candidate.get("todos")
        normalized["todos"] = todos_raw if isinstance(todos_raw, list) else []

    if "error" in candidate:
        error_raw = candidate.get("error")
        normalized["error"] = error_raw if isinstance(error_raw, str) else ""

    return normalized or None


def _new_structured_result_state() -> dict[str, Any]:
    return {
        "has_result": False,
        "result": "",
        "has_function_calls": False,
        "function_calls": [],
        "has_files": False,
        "files": {},
        "has_todos": False,
        "todos": [],
        "has_error": False,
        "error": "",
    }


def _merge_structured_result_payload(*, state: dict[str, Any], payload: dict[str, Any]) -> None:
    if "result" in payload:
        state["has_result"] = True
        state["result"] = payload.get("result", "")

    if "function_calls" in payload:
        state["has_function_calls"] = True
        existing_calls = state.get("function_calls")
        if not isinstance(existing_calls, list):
            existing_calls = []
            state["function_calls"] = existing_calls

        for name in _normalize_function_call_names(payload.get("function_calls")):
            if name not in existing_calls:
                existing_calls.append(name)

    if "files" in payload:
        state["has_files"] = True
        files_value = payload.get("files")
        state["files"] = files_value if isinstance(files_value, dict) else {}

    if "todos" in payload:
        state["has_todos"] = True
        todos_value = payload.get("todos")
        state["todos"] = todos_value if isinstance(todos_value, list) else []

    if "error" in payload:
        state["has_error"] = True
        error_value = payload.get("error")
        state["error"] = error_value if isinstance(error_value, str) else ""


def _append_text_if_new(*, parts: list[str], text: Any) -> None:
    if not isinstance(text, str) or not text:
        return

    if not parts or parts[-1] != text:
        parts.append(text)


def _consume_a2a_part(
    *,
    part: Any,
    response_text_parts: list[str],
    structured_state: dict[str, Any],
) -> None:
    root = part.get("root") if isinstance(part, dict) else getattr(part, "root", None)
    candidate = root if root is not None else part

    part_data = (
        candidate.get("data") if isinstance(candidate, dict) else getattr(candidate, "data", None)
    )
    payload = _parse_dynamic_task_result_payload(part_data)
    if payload is not None:
        _merge_structured_result_payload(state=structured_state, payload=payload)
        return

    part_text = (
        candidate.get("text") if isinstance(candidate, dict) else getattr(candidate, "text", None)
    )
    payload = _parse_dynamic_task_result_payload(part_text)
    if payload is not None:
        _merge_structured_result_payload(state=structured_state, payload=payload)
        return

    _append_text_if_new(parts=response_text_parts, text=part_text)


def _consume_a2a_object(
    *,
    value: Any,
    response_text_parts: list[str],
    structured_state: dict[str, Any],
    visited: set[int] | None = None,
) -> None:
    if value is None:
        return

    if visited is None:
        visited = set()

    value_id = id(value)
    if value_id in visited:
        return
    visited.add(value_id)

    payload = _parse_dynamic_task_result_payload(value)
    if payload is not None:
        _merge_structured_result_payload(state=structured_state, payload=payload)
        return

    if isinstance(value, dict):
        data_value = value.get("data")
        payload = _parse_dynamic_task_result_payload(data_value)
        if payload is not None:
            _merge_structured_result_payload(state=structured_state, payload=payload)

        parts = value.get("parts")
        if isinstance(parts, list):
            for part in parts:
                _consume_a2a_part(
                    part=part,
                    response_text_parts=response_text_parts,
                    structured_state=structured_state,
                )

        artifacts = value.get("artifacts")
        if isinstance(artifacts, list):
            for artifact in artifacts:
                _consume_a2a_object(
                    value=artifact,
                    response_text_parts=response_text_parts,
                    structured_state=structured_state,
                    visited=visited,
                )

        artifact = value.get("artifact")
        _consume_a2a_object(
            value=artifact,
            response_text_parts=response_text_parts,
            structured_state=structured_state,
            visited=visited,
        )

        message = value.get("message")
        _consume_a2a_object(
            value=message,
            response_text_parts=response_text_parts,
            structured_state=structured_state,
            visited=visited,
        )

        status = value.get("status")
        _consume_a2a_object(
            value=status,
            response_text_parts=response_text_parts,
            structured_state=structured_state,
            visited=visited,
        )

        text = value.get("text")
        payload = _parse_dynamic_task_result_payload(text)
        if payload is not None:
            _merge_structured_result_payload(state=structured_state, payload=payload)
            return
        _append_text_if_new(parts=response_text_parts, text=text)
        return

    parts = getattr(value, "parts", None)
    data_value = getattr(value, "data", None)
    payload = _parse_dynamic_task_result_payload(data_value)
    if payload is not None:
        _merge_structured_result_payload(state=structured_state, payload=payload)

    if isinstance(parts, list):
        for part in parts:
            _consume_a2a_part(
                part=part,
                response_text_parts=response_text_parts,
                structured_state=structured_state,
            )

    artifacts = getattr(value, "artifacts", None)
    if isinstance(artifacts, list):
        for artifact in artifacts:
            _consume_a2a_object(
                value=artifact,
                response_text_parts=response_text_parts,
                structured_state=structured_state,
                visited=visited,
            )

    artifact = getattr(value, "artifact", None)
    _consume_a2a_object(
        value=artifact,
        response_text_parts=response_text_parts,
        structured_state=structured_state,
        visited=visited,
    )

    message = getattr(value, "message", None)
    _consume_a2a_object(
        value=message,
        response_text_parts=response_text_parts,
        structured_state=structured_state,
        visited=visited,
    )

    status = getattr(value, "status", None)
    _consume_a2a_object(
        value=status,
        response_text_parts=response_text_parts,
        structured_state=structured_state,
        visited=visited,
    )

    text = getattr(value, "text", None)
    payload = _parse_dynamic_task_result_payload(text)
    if payload is not None:
        _merge_structured_result_payload(state=structured_state, payload=payload)
        return
    _append_text_if_new(parts=response_text_parts, text=text)


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
        from a2a.types import Message, Part, Role, TextPart
    except ImportError:
        raise ImportError(
            "A2A support requires the 'a2a-sdk' package. "
            "Install it with: pip install adk-deepagents[a2a]"
        ) from None

    message_id = f"{task_id}:{subagent_type}"
    request = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text=prompt))],
        message_id=message_id,
        context_id=task_id,
    )

    client = await ClientFactory.connect(
        a2a_config.agent_url,
        client_config=ClientConfig(streaming=True),
    )

    latest_task: Any | None = None
    response_text_parts: list[str] = []
    structured_state = _new_structured_result_state()
    timed_out = False
    error: str | None = None

    def _collect_task_text(task_obj: Any) -> None:
        _consume_a2a_object(
            value=task_obj,
            response_text_parts=response_text_parts,
            structured_state=structured_state,
        )

    async def _collect() -> None:
        nonlocal latest_task
        async for event in client.send_message(request):
            if isinstance(event, tuple) and len(event) == 2:
                task_obj, _update = event
                latest_task = task_obj
                _collect_task_text(task_obj)
            else:
                _consume_a2a_object(
                    value=event,
                    response_text_parts=response_text_parts,
                    structured_state=structured_state,
                )

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
    if structured_state.get("has_result"):
        result_text = structured_state.get("result", "")

    if not result_text and latest_task is not None:
        _consume_a2a_object(
            value=latest_task,
            response_text_parts=response_text_parts,
            structured_state=structured_state,
        )
        result_text = "\n".join(response_text_parts).strip()
        if structured_state.get("has_result"):
            result_text = structured_state.get("result", "")

    function_calls = (
        structured_state.get("function_calls") if structured_state.get("has_function_calls") else []
    )
    if not isinstance(function_calls, list):
        function_calls = []

    files = structured_state.get("files") if structured_state.get("has_files") else {}
    if not isinstance(files, dict):
        files = {}

    todos = structured_state.get("todos") if structured_state.get("has_todos") else []
    if not isinstance(todos, list):
        todos = []

    if not error and structured_state.get("has_error"):
        structured_error = structured_state.get("error")
        if isinstance(structured_error, str) and structured_error:
            error = structured_error

    return {
        "result": result_text,
        "function_calls": function_calls,
        "files": files,
        "todos": todos,
        "timed_out": timed_out,
        "error": error,
    }
