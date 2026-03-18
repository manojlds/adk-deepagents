"""Temporal worker factory for dynamic task execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from adk_deepagents.types import DynamicTaskConfig, SkillsConfig, SubAgentSpec

if TYPE_CHECKING:
    from google.adk.agents import LlmAgent


async def create_temporal_worker(
    *,
    default_model: str | Any = "gemini-2.5-flash",
    default_tools: Sequence[Callable] | None = None,
    subagents: list[SubAgentSpec | LlmAgent] | None = None,
    skills_config: SkillsConfig | None = None,
    dynamic_task_config: DynamicTaskConfig | None = None,
    before_agent_callback: Callable | None = None,
    before_model_callback: Callable | None = None,
    after_tool_callback: Callable | None = None,
    default_interrupt_on: dict[str, bool] | None = None,
) -> Any:
    """Create a Temporal worker aligned with dynamic task sub-agent config."""
    from temporalio.client import Client
    from temporalio.worker import UnsandboxedWorkflowRunner, Worker

    from adk_deepagents.temporal.activities import create_run_task_activity
    from adk_deepagents.temporal.workflows import DynamicTaskWorkflow, configure_workflow

    config = dynamic_task_config or DynamicTaskConfig()
    temporal_config = config.temporal
    if temporal_config is None:
        raise ValueError("dynamic_task_config.temporal must be set")

    from google.adk.agents import LlmAgent

    from adk_deepagents.callbacks.before_tool import make_before_tool_callback
    from adk_deepagents.prompts import DEFAULT_SUBAGENT_PROMPT
    from adk_deepagents.tools.task import (
        GENERAL_PURPOSE_SUBAGENT,
        _resolve_skills_tools,
        _sanitize_agent_name,
    )

    registry: dict[str, SubAgentSpec | LlmAgent] = {
        "general_purpose": GENERAL_PURPOSE_SUBAGENT,
    }
    if subagents:
        for entry in subagents:
            if isinstance(entry, LlmAgent):
                registry[_sanitize_agent_name(entry.name)] = entry
                continue

            name = entry.get("name")
            if isinstance(name, str) and name:
                registry[_sanitize_agent_name(name)] = entry

    resolved_tools = list(default_tools or [])
    tool_index: dict[str, Any] = {}
    for tool in resolved_tools:
        raw_name = getattr(tool, "__name__", getattr(tool, "name", None))
        if not isinstance(raw_name, str):
            continue
        name = raw_name.strip()
        if name and name not in tool_index:
            tool_index[name] = tool

    def _spec_from_payload(
        payload: dict[str, Any] | None,
        *,
        subagent_type: str,
    ) -> SubAgentSpec | None:
        if not isinstance(payload, dict):
            return None

        raw_name = payload.get("name", subagent_type)
        raw_description = payload.get("description")
        if not isinstance(raw_name, str) or not raw_name.strip():
            return None
        if not isinstance(raw_description, str) or not raw_description.strip():
            return None

        spec: SubAgentSpec = SubAgentSpec(
            name=_sanitize_agent_name(raw_name),
            description=raw_description.strip(),
        )

        raw_system_prompt = payload.get("system_prompt")
        if isinstance(raw_system_prompt, str) and raw_system_prompt.strip():
            spec["system_prompt"] = raw_system_prompt.strip()

        raw_model = payload.get("model")
        if isinstance(raw_model, str) and raw_model.strip():
            spec["model"] = raw_model.strip()

        raw_tool_names = payload.get("tool_names")
        if isinstance(raw_tool_names, list):
            spec["tools"] = [
                tool_index[name]
                for name in raw_tool_names
                if isinstance(name, str) and name in tool_index
            ]

        raw_skills = payload.get("skills")
        if isinstance(raw_skills, list):
            spec["skills"] = [s for s in raw_skills if isinstance(s, str) and s.strip()]

        return spec

    def agent_builder(
        subagent_type: str,
        model_override: str | None,
        subagent_spec: dict[str, Any] | None = None,
    ) -> LlmAgent:
        selected = _spec_from_payload(subagent_spec, subagent_type=subagent_type)
        if selected is None:
            selected = registry.get(subagent_type, GENERAL_PURPOSE_SUBAGENT)

        if isinstance(selected, LlmAgent):
            return selected

        sub_tools: list[Any] = list(selected.get("tools", resolved_tools))
        sub_skills = selected.get("skills")
        if sub_skills:
            sub_tools.extend(_resolve_skills_tools(sub_skills, skills_config))

        before_tool_cb = make_before_tool_callback(
            interrupt_on=selected.get("interrupt_on", default_interrupt_on)
        )

        spec_name = str(selected.get("name", subagent_type))
        spec_description = str(selected.get("description", subagent_type))
        resolved_model = model_override or selected.get("model", default_model)

        return LlmAgent(
            name=_sanitize_agent_name(spec_name),
            model=resolved_model,
            instruction=selected.get("system_prompt", DEFAULT_SUBAGENT_PROMPT),
            description=spec_description,
            tools=sub_tools,
            before_agent_callback=before_agent_callback,
            before_model_callback=before_model_callback,
            after_tool_callback=after_tool_callback,
            before_tool_callback=before_tool_cb,
        )

    configure_workflow(
        activity_timeout_seconds=temporal_config.activity_timeout_seconds or config.timeout_seconds,
        retry_max_attempts=temporal_config.retry_max_attempts,
        idle_timeout_seconds=temporal_config.idle_timeout_seconds,
    )

    client = await Client.connect(
        temporal_config.target_host,
        namespace=temporal_config.namespace,
    )

    return Worker(
        client,
        task_queue=temporal_config.task_queue,
        workflows=[DynamicTaskWorkflow],
        activities=[create_run_task_activity(agent_builder=agent_builder)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    )
