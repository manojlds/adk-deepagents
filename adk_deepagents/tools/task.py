"""Sub-agent builder — creates AgentTool instances from SubAgentSpec.

In deepagents, there's a single ``task`` tool that routes by ``subagent_type``.
In ADK, each sub-agent becomes its own ``AgentTool``. We support both patterns.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool

from adk_deepagents.callbacks.before_tool import make_before_tool_callback
from adk_deepagents.prompts import (
    DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    DEFAULT_SUBAGENT_PROMPT,
)
from adk_deepagents.types import SkillsConfig, SubAgentSpec


def _sanitize_agent_name(name: str) -> str:
    """Sanitize an agent name to be a valid Python identifier.

    ADK requires agent names to match ``[a-zA-Z_][a-zA-Z0-9_]*``.
    """
    sanitized = name.replace("-", "_").replace(" ", "_")
    # Strip any remaining invalid chars
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized or "agent"


# ---------------------------------------------------------------------------
# Default sub-agents
# ---------------------------------------------------------------------------

GENERAL_PURPOSE_SUBAGENT: SubAgentSpec = SubAgentSpec(
    name="general_purpose",
    description=DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    system_prompt=DEFAULT_SUBAGENT_PROMPT,
)

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _resolve_skills_tools(
    skills_dirs: list[str],
    skills_config: SkillsConfig | None = None,
) -> list[Callable]:
    """Resolve skills tools for a sub-agent.

    Raises ``ImportError`` if adk-skills-agent is not installed.
    """
    try:
        import adk_skills_agent  # noqa: F401
    except ImportError:
        raise ImportError(
            "adk-skills-agent is required for skills support. "
            "Install it with: pip install adk-skills-agent"
        ) from None

    from adk_deepagents.skills.integration import add_skills_tools

    skills_tools: list[Callable] = []
    return add_skills_tools(
        skills_tools,
        skills_dirs=skills_dirs,
        skills_config=skills_config,
    )


def build_subagent_tools(
    subagents: list[SubAgentSpec | LlmAgent],
    default_model: str,
    default_tools: list[Callable],
    *,
    include_general_purpose: bool = True,
    skills_config: SkillsConfig | None = None,
) -> list[AgentTool]:
    """Build ``AgentTool`` instances from sub-agent specifications.

    Parameters
    ----------
    subagents:
        User-provided sub-agent specs or pre-built ``LlmAgent`` instances.
    default_model:
        Model string to use when the spec doesn't specify one.
    default_tools:
        Default tool functions to give sub-agents that don't specify their own.
    include_general_purpose:
        If ``True`` (default), prepend the general-purpose sub-agent
        unless one already exists in *subagents*.
    skills_config:
        Optional skills configuration for sub-agents that have ``skills`` set.

    Returns
    -------
    list[AgentTool]
        One ``AgentTool`` per sub-agent, ready to add to the parent agent's tools.
    """
    # Separate pre-built agents from specs
    pre_built: list[LlmAgent] = []
    specs_input: list[SubAgentSpec] = []
    for item in subagents:
        if isinstance(item, LlmAgent):
            pre_built.append(item)
        else:
            specs_input.append(item)

    specs: list[SubAgentSpec] = []

    if include_general_purpose:
        all_names = {_sanitize_agent_name(s["name"]) for s in specs_input}
        all_names.update(a.name for a in pre_built)
        if "general_purpose" not in all_names:
            specs.append(GENERAL_PURPOSE_SUBAGENT)

    specs.extend(specs_input)

    tools: list[AgentTool] = []

    # Add pre-built agents first
    for agent in pre_built:
        tools.append(AgentTool(agent=agent))

    # Build agents from specs
    for spec in specs:
        sub_tools: list[Any] = list(spec.get("tools", default_tools))

        # Add skills tools if specified
        sub_skills = spec.get("skills")
        if sub_skills:
            skill_tools = _resolve_skills_tools(sub_skills, skills_config)
            sub_tools.extend(skill_tools)

        # Build before_tool_callback for HITL if specified
        sub_interrupt_on = spec.get("interrupt_on")
        before_tool_cb = make_before_tool_callback(interrupt_on=sub_interrupt_on)

        sub_agent = LlmAgent(
            name=_sanitize_agent_name(spec["name"]),
            model=spec.get("model", default_model),
            instruction=spec.get("system_prompt", DEFAULT_SUBAGENT_PROMPT),
            description=spec["description"],
            tools=sub_tools,
            before_tool_callback=before_tool_cb,
        )
        tools.append(AgentTool(agent=sub_agent))

    return tools
