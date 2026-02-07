"""Sub-agent builder — creates AgentTool instances from SubAgentSpec.

In deepagents, there's a single ``task`` tool that routes by ``subagent_type``.
In ADK, each sub-agent becomes its own ``AgentTool``. We support both patterns.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool

from adk_deepagents.prompts import (
    DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    DEFAULT_SUBAGENT_PROMPT,
)
from adk_deepagents.types import SubAgentSpec

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


def build_subagent_tools(
    subagents: list[SubAgentSpec],
    default_model: str,
    default_tools: list[Callable],
    *,
    include_general_purpose: bool = True,
) -> list[AgentTool]:
    """Build ``AgentTool`` instances from sub-agent specifications.

    Parameters
    ----------
    subagents:
        User-provided sub-agent specs.
    default_model:
        Model string to use when the spec doesn't specify one.
    default_tools:
        Default tool functions to give sub-agents that don't specify their own.
    include_general_purpose:
        If ``True`` (default), prepend the general-purpose sub-agent
        unless one already exists in *subagents*.

    Returns
    -------
    list[AgentTool]
        One ``AgentTool`` per sub-agent, ready to add to the parent agent's tools.
    """
    specs: list[SubAgentSpec] = []

    if include_general_purpose:
        has_gp = any(
            _sanitize_agent_name(s["name"]) == "general_purpose" for s in subagents
        )
        if not has_gp:
            specs.append(GENERAL_PURPOSE_SUBAGENT)

    specs.extend(subagents)

    tools: list[AgentTool] = []
    for spec in specs:
        sub_agent = LlmAgent(
            name=_sanitize_agent_name(spec["name"]),
            model=spec.get("model", default_model),
            instruction=spec.get("system_prompt", DEFAULT_SUBAGENT_PROMPT),
            description=spec["description"],
            tools=spec.get("tools", default_tools),
        )
        tools.append(AgentTool(agent=sub_agent))

    return tools
