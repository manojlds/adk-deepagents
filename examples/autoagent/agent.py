"""Autoagent example — deepagent harness for Harbor benchmarking.

This is the meta-agent's edit surface. The editable section below controls
what the agent does. The fixed boundary (Harbor adapter, ATIF, ADK runner)
lives in adk_deepagents.harbor and should not be modified unless the human
explicitly asks.

Usage:
    harbor run --agent-import-path agent:AutoAgent ...
"""

from __future__ import annotations

from pathlib import Path

from harbor.environments.base import BaseEnvironment

from adk_deepagents import create_deep_agent
from adk_deepagents.harbor import HarborAdapter, HarborBackend, create_harbor_execute_tool

try:
    from adk_skills_agent import SkillsRegistry

    _SKILLS_AVAILABLE = True
except ImportError:
    _SKILLS_AVAILABLE = False


# ============================================================================
# EDITABLE HARNESS — prompt, skills, tools, agent construction
# ============================================================================

SYSTEM_PROMPT = "You are an agent that executes tasks autonomously in a sandboxed environment."
MODEL = "gemini-2.0-flash"
SKILLS_DIRS: list[str] = ["./skills"]


def _build_instruction() -> str:
    if not _SKILLS_AVAILABLE:
        return SYSTEM_PROMPT
    registry = SkillsRegistry()
    dirs = [str(Path(__file__).parent / d) for d in SKILLS_DIRS]
    discovered = registry.discover(dirs)
    if discovered > 0:
        return registry.inject_skills_prompt(SYSTEM_PROMPT, format="xml")
    return SYSTEM_PROMPT


def create_agent(environment: BaseEnvironment):
    """Build the deepagent. Modify to add skills, sub-agents, browser, etc."""
    return create_deep_agent(
        name="adk-autoagent",
        model=MODEL,
        instruction=_build_instruction(),
        backend=HarborBackend(environment),
        tools=[create_harbor_execute_tool(environment)],
        execution=None,  # HarborBackend handles execution via execute tool
        browser=None,  # enable with BrowserConfig when needed
    )


# ============================================================================
# Harbor entry point — do not modify
# ============================================================================

AutoAgent = HarborAdapter.create(create_agent, agent_name="adk-autoagent", model=MODEL)
