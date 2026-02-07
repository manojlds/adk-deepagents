"""Skills integration — wires adk-skills into create_deep_agent().

Placeholder for Phase 4 implementation. Will integrate adk-skills
SkillsRegistry for SKILL.md discovery, progressive disclosure, and
script execution.
"""

from __future__ import annotations

from typing import Callable

from adk_deepagents.types import SkillsConfig


def add_skills_tools(
    tools: list[Callable],
    skills_dirs: list[str],
    skills_config: SkillsConfig | None = None,
) -> list[Callable]:
    """Add adk-skills tools to the tool list.

    Parameters
    ----------
    tools:
        Existing tool list to extend.
    skills_dirs:
        Directories to discover skills from.
    skills_config:
        Optional skills configuration.

    Returns
    -------
    list[Callable]
        The extended tool list.
    """
    try:
        from adk_skills_agent import SkillsRegistry

        registry = SkillsRegistry()
        for d in skills_dirs:
            registry.discover(d)

        tools.append(registry.create_use_skill_tool())
        tools.append(registry.create_run_script_tool())
        tools.append(registry.create_read_reference_tool())
    except ImportError:
        # adk-skills-agent not installed
        pass

    return tools
