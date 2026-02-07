"""Skills integration — wires adk-skills into create_deep_agent().

Integrates the adk-skills library (``adk-skills-agent`` on PyPI) for
Agent Skills support: SKILL.md discovery, progressive disclosure via
``use_skill``, script execution via ``run_script``, and reference
loading via ``read_reference``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from adk_deepagents.types import SkillsConfig

logger = logging.getLogger(__name__)


def add_skills_tools(
    tools: list[Callable],
    skills_dirs: list[str],
    skills_config: SkillsConfig | None = None,
    state: dict[str, Any] | None = None,
) -> list[Callable]:
    """Discover skills and add adk-skills tools to the tool list.

    Parameters
    ----------
    tools:
        Existing tool list to extend.
    skills_dirs:
        Directories to discover skills from.
    skills_config:
        Optional skills configuration (passed to SkillsRegistry).
    state:
        Optional session state dict. If provided, the ``SkillsRegistry``
        instance and discovered skills metadata are stored in state for
        later access by callbacks.

    Returns
    -------
    list[Callable]
        The extended tool list with use_skill, run_script, read_reference.
    """
    try:
        from adk_skills_agent import SkillsRegistry
    except ImportError:
        logger.warning(
            "adk-skills-agent is not installed. Skills will not be available. "
            "Install with: pip install adk-skills-agent"
        )
        return tools

    # Create the registry with optional config
    config_kwargs: dict[str, Any] = {}
    if skills_config and skills_config.extra:
        config_kwargs.update(skills_config.extra)

    try:
        registry = SkillsRegistry(**config_kwargs)
    except Exception:
        logger.exception("Failed to create SkillsRegistry")
        return tools

    # Discover skills from all provided directories
    discovered_count = 0
    for directory in skills_dirs:
        try:
            registry.discover(directory)
            discovered_count += 1
        except Exception:
            logger.exception("Failed to discover skills from %s", directory)

    if discovered_count == 0:
        logger.warning("No skills directories were successfully discovered")
        return tools

    # Store the registry and metadata in state for callback access
    if state is not None:
        state["_skills_registry"] = registry
        try:
            state["skills_metadata"] = registry.list_metadata()
        except Exception:
            logger.exception("Failed to list skills metadata")
            state["skills_metadata"] = []

    # Add the adk-skills tools
    try:
        tools.append(registry.create_use_skill_tool())
    except Exception:
        logger.exception("Failed to create use_skill tool")

    try:
        tools.append(registry.create_run_script_tool())
    except Exception:
        logger.exception("Failed to create run_script tool")

    try:
        tools.append(registry.create_read_reference_tool())
    except Exception:
        logger.exception("Failed to create read_reference tool")

    return tools


def inject_skills_into_prompt(
    instruction: str,
    state: dict[str, Any],
    format: str = "xml",
) -> str:
    """Inject skills listing into the system prompt.

    Alternative to tool-based discovery — appends an ``<available_skills>``
    block directly to the instruction string.

    Parameters
    ----------
    instruction:
        The existing system instruction.
    state:
        Session state dict containing ``_skills_registry``.
    format:
        Output format for skills listing (default ``"xml"``).

    Returns
    -------
    str
        The instruction with skills listing appended.
    """
    registry = state.get("_skills_registry")
    if registry is None:
        return instruction

    try:
        return registry.inject_skills_prompt(instruction, format=format)
    except Exception:
        logger.exception("Failed to inject skills prompt")
        return instruction
