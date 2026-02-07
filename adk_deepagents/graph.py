"""Main factory — ``create_deep_agent()``.

Mirrors ``deepagents.graph.create_deep_agent()`` using Google ADK primitives.
Wires together tools, callbacks, sub-agents, memory, and skills into a
configured ``LlmAgent``.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from google.adk.agents import LlmAgent

from adk_deepagents.backends.protocol import Backend, BackendFactory
from adk_deepagents.backends.state import StateBackend
from adk_deepagents.callbacks.after_tool import make_after_tool_callback
from adk_deepagents.callbacks.before_agent import make_before_agent_callback
from adk_deepagents.callbacks.before_model import make_before_model_callback
from adk_deepagents.callbacks.before_tool import make_before_tool_callback
from adk_deepagents.prompts import BASE_AGENT_PROMPT
from adk_deepagents.tools.filesystem import edit_file, glob, grep, ls, read_file, write_file
from adk_deepagents.tools.task import build_subagent_tools
from adk_deepagents.tools.todos import read_todos, write_todos
from adk_deepagents.types import SkillsConfig, SubAgentSpec, SummarizationConfig

# ---------------------------------------------------------------------------
# Default backend factory
# ---------------------------------------------------------------------------


def _default_backend_factory(state: dict[str, Any]) -> Backend:
    """Create a ``StateBackend`` from session state."""
    return StateBackend(state)


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------


def create_deep_agent(
    model: str = "gemini-2.5-flash",
    tools: Sequence[Callable] | None = None,
    *,
    instruction: str | None = None,
    subagents: list[SubAgentSpec] | None = None,
    skills: list[str] | None = None,
    skills_config: SkillsConfig | None = None,
    memory: list[str] | None = None,
    output_schema: type | None = None,
    backend: Backend | BackendFactory | None = None,
    execution: str | dict | None = None,
    interrupt_on: dict[str, bool] | None = None,
    name: str = "deep_agent",
) -> LlmAgent:
    """Create a deep agent with ADK primitives.

    This is the main entry point — analogous to
    ``deepagents.graph.create_deep_agent()``.

    Parameters
    ----------
    model:
        Model string (default ``"gemini-2.5-flash"``).
    tools:
        Additional user-provided tool functions.
    instruction:
        Custom system instruction. ``BASE_AGENT_PROMPT`` is prepended.
    subagents:
        Sub-agent specifications. A general-purpose sub-agent is always
        included by default.
    skills:
        List of directory paths to discover Agent Skills from (via adk-skills).
    skills_config:
        Optional configuration for adk-skills ``SkillsRegistry``.
    memory:
        List of AGENTS.md file paths to load as persistent memory.
    output_schema:
        Optional Pydantic model or type for structured output.
    backend:
        A ``Backend`` instance or a ``BackendFactory``. Defaults to
        ``StateBackend`` backed by session state.
    execution:
        Code execution backend. ``"heimdall"`` for sandboxed execution,
        ``"local"`` for subprocess, or a dict of MCP config.
    interrupt_on:
        Tool names that require human approval before execution.
    name:
        Agent name (default ``"deep_agent"``).

    Returns
    -------
    LlmAgent
        A fully configured ADK agent.
    """

    # 1. Resolve backend
    backend_factory: BackendFactory
    if backend is None:
        backend_factory = _default_backend_factory
    elif callable(backend) and not isinstance(backend, Backend):
        backend_factory = backend
    else:
        # Wrap a concrete backend instance in a factory
        backend_factory = lambda _state, _b=backend: _b  # type: ignore[assignment]

    # 2. Build core tool list
    core_tools: list[Callable] = [
        write_todos,
        read_todos,
        ls,
        read_file,
        write_file,
        edit_file,
        glob,
        grep,
    ]

    # Add user-provided tools
    if tools:
        core_tools.extend(tools)

    # 3. Skills integration (adk-skills)
    if skills:
        try:
            from adk_deepagents.skills.integration import add_skills_tools

            core_tools = add_skills_tools(
                core_tools,
                skills_dirs=skills,
                skills_config=skills_config,
            )
        except ImportError:
            pass  # adk-skills not installed; skip silently

    # 4. Execution tools
    has_execution = False
    if execution:
        has_execution = True
        if execution == "local":
            from adk_deepagents.execution.local import create_local_execute_tool

            core_tools.append(create_local_execute_tool())
        # "heimdall" and dict configs are handled in Phase 5

    # 5. Build sub-agent tools
    subagent_descriptions: list[dict[str, str]] = []
    subagent_tools = []
    if subagents is not None:
        subagent_tools = build_subagent_tools(
            subagents,
            default_model=model,
            default_tools=list(core_tools),
            include_general_purpose=True,
        )
        subagent_descriptions = [
            {"name": s["name"], "description": s["description"]}
            for s in subagents
        ]
        # Include general-purpose in descriptions if added
        has_gp = any(s["name"] in ("general-purpose", "general_purpose") for s in subagents)
        if not has_gp:
            from adk_deepagents.tools.task import GENERAL_PURPOSE_SUBAGENT

            subagent_descriptions.insert(
                0,
                {
                    "name": GENERAL_PURPOSE_SUBAGENT["name"],
                    "description": GENERAL_PURPOSE_SUBAGENT["description"],
                },
            )

    # 6. Compose callbacks
    before_agent_cb = make_before_agent_callback(
        memory_sources=memory,
        backend_factory=backend_factory if memory else None,
    )

    before_model_cb = make_before_model_callback(
        memory_sources=memory,
        has_execution=has_execution,
        subagent_descriptions=subagent_descriptions or None,
    )

    after_tool_cb = make_after_tool_callback(
        backend_factory=backend_factory,
    )

    before_tool_cb = make_before_tool_callback(
        interrupt_on=interrupt_on,
    )

    # 7. Build instruction
    full_instruction = BASE_AGENT_PROMPT
    if instruction:
        full_instruction = instruction + "\n\n" + BASE_AGENT_PROMPT

    # 8. Assemble all tools
    all_tools: list = list(core_tools) + subagent_tools

    # 9. Create and return the agent
    agent = LlmAgent(
        name=name,
        model=model,
        instruction=full_instruction,
        tools=all_tools,
        output_schema=output_schema,
        before_agent_callback=before_agent_cb,
        before_model_callback=before_model_cb,
        after_tool_callback=after_tool_cb,
        before_tool_callback=before_tool_cb,
    )

    return agent
