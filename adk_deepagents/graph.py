"""Main factory — ``create_deep_agent()``.

Mirrors ``deepagents.graph.create_deep_agent()`` using Google ADK primitives.
Wires together tools, callbacks, sub-agents, memory, skills, execution,
and summarization into a configured ``LlmAgent``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

from google.adk.agents import LlmAgent

from adk_deepagents.backends.protocol import Backend, BackendFactory
from adk_deepagents.backends.state import StateBackend
from adk_deepagents.callbacks.after_tool import make_after_tool_callback
from adk_deepagents.callbacks.before_agent import make_before_agent_callback
from adk_deepagents.callbacks.before_model import make_before_model_callback
from adk_deepagents.callbacks.before_tool import make_before_tool_callback
from adk_deepagents.prompts import BASE_AGENT_PROMPT
from adk_deepagents.tools.filesystem import edit_file, glob, grep, ls, read_file, write_file
from adk_deepagents.tools.task import (
    GENERAL_PURPOSE_SUBAGENT,
    _sanitize_agent_name,
    build_subagent_tools,
)
from adk_deepagents.tools.todos import read_todos, write_todos
from adk_deepagents.types import SkillsConfig, SubAgentSpec, SummarizationConfig

# ---------------------------------------------------------------------------
# Default backend factory
# ---------------------------------------------------------------------------


def _default_backend_factory(state: dict[str, Any]) -> Backend:
    """Create a ``StateBackend`` from session state."""
    return StateBackend(state)


def _compose_callbacks(
    builtin: Callable | None,
    extra: Callable | None,
) -> Callable | None:
    """Compose *builtin* and *extra* callbacks.

    The built-in callback runs first.  If it returns a non-``None`` value
    (short-circuit), the extra callback is **not** called and the built-in
    result is returned.  Otherwise, the extra callback is called with the
    same arguments and its result is returned.

    If either side is ``None``, the other is returned as-is.
    """
    if extra is None:
        return builtin
    if builtin is None:
        return extra

    def composed(*args: Any, **kwargs: Any) -> Any:
        result = builtin(*args, **kwargs)
        if result is not None:
            return result
        return extra(*args, **kwargs)

    return composed


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------


def create_deep_agent(
    model: str = "gemini-2.5-flash",
    tools: Sequence[Callable] | None = None,
    *,
    instruction: str | None = None,
    subagents: list[SubAgentSpec | LlmAgent] | None = None,
    skills: list[str] | None = None,
    skills_config: SkillsConfig | None = None,
    memory: list[str] | None = None,
    output_schema: type | None = None,
    backend: Backend | BackendFactory | None = None,
    execution: str | dict | None = None,
    summarization: SummarizationConfig | None = None,
    interrupt_on: dict[str, bool] | None = None,
    extra_callbacks: dict[str, Callable] | None = None,
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
        ``"local"`` for subprocess, or a dict of MCP config. For
        ``"heimdall"`` or dict configs, use ``create_deep_agent_async()``
        or pre-resolve MCP tools via ``tools``.
    summarization:
        Optional ``SummarizationConfig`` for context window management.
    interrupt_on:
        Tool names that require human approval before execution.
    extra_callbacks:
        Optional dict with keys ``before_agent``, ``before_model``,
        ``before_tool``, ``after_tool``.  Each value is a callback that
        is composed **after** the built-in callback.  If the built-in
        callback short-circuits (returns a non-``None`` value), the
        extra callback is **not** called.
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
        def backend_factory(_state: dict, _b: Backend = backend) -> Backend:
            return _b

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
            import adk_skills_agent  # noqa: F401
        except ImportError:
            raise ImportError(
                "adk-skills-agent is required for skills support. "
                "Install it with: pip install adk-skills-agent"
            ) from None

        from adk_deepagents.skills.integration import add_skills_tools

        core_tools = add_skills_tools(
            core_tools,
            skills_dirs=skills,
            skills_config=skills_config,
        )

    # 4. Execution tools
    has_execution = False
    if execution:
        has_execution = True
        if execution == "local":
            from adk_deepagents.execution.local import create_local_execute_tool

            core_tools.append(create_local_execute_tool())
        elif execution == "heimdall" or isinstance(execution, dict):
            # Heimdall/MCP tools must be resolved async.
            warnings.warn(
                f"execution={execution!r} requires async MCP tool resolution. "
                "Use create_deep_agent_async() or pre-resolve MCP tools "
                "and pass them via the `tools` parameter.",
                stacklevel=2,
            )

    # 5. Build sub-agent tools
    subagent_descriptions: list[dict[str, str]] = []
    subagent_tools = []
    if subagents is not None:
        subagent_tools = build_subagent_tools(
            subagents,
            default_model=model,
            default_tools=list(core_tools),
            include_general_purpose=True,
            skills_config=skills_config,
        )
        subagent_descriptions = []
        for s in subagents:
            if isinstance(s, LlmAgent):
                subagent_descriptions.append(
                    {"name": s.name, "description": s.description or s.name}
                )
            else:
                subagent_descriptions.append({"name": s["name"], "description": s["description"]})
        # Include general-purpose in descriptions if added
        all_names = set()
        for s in subagents:
            if isinstance(s, LlmAgent):
                all_names.add(s.name)
            else:
                all_names.add(s["name"])
        has_gp = any(n in ("general-purpose", "general_purpose") for n in all_names)
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
        backend_factory=backend_factory,
    )

    before_model_cb = make_before_model_callback(
        memory_sources=memory,
        has_execution=has_execution,
        subagent_descriptions=subagent_descriptions or None,
        summarization_config=summarization,
        backend_factory=backend_factory if summarization else None,
    )

    after_tool_cb = make_after_tool_callback(
        backend_factory=backend_factory,
    )

    before_tool_cb = make_before_tool_callback(
        interrupt_on=interrupt_on,
    )

    # 6b. Compose extra callbacks (if provided)
    if extra_callbacks:
        before_agent_cb = _compose_callbacks(before_agent_cb, extra_callbacks.get("before_agent"))
        before_model_cb = _compose_callbacks(before_model_cb, extra_callbacks.get("before_model"))
        after_tool_cb = _compose_callbacks(after_tool_cb, extra_callbacks.get("after_tool"))
        before_tool_cb = _compose_callbacks(before_tool_cb, extra_callbacks.get("before_tool"))

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


# ---------------------------------------------------------------------------
# Async factory (for Heimdall MCP)
# ---------------------------------------------------------------------------


async def create_deep_agent_async(
    model: str = "gemini-2.5-flash",
    tools: Sequence[Callable] | None = None,
    *,
    instruction: str | None = None,
    subagents: list[SubAgentSpec | LlmAgent] | None = None,
    skills: list[str] | None = None,
    skills_config: SkillsConfig | None = None,
    memory: list[str] | None = None,
    output_schema: type | None = None,
    backend: Backend | BackendFactory | None = None,
    execution: str | dict | None = None,
    summarization: SummarizationConfig | None = None,
    interrupt_on: dict[str, bool] | None = None,
    extra_callbacks: dict[str, Callable] | None = None,
    name: str = "deep_agent",
) -> tuple[LlmAgent, Callable | None]:
    """Async variant of ``create_deep_agent()`` that resolves MCP tools.

    Use this when ``execution="heimdall"`` or ``execution=dict(...)``.
    Returns ``(agent, cleanup_fn)`` where ``cleanup_fn`` must be awaited
    when the agent is no longer needed.

    Parameters
    ----------
    (same as ``create_deep_agent``)

    Returns
    -------
    tuple[LlmAgent, Callable | None]
        The agent and an optional async cleanup function for MCP connections.
    """
    cleanup_fn: Callable | None = None
    mcp_tools: list = []

    # Resolve MCP tools if needed
    if execution == "heimdall":
        from adk_deepagents.execution.heimdall import get_heimdall_tools

        mcp_tools, cleanup_fn = await get_heimdall_tools()
    elif isinstance(execution, dict):
        from adk_deepagents.execution.heimdall import get_heimdall_tools_from_config

        mcp_tools, cleanup_fn = await get_heimdall_tools_from_config(execution)

    # Merge MCP tools with user tools
    combined_tools: list[Callable] = list(tools or []) + mcp_tools

    # Signal that execution tools are available (for prompt injection) without
    # adding a duplicate local execute tool — MCP tools are already in combined_tools.
    effective_execution: str | dict | None = execution
    if execution == "heimdall" or isinstance(execution, dict):
        effective_execution = "_resolved" if mcp_tools else None

    agent = create_deep_agent(
        model=model,
        tools=combined_tools,
        instruction=instruction,
        subagents=subagents,
        skills=skills,
        skills_config=skills_config,
        memory=memory,
        output_schema=output_schema,
        backend=backend,
        execution=effective_execution,
        summarization=summarization,
        interrupt_on=interrupt_on,
        extra_callbacks=extra_callbacks,
        name=name,
    )

    return agent, cleanup_fn
