"""Main factory — ``create_deep_agent()``.

Mirrors ``deepagents.graph.create_deep_agent()`` using Google ADK primitives.
Wires together tools, callbacks, sub-agents, memory, skills, execution,
and summarization into a configured ``LlmAgent``.
"""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

from google.adk.agents import LlmAgent

from adk_deepagents.backends.protocol import Backend, BackendFactory
from adk_deepagents.backends.state import StateBackend
from adk_deepagents.callbacks.after_model import make_after_model_callback
from adk_deepagents.callbacks.after_tool import make_after_tool_callback
from adk_deepagents.callbacks.before_agent import make_before_agent_callback
from adk_deepagents.callbacks.before_model import make_before_model_callback
from adk_deepagents.callbacks.before_tool import make_before_tool_callback
from adk_deepagents.prompts import BASE_AGENT_PROMPT
from adk_deepagents.tools.compact import create_compact_conversation_tool
from adk_deepagents.tools.filesystem import edit_file, glob, grep, ls, read_file, write_file
from adk_deepagents.tools.task import (
    GENERAL_PURPOSE_SUBAGENT,
    _sanitize_agent_name,
    build_subagent_tools,
)
from adk_deepagents.tools.task_dynamic import (
    create_dynamic_task_tool,
    create_register_subagent_tool,
)
from adk_deepagents.tools.todos import read_todos, write_todos
from adk_deepagents.types import (
    BrowserConfig,
    DynamicTaskConfig,
    SkillsConfig,
    SubAgentSpec,
    SummarizationConfig,
)

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

    Handles async callbacks: if either callback is async, the composed
    callback is also async.
    """
    if extra is None:
        return builtin
    if builtin is None:
        return extra

    either_async = asyncio.iscoroutinefunction(builtin) or asyncio.iscoroutinefunction(extra)

    if either_async:

        async def composed_async(*args: Any, **kwargs: Any) -> Any:
            if asyncio.iscoroutinefunction(builtin):
                result = await builtin(*args, **kwargs)
            else:
                result = builtin(*args, **kwargs)
            if result is not None:
                return result
            if asyncio.iscoroutinefunction(extra):
                return await extra(*args, **kwargs)
            return extra(*args, **kwargs)

        return composed_async

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
    model: str | Any = "gemini-2.5-flash",
    tools: Sequence[Callable] | None = None,
    *,
    instruction: str | None = None,
    subagents: list[SubAgentSpec | LlmAgent] | None = None,
    skills: list[str] | None = None,
    skills_config: SkillsConfig | None = None,
    memory: list[str] | None = None,
    output_schema: Any = None,
    backend: Backend | BackendFactory | None = None,
    execution: str | dict | None = None,
    browser: BrowserConfig | str | None = None,
    summarization: SummarizationConfig | None = None,
    delegation_mode: Literal["static", "dynamic", "both"] = "static",
    dynamic_task_config: DynamicTaskConfig | None = None,
    interrupt_on: dict[str, bool] | None = None,
    extra_callbacks: dict[str, Callable] | None = None,
    name: str = "deep_agent",
    error_handling: bool = True,
    message_queue: bool = False,
    message_queue_provider: Callable[[], list[dict[str, Any]]] | None = None,
    multimodal: bool = False,
    http_tools: bool = False,
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
    browser:
        Browser automation backend. ``"playwright"`` for Playwright MCP,
        or a ``BrowserConfig`` for custom configuration. Requires
        ``create_deep_agent_async()`` to resolve MCP tools.
    summarization:
        Optional ``SummarizationConfig`` for context window management.
    delegation_mode:
        Sub-agent delegation style. ``"static"`` uses one ``AgentTool`` per
        configured sub-agent (default behavior). ``"dynamic"`` exposes a single
        ``task`` tool with runtime sub-agent/session routing. ``"both"`` exposes
        both interfaces.
    dynamic_task_config:
        Optional configuration for the dynamic ``task`` delegation tool.
    interrupt_on:
        Tool names that require human approval before execution.
    extra_callbacks:
        Optional dict with keys ``before_agent``, ``before_model``,
        ``after_model``, ``before_tool``, ``after_tool``.  Each value is a callback that
        is composed **after** the built-in callback.  If the built-in
        callback short-circuits (returns a non-``None`` value), the
        extra callback is **not** called.
    name:
        Agent name (default ``"deep_agent"``).
    error_handling:
        Wrap tool functions with error handlers so exceptions return
        structured error dicts to the LLM instead of crashing.
    message_queue:
        When ``True``, enable message queue support. External systems
        can inject messages mid-run by writing to
        ``state["_message_queue"]``.
    multimodal:
        When ``True``, scan user messages for image URLs and fetch them
        as inline base64 data parts for multimodal model support.
    http_tools:
        Include ``fetch_url`` and ``http_request`` tools with SSRF protection.

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
        backend_factory = cast(BackendFactory, backend)
    else:
        # Wrap a concrete backend instance in a factory
        concrete_backend = backend

        def _instance_backend_factory(_state: dict[str, Any]) -> Backend:
            return concrete_backend

        backend_factory = _instance_backend_factory

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

    # 2b. Wrap tools with error handler
    if error_handling:
        from adk_deepagents.tools.error_handler import wrap_tools_with_error_handler

        core_tools = wrap_tools_with_error_handler(core_tools)

    # 2c. Conversation compaction tool (manual trigger for summarization)
    if summarization is not None:
        core_tools.append(
            create_compact_conversation_tool(
                summarization_config=summarization,
            )
        )

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

    # 4b. Browser tools
    has_browser = False
    if browser:
        has_browser = True
        if browser != "_resolved":
            warnings.warn(
                f"browser={browser!r} requires async MCP tool resolution. "
                "Use create_deep_agent_async() or pre-resolve browser MCP tools "
                "and pass them via the `tools` parameter.",
                stacklevel=2,
            )

    # 4c. HTTP tools
    has_http_tools = False
    if http_tools:
        has_http_tools = True
        from adk_deepagents.tools.http import fetch_url, http_request

        core_tools.extend([fetch_url, http_request])

    # 5. Build sub-agent tools
    subagent_descriptions: list[dict[str, str]] = []
    subagent_tools = []
    if delegation_mode not in {"static", "dynamic", "both"}:
        raise ValueError(
            f"Invalid delegation_mode={delegation_mode!r}. "
            "Expected one of: 'static', 'dynamic', 'both'."
        )

    # Sub-agents use the same core callback stack as the parent agent, but
    # without nested sub-agent prompt docs to avoid recursion noise.
    subagent_before_agent_cb = make_before_agent_callback(
        memory_sources=memory,
        backend_factory=backend_factory,
    )
    subagent_before_model_cb = make_before_model_callback(
        memory_sources=memory,
        has_execution=has_execution,
        subagent_descriptions=None,
        dynamic_task_config=None,
        summarization_config=summarization,
        backend_factory=backend_factory if summarization else None,
    )
    subagent_after_tool_cb = make_after_tool_callback(
        backend_factory=backend_factory,
    )

    if delegation_mode in {"static", "both"}:
        static_subagents = subagents or []
        subagent_tools = build_subagent_tools(
            static_subagents,
            default_model=model,
            default_tools=list(core_tools),
            include_general_purpose=True,
            skills_config=skills_config,
            before_agent_callback=subagent_before_agent_cb,
            before_model_callback=subagent_before_model_cb,
            after_tool_callback=subagent_after_tool_cb,
            default_interrupt_on=interrupt_on,
        )

    resolved_dynamic_task_config: DynamicTaskConfig | None = None
    if delegation_mode in {"dynamic", "both"}:
        resolved_dynamic_task_config = dynamic_task_config or DynamicTaskConfig()
        core_tools.append(
            create_register_subagent_tool(
                default_model=model,
                default_tools=list(core_tools),
                config=resolved_dynamic_task_config,
            )
        )
        core_tools.append(
            create_dynamic_task_tool(
                default_model=model,
                default_tools=list(core_tools),
                subagents=subagents,
                skills_config=skills_config,
                config=resolved_dynamic_task_config,
                before_agent_callback=subagent_before_agent_cb,
                before_model_callback=subagent_before_model_cb,
                after_tool_callback=subagent_after_tool_cb,
                default_interrupt_on=interrupt_on,
            )
        )

    gp_name = _sanitize_agent_name(GENERAL_PURPOSE_SUBAGENT.get("name", "general_purpose"))
    gp_description = GENERAL_PURPOSE_SUBAGENT.get(
        "description",
        "General-purpose sub-agent for research and multi-step tasks.",
    )

    if subagents is not None:
        subagent_descriptions = []
        for s in subagents:
            if isinstance(s, LlmAgent):
                sanitized_name = _sanitize_agent_name(s.name)
                subagent_descriptions.append(
                    {"name": sanitized_name, "description": s.description or s.name}
                )
            else:
                spec_name: str | None = s.get("name")
                spec_description: str | None = s.get("description")
                if isinstance(spec_name, str) and isinstance(spec_description, str):
                    sanitized_name = _sanitize_agent_name(spec_name)
                    subagent_descriptions.append(
                        {"name": sanitized_name, "description": spec_description}
                    )

    # Include general-purpose in docs whenever delegation is enabled.
    if delegation_mode in {"static", "dynamic", "both"}:
        names = {d["name"] for d in subagent_descriptions}
        if gp_name not in names:
            subagent_descriptions.insert(
                0,
                {
                    "name": gp_name,
                    "description": gp_description,
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
        has_http_tools=has_http_tools,
        subagent_descriptions=subagent_descriptions or None,
        dynamic_task_config=resolved_dynamic_task_config,
        summarization_config=summarization,
        backend_factory=backend_factory if summarization else None,
        message_queue=message_queue,
        message_queue_provider=message_queue_provider,
        multimodal=multimodal,
    )

    after_tool_cb = make_after_tool_callback(
        backend_factory=backend_factory,
    )

    before_tool_cb = make_before_tool_callback(
        interrupt_on=interrupt_on,
    )

    after_model_cb = make_after_model_callback()

    # 6b. Compose extra callbacks (if provided)
    if extra_callbacks:
        before_agent_cb = _compose_callbacks(before_agent_cb, extra_callbacks.get("before_agent"))
        before_model_cb = _compose_callbacks(before_model_cb, extra_callbacks.get("before_model"))
        after_model_cb = _compose_callbacks(after_model_cb, extra_callbacks.get("after_model"))
        after_tool_cb = _compose_callbacks(after_tool_cb, extra_callbacks.get("after_tool"))
        before_tool_cb = _compose_callbacks(before_tool_cb, extra_callbacks.get("before_tool"))

    # 7. Build instruction
    full_instruction = BASE_AGENT_PROMPT
    if instruction:
        full_instruction = instruction + "\n\n" + BASE_AGENT_PROMPT

    if has_browser:
        from adk_deepagents.browser.prompts import BROWSER_SYSTEM_PROMPT

        full_instruction = full_instruction + "\n\n" + BROWSER_SYSTEM_PROMPT

    # 8. Assemble all tools
    all_tools: list[Any] = list(core_tools) + subagent_tools

    # 9. Create and return the agent
    agent = LlmAgent(
        name=name,
        model=model,
        instruction=full_instruction,
        tools=all_tools,  # type: ignore[invalid-argument-type]
        output_schema=output_schema,
        before_agent_callback=before_agent_cb,
        before_model_callback=before_model_cb,
        after_model_callback=after_model_cb,
        after_tool_callback=after_tool_cb,
        before_tool_callback=before_tool_cb,
    )

    return agent


# ---------------------------------------------------------------------------
# Async factory (for Heimdall MCP)
# ---------------------------------------------------------------------------


async def create_deep_agent_async(
    model: str | Any = "gemini-2.5-flash",
    tools: Sequence[Callable] | None = None,
    *,
    instruction: str | None = None,
    subagents: list[SubAgentSpec | LlmAgent] | None = None,
    skills: list[str] | None = None,
    skills_config: SkillsConfig | None = None,
    memory: list[str] | None = None,
    output_schema: Any = None,
    backend: Backend | BackendFactory | None = None,
    execution: str | dict | None = None,
    browser: BrowserConfig | str | None = None,
    summarization: SummarizationConfig | None = None,
    delegation_mode: Literal["static", "dynamic", "both"] = "static",
    dynamic_task_config: DynamicTaskConfig | None = None,
    interrupt_on: dict[str, bool] | None = None,
    extra_callbacks: dict[str, Callable] | None = None,
    name: str = "deep_agent",
    error_handling: bool = True,
    message_queue: bool = False,
    message_queue_provider: Callable[[], list[dict[str, Any]]] | None = None,
    multimodal: bool = False,
    http_tools: bool = False,
) -> tuple[LlmAgent, Callable | None]:
    """Async variant of ``create_deep_agent()`` that resolves MCP tools.

    Use this when ``execution="heimdall"``, ``execution=dict(...)``,
    or ``browser="playwright"``.
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
    cleanup_fns: list[Callable] = []
    mcp_tools: list = []

    # Resolve execution MCP tools if needed
    if execution == "heimdall":
        from adk_deepagents.execution.heimdall import get_heimdall_tools

        exec_tools, exec_cleanup = await get_heimdall_tools()
        mcp_tools.extend(exec_tools)
        cleanup_fns.append(exec_cleanup)
    elif isinstance(execution, dict):
        from adk_deepagents.execution.heimdall import get_heimdall_tools_from_config

        exec_tools, exec_cleanup = await get_heimdall_tools_from_config(execution)
        mcp_tools.extend(exec_tools)
        cleanup_fns.append(exec_cleanup)

    # Resolve browser MCP tools if needed
    browser_config: BrowserConfig | None = None
    if browser == "playwright" or isinstance(browser, BrowserConfig):
        from adk_deepagents.browser.playwright_mcp import get_playwright_browser_tools

        browser_config = browser if isinstance(browser, BrowserConfig) else BrowserConfig()

        browser_tools, browser_cleanup = await get_playwright_browser_tools(
            config=browser_config,
        )
        mcp_tools.extend(browser_tools)
        cleanup_fns.append(browser_cleanup)

    # Merge MCP tools with user tools
    combined_tools: list[Callable] = list(tools or []) + mcp_tools

    # Signal that execution tools are available (for prompt injection) without
    # adding a duplicate local execute tool — MCP tools are already in combined_tools.
    effective_execution: str | dict | None = execution
    if execution == "heimdall" or isinstance(execution, dict):
        effective_execution = "_resolved" if mcp_tools else None

    # Signal that browser tools are resolved
    effective_browser: BrowserConfig | str | None = browser
    if browser == "playwright" or isinstance(browser, BrowserConfig):
        effective_browser = "_resolved" if browser_config else None

    # Build a single cleanup function
    async def cleanup() -> None:
        for fn in cleanup_fns:
            await fn()

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
        browser=effective_browser,
        summarization=summarization,
        delegation_mode=delegation_mode,
        dynamic_task_config=dynamic_task_config,
        interrupt_on=interrupt_on,
        extra_callbacks=extra_callbacks,
        name=name,
        error_handling=error_handling,
        message_queue=message_queue,
        message_queue_provider=message_queue_provider,
        multimodal=multimodal,
        http_tools=http_tools,
    )

    return agent, cleanup
