"""Shared type definitions for adk-deepagents."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


class SubAgentSpec(TypedDict, total=False):
    """Specification for a sub-agent.

    Mirrors deepagents' ``SubAgent`` TypedDict.
    """

    name: str  # required
    description: str  # required
    system_prompt: str
    tools: Sequence[Callable]
    model: str
    skills: list[str]
    interrupt_on: dict[str, bool]


SubAgentSpec.__required_keys__ = frozenset({"name", "description"})


@dataclass
class TemporalTaskConfig:
    """Configuration for running dynamic tasks on Temporal workers.

    Attach this to ``DynamicTaskConfig.temporal`` to dispatch dynamic
    sub-agent turns through Temporal workflows instead of in-process
    ``InMemoryRunner`` execution.

    Requires ``pip install adk-deepagents[temporal]``.
    """

    target_host: str = "localhost:7233"
    """Temporal server gRPC endpoint."""

    namespace: str = "default"
    """Temporal namespace."""

    task_queue: str = "adk-deepagents-tasks"
    """Task queue used by Temporal workers for dynamic task workflows."""

    workflow_id_prefix: str = "dynamic-task"
    """Workflow ID prefix. IDs are ``{prefix}:{parent_id}:{task_id}``."""

    activity_timeout_seconds: float | None = None
    """Optional activity timeout override.

    If ``None``, the worker uses ``DynamicTaskConfig.timeout_seconds``.
    """

    retry_max_attempts: int = 1
    """Maximum activity retry attempts (default: no retries)."""

    idle_timeout_seconds: float = 600.0
    """Seconds a workflow may sit idle (no ``run_turn`` updates) before
    it automatically completes.  Prevents leaked workflows."""


@dataclass
class A2ATaskConfig:
    """Configuration for running dynamic tasks via an external A2A agent."""

    agent_url: str = "http://localhost:8000"
    """Base URL of the target A2A agent endpoint."""

    timeout_seconds: float = 120.0
    """Per-request timeout used by the A2A transport."""

    poll_interval_seconds: float = 1.0
    """Polling interval when waiting for async A2A task completion."""

    max_polls: int = 120
    """Maximum number of polling attempts before timing out."""


@dataclass
class DynamicTaskConfig:
    """Configuration for dynamic task-based sub-agent delegation."""

    max_parallel: int = 4
    """Maximum number of concurrent dynamic tasks per parent session."""

    concurrency_policy: Literal["error", "wait"] = "error"
    """Behavior when `max_parallel` is reached.

    - ``"error"``: immediately return a tool-level error
    - ``"wait"``: queue/wait until a slot is free or queue timeout is reached
    """

    queue_timeout_seconds: float = 30.0
    """Maximum seconds to wait for a concurrency slot when policy is ``"wait"``."""

    max_depth: int = 2
    """Maximum delegation depth for dynamically spawned sub-agents."""

    timeout_seconds: float = 120.0
    """Per-task timeout when running a dynamic sub-agent."""

    allow_model_override: bool = False
    """Allow callers to override the sub-agent model per task invocation."""

    temporal: TemporalTaskConfig | None = None
    """Optional Temporal backend for dynamic task execution.

    When set, ``task()`` dispatches sub-agent turns to Temporal workers
    instead of running child sessions in-process.
    """

    a2a: A2ATaskConfig | None = None
    """Optional A2A backend for dynamic task execution.

    When set, ``task()`` dispatches delegated turns to an external A2A
    agent endpoint.
    """


@dataclass
class TruncateArgsConfig:
    """Settings for truncating large tool arguments in older messages.

    Ported from deepagents ``TruncateArgsSettings``.  Before summarization
    is triggered, arguments to ``write_file`` and ``edit_file`` calls in
    older messages are replaced with a truncated preview.  This frees
    context window space without losing the record of *which* tool was
    called.
    """

    trigger: tuple[str, float | int] | None = None
    """When to start truncating.  Same format as ``SummarizationConfig.trigger``."""

    keep: tuple[str, int | float] = ("messages", 20)
    """How many recent messages to leave untouched."""

    max_length: int = 2000
    """Maximum character length for a single tool argument before truncation."""

    truncation_text: str = "...(argument truncated)"
    """Replacement text appended after the 20-char prefix of a truncated arg."""


@dataclass
class SummarizationConfig:
    """Configuration for conversation summarization."""

    model: str = "gemini-2.5-flash"
    trigger: tuple[str, float] = ("fraction", 0.85)
    keep: tuple[str, int] = ("messages", 6)
    history_path_prefix: str = "/conversation_history"
    use_llm_summary: bool = True
    """If True (default), use the configured model to generate summaries.
    If False, fall back to inline text truncation (faster, no extra API call)."""
    truncate_args: TruncateArgsConfig | None = None
    """Optional settings for truncating large tool arguments in older messages."""
    context_window: int | None = None
    """Explicit context window size in tokens. If set, overrides the model-based
    lookup. If None, the context window is resolved from the model name."""


@dataclass
class BrowserConfig:
    """Configuration for browser automation via Playwright MCP.

    Controls how the ``@playwright/mcp`` server is launched and configured.
    """

    provider: str = "playwright"
    """Browser automation provider. Currently only ``"playwright"`` is supported."""

    headless: bool = True
    """Run the browser in headless mode (no visible window)."""

    browser: str = "chromium"
    """Browser engine: ``"chromium"``, ``"firefox"``, or ``"webkit"``."""

    viewport: tuple[int, int] = (1280, 720)
    """Browser viewport size as ``(width, height)``."""

    caps: list[str] = field(default_factory=list)
    """Extra capabilities to enable: ``"vision"``, ``"pdf"``, ``"testing"``."""

    cdp_endpoint: str | None = None
    """Connect to an existing browser via Chrome DevTools Protocol URL."""

    storage_state: str | None = None
    """Path to a saved browser authentication state file."""


@dataclass
class SkillsConfig:
    """Configuration passed to adk-skills SkillsRegistry."""

    # Placeholder — populated when adk-skills integration is wired up.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CallbackHooks:
    """Typed callback hooks for composing with built-in callbacks.

    Each callback is composed **after** the built-in callback.  If the
    built-in short-circuits (returns non-``None``), the hook is skipped.
    """

    before_agent: Callable | None = None
    before_model: Callable | None = None
    after_model: Callable | None = None
    before_tool: Callable | None = None
    after_tool: Callable | None = None


@dataclass
class DeepAgentConfig:
    """Advanced configuration for ``create_deep_agent()``.

    Groups rarely-used settings, feature flags, and hooks so the main
    factory signature stays clean.
    """

    output_schema: Any = None
    """Optional Pydantic model or type for structured output."""

    summarization: SummarizationConfig | None = None
    """Context window management configuration."""

    delegation_mode: Literal["static", "dynamic", "both"] = "static"
    """Sub-agent delegation style."""

    dynamic_task_config: DynamicTaskConfig | None = None
    """Configuration for the dynamic ``task`` delegation tool."""

    skills_config: SkillsConfig | None = None
    """Optional configuration for adk-skills ``SkillsRegistry``."""

    interrupt_on: dict[str, bool] | None = None
    """Tool names that require human approval before execution."""

    callbacks: CallbackHooks | None = None
    """Extra callback hooks composed after the built-in callbacks."""

    error_handling: bool = True
    """Wrap tools with error handlers (structured error dicts)."""

    message_queue: bool = False
    """Enable message queue support via ``state["_message_queue"]``."""

    message_queue_provider: Callable[[], list[dict[str, Any]]] | None = None
    """Optional callable that drains in-process queued messages."""

    multimodal: bool = False
    """Scan user messages for image URLs and inline them."""

    http_tools: bool = False
    """Include ``fetch_url`` and ``http_request`` tools."""
