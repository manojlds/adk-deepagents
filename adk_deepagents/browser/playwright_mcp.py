"""Playwright MCP integration — browser automation via ``@playwright/mcp``.

Connects to the official Playwright MCP server for structured browser
automation using accessibility tree snapshots and element refs.

The Playwright MCP server exposes tools like ``browser_navigate``,
``browser_snapshot``, ``browser_click``, ``browser_type``, etc.  The agent
interacts with pages through ARIA snapshots (not screenshots), making it
token-efficient and deterministic.

Usage::

    # Async — resolve tools before creating agent
    tools, cleanup = await get_playwright_browser_tools()
    agent = create_deep_agent(tools=tools)
    # ... later ...
    await cleanup()

    # Or use create_deep_agent_async for convenience
    agent, cleanup = await create_deep_agent_async(browser="playwright")
"""

from __future__ import annotations

import logging
from typing import Any

from adk_deepagents.types import BrowserConfig

logger = logging.getLogger(__name__)

# Core browser tools exposed by @playwright/mcp (always available)
PLAYWRIGHT_CORE_TOOL_NAMES = frozenset(
    {
        "browser_navigate",
        "browser_navigate_back",
        "browser_snapshot",
        "browser_click",
        "browser_type",
        "browser_fill_form",
        "browser_select_option",
        "browser_hover",
        "browser_drag",
        "browser_press_key",
        "browser_wait_for",
        "browser_handle_dialog",
        "browser_file_upload",
        "browser_evaluate",
        "browser_take_screenshot",
        "browser_network_requests",
        "browser_console_messages",
        "browser_close",
        "browser_resize",
        "browser_tabs",
        "browser_install",
    }
)

# Vision tools (opt-in via caps=["vision"])
PLAYWRIGHT_VISION_TOOL_NAMES = frozenset(
    {
        "browser_mouse_click_xy",
        "browser_mouse_move_xy",
        "browser_mouse_drag_xy",
        "browser_mouse_down",
        "browser_mouse_up",
        "browser_mouse_wheel",
    }
)

# Testing tools (opt-in via caps=["testing"])
PLAYWRIGHT_TESTING_TOOL_NAMES = frozenset(
    {
        "browser_verify_element_visible",
        "browser_verify_text_visible",
        "browser_verify_list_visible",
        "browser_verify_value",
        "browser_generate_locator",
    }
)


def _build_server_args(config: BrowserConfig) -> list[str]:
    """Build the ``@playwright/mcp`` CLI arguments from a ``BrowserConfig``."""
    args: list[str] = ["@playwright/mcp@latest"]

    if config.browser != "chromium":
        args.extend(["--browser", config.browser])

    if config.headless:
        args.append("--headless")

    if config.viewport:
        w, h = config.viewport
        args.extend(["--viewport-size", f"{w}x{h}"])

    if config.caps:
        args.extend(["--caps", ",".join(config.caps)])

    if config.cdp_endpoint:
        args.extend(["--cdp-endpoint", config.cdp_endpoint])

    if config.storage_state:
        args.extend(["--storage-state", config.storage_state])

    return args


def _allowed_tool_names(config: BrowserConfig) -> frozenset[str]:
    """Return the set of tool names to include based on capabilities."""
    allowed = set(PLAYWRIGHT_CORE_TOOL_NAMES)
    if "vision" in config.caps:
        allowed |= PLAYWRIGHT_VISION_TOOL_NAMES
    if "testing" in config.caps:
        allowed |= PLAYWRIGHT_TESTING_TOOL_NAMES
    return frozenset(allowed)


async def get_playwright_browser_tools(
    config: BrowserConfig | None = None,
    *,
    command: str = "npx",
    filter_tools: bool = True,
) -> tuple[list[Any], Any]:
    """Connect to ``@playwright/mcp`` and return its tools.

    Parameters
    ----------
    config:
        Browser configuration. Defaults to headless Chromium at 1280×720.
    command:
        Command to start the MCP server (default ``"npx"``).
    filter_tools:
        If ``True``, only include known browser tools.
        If ``False``, include all tools from the server.

    Returns
    -------
    tuple[list, Any]
        ``(tools, cleanup)`` — tools are ready to add to an agent,
        and ``cleanup`` must be awaited when done.

    Raises
    ------
    ImportError
        If ``google.adk.tools.mcp_tool`` is not available.
    RuntimeError
        If connection to the Playwright MCP server fails.
    """
    try:
        from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
    except ImportError as e:
        raise ImportError(
            "MCP toolset not available. Ensure google-adk[mcp] is installed."
        ) from e

    if config is None:
        config = BrowserConfig()

    server_args = _build_server_args(config)

    try:
        tools, exit_stack = await MCPToolset.from_server(
            connection_params=StdioServerParameters(
                command=command,
                args=server_args,
            ),
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to Playwright MCP server: {e}. "
            "Ensure @playwright/mcp is available (npx @playwright/mcp@latest)."
        ) from e

    if filter_tools:
        allowed = _allowed_tool_names(config)
        tools = [t for t in tools if getattr(t, "name", "") in allowed]

    logger.info(
        "Connected to Playwright MCP server with %d tools: %s",
        len(tools),
        [getattr(t, "name", "?") for t in tools],
    )

    async def cleanup() -> None:
        """Close the MCP server connection."""
        try:
            await exit_stack.aclose()
        except Exception:
            logger.exception("Error closing Playwright MCP connection")

    return tools, cleanup
