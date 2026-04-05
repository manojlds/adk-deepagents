"""Heimdall MCP integration — sandboxed code execution.

Integrates the Heimdall MCP server for sandboxed Python (Pyodide/WASM)
and Bash (just-bash) execution via ADK's ``McpToolset``.

Heimdall provides:
- ``execute_python``: Sandboxed Python via Pyodide WebAssembly
- ``execute_bash``: Bash command simulation via just-bash
- ``write_file`` / ``read_file`` / ``list_files``: Virtual workspace filesystem
- ``install_packages``: Python package installation via micropip

Usage::

    # Async — resolve tools before creating agent
    tools, cleanup = await get_heimdall_tools("/workspace")
    agent = create_deep_agent(tools=tools)
    # ... later ...
    await cleanup()

    # Or use create_deep_agent_async for convenience
    agent, cleanup = await create_deep_agent_async(execution="heimdall")
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _import_mcp_toolset_module() -> Any:
    try:
        return importlib.import_module("google.adk.tools.mcp_tool.mcp_toolset")
    except ImportError as e:
        raise ImportError(
            "MCP toolset not available. Ensure google-adk>=1.21.0 and mcp are installed."
        ) from e


# Tool names exposed by Heimdall that we want to include
HEIMDALL_TOOL_NAMES = frozenset(
    {
        "execute_python",
        "execute_bash",
        "install_packages",
    }
)

# Heimdall workspace tools — namespaced to avoid collision with filesystem tools
HEIMDALL_WORKSPACE_TOOL_NAMES = frozenset(
    {
        "write_file",
        "read_file",
        "list_files",
        "delete_file",
    }
)


async def get_heimdall_tools(
    workspace_path: str = "/workspace",
    *,
    command: str = "npx",
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    filter_tools: bool = True,
) -> tuple[list[Any], Any]:
    """Connect to Heimdall MCP server and return its tools.

    Parameters
    ----------
    workspace_path:
        Path to the Heimdall workspace directory.
    command:
        Command to start the MCP server (default ``"npx"``).
    args:
        Arguments for the MCP server command.
    env:
        Environment variables for the MCP server process.
    filter_tools:
        If ``True``, only include execution and workspace tools.
        If ``False``, include all tools from the server.

    Returns
    -------
    tuple[list, Any]
        ``(tools, exit_stack)`` — tools are ready to add to an agent,
        and ``exit_stack`` must be closed for cleanup.

    Raises
    ------
    ImportError
        If ``google.adk.tools.mcp_tool`` is not available.
    RuntimeError
        If connection to the Heimdall MCP server fails.
    """
    mcp_module = _import_mcp_toolset_module()

    server_args = args or ["@heimdall-ai/heimdall"]
    server_env = {"HEIMDALL_WORKSPACE": workspace_path}
    if env:
        server_env.update(env)

    legacy_toolset = getattr(mcp_module, "MCPToolset", None)
    legacy_from_server = getattr(legacy_toolset, "from_server", None)

    if callable(legacy_from_server):
        try:
            stdio_server_parameters = mcp_module.StdioServerParameters
            tools, exit_stack = await legacy_from_server(
                connection_params=stdio_server_parameters(
                    command=command,
                    args=server_args,
                    env=server_env,
                )
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Heimdall MCP server: {e}. "
                "Ensure @heimdall-ai/heimdall is installed (npm i -g @heimdall-ai/heimdall)."
            ) from e

        async def cleanup() -> None:
            """Close the MCP server connection."""
            try:
                await exit_stack.aclose()
            except Exception:
                logger.exception("Error closing Heimdall MCP connection")

    else:
        try:
            mcp_toolset = mcp_module.McpToolset
            stdio_connection_params = mcp_module.StdioConnectionParams
            stdio_server_parameters = mcp_module.StdioServerParameters

            toolset = mcp_toolset(
                connection_params=stdio_connection_params(
                    server_params=stdio_server_parameters(
                        command=command,
                        args=server_args,
                        env=server_env,
                    ),
                    timeout=60.0,
                )
            )
            tools = await toolset.get_tools()
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Heimdall MCP server: {e}. "
                "Ensure @heimdall-ai/heimdall is installed (npm i -g @heimdall-ai/heimdall)."
            ) from e

        async def cleanup() -> None:
            """Close the MCP server connection."""
            try:
                await toolset.close()
            except Exception:
                logger.exception("Error closing Heimdall MCP connection")

    if filter_tools:
        allowed_names = HEIMDALL_TOOL_NAMES | HEIMDALL_WORKSPACE_TOOL_NAMES
        tools = [t for t in tools if getattr(t, "name", "") in allowed_names]

    logger.info(
        "Connected to Heimdall MCP server with %d tools: %s",
        len(tools),
        [getattr(t, "name", "?") for t in tools],
    )

    return tools, cleanup


async def get_heimdall_tools_from_config(
    config: dict[str, Any],
) -> tuple[list[Any], Any]:
    """Connect to a Heimdall MCP server using a custom configuration dict.

    The *config* dict configures an ADK ``McpToolset`` instance.
    Supports stdio, SSE, and streamable HTTP transports.

    Parameters
    ----------
    config:
        Configuration dict. For stdio: ``{"command": ..., "args": ..., "env": ...}``.
        For SSE: ``{"uri": "http://..."}``.

    Returns
    -------
    tuple[list, Any]
        ``(tools, cleanup)`` — same as ``get_heimdall_tools()``.
    """
    mcp_module = _import_mcp_toolset_module()

    legacy_toolset = getattr(mcp_module, "MCPToolset", None)
    legacy_from_server = getattr(legacy_toolset, "from_server", None)

    if callable(legacy_from_server):
        try:
            stdio_server_parameters = mcp_module.StdioServerParameters
            if "uri" in config:
                sse_server_params = mcp_module.SseServerParams
                connection_params = sse_server_params(url=config["uri"])
            else:
                connection_params = stdio_server_parameters(
                    command=config.get("command", "npx"),
                    args=config.get("args", ["@heimdall-ai/heimdall"]),
                    env=config.get("env", {}),
                )

            tools, exit_stack = await legacy_from_server(connection_params=connection_params)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to MCP server: {e}") from e

        async def cleanup() -> None:
            """Close the MCP server connection."""
            try:
                await exit_stack.aclose()
            except Exception:
                logger.exception("Error closing MCP connection")

    else:
        if "uri" in config:
            # Supports both /sse and streamable HTTP endpoints.
            uri = str(config["uri"])
            if uri.rstrip("/").endswith("/sse"):
                sse_connection_params = mcp_module.SseConnectionParams
                connection_params = sse_connection_params(url=uri, timeout=60.0)
            else:
                streamable_http_connection_params = mcp_module.StreamableHTTPConnectionParams
                connection_params = streamable_http_connection_params(url=uri, timeout=60.0)
        else:
            # Stdio transport
            stdio_connection_params = mcp_module.StdioConnectionParams
            stdio_server_parameters = mcp_module.StdioServerParameters
            connection_params = stdio_connection_params(
                server_params=stdio_server_parameters(
                    command=config.get("command", "npx"),
                    args=config.get("args", ["@heimdall-ai/heimdall"]),
                    env=config.get("env", {}),
                ),
                timeout=60.0,
            )

        try:
            mcp_toolset = mcp_module.McpToolset
            toolset = mcp_toolset(
                connection_params=connection_params,
            )
            tools = await toolset.get_tools()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to MCP server: {e}") from e

        async def cleanup() -> None:
            """Close the MCP server connection."""
            try:
                await toolset.close()
            except Exception:
                logger.exception("Error closing MCP connection")

    logger.info(
        "Connected to MCP server with %d tools: %s",
        len(tools),
        [getattr(t, "name", "?") for t in tools],
    )

    return tools, cleanup
