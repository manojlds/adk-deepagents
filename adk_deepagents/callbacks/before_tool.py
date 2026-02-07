"""Before-tool callback — human-in-the-loop approval.

Ported from deepagents.middleware.human_in_the_loop.
When configured, certain tools require explicit approval before execution.
"""

from __future__ import annotations

from typing import Any, Callable

from google.adk.tools import BaseTool, ToolContext


def make_before_tool_callback(
    *,
    interrupt_on: dict[str, bool] | None = None,
) -> Callable | None:
    """Create a ``before_tool_callback`` for human-in-the-loop approval.

    Parameters
    ----------
    interrupt_on:
        Mapping of tool name → whether to require approval.
        Example: ``{"write_file": True, "execute": True}``.

    Returns
    -------
    Callable or None
        The callback function, or ``None`` if no tools need approval.
    """
    if not interrupt_on:
        return None

    tools_requiring_approval = {
        name for name, requires in interrupt_on.items() if requires
    }

    if not tools_requiring_approval:
        return None

    def before_tool_callback(
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict | None:
        tool_name = getattr(tool, "name", "")

        if tool_name in tools_requiring_approval:
            # Store pending approval in state for the caller to handle
            tool_context.state["_pending_approval"] = {
                "tool": tool_name,
                "args": args,
            }
            # Return a result that signals approval is needed,
            # effectively skipping the tool execution
            return {
                "status": "awaiting_approval",
                "tool": tool_name,
                "message": (
                    f"Tool '{tool_name}' requires approval before execution. "
                    "The request has been recorded."
                ),
            }

        return None  # Proceed with tool execution

    return before_tool_callback
