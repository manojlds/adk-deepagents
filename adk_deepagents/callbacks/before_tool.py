"""Before-tool callback — human-in-the-loop approval via ADK ToolConfirmation.

Uses ADK's native ``tool_context.request_confirmation()`` to truly pause the
agent execution loop.  When a tool that requires approval is invoked:

1. **First invocation** (``tool_confirmation is None``):
   ``request_confirmation()`` is called, halting the agent.  The approval
   request includes a unique ``approval_id`` (the ``function_call_id``),
   the tool name, and its arguments.

2. **Resumed invocation** (``tool_confirmation`` is present):
   - ``confirmed=True``:  the tool proceeds.  If ``payload`` contains
     ``modified_args``, those replace the original arguments.
   - ``confirmed=False``: a rejection message is returned and the tool is
     skipped.

The ``resume_approval()`` helper creates a ``ToolConfirmation`` object that
callers (CLI, web UI, tests) can feed back to ADK to resume the agent.

Ported from deepagents.middleware.human_in_the_loop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from google.adk.tools import BaseTool, ToolContext
from google.adk.tools.tool_confirmation import ToolConfirmation


def resume_approval(
    *,
    approved: bool,
    modified_args: dict[str, Any] | None = None,
) -> ToolConfirmation:
    """Create a ``ToolConfirmation`` to resume an interrupted tool call.

    Parameters
    ----------
    approved:
        Whether the human approved the tool call.
    modified_args:
        Optional replacement arguments for the tool call.  Only used when
        ``approved=True``.

    Returns
    -------
    ToolConfirmation
        The confirmation object to pass back to ADK when resuming the agent.
    """
    payload: dict[str, Any] | None = None
    if approved and modified_args is not None:
        payload = {"modified_args": modified_args}

    return ToolConfirmation(
        confirmed=approved,
        payload=payload,
    )


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

    tools_requiring_approval = {name for name, requires in interrupt_on.items() if requires}

    if not tools_requiring_approval:
        return None

    def before_tool_callback(
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict | None:
        tool_name = getattr(tool, "name", "")

        if tool_name not in tools_requiring_approval:
            return None  # Proceed with tool execution

        # --- Confirmation already received (resume path) ------------------
        confirmation: ToolConfirmation | None = getattr(tool_context, "tool_confirmation", None)

        if confirmation is not None:
            if not confirmation.confirmed:
                # Rejected — skip tool execution
                return {
                    "status": "rejected",
                    "tool": tool_name,
                    "message": (f"Tool '{tool_name}' was rejected by the human reviewer."),
                }

            # Approved — apply modified args if provided
            if confirmation.payload and isinstance(confirmation.payload, dict):
                modified = confirmation.payload.get("modified_args")
                if isinstance(modified, dict):
                    args.update(modified)

            return None  # Proceed with (possibly modified) original args

        # --- First invocation — request confirmation (true pause) ---------
        approval_id = getattr(tool_context, "function_call_id", None) or "unknown"

        tool_context.request_confirmation(
            hint=(f"Tool '{tool_name}' requires human approval before execution."),
            payload={
                "approval_id": approval_id,
                "tool": tool_name,
                "args": args,
            },
        )
        tool_context.actions.skip_summarization = True

        return {
            "status": "awaiting_approval",
            "approval_id": approval_id,
            "tool": tool_name,
            "message": (
                f"Tool '{tool_name}' requires approval before execution. "
                "The request has been recorded."
            ),
        }

    return before_tool_callback
