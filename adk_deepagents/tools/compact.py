"""Manual conversation compaction tool.

The tool does not summarize immediately. Instead, it marks the session state so
the next ``before_model_callback`` forces one summarization pass.
"""

from __future__ import annotations

from typing import Any, cast

from google.adk.tools.tool_context import ToolContext

from adk_deepagents.types import SummarizationConfig

COMPACT_CONVERSATION_REQUEST_KEY = "_compact_conversation_requested"


def create_compact_conversation_tool(
    *,
    summarization_config: SummarizationConfig,
):
    """Create a ``compact_conversation`` tool.

    The tool stores a state flag consumed by ``before_model_callback``. On the
    next model invocation, summarization runs in forced mode.
    """
    keep_kind, keep_value = summarization_config.keep

    def compact_conversation(tool_context: ToolContext) -> dict:
        """Request conversation compaction on the next model turn."""
        state = cast(dict[str, Any], tool_context.state)
        if state.get(COMPACT_CONVERSATION_REQUEST_KEY):
            return {
                "status": "queued",
                "message": "Conversation compaction is already queued.",
            }

        state[COMPACT_CONVERSATION_REQUEST_KEY] = True
        return {
            "status": "queued",
            "message": (
                "Conversation compaction queued. The next model turn will summarize "
                f"older context (keep={keep_kind}:{keep_value})."
            ),
        }

    compact_conversation.__name__ = "compact_conversation"
    return compact_conversation
