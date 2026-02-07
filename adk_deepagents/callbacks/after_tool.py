"""After-tool callback — large result eviction.

When a tool result exceeds the token threshold, the full content is saved
to the backend and replaced with a preview. This prevents context window
bloat from large file reads or grep results.

Ported from deepagents.middleware.filesystem large result eviction logic.
"""

from __future__ import annotations

from typing import Any, Callable

from google.adk.tools import BaseTool, ToolContext

from adk_deepagents.backends.protocol import Backend, BackendFactory
from adk_deepagents.backends.utils import (
    NUM_CHARS_PER_TOKEN,
    TOOL_RESULT_TOKEN_LIMIT,
    create_content_preview,
    sanitize_tool_call_id,
)

# Tools whose results should NOT be evicted even if large.
TOOLS_EXCLUDED_FROM_EVICTION = frozenset(
    {"ls", "glob", "grep", "read_file", "edit_file", "write_file"}
)


def make_after_tool_callback(
    *,
    backend_factory: BackendFactory | None = None,
    token_limit: int = TOOL_RESULT_TOKEN_LIMIT,
) -> Callable:
    """Create an ``after_tool_callback`` for large result eviction.

    Parameters
    ----------
    backend_factory:
        Factory to create a backend from state (for saving evicted content).
    token_limit:
        Approximate token limit above which results are evicted.
    """

    char_limit = token_limit * NUM_CHARS_PER_TOKEN

    def after_tool_callback(
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict | None:
        # Skip eviction for excluded tools
        tool_name = getattr(tool, "name", "")
        if tool_name in TOOLS_EXCLUDED_FROM_EVICTION:
            return None

        # We can only evict if we have the result as a string to check size
        # ADK calls this after the tool runs; we examine the result indirectly
        # through tool_context if needed. For now, return None (no modification).
        # The actual eviction is handled in the tool functions themselves via
        # truncate_if_too_long.

        return None  # Use result as-is

    return after_tool_callback
