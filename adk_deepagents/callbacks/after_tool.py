"""After-tool callback — large result eviction.

When a tool result exceeds the token threshold, the full content is saved
to the backend and replaced with a preview. This prevents context window
bloat from large file reads or grep results.

Ported from deepagents.middleware.filesystem large result eviction logic.

**ADK limitation:** ADK's ``after_tool_callback`` receives ``(tool, args,
tool_context)`` but does NOT receive the tool's return value. Therefore,
large-result eviction for built-in filesystem tools is handled inline
within the tool functions via ``truncate_if_too_long()``.

This callback handles eviction for *custom/external tools* by tracking
the last tool result in ``tool_context.state`` (a cooperative pattern:
tools that opt in can store their raw result under ``_last_tool_result``
before returning the truncated version, allowing this callback to save
the full content).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from google.adk.tools import BaseTool, ToolContext

from adk_deepagents.backends.protocol import BackendFactory
from adk_deepagents.backends.utils import (
    NUM_CHARS_PER_TOKEN,
    TOOL_RESULT_TOKEN_LIMIT,
    create_content_preview,
    sanitize_tool_call_id,
)

logger = logging.getLogger(__name__)

# Tools whose results should NOT be evicted even if large.
# These either have built-in truncation or would cause issues if evicted.
TOOLS_EXCLUDED_FROM_EVICTION = frozenset(
    {"ls", "glob", "grep", "read_file", "edit_file", "write_file"}
)

# Sentinel key for cooperative large-result tracking
LAST_TOOL_RESULT_KEY = "_last_tool_result"

# Template for the replacement message when a result is evicted
TOO_LARGE_TOOL_MSG = """\
Tool result was too large ({char_count} characters, ~{token_count} tokens).
Full result saved to: {file_path}

Preview (first and last lines):
{preview}"""


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
        # Skip eviction for excluded tools (they handle truncation inline)
        tool_name = getattr(tool, "name", "")
        if tool_name in TOOLS_EXCLUDED_FROM_EVICTION:
            return None

        # Check for cooperative large-result tracking
        state = tool_context.state
        raw_result = state.pop(LAST_TOOL_RESULT_KEY, None)
        if raw_result is None:
            return None

        # Check if result exceeds token threshold
        result_str = str(raw_result)
        if len(result_str) <= char_limit:
            return None

        # Evict: save full result to backend and return preview
        if backend_factory is None:
            # No backend — just log and let inline truncation handle it
            logger.debug(
                "Large result from %s (%d chars) but no backend for eviction",
                tool_name,
                len(result_str),
            )
            return None

        backend = backend_factory(state)
        call_id = getattr(tool_context, "function_call_id", "") or "unknown"
        safe_id = sanitize_tool_call_id(call_id)
        file_path = f"/large_tool_results/{tool_name}_{safe_id}"

        try:
            backend.write(file_path, result_str)
        except Exception:
            logger.exception("Failed to evict large result to %s", file_path)
            return None

        preview = create_content_preview(result_str, max_lines=10)
        char_count = len(result_str)
        token_count = char_count // NUM_CHARS_PER_TOKEN

        return {
            "status": "result_too_large",
            "saved_to": file_path,
            "message": TOO_LARGE_TOOL_MSG.format(
                char_count=char_count,
                token_count=token_count,
                file_path=file_path,
                preview=preview,
            ),
        }

    return after_tool_callback
