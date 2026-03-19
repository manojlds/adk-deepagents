"""Tool error handler — wraps tools to catch exceptions gracefully.

When a tool raises an exception, the wrapper catches it and returns a
structured error dict to the LLM, enabling self-correction instead of
crashing the agent loop.

Ported from OpenSWE's ToolErrorMiddleware pattern.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import traceback
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Maximum traceback lines to include in the error message
MAX_TRACEBACK_LINES = 20

# Tools that already handle their own errors internally
TOOLS_WITH_INTERNAL_ERROR_HANDLING = frozenset(
    {"ls", "read_file", "write_file", "edit_file", "glob", "grep"}
)


def _format_error(exc: Exception, max_tb_lines: int = MAX_TRACEBACK_LINES) -> dict[str, str]:
    """Format an exception into a structured error dict."""
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tb_text = "".join(tb_lines)
    # Truncate traceback if too long
    tb_split = tb_text.split("\n")
    if len(tb_split) > max_tb_lines:
        tb_text = "\n".join(tb_split[:max_tb_lines]) + "\n... (truncated)"

    return {
        "status": "error",
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": tb_text,
    }


def wrap_tool_with_error_handler(
    fn: Callable,
    *,
    max_traceback_lines: int = MAX_TRACEBACK_LINES,
) -> Callable:
    """Wrap a tool function to catch exceptions and return error dicts.

    Preserves the function's ``__name__``, ``__doc__``, and signature so
    ADK tool introspection continues to work.

    Parameters
    ----------
    fn:
        The tool function to wrap.
    max_traceback_lines:
        Maximum number of traceback lines in the error response.
    """
    tool_name = getattr(fn, "__name__", str(fn))

    # Skip tools that already handle errors internally
    if tool_name in TOOLS_WITH_INTERNAL_ERROR_HANDLING:
        return fn

    if asyncio.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                logger.warning(
                    "Tool %s raised %s: %s",
                    tool_name,
                    type(exc).__name__,
                    exc,
                )
                return _format_error(exc, max_traceback_lines)

        # ADK uses typing.get_type_hints() on the wrapped function to build
        # its tool declaration.  functools.wraps copies __annotations__ and
        # __globals__ from the original function.  When the original uses
        # `from __future__ import annotations`, annotation strings (e.g.
        # "ToolContext") are evaluated in the *wrapper's* module globals,
        # which lacks those imports, causing NameError.
        #
        # Fix: copy the original function's __globals__ onto the wrapper so
        # deferred annotations can be resolved in the correct namespace.
        async_wrapper.__globals__.update(fn.__globals__)  # type: ignore[union-attr]
        return async_wrapper

    @functools.wraps(fn)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.warning(
                "Tool %s raised %s: %s",
                tool_name,
                type(exc).__name__,
                exc,
            )
            return _format_error(exc, max_traceback_lines)

    sync_wrapper.__globals__.update(fn.__globals__)  # type: ignore[union-attr]
    return sync_wrapper


def wrap_tools_with_error_handler(
    tools: list[Callable],
    *,
    max_traceback_lines: int = MAX_TRACEBACK_LINES,
) -> list[Callable]:
    """Wrap a list of tools with error handlers.

    Returns a new list; the original is not modified.
    """
    return [wrap_tool_with_error_handler(t, max_traceback_lines=max_traceback_lines) for t in tools]
