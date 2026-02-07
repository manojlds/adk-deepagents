"""Before-agent callback — memory loading and dangling tool call patching.

Composes:
1. Patch dangling tool calls (from PatchToolCallsMiddleware)
2. Load memory files (from MemoryMiddleware.before_agent)
"""

from __future__ import annotations

from collections.abc import Callable

from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from adk_deepagents.backends.protocol import Backend, BackendFactory


def _load_memory_files(
    backend: Backend,
    sources: list[str],
) -> dict[str, str]:
    """Load memory files from the backend."""
    contents: dict[str, str] = {}
    results = backend.download_files(sources)
    for resp in results:
        if resp.content is not None:
            contents[resp.path] = resp.content.decode("utf-8")
    return contents


def make_before_agent_callback(
    *,
    memory_sources: list[str] | None = None,
    backend_factory: BackendFactory | None = None,
) -> Callable:
    """Create a ``before_agent_callback``.

    Parameters
    ----------
    memory_sources:
        Paths to AGENTS.md files to load into state.
    backend_factory:
        Factory to create a backend from session state (for memory loading).
    """

    def before_agent_callback(
        callback_context: CallbackContext,
    ) -> types.Content | None:
        state = callback_context.state

        # 1. Load memory files (once per session)
        if memory_sources and backend_factory and "memory_contents" not in state:
            backend = backend_factory(state)
            contents = _load_memory_files(backend, memory_sources)
            state["memory_contents"] = contents

        return None  # Continue with normal agent execution

    return before_agent_callback
