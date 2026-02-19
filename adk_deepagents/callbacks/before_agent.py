"""Before-agent callback — memory loading and dangling tool call patching.

Composes:
1. Patch dangling tool calls (from PatchToolCallsMiddleware)
2. Load memory files (from MemoryMiddleware.before_agent)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast

from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from adk_deepagents.backends.protocol import Backend, BackendFactory
from adk_deepagents.backends.runtime import register_backend_factory

logger = logging.getLogger(__name__)


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


def _patch_dangling_tool_calls(
    callback_context: CallbackContext,
) -> bool:
    """Scan message history for orphaned tool calls and patch them.

    Ported from deepagents ``PatchToolCallsMiddleware``.

    An orphaned tool call is a ``function_call`` part in a model message
    that has no corresponding ``function_response`` part anywhere later
    in the message history.  This can happen when the agent is interrupted,
    a tool execution crashes, or a session is resumed from a checkpoint.

    Without patching, these dangling calls cause LLM API errors because
    most models require every function_call to have a matching response.

    Returns ``True`` if any patching was performed.
    """
    # Access the session events through the invocation context
    # ADK stores conversation messages on the session
    session = getattr(callback_context, "session", None)
    if session is None:
        return False

    events = getattr(session, "events", None)
    if events is None:
        return False

    # Collect all function_call ids and all function_response ids
    call_ids: dict[str, tuple[str, int]] = {}  # id -> (name, event_index)
    response_ids: set[str] = set()

    for idx, event in enumerate(events):
        content = getattr(event, "content", None)
        if content is None or not content.parts:
            continue
        for part in content.parts:
            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "id", None):
                call_ids[fc.id] = (getattr(fc, "name", "unknown"), idx)
            fr = getattr(part, "function_response", None)
            if fr is not None and getattr(fr, "id", None):
                response_ids.add(fr.id)

    # Find dangling calls (calls without responses)
    dangling = {cid: info for cid, info in call_ids.items() if cid not in response_ids}

    if not dangling:
        return False

    # We can't directly inject events into the session in all ADK backends,
    # so we store the dangling call info in state for before_model_callback
    # to inject synthetic function_response parts into the LLM request.
    state = callback_context.state
    dangling_info = []
    for cid, (name, _idx) in dangling.items():
        dangling_info.append({"id": cid, "name": name})

    state["_dangling_tool_calls"] = dangling_info
    logger.info("Found %d dangling tool call(s), stored for patching", len(dangling))
    return True


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
        session = getattr(callback_context, "session", None)

        # 0. Register backend factory for this ADK session without writing it
        #    into state (functions are not JSON-serializable in sqlite-backed
        #    session services used by adk web/api_server).
        if backend_factory is not None:
            session_id = getattr(session, "id", None)
            if isinstance(session_id, str) and session_id:
                register_backend_factory(session_id, backend_factory)

        # 1. Patch dangling tool calls
        _patch_dangling_tool_calls(callback_context)

        # 2. Load memory files (once per session)
        if memory_sources and backend_factory and "memory_contents" not in state:
            session_state = getattr(session, "state", None)
            if isinstance(session_state, dict):
                backend = backend_factory(session_state)
            else:
                backend = backend_factory(cast(dict, state))
            contents = _load_memory_files(backend, memory_sources)
            state["memory_contents"] = contents

        return None  # Continue with normal agent execution

    return before_agent_callback
