"""Runtime backend registry keyed by ADK session id.

This avoids storing non-serializable backend factories/instances in session
state, which is required for sqlite-backed ADK session services.
"""

from __future__ import annotations

from threading import Lock

from adk_deepagents.backends.protocol import Backend, BackendFactory

_lock = Lock()
_backend_factory_by_session: dict[str, BackendFactory] = {}


def register_backend_factory(session_id: str, backend_factory: BackendFactory) -> None:
    """Register a backend factory for an ADK session id."""
    with _lock:
        _backend_factory_by_session[session_id] = backend_factory


def get_registered_backend_factory(session_id: str) -> BackendFactory | None:
    """Return registered backend factory for *session_id*, if any."""
    with _lock:
        return _backend_factory_by_session.get(session_id)


def get_or_create_backend_for_session(session_id: str, state: dict) -> Backend | None:
    """Create backend for *session_id* from the registered factory."""
    with _lock:
        factory = _backend_factory_by_session.get(session_id)

    if factory is None:
        return None
    return factory(state)


def clear_session_backend(session_id: str) -> None:
    """Remove runtime backend registry entries for *session_id*."""
    with _lock:
        _backend_factory_by_session.pop(session_id, None)
