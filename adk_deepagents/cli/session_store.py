"""Persistent CLI thread/session helpers backed by ADK SQLite sessions."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.adk.sessions.sqlite_session_service import SqliteSessionService

CLI_SESSIONS_APP_NAME = "adk-deepagents"

THREAD_META_AGENT_KEY = "adk_deepagents.cli.agent"
THREAD_META_MODEL_KEY = "adk_deepagents.cli.model"
THREAD_META_CREATED_AT_KEY = "adk_deepagents.cli.created_at"
THREAD_META_UPDATED_AT_KEY = "adk_deepagents.cli.updated_at"


@dataclass(frozen=True)
class ThreadRecord:
    """Summary metadata for a persisted thread."""

    session_id: str
    user_id: str
    agent_name: str
    model: str | None
    created_at: float | None
    updated_at: float


def create_thread(
    *,
    db_path: Path | str,
    user_id: str,
    agent_name: str,
    model: str | None,
) -> ThreadRecord:
    """Create and persist a new thread session."""
    normalized_agent = _normalize_text(agent_name)
    if normalized_agent is None:
        raise ValueError("agent_name cannot be empty.")

    normalized_user = _normalize_text(user_id)
    if normalized_user is None:
        raise ValueError("user_id cannot be empty.")

    now = time.time()
    service = _build_service(db_path)

    session = asyncio.run(
        service.create_session(
            app_name=CLI_SESSIONS_APP_NAME,
            user_id=normalized_user,
            state={
                THREAD_META_AGENT_KEY: normalized_agent,
                THREAD_META_MODEL_KEY: _normalize_text(model),
                THREAD_META_CREATED_AT_KEY: now,
                THREAD_META_UPDATED_AT_KEY: now,
            },
        )
    )

    return _session_to_record(session, default_agent=normalized_agent)


def get_thread(*, db_path: Path | str, user_id: str, session_id: str) -> ThreadRecord | None:
    """Load a persisted thread by session id."""
    normalized_user = _normalize_text(user_id)
    if normalized_user is None:
        raise ValueError("user_id cannot be empty.")

    normalized_session = _normalize_text(session_id)
    if normalized_session is None:
        return None

    service = _build_service(db_path)
    session = asyncio.run(
        service.get_session(
            app_name=CLI_SESSIONS_APP_NAME,
            user_id=normalized_user,
            session_id=normalized_session,
        )
    )

    if session is None:
        return None

    return _session_to_record(session)


def list_threads(
    *,
    db_path: Path | str,
    user_id: str,
    agent_name: str,
    limit: int = 20,
) -> list[ThreadRecord]:
    """Return most-recent threads for a specific agent profile."""
    normalized_user = _normalize_text(user_id)
    if normalized_user is None:
        raise ValueError("user_id cannot be empty.")

    normalized_agent = _normalize_text(agent_name)
    if normalized_agent is None:
        raise ValueError("agent_name cannot be empty.")

    if limit < 1:
        return []

    service = _build_service(db_path)
    response = asyncio.run(
        service.list_sessions(
            app_name=CLI_SESSIONS_APP_NAME,
            user_id=normalized_user,
        )
    )

    threads = [
        _session_to_record(session, default_agent=normalized_agent) for session in response.sessions
    ]
    filtered = [thread for thread in threads if thread.agent_name == normalized_agent]
    filtered.sort(key=lambda thread: thread.updated_at, reverse=True)

    return filtered[:limit]


def get_latest_thread(*, db_path: Path | str, user_id: str, agent_name: str) -> ThreadRecord | None:
    """Return the most recently updated thread for an agent profile."""
    threads = list_threads(db_path=db_path, user_id=user_id, agent_name=agent_name, limit=1)
    return threads[0] if threads else None


def delete_thread(*, db_path: Path | str, user_id: str, session_id: str) -> bool:
    """Delete a persisted thread. Returns True only when a thread was removed."""
    normalized_user = _normalize_text(user_id)
    if normalized_user is None:
        raise ValueError("user_id cannot be empty.")

    existing = get_thread(db_path=db_path, user_id=normalized_user, session_id=session_id)
    if existing is None:
        return False

    service = _build_service(db_path)
    asyncio.run(
        service.delete_session(
            app_name=CLI_SESSIONS_APP_NAME,
            user_id=normalized_user,
            session_id=existing.session_id,
        )
    )
    return True


def _build_service(db_path: Path | str) -> SqliteSessionService:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSessionService(str(path))


def _session_to_record(session: Any, default_agent: str | None = None) -> ThreadRecord:
    state = session.state if isinstance(session.state, dict) else {}

    agent_name = (
        _normalize_text(state.get(THREAD_META_AGENT_KEY)) or default_agent or session.user_id
    )
    model = _normalize_text(state.get(THREAD_META_MODEL_KEY))

    created_at = _normalize_timestamp(state.get(THREAD_META_CREATED_AT_KEY))
    updated_at = _normalize_timestamp(state.get(THREAD_META_UPDATED_AT_KEY))
    if updated_at is None:
        updated_at = float(session.last_update_time)

    return ThreadRecord(
        session_id=session.id,
        user_id=session.user_id,
        agent_name=agent_name,
        model=model,
        created_at=created_at,
        updated_at=updated_at,
    )


def _normalize_text(value: Any) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def _normalize_timestamp(value: Any) -> float | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None

    return None
