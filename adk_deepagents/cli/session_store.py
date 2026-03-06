"""Persistent thread/session helpers backed by ADK SQLite sessions."""

from __future__ import annotations

import asyncio
import getpass
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from google.adk.sessions.session import Session
from google.adk.sessions.sqlite_session_service import SqliteSessionService

CLI_APP_NAME = "adk_deepagents_cli"
CLI_USER_ENV = "ADK_DEEPAGENTS_USER"

STATE_AGENT_KEY = "_cli_agent_name"
STATE_MODEL_KEY = "_cli_model"
STATE_CREATED_AT_KEY = "_cli_created_at"


@dataclass
class ThreadInfo:
    """Summary view of a persisted thread/session."""

    session_id: str
    user_id: str
    agent_name: str
    model: str | None
    last_update_time: float

    @property
    def updated_at_iso(self) -> str:
        """Return last update timestamp in ISO format."""
        return datetime.fromtimestamp(self.last_update_time, tz=UTC).isoformat()


def get_cli_user_id() -> str:
    """Return CLI user identifier for session storage."""
    from_env = os.environ.get(CLI_USER_ENV, "").strip()
    if from_env:
        return from_env

    try:
        system_user = getpass.getuser().strip()
    except Exception:
        system_user = ""

    return system_user or "user"


def _create_service(db_path: Path) -> SqliteSessionService:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSessionService(str(db_path))


def _to_thread_info(session: Session) -> ThreadInfo:
    raw_agent = session.state.get(STATE_AGENT_KEY)
    raw_model = session.state.get(STATE_MODEL_KEY)

    return ThreadInfo(
        session_id=session.id,
        user_id=session.user_id,
        agent_name=raw_agent if isinstance(raw_agent, str) and raw_agent else "unknown",
        model=raw_model if isinstance(raw_model, str) and raw_model else None,
        last_update_time=session.last_update_time,
    )


async def _create_thread_async(
    *,
    db_path: Path,
    user_id: str,
    agent_name: str,
    model: str | None,
    app_name: str,
) -> ThreadInfo:
    service = _create_service(db_path)
    state = {
        STATE_AGENT_KEY: agent_name,
        STATE_CREATED_AT_KEY: datetime.now(tz=UTC).isoformat(),
    }
    if model:
        state[STATE_MODEL_KEY] = model

    session = await service.create_session(
        app_name=app_name,
        user_id=user_id,
        state=state,
    )
    return _to_thread_info(session)


async def _list_threads_async(
    *,
    db_path: Path,
    user_id: str,
    agent_name: str | None,
    limit: int,
    app_name: str,
) -> list[ThreadInfo]:
    service = _create_service(db_path)
    response = await service.list_sessions(app_name=app_name, user_id=user_id)

    sessions = response.sessions
    if agent_name:
        sessions = [
            session
            for session in sessions
            if isinstance(session.state.get(STATE_AGENT_KEY), str)
            and session.state.get(STATE_AGENT_KEY) == agent_name
        ]

    sessions.sort(key=lambda session: session.last_update_time, reverse=True)
    if limit > 0:
        sessions = sessions[:limit]

    return [_to_thread_info(session) for session in sessions]


async def _get_thread_async(
    *,
    db_path: Path,
    user_id: str,
    session_id: str,
    app_name: str,
) -> ThreadInfo | None:
    service = _create_service(db_path)
    session = await service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )
    if session is None:
        return None
    return _to_thread_info(session)


async def _delete_thread_async(
    *,
    db_path: Path,
    user_id: str,
    session_id: str,
    app_name: str,
) -> bool:
    service = _create_service(db_path)
    existing = await service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )
    if existing is None:
        return False

    await service.delete_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )
    return True


def create_thread(
    *,
    db_path: Path,
    user_id: str,
    agent_name: str,
    model: str | None,
    app_name: str = CLI_APP_NAME,
) -> ThreadInfo:
    """Create and return a new thread for ``agent_name``."""
    return asyncio.run(
        _create_thread_async(
            db_path=db_path,
            user_id=user_id,
            agent_name=agent_name,
            model=model,
            app_name=app_name,
        )
    )


def list_threads(
    *,
    db_path: Path,
    user_id: str,
    agent_name: str | None = None,
    limit: int = 20,
    app_name: str = CLI_APP_NAME,
) -> list[ThreadInfo]:
    """List threads for a user, optionally filtered by ``agent_name``."""
    return asyncio.run(
        _list_threads_async(
            db_path=db_path,
            user_id=user_id,
            agent_name=agent_name,
            limit=limit,
            app_name=app_name,
        )
    )


def get_thread(
    *,
    db_path: Path,
    user_id: str,
    session_id: str,
    app_name: str = CLI_APP_NAME,
) -> ThreadInfo | None:
    """Return a specific thread by ``session_id``."""
    return asyncio.run(
        _get_thread_async(
            db_path=db_path,
            user_id=user_id,
            session_id=session_id,
            app_name=app_name,
        )
    )


def get_latest_thread(
    *,
    db_path: Path,
    user_id: str,
    agent_name: str | None = None,
    app_name: str = CLI_APP_NAME,
) -> ThreadInfo | None:
    """Return most recently updated thread, optionally filtered by agent."""
    threads = list_threads(
        db_path=db_path,
        user_id=user_id,
        agent_name=agent_name,
        limit=1,
        app_name=app_name,
    )
    return threads[0] if threads else None


def delete_thread(
    *,
    db_path: Path,
    user_id: str,
    session_id: str,
    app_name: str = CLI_APP_NAME,
) -> bool:
    """Delete a thread by ``session_id``. Returns True if deleted."""
    return asyncio.run(
        _delete_thread_async(
            db_path=db_path,
            user_id=user_id,
            session_id=session_id,
            app_name=app_name,
        )
    )
