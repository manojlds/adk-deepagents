"""Unit tests for CLI thread/session persistence helpers."""

from __future__ import annotations

from pathlib import Path

from adk_deepagents.cli.session_store import (
    create_thread,
    delete_thread,
    get_latest_thread,
    get_thread,
    list_threads,
)


def test_create_and_get_thread(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"

    created = create_thread(
        db_path=db_path,
        user_id="u1",
        agent_name="demo",
        model="gemini-2.5-flash",
    )
    loaded = get_thread(
        db_path=db_path,
        user_id="u1",
        session_id=created.session_id,
    )

    assert loaded is not None
    assert loaded.session_id == created.session_id
    assert loaded.agent_name == "demo"
    assert loaded.model == "gemini-2.5-flash"


def test_list_threads_filters_by_agent(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"

    create_thread(db_path=db_path, user_id="u1", agent_name="alpha", model="m1")
    create_thread(db_path=db_path, user_id="u1", agent_name="beta", model="m2")

    alpha_threads = list_threads(db_path=db_path, user_id="u1", agent_name="alpha", limit=20)

    assert len(alpha_threads) == 1
    assert alpha_threads[0].agent_name == "alpha"


def test_get_latest_thread_returns_most_recent_for_agent(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"

    first = create_thread(db_path=db_path, user_id="u1", agent_name="alpha", model="m1")
    second = create_thread(db_path=db_path, user_id="u1", agent_name="alpha", model="m1")

    latest = get_latest_thread(db_path=db_path, user_id="u1", agent_name="alpha")

    assert latest is not None
    assert latest.session_id in {first.session_id, second.session_id}
    assert latest.session_id == second.session_id


def test_delete_thread_removes_session(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"

    created = create_thread(db_path=db_path, user_id="u1", agent_name="alpha", model="m1")

    deleted = delete_thread(db_path=db_path, user_id="u1", session_id=created.session_id)
    loaded = get_thread(db_path=db_path, user_id="u1", session_id=created.session_id)

    assert deleted is True
    assert loaded is None
