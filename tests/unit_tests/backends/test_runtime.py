"""Tests for backend runtime registry (backends/runtime.py)."""

from __future__ import annotations

import threading

from adk_deepagents.backends.runtime import (
    clear_session_backend,
    get_or_create_backend_for_session,
    get_registered_backend_factory,
    register_backend_factory,
)
from adk_deepagents.backends.state import StateBackend


def _make_factory():
    def factory(state):
        return StateBackend(state)

    return factory


class TestRegisterBackendFactory:
    def test_register_and_retrieve(self):
        factory = _make_factory()
        register_backend_factory("sess-1", factory)
        assert get_registered_backend_factory("sess-1") is factory
        # Cleanup
        clear_session_backend("sess-1")

    def test_missing_session_returns_none(self):
        assert get_registered_backend_factory("nonexistent-session") is None

    def test_overwrite_registration(self):
        factory1 = _make_factory()
        factory2 = _make_factory()
        register_backend_factory("sess-2", factory1)
        register_backend_factory("sess-2", factory2)
        assert get_registered_backend_factory("sess-2") is factory2
        clear_session_backend("sess-2")


class TestGetOrCreateBackendForSession:
    def test_creates_backend_from_factory(self):
        state = {"files": {}}
        register_backend_factory("sess-3", _make_factory())
        backend = get_or_create_backend_for_session("sess-3", state)
        assert isinstance(backend, StateBackend)
        clear_session_backend("sess-3")

    def test_factory_receives_state(self):
        received_state = {}

        def tracking_factory(state):
            received_state.update(state)
            return StateBackend(state)

        state = {"files": {}, "marker": True}
        register_backend_factory("sess-4", tracking_factory)
        get_or_create_backend_for_session("sess-4", state)
        assert received_state.get("marker") is True
        clear_session_backend("sess-4")

    def test_missing_session_returns_none(self):
        assert get_or_create_backend_for_session("nonexistent", {}) is None


class TestClearSessionBackend:
    def test_clear_existing(self):
        register_backend_factory("sess-5", _make_factory())
        assert get_registered_backend_factory("sess-5") is not None
        clear_session_backend("sess-5")
        assert get_registered_backend_factory("sess-5") is None

    def test_clear_nonexistent_no_error(self):
        """Clearing a non-existent session should not raise."""
        clear_session_backend("does-not-exist")


class TestThreadSafety:
    def test_concurrent_register_and_retrieve(self):
        """Multiple threads can register and retrieve without errors."""
        errors: list[Exception] = []
        num_threads = 20
        barrier = threading.Barrier(num_threads)

        def worker(thread_id: int) -> None:
            try:
                session_id = f"thread-{thread_id}"

                def factory(state, tid=thread_id):
                    return StateBackend(state)

                barrier.wait()
                register_backend_factory(session_id, factory)
                retrieved = get_registered_backend_factory(session_id)
                assert retrieved is not None
                clear_session_backend(session_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_clear(self):
        """Multiple threads clearing the same session don't raise."""
        errors: list[Exception] = []
        register_backend_factory("shared-sess", _make_factory())

        def clearer() -> None:
            try:
                clear_session_backend("shared-sess")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=clearer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
