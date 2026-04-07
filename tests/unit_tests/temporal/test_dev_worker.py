"""Unit tests for Temporal development worker entrypoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adk_deepagents.types import DynamicTaskConfig, TemporalTaskConfig


class TestResolveWorkerModel:
    def test_prefers_temporal_worker_model_env(self, monkeypatch) -> None:
        from adk_deepagents.temporal.dev_worker import _resolve_worker_model

        monkeypatch.setenv("ADK_DEEPAGENTS_TEMPORAL_WORKER_MODEL", "model.worker")
        monkeypatch.setenv("ADK_DEEPAGENTS_MODEL", "model.general")

        assert _resolve_worker_model() == "model.worker"

    def test_uses_general_model_env_when_worker_model_missing(self, monkeypatch) -> None:
        from adk_deepagents.temporal.dev_worker import _resolve_worker_model

        monkeypatch.delenv("ADK_DEEPAGENTS_TEMPORAL_WORKER_MODEL", raising=False)
        monkeypatch.setenv("ADK_DEEPAGENTS_MODEL", "model.general")

        assert _resolve_worker_model() == "model.general"

    def test_falls_back_to_default_model(self, monkeypatch) -> None:
        from adk_deepagents.temporal.dev_worker import _resolve_worker_model

        monkeypatch.delenv("ADK_DEEPAGENTS_TEMPORAL_WORKER_MODEL", raising=False)
        monkeypatch.delenv("ADK_DEEPAGENTS_MODEL", raising=False)

        assert _resolve_worker_model() == "gemini-2.5-flash"


class TestRunWorker:
    async def test_requires_temporal_config(self) -> None:
        from adk_deepagents.temporal.dev_worker import run_worker

        with (
            patch("adk_deepagents.temporal.dev_worker._load_workspace_env"),
            patch(
                "adk_deepagents.temporal.dev_worker.build_dynamic_task_config",
                return_value=DynamicTaskConfig(),
            ),
            pytest.raises(RuntimeError, match="ADK_DEEPAGENTS_TEMPORAL"),
        ):
            await run_worker()

    async def test_creates_worker_with_default_tools(self, monkeypatch) -> None:
        from adk_deepagents.temporal.dev_worker import run_worker

        worker = MagicMock()
        worker.run = AsyncMock()
        create_worker = AsyncMock(return_value=worker)

        monkeypatch.setenv("ADK_DEEPAGENTS_TEMPORAL_WORKER_MODEL", "model.worker")

        with (
            patch("adk_deepagents.temporal.dev_worker._load_workspace_env"),
            patch(
                "adk_deepagents.temporal.dev_worker.build_dynamic_task_config",
                return_value=DynamicTaskConfig(temporal=TemporalTaskConfig()),
            ),
            patch(
                "adk_deepagents.temporal.dev_worker.create_temporal_worker",
                create_worker,
            ),
        ):
            await run_worker()

        create_worker.assert_awaited_once()
        await_args = create_worker.await_args
        assert await_args is not None
        call_kwargs = await_args.kwargs
        assert call_kwargs["default_model"] == "model.worker"
        assert [tool.__name__ for tool in call_kwargs["default_tools"]] == [
            "write_todos",
            "read_todos",
            "ls",
            "read_file",
            "write_file",
            "edit_file",
            "glob",
            "grep",
        ]
        worker.run.assert_awaited_once()

    async def test_health_port_server_is_started_and_closed(self) -> None:
        from adk_deepagents.temporal.dev_worker import run_worker

        worker = MagicMock()
        worker.run = AsyncMock()
        create_worker = AsyncMock(return_value=worker)

        mock_server = MagicMock()
        mock_server.wait_closed = AsyncMock()
        start_server = AsyncMock(return_value=mock_server)

        with (
            patch("adk_deepagents.temporal.dev_worker._load_workspace_env"),
            patch(
                "adk_deepagents.temporal.dev_worker.build_dynamic_task_config",
                return_value=DynamicTaskConfig(temporal=TemporalTaskConfig()),
            ),
            patch("adk_deepagents.temporal.dev_worker.create_temporal_worker", create_worker),
            patch("adk_deepagents.temporal.dev_worker.asyncio.start_server", start_server),
        ):
            await run_worker(health_port=17451)

        start_server.assert_awaited_once()
        await_args = start_server.await_args
        assert await_args is not None
        assert await_args.kwargs["host"] == "127.0.0.1"
        assert await_args.kwargs["port"] == 17451
        mock_server.close.assert_called_once()
        mock_server.wait_closed.assert_awaited_once()
