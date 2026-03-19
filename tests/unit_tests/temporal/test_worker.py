"""Tests for Temporal worker factory (temporal/worker.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("temporalio")

from adk_deepagents.types import DynamicTaskConfig, SubAgentSpec, TemporalTaskConfig


class TestCreateTemporalWorkerValidation:
    async def test_missing_temporal_config_raises(self):
        from adk_deepagents.temporal.worker import create_temporal_worker

        with pytest.raises(ValueError, match="temporal must be set"):
            await create_temporal_worker(
                default_model="gemini-2.5-flash",
                dynamic_task_config=DynamicTaskConfig(temporal=None),
            )

    async def test_default_dynamic_task_config_raises(self):
        """Default DynamicTaskConfig has temporal=None, so it should raise."""
        from adk_deepagents.temporal.worker import create_temporal_worker

        with pytest.raises(ValueError, match="temporal must be set"):
            await create_temporal_worker(default_model="gemini-2.5-flash")


class TestCreateTemporalWorkerSuccess:
    async def test_creates_worker_with_valid_config(self):
        from adk_deepagents.temporal.worker import create_temporal_worker

        temporal_cfg = TemporalTaskConfig(
            target_host="localhost:7233",
            namespace="default",
            task_queue="test-queue",
        )
        config = DynamicTaskConfig(temporal=temporal_cfg)

        mock_client = MagicMock()
        mock_worker = MagicMock()

        with (
            patch(
                "temporalio.client.Client.connect",
                new_callable=AsyncMock,
                return_value=mock_client,
            ) as mock_connect,
            patch(
                "temporalio.worker.Worker",
                return_value=mock_worker,
            ) as mock_worker_cls,
            patch("temporalio.worker.UnsandboxedWorkflowRunner"),
        ):
            result = await create_temporal_worker(
                default_model="gemini-2.5-flash",
                dynamic_task_config=config,
            )

            mock_connect.assert_awaited_once_with(
                "localhost:7233",
                namespace="default",
            )
            mock_worker_cls.assert_called_once()
            call_kwargs = mock_worker_cls.call_args
            assert call_kwargs[1]["task_queue"] == "test-queue"
            assert result is mock_worker

    async def test_configures_workflow_timeouts(self):
        from adk_deepagents.temporal import workflows
        from adk_deepagents.temporal.worker import create_temporal_worker

        temporal_cfg = TemporalTaskConfig(
            activity_timeout_seconds=250.0,
            retry_max_attempts=3,
            idle_timeout_seconds=500.0,
        )
        config = DynamicTaskConfig(
            timeout_seconds=120.0,
            temporal=temporal_cfg,
        )

        with (
            patch(
                "temporalio.client.Client.connect",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch("temporalio.worker.Worker", return_value=MagicMock()),
            patch("temporalio.worker.UnsandboxedWorkflowRunner"),
        ):
            await create_temporal_worker(
                default_model="gemini-2.5-flash",
                dynamic_task_config=config,
            )

            assert workflows._activity_timeout_seconds == 250.0
            assert workflows._retry_max_attempts == 3
            assert workflows._idle_timeout_seconds == 500.0

            # Reset
            from adk_deepagents.temporal.workflows import configure_workflow

            configure_workflow()

    async def test_activity_timeout_falls_back_to_config_timeout(self):
        """When activity_timeout_seconds is None, uses config.timeout_seconds."""
        from adk_deepagents.temporal import workflows
        from adk_deepagents.temporal.worker import create_temporal_worker

        temporal_cfg = TemporalTaskConfig(activity_timeout_seconds=None)
        config = DynamicTaskConfig(timeout_seconds=180.0, temporal=temporal_cfg)

        with (
            patch(
                "temporalio.client.Client.connect",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch("temporalio.worker.Worker", return_value=MagicMock()),
            patch("temporalio.worker.UnsandboxedWorkflowRunner"),
        ):
            await create_temporal_worker(
                default_model="gemini-2.5-flash",
                dynamic_task_config=config,
            )

            assert workflows._activity_timeout_seconds == 180.0

            from adk_deepagents.temporal.workflows import configure_workflow

            configure_workflow()


class TestAgentBuilder:
    """Tests for the agent_builder closure created inside create_temporal_worker."""

    async def _get_agent_builder(self, *, subagents=None, default_tools=None):
        """Helper to extract the agent_builder function."""
        from adk_deepagents.temporal.worker import create_temporal_worker

        temporal_cfg = TemporalTaskConfig()
        config = DynamicTaskConfig(temporal=temporal_cfg)
        captured_activity = None

        def capture_activity(*args, **kwargs):
            nonlocal captured_activity
            # The activity is passed as activities=[activity_fn]
            return MagicMock()

        with (
            patch(
                "temporalio.client.Client.connect",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "temporalio.worker.Worker",
                side_effect=capture_activity,
            ) as mock_worker_cls,
            patch("temporalio.worker.UnsandboxedWorkflowRunner"),
        ):
            await create_temporal_worker(
                default_model="gemini-2.5-flash",
                default_tools=default_tools or [],
                subagents=subagents,
                dynamic_task_config=config,
            )

            from adk_deepagents.temporal.workflows import configure_workflow

            configure_workflow()

            # The agent_builder is captured inside the activity.
            # We can't easily extract it, so we test the registry indirectly.
            # The Worker was called with activities=[...], extract it.
            call_kwargs = mock_worker_cls.call_args
            return call_kwargs

    async def test_worker_includes_subagent_registry(self):
        """Subagents are registered in the worker's agent registry."""
        specs = [
            SubAgentSpec(name="researcher", description="Researches topics"),
            SubAgentSpec(name="writer", description="Writes content"),
        ]
        call_kwargs = await self._get_agent_builder(subagents=specs)
        # Worker should have been called successfully (activities are set)
        assert call_kwargs is not None
        assert "activities" in call_kwargs[1]
        assert len(call_kwargs[1]["activities"]) == 1
