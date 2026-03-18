"""Tests for Temporal configuration types."""

from __future__ import annotations

from adk_deepagents.types import DynamicTaskConfig, TemporalTaskConfig


def test_temporal_task_config_defaults():
    config = TemporalTaskConfig()
    assert config.target_host == "localhost:7233"
    assert config.namespace == "default"
    assert config.task_queue == "adk-deepagents-tasks"
    assert config.workflow_id_prefix == "dynamic-task"
    assert config.activity_timeout_seconds is None
    assert config.retry_max_attempts == 1
    assert config.idle_timeout_seconds == 600.0


def test_temporal_task_config_custom_values():
    config = TemporalTaskConfig(
        target_host="temporal.prod:7233",
        namespace="production",
        task_queue="task-queue",
        workflow_id_prefix="my-prefix",
        activity_timeout_seconds=300.0,
        retry_max_attempts=3,
    )
    assert config.target_host == "temporal.prod:7233"
    assert config.namespace == "production"
    assert config.task_queue == "task-queue"
    assert config.workflow_id_prefix == "my-prefix"
    assert config.activity_timeout_seconds == 300.0
    assert config.retry_max_attempts == 3


def test_dynamic_task_temporal_is_none_by_default():
    config = DynamicTaskConfig()
    assert config.temporal is None


def test_dynamic_task_temporal_can_be_enabled():
    config = DynamicTaskConfig(temporal=TemporalTaskConfig(task_queue="custom"))
    assert config.temporal is not None
    assert config.temporal.task_queue == "custom"
