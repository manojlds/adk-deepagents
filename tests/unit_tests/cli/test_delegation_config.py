"""Unit tests for CLI dynamic delegation config helpers."""

from __future__ import annotations

from adk_deepagents.cli.config import CliDefaults
from adk_deepagents.cli.delegation_config import build_cli_dynamic_task_config


def _clear_dynamic_task_env(monkeypatch) -> None:
    for key in (
        "ADK_DYNAMIC_TASK_MAX_PARALLEL",
        "ADK_DYNAMIC_TASK_CONCURRENCY_POLICY",
        "ADK_DYNAMIC_TASK_QUEUE_TIMEOUT_SECONDS",
        "ADK_DEEPAGENTS_TEMPORAL_TARGET_HOST",
        "ADK_DEEPAGENTS_TEMPORAL_NAMESPACE",
        "ADK_DEEPAGENTS_TEMPORAL_TASK_QUEUE",
        "ADK_DEEPAGENTS_TEMPORAL_WORKFLOW_ID_PREFIX",
        "ADK_DEEPAGENTS_TEMPORAL_ACTIVITY_TIMEOUT_SECONDS",
        "ADK_DEEPAGENTS_TEMPORAL_RETRY_MAX_ATTEMPTS",
        "ADK_DEEPAGENTS_TEMPORAL_IDLE_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)


def test_build_cli_dynamic_task_config_defaults_to_wait_policy(monkeypatch) -> None:
    _clear_dynamic_task_env(monkeypatch)

    config = build_cli_dynamic_task_config()

    assert config.concurrency_policy == "wait"
    assert config.queue_timeout_seconds == 30.0
    assert config.temporal is None


def test_build_cli_dynamic_task_config_reads_env_overrides(monkeypatch) -> None:
    _clear_dynamic_task_env(monkeypatch)
    monkeypatch.setenv("ADK_DYNAMIC_TASK_MAX_PARALLEL", "6")
    monkeypatch.setenv("ADK_DYNAMIC_TASK_CONCURRENCY_POLICY", "error")
    monkeypatch.setenv("ADK_DYNAMIC_TASK_QUEUE_TIMEOUT_SECONDS", "7.5")

    config = build_cli_dynamic_task_config()

    assert config.max_parallel == 6
    assert config.concurrency_policy == "error"
    assert config.queue_timeout_seconds == 7.5


def test_build_cli_dynamic_task_config_uses_cli_defaults_when_provided(monkeypatch) -> None:
    _clear_dynamic_task_env(monkeypatch)

    defaults = CliDefaults(
        dynamic_task_max_parallel=9,
        dynamic_task_concurrency_policy="error",
        dynamic_task_queue_timeout_seconds=12.5,
    )

    config = build_cli_dynamic_task_config(defaults)

    assert config.max_parallel == 9
    assert config.concurrency_policy == "error"
    assert config.queue_timeout_seconds == 12.5


def test_build_cli_dynamic_task_config_env_overrides_cli_defaults(monkeypatch) -> None:
    _clear_dynamic_task_env(monkeypatch)
    defaults = CliDefaults(
        dynamic_task_max_parallel=9,
        dynamic_task_concurrency_policy="error",
        dynamic_task_queue_timeout_seconds=12.5,
    )

    monkeypatch.setenv("ADK_DYNAMIC_TASK_MAX_PARALLEL", "3")
    monkeypatch.setenv("ADK_DYNAMIC_TASK_CONCURRENCY_POLICY", "wait")
    monkeypatch.setenv("ADK_DYNAMIC_TASK_QUEUE_TIMEOUT_SECONDS", "1.5")

    config = build_cli_dynamic_task_config(defaults)

    assert config.max_parallel == 3
    assert config.concurrency_policy == "wait"
    assert config.queue_timeout_seconds == 1.5


def test_build_cli_dynamic_task_config_ignores_invalid_env_values(monkeypatch) -> None:
    _clear_dynamic_task_env(monkeypatch)
    monkeypatch.setenv("ADK_DYNAMIC_TASK_MAX_PARALLEL", "0")
    monkeypatch.setenv("ADK_DYNAMIC_TASK_CONCURRENCY_POLICY", "invalid")
    monkeypatch.setenv("ADK_DYNAMIC_TASK_QUEUE_TIMEOUT_SECONDS", "-5")

    config = build_cli_dynamic_task_config()

    assert config.max_parallel == 4
    assert config.concurrency_policy == "wait"
    assert config.queue_timeout_seconds == 30.0


def test_build_cli_dynamic_task_config_enables_temporal_from_env(monkeypatch) -> None:
    _clear_dynamic_task_env(monkeypatch)
    monkeypatch.setenv("ADK_DEEPAGENTS_TEMPORAL_TARGET_HOST", "127.0.0.1:7233")

    config = build_cli_dynamic_task_config()

    assert config.temporal is not None
    assert config.temporal.target_host == "127.0.0.1:7233"
    assert config.temporal.namespace == "default"
    assert config.temporal.task_queue == "adk-deepagents-tasks"


def test_build_cli_dynamic_task_config_reads_temporal_env_overrides(monkeypatch) -> None:
    _clear_dynamic_task_env(monkeypatch)
    monkeypatch.setenv("ADK_DEEPAGENTS_TEMPORAL_TARGET_HOST", "temporal.internal:7233")
    monkeypatch.setenv("ADK_DEEPAGENTS_TEMPORAL_NAMESPACE", "ci")
    monkeypatch.setenv("ADK_DEEPAGENTS_TEMPORAL_TASK_QUEUE", "custom-queue")
    monkeypatch.setenv("ADK_DEEPAGENTS_TEMPORAL_WORKFLOW_ID_PREFIX", "ci-dynamic")
    monkeypatch.setenv("ADK_DEEPAGENTS_TEMPORAL_ACTIVITY_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("ADK_DEEPAGENTS_TEMPORAL_RETRY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("ADK_DEEPAGENTS_TEMPORAL_IDLE_TIMEOUT_SECONDS", "120")

    config = build_cli_dynamic_task_config()

    assert config.temporal is not None
    assert config.temporal.target_host == "temporal.internal:7233"
    assert config.temporal.namespace == "ci"
    assert config.temporal.task_queue == "custom-queue"
    assert config.temporal.workflow_id_prefix == "ci-dynamic"
    assert config.temporal.activity_timeout_seconds == 90.0
    assert config.temporal.retry_max_attempts == 3
    assert config.temporal.idle_timeout_seconds == 120.0


def test_build_cli_dynamic_task_config_ignores_invalid_temporal_numeric_env(monkeypatch) -> None:
    _clear_dynamic_task_env(monkeypatch)
    monkeypatch.setenv("ADK_DEEPAGENTS_TEMPORAL_TARGET_HOST", "127.0.0.1:7233")
    monkeypatch.setenv("ADK_DEEPAGENTS_TEMPORAL_ACTIVITY_TIMEOUT_SECONDS", "-1")
    monkeypatch.setenv("ADK_DEEPAGENTS_TEMPORAL_RETRY_MAX_ATTEMPTS", "0")
    monkeypatch.setenv("ADK_DEEPAGENTS_TEMPORAL_IDLE_TIMEOUT_SECONDS", "-10")

    config = build_cli_dynamic_task_config()

    assert config.temporal is not None
    assert config.temporal.target_host == "127.0.0.1:7233"
    assert config.temporal.activity_timeout_seconds is None
    assert config.temporal.retry_max_attempts == 1
    assert config.temporal.idle_timeout_seconds == 600.0
