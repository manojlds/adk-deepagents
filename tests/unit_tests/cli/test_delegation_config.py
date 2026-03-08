"""Unit tests for CLI dynamic delegation config helpers."""

from __future__ import annotations

from adk_deepagents.cli.config import CliDefaults
from adk_deepagents.cli.delegation_config import build_cli_dynamic_task_config


def test_build_cli_dynamic_task_config_defaults_to_wait_policy(monkeypatch) -> None:
    monkeypatch.delenv("ADK_DYNAMIC_TASK_MAX_PARALLEL", raising=False)
    monkeypatch.delenv("ADK_DYNAMIC_TASK_CONCURRENCY_POLICY", raising=False)
    monkeypatch.delenv("ADK_DYNAMIC_TASK_QUEUE_TIMEOUT_SECONDS", raising=False)

    config = build_cli_dynamic_task_config()

    assert config.concurrency_policy == "wait"
    assert config.queue_timeout_seconds == 30.0


def test_build_cli_dynamic_task_config_reads_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("ADK_DYNAMIC_TASK_MAX_PARALLEL", "6")
    monkeypatch.setenv("ADK_DYNAMIC_TASK_CONCURRENCY_POLICY", "error")
    monkeypatch.setenv("ADK_DYNAMIC_TASK_QUEUE_TIMEOUT_SECONDS", "7.5")

    config = build_cli_dynamic_task_config()

    assert config.max_parallel == 6
    assert config.concurrency_policy == "error"
    assert config.queue_timeout_seconds == 7.5


def test_build_cli_dynamic_task_config_uses_cli_defaults_when_provided(monkeypatch) -> None:
    monkeypatch.delenv("ADK_DYNAMIC_TASK_MAX_PARALLEL", raising=False)
    monkeypatch.delenv("ADK_DYNAMIC_TASK_CONCURRENCY_POLICY", raising=False)
    monkeypatch.delenv("ADK_DYNAMIC_TASK_QUEUE_TIMEOUT_SECONDS", raising=False)

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
    monkeypatch.setenv("ADK_DYNAMIC_TASK_MAX_PARALLEL", "0")
    monkeypatch.setenv("ADK_DYNAMIC_TASK_CONCURRENCY_POLICY", "invalid")
    monkeypatch.setenv("ADK_DYNAMIC_TASK_QUEUE_TIMEOUT_SECONDS", "-5")

    config = build_cli_dynamic_task_config()

    assert config.max_parallel == 4
    assert config.concurrency_policy == "wait"
    assert config.queue_timeout_seconds == 30.0
