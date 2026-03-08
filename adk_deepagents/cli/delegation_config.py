"""CLI dynamic delegation configuration helpers."""

from __future__ import annotations

import os
from typing import Literal

from adk_deepagents.cli.config import CliDefaults
from adk_deepagents.types import DynamicTaskConfig

_ENV_MAX_PARALLEL = "ADK_DYNAMIC_TASK_MAX_PARALLEL"
_ENV_CONCURRENCY_POLICY = "ADK_DYNAMIC_TASK_CONCURRENCY_POLICY"
_ENV_QUEUE_TIMEOUT_SECONDS = "ADK_DYNAMIC_TASK_QUEUE_TIMEOUT_SECONDS"


def build_cli_dynamic_task_config(defaults: CliDefaults | None = None) -> DynamicTaskConfig:
    """Build dynamic task config for CLI/TUI harnesses.

    Defaults are tuned for interactive reliability:
    - ``concurrency_policy`` defaults to ``"wait"``
    - ``queue_timeout_seconds`` defaults to ``30``
    - ``max_parallel`` defaults to the library default unless overridden

    Precedence:
    - environment variables
    - CLI config defaults (`CliDefaults`)
    - built-in defaults
    """
    config = DynamicTaskConfig(concurrency_policy="wait", queue_timeout_seconds=30.0)

    if defaults is not None:
        if defaults.dynamic_task_max_parallel is not None:
            config.max_parallel = defaults.dynamic_task_max_parallel
        if defaults.dynamic_task_concurrency_policy is not None:
            config.concurrency_policy = defaults.dynamic_task_concurrency_policy
        if defaults.dynamic_task_queue_timeout_seconds is not None:
            config.queue_timeout_seconds = defaults.dynamic_task_queue_timeout_seconds

    max_parallel = _read_int_env(_ENV_MAX_PARALLEL, minimum=1)
    if max_parallel is not None:
        config.max_parallel = max_parallel

    concurrency_policy = _read_policy_env(_ENV_CONCURRENCY_POLICY)
    if concurrency_policy is not None:
        config.concurrency_policy = concurrency_policy

    queue_timeout_seconds = _read_float_env(_ENV_QUEUE_TIMEOUT_SECONDS, minimum=0.0)
    if queue_timeout_seconds is not None:
        config.queue_timeout_seconds = queue_timeout_seconds

    return config


def _read_int_env(env_name: str, *, minimum: int) -> int | None:
    raw_value = os.environ.get(env_name)
    if raw_value is None:
        return None

    normalized = raw_value.strip()
    if not normalized:
        return None

    try:
        value = int(normalized)
    except ValueError:
        return None

    if value < minimum:
        return None
    return value


def _read_float_env(env_name: str, *, minimum: float) -> float | None:
    raw_value = os.environ.get(env_name)
    if raw_value is None:
        return None

    normalized = raw_value.strip()
    if not normalized:
        return None

    try:
        value = float(normalized)
    except ValueError:
        return None

    if value < minimum:
        return None
    return value


def _read_policy_env(env_name: str) -> Literal["error", "wait"] | None:
    raw_value = os.environ.get(env_name)
    if raw_value is None:
        return None

    normalized = raw_value.strip().lower()
    if normalized == "error":
        return "error"
    if normalized == "wait":
        return "wait"

    return None
