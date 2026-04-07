"""Helpers for dynamic task configuration."""

from __future__ import annotations

import os
from typing import Literal

from adk_deepagents.types import DynamicTaskConfig, TemporalTaskConfig

_ENV_MAX_PARALLEL = "ADK_DYNAMIC_TASK_MAX_PARALLEL"
_ENV_CONCURRENCY_POLICY = "ADK_DYNAMIC_TASK_CONCURRENCY_POLICY"
_ENV_QUEUE_TIMEOUT_SECONDS = "ADK_DYNAMIC_TASK_QUEUE_TIMEOUT_SECONDS"

_ENV_TEMPORAL_TARGET_HOST = "ADK_DEEPAGENTS_TEMPORAL_TARGET_HOST"
_ENV_TEMPORAL_NAMESPACE = "ADK_DEEPAGENTS_TEMPORAL_NAMESPACE"
_ENV_TEMPORAL_TASK_QUEUE = "ADK_DEEPAGENTS_TEMPORAL_TASK_QUEUE"
_ENV_TEMPORAL_WORKFLOW_ID_PREFIX = "ADK_DEEPAGENTS_TEMPORAL_WORKFLOW_ID_PREFIX"
_ENV_TEMPORAL_ACTIVITY_TIMEOUT_SECONDS = "ADK_DEEPAGENTS_TEMPORAL_ACTIVITY_TIMEOUT_SECONDS"
_ENV_TEMPORAL_RETRY_MAX_ATTEMPTS = "ADK_DEEPAGENTS_TEMPORAL_RETRY_MAX_ATTEMPTS"
_ENV_TEMPORAL_IDLE_TIMEOUT_SECONDS = "ADK_DEEPAGENTS_TEMPORAL_IDLE_TIMEOUT_SECONDS"

_TEMPORAL_ENV_KEYS = (
    _ENV_TEMPORAL_TARGET_HOST,
    _ENV_TEMPORAL_NAMESPACE,
    _ENV_TEMPORAL_TASK_QUEUE,
    _ENV_TEMPORAL_WORKFLOW_ID_PREFIX,
    _ENV_TEMPORAL_ACTIVITY_TIMEOUT_SECONDS,
    _ENV_TEMPORAL_RETRY_MAX_ATTEMPTS,
    _ENV_TEMPORAL_IDLE_TIMEOUT_SECONDS,
)


def build_dynamic_task_config() -> DynamicTaskConfig:
    """Build dynamic task config from environment variables.

    Defaults are tuned for interactive reliability:
    - ``concurrency_policy`` defaults to ``"wait"``
    - ``queue_timeout_seconds`` defaults to ``30``
    - ``max_parallel`` defaults to the ``DynamicTaskConfig`` default
    """
    config = DynamicTaskConfig(concurrency_policy="wait", queue_timeout_seconds=30.0)

    max_parallel = _read_int_env(_ENV_MAX_PARALLEL, minimum=1)
    if max_parallel is not None:
        config.max_parallel = max_parallel

    concurrency_policy = _read_policy_env(_ENV_CONCURRENCY_POLICY)
    if concurrency_policy is not None:
        config.concurrency_policy = concurrency_policy

    queue_timeout_seconds = _read_float_env(_ENV_QUEUE_TIMEOUT_SECONDS, minimum=0.0)
    if queue_timeout_seconds is not None:
        config.queue_timeout_seconds = queue_timeout_seconds

    temporal_config = _build_temporal_config_from_env()
    if temporal_config is not None:
        config.temporal = temporal_config

    return config


def _build_temporal_config_from_env() -> TemporalTaskConfig | None:
    temporal_enabled = any(os.environ.get(key, "").strip() for key in _TEMPORAL_ENV_KEYS)
    if not temporal_enabled:
        return None

    config = TemporalTaskConfig()

    target_host = _read_str_env(_ENV_TEMPORAL_TARGET_HOST)
    if target_host is not None:
        config.target_host = target_host

    namespace = _read_str_env(_ENV_TEMPORAL_NAMESPACE)
    if namespace is not None:
        config.namespace = namespace

    task_queue = _read_str_env(_ENV_TEMPORAL_TASK_QUEUE)
    if task_queue is not None:
        config.task_queue = task_queue

    workflow_id_prefix = _read_str_env(_ENV_TEMPORAL_WORKFLOW_ID_PREFIX)
    if workflow_id_prefix is not None:
        config.workflow_id_prefix = workflow_id_prefix

    activity_timeout_seconds = _read_float_env(_ENV_TEMPORAL_ACTIVITY_TIMEOUT_SECONDS, minimum=0.0)
    if activity_timeout_seconds is not None:
        config.activity_timeout_seconds = activity_timeout_seconds

    retry_max_attempts = _read_int_env(_ENV_TEMPORAL_RETRY_MAX_ATTEMPTS, minimum=1)
    if retry_max_attempts is not None:
        config.retry_max_attempts = retry_max_attempts

    idle_timeout_seconds = _read_float_env(_ENV_TEMPORAL_IDLE_TIMEOUT_SECONDS, minimum=0.0)
    if idle_timeout_seconds is not None:
        config.idle_timeout_seconds = idle_timeout_seconds

    return config


def _read_str_env(env_name: str) -> str | None:
    raw_value = os.environ.get(env_name)
    if raw_value is None:
        return None

    normalized = raw_value.strip()
    if not normalized:
        return None

    return normalized


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
