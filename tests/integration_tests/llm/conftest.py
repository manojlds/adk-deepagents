"""Shared helpers and fixtures for LLM integration tests."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
from collections.abc import AsyncGenerator
from contextlib import suppress

import pytest

from adk_deepagents.types import TemporalTaskConfig
from tests.integration_tests.conftest import (
    backend_factory,
    get_file_content,
    make_litellm_model,
    run_agent,
    run_agent_with_events,
    send_followup,
    send_followup_with_events,
)


def _parse_target_host(target_host: str) -> tuple[str, int]:
    host, sep, port_raw = target_host.rpartition(":")
    if not sep:
        return target_host, 7233

    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]

    try:
        port = int(port_raw)
    except ValueError:
        return target_host, 7233

    return host, port


async def _can_connect_temporal_port(target_host: str) -> bool:
    host, port = _parse_target_host(target_host)

    try:
        _reader, writer = await asyncio.open_connection(host, port)
    except OSError:
        return False

    writer.close()
    with suppress(Exception):
        await writer.wait_closed()
    return True


async def _wait_for_temporal_port(
    *,
    target_host: str,
    timeout_seconds: float,
    process: asyncio.subprocess.Process | None = None,
) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds

    while loop.time() < deadline:
        if process is not None and process.returncode is not None:
            return False
        if await _can_connect_temporal_port(target_host):
            return True
        await asyncio.sleep(0.5)

    return False


async def _terminate_process_tree(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return

    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)

    try:
        await asyncio.wait_for(process.wait(), timeout=15.0)
    except TimeoutError:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        with suppress(Exception):
            await process.wait()


@pytest.fixture
async def ensure_temporal_server() -> AsyncGenerator[TemporalTaskConfig, None]:
    """Ensure Temporal dev server is reachable for LLM Temporal tests.

    Behavior:
    - If server is already reachable at configured host: use it as-is.
    - Otherwise, try `devenv up temporal-server`, wait until reachable, and
      tear it down after the test.
    - If neither is possible, skip the test with guidance.
    """
    config = TemporalTaskConfig(
        target_host=os.environ.get("ADK_DEEPAGENTS_TEMPORAL_TARGET_HOST", "127.0.0.1:7233"),
        namespace=os.environ.get("ADK_DEEPAGENTS_TEMPORAL_NAMESPACE", "default"),
        task_queue=os.environ.get("ADK_DEEPAGENTS_TEMPORAL_TASK_QUEUE", "adk-deepagents-tasks"),
    )

    if await _wait_for_temporal_port(target_host=config.target_host, timeout_seconds=1.0):
        yield config
        return

    if shutil.which("devenv") is None:
        # Fall back to the SDK-bundled Temporal test server.
        try:
            from temporalio.testing import WorkflowEnvironment
        except ImportError:
            pytest.skip(
                "Temporal server is not reachable, `devenv` is not installed, "
                "and `temporalio` is not available. Start Temporal manually at "
                f"{config.target_host} or install the temporal extra."
            )

        env = await WorkflowEnvironment.start_local(ui=False, dev_server_log_level="error")
        env_config = env.client.config()
        service_config = env_config["service_client"].config
        local_config = TemporalTaskConfig(
            target_host=service_config.target_host,
            namespace=env_config["namespace"],
            task_queue=config.task_queue,
        )
        try:
            yield local_config
        finally:
            await env.shutdown()
        return

    process = await asyncio.create_subprocess_exec(
        "devenv",
        "up",
        "temporal-server",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )

    ready = await _wait_for_temporal_port(
        target_host=config.target_host,
        timeout_seconds=60.0,
        process=process,
    )
    if not ready:
        await _terminate_process_tree(process)
        pytest.skip(
            "Failed to auto-start Temporal with `devenv up temporal-server`. "
            "Start Temporal manually and retry."
        )

    try:
        yield config
    finally:
        await _terminate_process_tree(process)


__all__ = [
    "backend_factory",
    "get_file_content",
    "make_litellm_model",
    "run_agent",
    "run_agent_with_events",
    "send_followup",
    "send_followup_with_events",
    "ensure_temporal_server",
]
