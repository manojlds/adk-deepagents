"""Shared helpers and fixtures for LLM integration tests."""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest
from google.adk.agents import LlmAgent

from adk_deepagents import BrowserConfig, create_deep_agent
from adk_deepagents.browser.playwright_mcp import get_playwright_browser_tools
from adk_deepagents.types import TemporalTaskConfig
from tests.integration_tests.conftest import (
    backend_factory,
    get_file_content,
    make_litellm_model,
    run_agent,
    run_agent_with_events,
    run_agent_with_task_payloads,
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
) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds

    while loop.time() < deadline:
        if await _can_connect_temporal_port(target_host):
            return True
        await asyncio.sleep(0.5)

    return False


@pytest.fixture(scope="session")
async def ensure_temporal_server() -> AsyncGenerator[TemporalTaskConfig, None]:
    """Ensure Temporal dev server is reachable for LLM Temporal tests.

    Behavior:
    - If server is already reachable at configured host: use it as-is.
    - Otherwise, try `pitchfork start temporal-server`, wait until reachable,
      and tear it down after the test.
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

    if shutil.which("pitchfork") is None:
        # Fall back to the SDK-bundled Temporal test server.
        try:
            from temporalio.testing import WorkflowEnvironment
        except ImportError:
            pytest.skip(
                "Temporal server is not reachable, `pitchfork` is not installed, "
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

    repo_root = Path(__file__).resolve().parents[3]

    start = await asyncio.create_subprocess_exec(
        "pitchfork",
        "start",
        "temporal-server",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=str(repo_root),
    )
    start_returncode = await start.wait()

    if start_returncode != 0 and not await _wait_for_temporal_port(
        target_host=config.target_host,
        timeout_seconds=5.0,
    ):
        pytest.skip(
            "`pitchfork start temporal-server` failed and Temporal is still unreachable. "
            "Start Temporal manually and retry."
        )

    ready = await _wait_for_temporal_port(
        target_host=config.target_host,
        timeout_seconds=60.0,
    )
    if not ready:
        with suppress(Exception):
            await asyncio.create_subprocess_exec(
                "pitchfork",
                "stop",
                "temporal-server",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=str(repo_root),
            )
        pytest.skip(
            "Failed to auto-start Temporal with `pitchfork start temporal-server`. "
            "Start Temporal manually and retry."
        )

    try:
        yield config
    finally:
        # Under xdist each worker has its own session fixture; stopping the
        # shared daemon from one worker can flap other workers.
        if "PYTEST_XDIST_WORKER" not in os.environ:
            with suppress(Exception):
                await asyncio.create_subprocess_exec(
                    "pitchfork",
                    "stop",
                    "temporal-server",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    cwd=str(repo_root),
                )


@pytest.fixture
async def ensure_browser_tools() -> AsyncGenerator[tuple[list[Any], Callable], None]:
    """Provide Playwright MCP browser tools for integration tests.

    Skips the test if ``npx`` is not available.  Yields ``(tools, cleanup)``
    and awaits the cleanup coroutine during teardown.
    """
    if shutil.which("npx") is None:
        pytest.skip("npx not found — cannot run Playwright MCP")

    tools, cleanup = await get_playwright_browser_tools(config=BrowserConfig(headless=True))
    try:
        yield tools, cleanup
    finally:
        await cleanup()


@pytest.fixture
async def browser_agent_factory(
    ensure_browser_tools: tuple[list[Any], Callable],
) -> Callable[..., LlmAgent]:
    """Factory fixture that creates browser-enabled agents.

    The Playwright MCP lifecycle (startup & cleanup) is managed by the
    ``ensure_browser_tools`` fixture — callers only need to create agents.

    Usage::

        agent = browser_agent_factory("my_agent", "Do browser things.")
    """
    tools, _cleanup = ensure_browser_tools

    def _factory(
        name: str,
        instruction: str,
        extra_tools: Sequence[Any] | None = None,
    ) -> LlmAgent:
        all_tools: list[Any] = list(tools)
        if extra_tools:
            all_tools.extend(extra_tools)
        return create_deep_agent(
            model=make_litellm_model(),
            name=name,
            tools=all_tools,
            instruction=instruction,
            browser="_resolved",
        )

    return _factory


__all__ = [
    "backend_factory",
    "get_file_content",
    "make_litellm_model",
    "run_agent",
    "run_agent_with_task_payloads",
    "run_agent_with_events",
    "send_followup",
    "send_followup_with_events",
    "ensure_temporal_server",
    "ensure_browser_tools",
    "browser_agent_factory",
]
