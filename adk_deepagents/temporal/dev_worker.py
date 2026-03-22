"""Development Temporal worker entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from adk_deepagents.cli.delegation_config import build_cli_dynamic_task_config
from adk_deepagents.temporal.worker import create_temporal_worker
from adk_deepagents.tools.filesystem import edit_file, glob, grep, ls, read_file, write_file
from adk_deepagents.tools.todos import read_todos, write_todos


def _load_workspace_env() -> None:
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)


def _resolve_worker_model() -> str:
    for key in ("ADK_DEEPAGENTS_TEMPORAL_WORKER_MODEL", "ADK_DEEPAGENTS_MODEL"):
        raw_value = os.environ.get(key)
        if raw_value is not None:
            value = raw_value.strip()
            if value:
                return value

    return "gemini-2.5-flash"


def _default_tools() -> list[Any]:
    return [write_todos, read_todos, ls, read_file, write_file, edit_file, glob, grep]


async def _handle_health_probe(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    await reader.read(1)
    writer.close()
    await writer.wait_closed()


async def run_worker(*, health_port: int | None = None) -> None:
    _load_workspace_env()
    dynamic_task_config = build_cli_dynamic_task_config()
    if dynamic_task_config.temporal is None:
        raise RuntimeError(
            "Temporal worker requires ADK_DEEPAGENTS_TEMPORAL_* environment variables."
        )

    worker = await create_temporal_worker(
        default_model=_resolve_worker_model(),
        default_tools=_default_tools(),
        dynamic_task_config=dynamic_task_config,
    )

    health_server: asyncio.AbstractServer | None = None
    if health_port is not None:
        health_server = await asyncio.start_server(
            _handle_health_probe,
            host="127.0.0.1",
            port=health_port,
        )

    try:
        await worker.run()
    finally:
        if health_server is not None:
            health_server.close()
            await health_server.wait_closed()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m adk_deepagents.temporal.dev_worker")
    parser.add_argument(
        "--health-port",
        type=int,
        default=None,
        help="Optional localhost TCP port for liveness probes.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(run_worker(health_port=args.health_port))


if __name__ == "__main__":
    main()
