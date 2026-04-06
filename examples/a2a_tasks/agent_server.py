"""A2A worker server example.

This module exposes a deep agent as an A2A app so other agents can delegate
dynamic `task()` calls to it.

Usage:
    uv run python -m examples.a2a_tasks.agent_server
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from adk_deepagents import create_deep_agent, to_a2a_app

load_dotenv()

CARD_HOST = os.environ.get("A2A_CARD_HOST", "127.0.0.1")
PORT = int(os.environ.get("A2A_PORT", "8000"))
PROTOCOL = os.environ.get("A2A_PROTOCOL", "http")

root_agent = create_deep_agent(
    name="a2a_worker",
    instruction=(
        "You are an A2A task worker. Use your tools to complete delegated work "
        "and return concise, actionable results."
    ),
)

app = to_a2a_app(
    root_agent,
    host=CARD_HOST,
    port=PORT,
    protocol=PROTOCOL,
)


def main() -> None:
    """Run the A2A server with uvicorn."""
    import uvicorn

    bind_host = os.environ.get("A2A_BIND_HOST", "127.0.0.1")
    uvicorn.run("examples.a2a_tasks.agent_server:app", host=bind_host, port=PORT)


if __name__ == "__main__":
    main()
