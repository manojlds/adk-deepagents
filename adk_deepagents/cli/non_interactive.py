"""Non-interactive CLI execution helpers."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TextIO

from google.adk.runners import Runner
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types

from adk_deepagents import create_deep_agent
from adk_deepagents.backends.filesystem import FilesystemBackend
from adk_deepagents.cli.session_store import CLI_SESSIONS_APP_NAME


def read_piped_stdin(stdin: TextIO | None = None) -> str | None:
    """Read stdin only when input is piped.

    Returns ``None`` for TTY input or empty piped data.
    """
    stream = stdin if stdin is not None else sys.stdin
    if stream.isatty():
        return None

    try:
        data = stream.read()
    except OSError:
        # Pytest capture and some environments provide a non-readable stdin.
        return None

    if not data or not data.strip():
        return None
    return data


def combine_non_interactive_prompt(flag_prompt: str | None, piped_text: str | None) -> str | None:
    """Combine flag prompt and piped stdin into a single task prompt."""
    flag_text = (flag_prompt or "").strip()
    piped = (piped_text or "").strip()

    if piped and flag_text:
        return f"{piped}\n\n{flag_text}"
    if piped:
        return piped
    if flag_text:
        return flag_text
    return None


def _build_cli_agent(agent_name: str, model: str | None, cwd: Path):
    """Build a minimal CLI agent for non-interactive turns."""
    backend = FilesystemBackend(root_dir=cwd, virtual_mode=True)

    if model is None:
        return create_deep_agent(name=f"{agent_name}_cli", backend=backend)

    return create_deep_agent(name=f"{agent_name}_cli", model=model, backend=backend)


async def _run_non_interactive_async(
    *,
    prompt: str,
    model: str | None,
    agent_name: str,
    user_id: str,
    session_id: str,
    db_path: Path,
    no_stream: bool,
) -> str:
    agent = _build_cli_agent(agent_name=agent_name, model=model, cwd=Path.cwd())
    session_service = SqliteSessionService(str(db_path))
    runner = Runner(
        app_name=CLI_SESSIONS_APP_NAME,
        agent=agent,
        session_service=session_service,
    )

    user_message = types.Content(role="user", parts=[types.Part(text=prompt)])
    chunks: list[str] = []

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message,
    ):
        if getattr(event, "author", None) == "user" or not event.content or not event.content.parts:
            continue

        for part in event.content.parts:
            text = getattr(part, "text", None)
            if not isinstance(text, str) or not text:
                continue

            chunks.append(text)
            if not no_stream:
                print(text, end="", flush=True)

    output = "".join(chunks)
    if not no_stream and output and not output.endswith("\n"):
        print()

    return output


def run_non_interactive(
    *,
    prompt: str,
    model: str | None,
    agent_name: str,
    user_id: str,
    session_id: str,
    db_path: Path,
    no_stream: bool,
) -> int:
    """Execute one non-interactive turn and print model output."""
    try:
        output = asyncio.run(
            _run_non_interactive_async(
                prompt=prompt,
                model=model,
                agent_name=agent_name,
                user_id=user_id,
                session_id=session_id,
                db_path=db_path,
                no_stream=no_stream,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI should map runtime errors to exit code 1.
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if no_stream and output:
        print(output, end="" if output.endswith("\n") else "\n")

    return 0
