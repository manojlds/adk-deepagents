"""Interactive CLI REPL helpers."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TextIO

from google.adk.runners import Runner
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types

from adk_deepagents import create_deep_agent
from adk_deepagents.backends.filesystem import FilesystemBackend
from adk_deepagents.cli.session_store import CLI_SESSIONS_APP_NAME

INPUT_PROMPT = "> "
INTERACTIVE_HELP_TEXT = (
    "Interactive commands:\n"
    "  /help  Show this help message\n"
    "  /quit  Exit interactive mode\n"
    "  /q     Exit interactive mode\n"
)

SlashCommandResult = Literal["not_command", "handled", "exit"]
InputReader = Callable[[str], str]


class _SupportsIsatty(Protocol):
    def isatty(self) -> bool: ...


@dataclass
class _TurnRenderer:
    """Render streamed interactive events with clear prefixes."""

    stdout: TextIO
    stderr: TextIO
    assistant_line_open: bool = False

    def assistant_text(self, text: str) -> None:
        if not text:
            return

        if not self.assistant_line_open:
            print("assistant> ", end="", file=self.stdout, flush=True)
            self.assistant_line_open = True

        print(text, end="", file=self.stdout, flush=True)

    def tool_call(self, tool_name: str) -> None:
        self.finish_assistant_line()
        print(f"[tool] {tool_name}", file=self.stdout)

    def error(self, message: str) -> None:
        self.finish_assistant_line()
        print(f"[error] {message}", file=self.stderr)

    def finish_assistant_line(self) -> None:
        if self.assistant_line_open:
            print(file=self.stdout)
            self.assistant_line_open = False


def _normalize_prompt(prompt: str | None) -> str | None:
    if prompt is None:
        return None

    normalized = prompt.strip()
    return normalized or None


def _build_cli_agent(agent_name: str, model: str | None, cwd: Path):
    backend = FilesystemBackend(root_dir=cwd, virtual_mode=True)

    if model is None:
        return create_deep_agent(
            name=f"{agent_name}_cli",
            backend=backend,
            execution="local",
        )

    return create_deep_agent(
        name=f"{agent_name}_cli",
        model=model,
        backend=backend,
        execution="local",
    )


def handle_slash_command(
    prompt: str,
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> SlashCommandResult:
    """Handle slash commands in interactive mode."""
    if not prompt.startswith("/"):
        return "not_command"

    command = prompt.strip().lower()
    if command == "/help":
        print(INTERACTIVE_HELP_TEXT, end="", file=stdout)
        return "handled"

    if command in {"/quit", "/q"}:
        return "exit"

    print(
        f"[error] Unknown slash command: {command}. Type /help for available commands.",
        file=stderr,
    )
    return "handled"


def _extract_event_error(event: Any) -> str | None:
    error_message = getattr(event, "error_message", None)
    if isinstance(error_message, str) and error_message.strip():
        return error_message.strip()

    error = getattr(event, "error", None)
    if isinstance(error, Exception):
        return str(error)
    if isinstance(error, str) and error.strip():
        return error.strip()

    error_code = getattr(event, "error_code", None)
    if isinstance(error_code, str) and error_code.strip():
        return error_code.strip()

    return None


def _extract_tool_error(function_response: Any) -> str | None:
    response = getattr(function_response, "response", None)
    if not isinstance(response, dict):
        return None

    for key in ("error", "stderr"):
        value = response.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _render_event(event: Any, *, renderer: _TurnRenderer) -> None:
    event_error = _extract_event_error(event)
    if event_error is not None:
        renderer.error(event_error)

    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return

    for part in parts:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text:
            renderer.assistant_text(text)

        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            tool_name = getattr(function_call, "name", None)
            if not isinstance(tool_name, str) or not tool_name:
                tool_name = "unknown_tool"
            renderer.tool_call(tool_name)

        function_response = getattr(part, "function_response", None)
        if function_response is not None:
            tool_name = getattr(function_response, "name", None)
            if not isinstance(tool_name, str) or not tool_name:
                tool_name = "unknown_tool"

            tool_error = _extract_tool_error(function_response)
            if tool_error is not None:
                renderer.error(f"{tool_name}: {tool_error}")


def _read_prompt(*, input_reader: InputReader, stderr: TextIO) -> str | None:
    try:
        return input_reader(INPUT_PROMPT)
    except (EOFError, OSError):
        return None
    except KeyboardInterrupt:
        print("\nUse /quit or /q to exit.", file=stderr)
        return ""


async def _run_interactive_turn(
    *,
    runner: Any,
    prompt: str,
    user_id: str,
    session_id: str,
    stdout: TextIO,
    stderr: TextIO,
) -> None:
    renderer = _TurnRenderer(stdout=stdout, stderr=stderr)
    user_message = types.Content(role="user", parts=[types.Part(text=prompt)])

    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_message,
        ):
            if getattr(event, "author", None) == "user":
                continue

            _render_event(event, renderer=renderer)
    except Exception as exc:  # noqa: BLE001 - keep REPL alive across turn errors.
        renderer.error(str(exc))
    finally:
        renderer.finish_assistant_line()


async def _run_interactive_async(
    *,
    first_prompt: str | None,
    model: str | None,
    agent_name: str,
    user_id: str,
    session_id: str,
    db_path: Path,
    auto_approve: bool,
    input_reader: InputReader = input,
    stdin: _SupportsIsatty | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run interactive REPL mode until the user exits."""
    del auto_approve  # Reserved for future HITL auto-approve behavior.

    input_stream = stdin if stdin is not None else sys.stdin
    out_stream = stdout if stdout is not None else sys.stdout
    err_stream = stderr if stderr is not None else sys.stderr

    pending_prompt = _normalize_prompt(first_prompt)
    if pending_prompt is None and not input_stream.isatty():
        return 0

    agent = _build_cli_agent(agent_name=agent_name, model=model, cwd=Path.cwd())
    session_service = SqliteSessionService(str(db_path))
    runner = Runner(
        app_name=CLI_SESSIONS_APP_NAME,
        agent=agent,
        session_service=session_service,
    )

    print(
        f"[thread {session_id}] interactive mode. Type /help for commands.",
        file=err_stream,
    )

    while True:
        prompt = pending_prompt
        pending_prompt = None

        if prompt is None:
            prompt = _read_prompt(input_reader=input_reader, stderr=err_stream)
            if prompt is None:
                break

        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            continue

        slash_result = handle_slash_command(
            normalized_prompt,
            stdout=out_stream,
            stderr=err_stream,
        )
        if slash_result == "handled":
            continue
        if slash_result == "exit":
            break

        await _run_interactive_turn(
            runner=runner,
            prompt=normalized_prompt,
            user_id=user_id,
            session_id=session_id,
            stdout=out_stream,
            stderr=err_stream,
        )

    return 0


def run_interactive(
    *,
    first_prompt: str | None,
    model: str | None,
    agent_name: str,
    user_id: str,
    session_id: str,
    db_path: Path,
    auto_approve: bool,
) -> int:
    """Execute interactive REPL mode."""
    try:
        return asyncio.run(
            _run_interactive_async(
                first_prompt=first_prompt,
                model=model,
                agent_name=agent_name,
                user_id=user_id,
                session_id=session_id,
                db_path=db_path,
                auto_approve=auto_approve,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI should map runtime errors to exit code 1.
        print(f"Error: {exc}", file=sys.stderr)
        return 1
