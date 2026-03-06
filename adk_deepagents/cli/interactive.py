"""Interactive CLI REPL helpers."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TextIO

from google.adk.runners import Runner
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types

from adk_deepagents import create_deep_agent
from adk_deepagents.backends.filesystem import FilesystemBackend
from adk_deepagents.cli.session_store import (
    CLI_SESSIONS_APP_NAME,
    ThreadRecord,
    create_thread,
    get_thread,
    list_threads,
)

INPUT_PROMPT = "> "
THREAD_LIST_LIMIT = 200
INTERACTIVE_HELP_TEXT = (
    "Interactive commands:\n"
    "  /help                      Show this help message\n"
    "  /threads                   List recent threads and show the active thread\n"
    "  /threads <selector>        Switch thread by index, thread id, or 'latest'\n"
    "  /clear                     Start a new thread and make it active\n"
    "  /model                     Show the active model for this REPL session\n"
    "  /model <name|default>      Switch model for subsequent turns\n"
    "  /quit                      Exit interactive mode\n"
    "  /q                         Exit interactive mode\n"
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


@dataclass
class _ThreadCommandContext:
    """Mutable interactive thread state used by slash commands."""

    db_path: Path
    user_id: str
    agent_name: str
    model: str | None
    active_session_id: str


@dataclass
class _ModelCommandContext:
    """Mutable interactive model state used by slash commands."""

    model: str | None
    switch_model: Callable[[str | None], None]


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


def _build_runner(*, agent_name: str, model: str | None, db_path: Path) -> Runner:
    agent = _build_cli_agent(agent_name=agent_name, model=model, cwd=Path.cwd())
    session_service = SqliteSessionService(str(db_path))
    return Runner(
        app_name=CLI_SESSIONS_APP_NAME,
        agent=agent,
        session_service=session_service,
    )


def _normalize_model_value(raw: str) -> str | None:
    normalized = raw.strip()
    return normalized or None


def _format_model_name(model: str | None) -> str:
    return model if model is not None else "default"


def _format_timestamp(timestamp: float | None) -> str:
    if timestamp is None:
        return "-"

    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat(timespec="seconds")


def _render_threads_list(
    *,
    thread_context: _ThreadCommandContext,
    threads: list[ThreadRecord],
    stdout: TextIO,
) -> None:
    if not threads:
        print(f"No threads found for profile '{thread_context.agent_name}'.", file=stdout)
        print("Use /clear to start a new thread.", file=stdout)
        return

    print(f"Threads for profile '{thread_context.agent_name}':", file=stdout)
    print("CUR\tINDEX\tTHREAD_ID\tUPDATED_AT\tCREATED_AT\tMODEL", file=stdout)
    for index, thread in enumerate(threads, start=1):
        is_active = thread.session_id == thread_context.active_session_id
        marker = "*" if is_active else "-"
        model_name = thread_context.model if is_active else thread.model
        model_label = model_name or "-"
        print(
            f"{marker}\t{index}\t{thread.session_id}\t"
            f"{_format_timestamp(thread.updated_at)}\t"
            f"{_format_timestamp(thread.created_at)}\t{model_label}",
            file=stdout,
        )

    print("Use /threads <index|thread_id|latest> to switch active thread.", file=stdout)


def _resolve_thread_selector(
    selector: str,
    *,
    thread_context: _ThreadCommandContext,
    threads: list[ThreadRecord],
) -> ThreadRecord | None:
    normalized_selector = selector.strip()
    if not normalized_selector:
        return None

    selector_lower = normalized_selector.lower()
    if selector_lower == "latest":
        return threads[0] if threads else None

    if normalized_selector.isdigit():
        index = int(normalized_selector)
        if index < 1:
            return None
        if index > len(threads):
            return None
        return threads[index - 1]

    thread = get_thread(
        db_path=thread_context.db_path,
        user_id=thread_context.user_id,
        session_id=normalized_selector,
    )
    if thread is None or thread.agent_name != thread_context.agent_name:
        return None

    return thread


def _handle_threads_slash_command(
    raw_command: str,
    *,
    thread_context: _ThreadCommandContext,
    stdout: TextIO,
    stderr: TextIO,
) -> SlashCommandResult:
    command_parts = raw_command.split(maxsplit=1)
    selector = command_parts[1].strip() if len(command_parts) == 2 else None

    try:
        threads = list_threads(
            db_path=thread_context.db_path,
            user_id=thread_context.user_id,
            agent_name=thread_context.agent_name,
            limit=THREAD_LIST_LIMIT,
        )
    except Exception as exc:  # noqa: BLE001 - slash command errors should not kill REPL.
        print(f"[error] Failed to list threads: {exc}", file=stderr)
        return "handled"

    if selector is None:
        _render_threads_list(
            thread_context=thread_context,
            threads=threads,
            stdout=stdout,
        )
        return "handled"

    target_thread = _resolve_thread_selector(
        selector,
        thread_context=thread_context,
        threads=threads,
    )
    if target_thread is None:
        print(
            f"[error] Could not resolve thread selector '{selector}'. "
            "Use /threads to list options.",
            file=stderr,
        )
        return "handled"

    if target_thread.session_id == thread_context.active_session_id:
        print(f"[thread {target_thread.session_id}] already active.", file=stderr)
        return "handled"

    thread_context.active_session_id = target_thread.session_id
    print(f"[thread {target_thread.session_id}] switched active thread.", file=stderr)
    return "handled"


def _handle_model_slash_command(
    raw_command: str,
    *,
    thread_context: _ThreadCommandContext,
    model_context: _ModelCommandContext,
    stdout: TextIO,
    stderr: TextIO,
) -> SlashCommandResult:
    command_parts = raw_command.split(maxsplit=1)
    model_selector = command_parts[1] if len(command_parts) == 2 else None

    if model_selector is None:
        print(f"Active model: {_format_model_name(model_context.model)}", file=stdout)
        print("Use /model <name|default> to switch models.", file=stdout)
        return "handled"

    requested_model = _normalize_model_value(model_selector)
    if requested_model is None:
        print("[error] /model requires a model name or 'default'.", file=stderr)
        return "handled"

    target_model = None if requested_model.lower() in {"default", "reset"} else requested_model

    if target_model == model_context.model:
        print(f"[model {_format_model_name(target_model)}] already active.", file=stderr)
        return "handled"

    try:
        model_context.switch_model(target_model)
    except Exception as exc:  # noqa: BLE001 - slash command errors should not kill REPL.
        print(f"[error] Failed to switch model: {exc}", file=stderr)
        return "handled"

    model_context.model = target_model
    thread_context.model = target_model
    print(f"[model {_format_model_name(target_model)}] switched active model.", file=stderr)
    return "handled"


def handle_slash_command(
    prompt: str,
    *,
    stdout: TextIO,
    stderr: TextIO,
    thread_context: _ThreadCommandContext | None = None,
    model_context: _ModelCommandContext | None = None,
) -> SlashCommandResult:
    """Handle slash commands in interactive mode."""
    if not prompt.startswith("/"):
        return "not_command"

    command = prompt.strip()
    command_lower = command.lower()

    if command_lower == "/help":
        print(INTERACTIVE_HELP_TEXT, end="", file=stdout)
        return "handled"

    if command_lower in {"/quit", "/q"}:
        return "exit"

    if command_lower == "/clear":
        if thread_context is None:
            print("[error] Thread controls are unavailable in this context.", file=stderr)
            return "handled"

        try:
            new_thread = create_thread(
                db_path=thread_context.db_path,
                user_id=thread_context.user_id,
                agent_name=thread_context.agent_name,
                model=thread_context.model,
            )
        except Exception as exc:  # noqa: BLE001 - slash command errors should not kill REPL.
            print(f"[error] Failed to create a new thread: {exc}", file=stderr)
            return "handled"

        thread_context.active_session_id = new_thread.session_id
        print(f"[thread {new_thread.session_id}] started a new thread.", file=stderr)
        return "handled"

    if command_lower == "/threads" or command_lower.startswith("/threads "):
        if thread_context is None:
            print("[error] Thread controls are unavailable in this context.", file=stderr)
            return "handled"

        return _handle_threads_slash_command(
            command,
            thread_context=thread_context,
            stdout=stdout,
            stderr=stderr,
        )

    if command_lower == "/model" or command_lower.startswith("/model "):
        if thread_context is None or model_context is None:
            print("[error] Model controls are unavailable in this context.", file=stderr)
            return "handled"

        return _handle_model_slash_command(
            command,
            thread_context=thread_context,
            model_context=model_context,
            stdout=stdout,
            stderr=stderr,
        )

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

    thread_context = _ThreadCommandContext(
        db_path=db_path,
        user_id=user_id,
        agent_name=agent_name,
        model=model,
        active_session_id=session_id,
    )

    runner = _build_runner(agent_name=agent_name, model=model, db_path=db_path)

    def _switch_model(new_model: str | None) -> None:
        nonlocal runner
        runner = _build_runner(agent_name=agent_name, model=new_model, db_path=db_path)

    model_context = _ModelCommandContext(model=model, switch_model=_switch_model)

    print(
        f"[thread {thread_context.active_session_id}] interactive mode. Type /help for commands.",
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
            thread_context=thread_context,
            model_context=model_context,
        )
        if slash_result == "handled":
            continue
        if slash_result == "exit":
            break

        await _run_interactive_turn(
            runner=runner,
            prompt=normalized_prompt,
            user_id=user_id,
            session_id=thread_context.active_session_id,
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
