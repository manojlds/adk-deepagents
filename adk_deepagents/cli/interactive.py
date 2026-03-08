"""Interactive CLI REPL helpers."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TextIO

from google.adk.runners import Runner
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types

from adk_deepagents import create_deep_agent
from adk_deepagents.callbacks.before_tool import resume_approval
from adk_deepagents.cli.delegation_config import build_cli_dynamic_task_config
from adk_deepagents.cli.resources import (
    MemoryMappedFilesystemBackend,
    build_missing_skills_dependency_error,
)
from adk_deepagents.cli.session_store import (
    CLI_SESSIONS_APP_NAME,
    ThreadRecord,
    create_thread,
    get_thread,
    list_threads,
)
from adk_deepagents.types import DynamicTaskConfig

INPUT_PROMPT = "> "
APPROVAL_PROMPT = "approval> "
THREAD_LIST_LIMIT = 200
REQUEST_CONFIRMATION_FUNCTION_CALL_NAME = "adk_request_confirmation"
INTERACTIVE_INTERRUPT_ON: dict[str, bool] = {
    "write_file": True,
    "edit_file": True,
    "delete_file": True,
    "execute": True,
    "execute_bash": True,
}
MAX_APPROVAL_ARGS_PREVIEW = 400
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
ApprovalDecision = Literal["approve", "reject", "auto"]
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

    def tool_result(self, tool_name: str, detail: str) -> None:
        self.finish_assistant_line()
        print(f"[tool] {tool_name} -> {detail}", file=self.stdout)

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


@dataclass
class _InteractiveApprovalContext:
    """Mutable approval state shared across interactive turns."""

    auto_approve: bool


@dataclass
class _ToolConfirmationRequest:
    """Parsed confirmation request payload from ADK function-call events."""

    request_id: str
    tool_name: str
    tool_args: dict[str, Any]
    hint: str | None


def _normalize_prompt(prompt: str | None) -> str | None:
    if prompt is None:
        return None

    normalized = prompt.strip()
    return normalized or None


def _build_cli_agent(
    agent_name: str,
    model: str | None,
    cwd: Path,
    *,
    dynamic_task_config: DynamicTaskConfig | None = None,
    memory_sources: Sequence[str] = (),
    memory_source_paths: Mapping[str, Path] | None = None,
    skills_dirs: Sequence[str] = (),
):
    backend = MemoryMappedFilesystemBackend(
        root_dir=cwd,
        memory_source_paths=memory_source_paths,
    )

    agent_kwargs: dict[str, Any] = {
        "name": f"{agent_name}_cli",
        "backend": backend,
        "execution": "local",
        "delegation_mode": "dynamic",
        "dynamic_task_config": dynamic_task_config or build_cli_dynamic_task_config(),
        "interrupt_on": INTERACTIVE_INTERRUPT_ON,
    }
    if model is not None:
        agent_kwargs["model"] = model
    if memory_sources:
        agent_kwargs["memory"] = list(memory_sources)
    if skills_dirs:
        agent_kwargs["skills"] = list(skills_dirs)

    try:
        return create_deep_agent(**agent_kwargs)
    except ImportError as exc:
        if skills_dirs and "adk-skills-agent is required for skills support" in str(exc):
            raise build_missing_skills_dependency_error(skills_dirs) from exc
        raise


def _build_runner(
    *,
    agent_name: str,
    model: str | None,
    db_path: Path,
    dynamic_task_config: DynamicTaskConfig | None = None,
    memory_sources: Sequence[str] = (),
    memory_source_paths: Mapping[str, Path] | None = None,
    skills_dirs: Sequence[str] = (),
) -> Runner:
    agent = _build_cli_agent(
        agent_name=agent_name,
        model=model,
        cwd=Path.cwd(),
        dynamic_task_config=dynamic_task_config,
        memory_sources=memory_sources,
        memory_source_paths=memory_source_paths,
        skills_dirs=skills_dirs,
    )
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


def _extract_task_queue_notice(function_response: Any) -> str | None:
    if getattr(function_response, "name", None) != "task":
        return None

    response = getattr(function_response, "response", None)
    if not isinstance(response, dict):
        return None

    if not bool(response.get("queued")):
        return None

    queue_wait = response.get("queue_wait_seconds")
    if isinstance(queue_wait, (int, float)):
        return f"queued ({queue_wait:.3f}s)"

    return "queued"


def _extract_confirmation_requests(event: Any) -> list[_ToolConfirmationRequest]:
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return []

    requests: list[_ToolConfirmationRequest] = []
    for part in parts:
        function_call = getattr(part, "function_call", None)
        if function_call is None:
            continue

        tool_name = getattr(function_call, "name", None)
        if tool_name != REQUEST_CONFIRMATION_FUNCTION_CALL_NAME:
            continue

        request_id = getattr(function_call, "id", None)
        if not isinstance(request_id, str) or not request_id.strip():
            continue

        args = getattr(function_call, "args", None)
        args_dict = args if isinstance(args, dict) else {}

        original_call = args_dict.get("originalFunctionCall")
        original_call_dict = original_call if isinstance(original_call, dict) else {}

        requested_tool = original_call_dict.get("name")
        requested_tool_name = requested_tool if isinstance(requested_tool, str) else "unknown_tool"

        requested_args = original_call_dict.get("args")
        requested_tool_args = requested_args if isinstance(requested_args, dict) else {}

        tool_confirmation = args_dict.get("toolConfirmation")
        confirmation_dict = tool_confirmation if isinstance(tool_confirmation, dict) else {}
        hint_value = confirmation_dict.get("hint")
        hint = hint_value.strip() if isinstance(hint_value, str) and hint_value.strip() else None

        requests.append(
            _ToolConfirmationRequest(
                request_id=request_id,
                tool_name=requested_tool_name,
                tool_args=requested_tool_args,
                hint=hint,
            )
        )

    return requests


def _format_approval_args_preview(tool_args: dict[str, Any]) -> str:
    if not tool_args:
        return "{}"

    rendered = json.dumps(tool_args, ensure_ascii=False, sort_keys=True, default=str)
    if len(rendered) <= MAX_APPROVAL_ARGS_PREVIEW:
        return rendered

    return rendered[: MAX_APPROVAL_ARGS_PREVIEW - 3] + "..."


def _read_approval_input(*, input_reader: InputReader, stderr: TextIO) -> str | None:
    try:
        return input_reader(APPROVAL_PROMPT)
    except (EOFError, OSError):
        return None
    except KeyboardInterrupt:
        print("\n[approval] Interrupted; rejecting pending tool call.", file=stderr)
        return "reject"


def _resolve_confirmation_decision(response: str) -> ApprovalDecision | None:
    normalized = response.strip().lower()
    if normalized in {"a", "approve", "y", "yes"}:
        return "approve"
    if normalized in {"r", "reject", "n", "no"}:
        return "reject"
    if normalized in {"auto", "aa", "always"}:
        return "auto"

    return None


def _prompt_for_tool_confirmation(
    request: _ToolConfirmationRequest,
    *,
    approval_context: _InteractiveApprovalContext,
    input_reader: InputReader,
    stderr: TextIO,
) -> bool:
    if approval_context.auto_approve:
        print(
            f"[approval {request.request_id}] auto-approved tool '{request.tool_name}'.",
            file=stderr,
        )
        return True

    print(
        f"[approval {request.request_id}] Tool '{request.tool_name}' requested confirmation.",
        file=stderr,
    )
    if request.hint is not None:
        print(f"Hint: {request.hint}", file=stderr)
    print(f"Args: {_format_approval_args_preview(request.tool_args)}", file=stderr)
    print("Choose: (a)pprove, (r)eject, or (auto)-approve remaining prompts.", file=stderr)

    while True:
        raw_response = _read_approval_input(input_reader=input_reader, stderr=stderr)
        if raw_response is None:
            print(
                f"[approval {request.request_id}] input unavailable; rejecting tool "
                f"'{request.tool_name}'.",
                file=stderr,
            )
            return False

        decision = _resolve_confirmation_decision(raw_response)
        if decision is None:
            print(
                "[error] Invalid approval response. Enter 'a', 'r', or 'auto'.",
                file=stderr,
            )
            continue

        if decision == "auto":
            approval_context.auto_approve = True
            print("[approval] Auto-approve enabled for remaining tool prompts.", file=stderr)
            print(
                f"[approval {request.request_id}] approved tool '{request.tool_name}'.",
                file=stderr,
            )
            return True

        if decision == "approve":
            print(
                f"[approval {request.request_id}] approved tool '{request.tool_name}'.", file=stderr
            )
            return True

        print(f"[approval {request.request_id}] rejected tool '{request.tool_name}'.", file=stderr)
        return False


def _build_confirmation_response_message(*, request_id: str, approved: bool) -> types.Content:
    confirmation = resume_approval(approved=approved)
    response_payload = confirmation.model_dump(by_alias=True, exclude_none=True)

    part = types.Part.from_function_response(
        name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
        response=response_payload,
    )
    if part.function_response is not None:
        part.function_response.id = request_id

    return types.Content(role="user", parts=[part])


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

            if tool_name != REQUEST_CONFIRMATION_FUNCTION_CALL_NAME:
                renderer.tool_call(tool_name)

        function_response = getattr(part, "function_response", None)
        if function_response is not None:
            tool_name = getattr(function_response, "name", None)
            if not isinstance(tool_name, str) or not tool_name:
                tool_name = "unknown_tool"

            if tool_name == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME:
                continue

            tool_error = _extract_tool_error(function_response)
            if tool_error is not None:
                renderer.error(f"{tool_name}: {tool_error}")

            queue_notice = _extract_task_queue_notice(function_response)
            if queue_notice is not None:
                renderer.tool_result(tool_name, queue_notice)


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
    input_reader: InputReader = input,
    approval_context: _InteractiveApprovalContext | None = None,
) -> None:
    renderer = _TurnRenderer(stdout=stdout, stderr=stderr)
    approval_context = approval_context or _InteractiveApprovalContext(auto_approve=False)
    pending_messages: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=prompt)])
    ]

    try:
        while pending_messages:
            next_message = pending_messages.pop(0)
            pending_confirmations: list[_ToolConfirmationRequest] = []
            seen_confirmation_ids: set[str] = set()

            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=next_message,
            ):
                if getattr(event, "author", None) == "user":
                    continue

                _render_event(event, renderer=renderer)

                for request in _extract_confirmation_requests(event):
                    if request.request_id in seen_confirmation_ids:
                        continue

                    seen_confirmation_ids.add(request.request_id)
                    pending_confirmations.append(request)

            for request in pending_confirmations:
                renderer.finish_assistant_line()
                approved = _prompt_for_tool_confirmation(
                    request,
                    approval_context=approval_context,
                    input_reader=input_reader,
                    stderr=stderr,
                )
                pending_messages.append(
                    _build_confirmation_response_message(
                        request_id=request.request_id,
                        approved=approved,
                    )
                )

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
    dynamic_task_config: DynamicTaskConfig | None = None,
    memory_sources: Sequence[str] = (),
    memory_source_paths: Mapping[str, Path] | None = None,
    skills_dirs: Sequence[str] = (),
    input_reader: InputReader = input,
    stdin: _SupportsIsatty | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run interactive REPL mode until the user exits."""
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

    runner = _build_runner(
        agent_name=agent_name,
        model=model,
        db_path=db_path,
        dynamic_task_config=dynamic_task_config,
        memory_sources=memory_sources,
        memory_source_paths=memory_source_paths,
        skills_dirs=skills_dirs,
    )

    def _switch_model(new_model: str | None) -> None:
        nonlocal runner
        runner = _build_runner(
            agent_name=agent_name,
            model=new_model,
            db_path=db_path,
            dynamic_task_config=dynamic_task_config,
            memory_sources=memory_sources,
            memory_source_paths=memory_source_paths,
            skills_dirs=skills_dirs,
        )

    model_context = _ModelCommandContext(model=model, switch_model=_switch_model)
    approval_context = _InteractiveApprovalContext(auto_approve=auto_approve)

    print(
        f"[thread {thread_context.active_session_id}] interactive mode. Type /help for commands.",
        file=err_stream,
    )
    if approval_context.auto_approve:
        print("[approval] Auto-approve enabled for this interactive session.", file=err_stream)

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
            input_reader=input_reader,
            approval_context=approval_context,
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
    dynamic_task_config: DynamicTaskConfig | None = None,
    memory_sources: Sequence[str] = (),
    memory_source_paths: Mapping[str, Path] | None = None,
    skills_dirs: Sequence[str] = (),
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
                dynamic_task_config=dynamic_task_config,
                memory_sources=memory_sources,
                memory_source_paths=memory_source_paths,
                skills_dirs=skills_dirs,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI should map runtime errors to exit code 1.
        print(f"Error: {exc}", file=sys.stderr)
        return 1
