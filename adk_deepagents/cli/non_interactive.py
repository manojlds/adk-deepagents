"""Non-interactive CLI execution helpers."""

from __future__ import annotations

import asyncio
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from google.adk.runners import Runner
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.adk.tools import BaseTool, ToolContext
from google.genai import types

from adk_deepagents import create_deep_agent
from adk_deepagents.cli.resources import (
    MemoryMappedFilesystemBackend,
    build_missing_skills_dependency_error,
)
from adk_deepagents.cli.session_store import CLI_SESSIONS_APP_NAME

SHELL_TOOL_NAMES = frozenset({"execute", "execute_bash"})
CONFIRMATION_REQUIRED_TOOLS = frozenset({"write_file", "edit_file", "delete_file"})
RECOMMENDED_SHELL_ALLOW_LIST = frozenset(
    {
        "cat",
        "echo",
        "find",
        "git",
        "grep",
        "ls",
        "pwd",
        "pytest",
        "python",
        "python3",
        "ruff",
        "uv",
    }
)
SHELL_CONTROL_TOKENS = ("&&", "||", ";", "|", "`", "$(", ">", "<")


class NonInteractivePolicyError(RuntimeError):
    """Raised when non-interactive policy blocks a tool invocation."""


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


def normalize_shell_allow_list(shell_allow_list: Sequence[str] | None) -> frozenset[str]:
    """Normalize ``--shell-allow-list`` values into lowercase command names."""
    if not shell_allow_list:
        return frozenset()

    normalized: set[str] = set()
    for raw_value in shell_allow_list:
        value = raw_value.strip().lower()
        if not value:
            continue

        if value == "recommended":
            normalized.update(RECOMMENDED_SHELL_ALLOW_LIST)
            continue

        normalized.add(value)

    return frozenset(normalized)


def _extract_shell_command_name(command: str) -> str:
    normalized_command = command.strip()
    if not normalized_command:
        raise NonInteractivePolicyError("Shell command cannot be empty.")

    if any(token in normalized_command for token in SHELL_CONTROL_TOKENS):
        raise NonInteractivePolicyError(
            "Shell control operators are not allowed in non-interactive mode. "
            "Run a single command without pipes, redirection, or chaining."
        )

    try:
        command_parts = shlex.split(normalized_command, posix=True)
    except ValueError as exc:
        raise NonInteractivePolicyError(f"Invalid shell command: {exc}") from exc

    if not command_parts:
        raise NonInteractivePolicyError("Shell command cannot be empty.")

    return Path(command_parts[0]).name.lower()


def build_non_interactive_before_tool_callback(
    *,
    shell_allow_list: Sequence[str] | None,
    auto_approve: bool,
):
    """Build CLI safety callback for non-interactive runs."""
    allowed_shell_commands = normalize_shell_allow_list(shell_allow_list)

    def before_tool_callback(
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
    ) -> None:
        del tool_context

        tool_name = getattr(tool, "name", "")

        if tool_name in CONFIRMATION_REQUIRED_TOOLS and not auto_approve:
            raise NonInteractivePolicyError(
                f"Tool '{tool_name}' requires confirmation and is blocked in "
                "non-interactive mode. Re-run interactively or pass --auto-approve."
            )

        if tool_name in SHELL_TOOL_NAMES:
            if not allowed_shell_commands:
                raise NonInteractivePolicyError(
                    "Shell execution is blocked in non-interactive mode by default. "
                    "Pass --shell-allow-list with explicit commands to allow execution."
                )

            raw_command = args.get("command")
            if not isinstance(raw_command, str):
                raise NonInteractivePolicyError(
                    f"Tool '{tool_name}' must receive a string 'command' argument."
                )

            command_name = _extract_shell_command_name(raw_command)
            if command_name not in allowed_shell_commands:
                allowed = ", ".join(sorted(allowed_shell_commands))
                raise NonInteractivePolicyError(
                    f"Command '{command_name}' is not in --shell-allow-list ({allowed})."
                )

        return None

    return before_tool_callback


def _build_cli_agent(
    agent_name: str,
    model: str | None,
    cwd: Path,
    *,
    shell_allow_list: Sequence[str] | None,
    auto_approve: bool,
    memory_sources: Sequence[str] = (),
    memory_source_paths: Mapping[str, Path] | None = None,
    skills_dirs: Sequence[str] = (),
):
    """Build a minimal CLI agent for non-interactive turns."""
    backend = MemoryMappedFilesystemBackend(
        root_dir=cwd,
        memory_source_paths=memory_source_paths,
    )
    before_tool_callback = build_non_interactive_before_tool_callback(
        shell_allow_list=shell_allow_list,
        auto_approve=auto_approve,
    )
    extra_callbacks = {"before_tool": before_tool_callback}

    agent_kwargs: dict[str, Any] = {
        "name": f"{agent_name}_cli",
        "backend": backend,
        "execution": "local",
        "delegation_mode": "dynamic",
        "extra_callbacks": extra_callbacks,
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


async def _run_non_interactive_async(
    *,
    prompt: str,
    model: str | None,
    agent_name: str,
    user_id: str,
    session_id: str,
    db_path: Path,
    no_stream: bool,
    shell_allow_list: Sequence[str] | None,
    auto_approve: bool,
    memory_sources: Sequence[str] = (),
    memory_source_paths: Mapping[str, Path] | None = None,
    skills_dirs: Sequence[str] = (),
) -> str:
    agent = _build_cli_agent(
        agent_name=agent_name,
        model=model,
        cwd=Path.cwd(),
        shell_allow_list=shell_allow_list,
        auto_approve=auto_approve,
        memory_sources=memory_sources,
        memory_source_paths=memory_source_paths,
        skills_dirs=skills_dirs,
    )
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
    shell_allow_list: Sequence[str] | None,
    auto_approve: bool,
    memory_sources: Sequence[str] = (),
    memory_source_paths: Mapping[str, Path] | None = None,
    skills_dirs: Sequence[str] = (),
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
                shell_allow_list=shell_allow_list,
                auto_approve=auto_approve,
                memory_sources=memory_sources,
                memory_source_paths=memory_source_paths,
                skills_dirs=skills_dirs,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI should map runtime errors to exit code 1.
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if no_stream and output:
        print(output, end="" if output.endswith("\n") else "\n")

    return 0
