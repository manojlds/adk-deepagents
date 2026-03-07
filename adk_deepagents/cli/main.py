"""Main command-line entrypoint for adk-deepagents."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from adk_deepagents import __version__
from adk_deepagents.cli.config import (
    DEFAULT_AGENT_NAME,
    CliDefaults,
    CliPaths,
    bootstrap_cli_home,
    ensure_profile_memory,
    list_profiles,
    load_cli_defaults,
    reset_profile_memory,
    resolve_cli_paths,
    save_cli_defaults,
)
from adk_deepagents.cli.interactive import run_interactive
from adk_deepagents.cli.non_interactive import (
    combine_non_interactive_prompt,
    read_piped_stdin,
    run_non_interactive,
)
from adk_deepagents.cli.resources import discover_cli_agent_resources
from adk_deepagents.cli.session_store import (
    create_thread,
    delete_thread,
    get_latest_thread,
    get_thread,
    list_threads,
)

MODEL_ENV_VAR = "ADK_DEEPAGENTS_MODEL"
CLI_USER_ID = "local"


def _normalize_model(raw: str | None) -> str | None:
    if raw is None:
        return None

    stripped = raw.strip()
    return stripped or None


def _load_workspace_env() -> None:
    """Load environment variables from the current workspace `.env` file."""
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)


def resolve_model(cli_model: str | None, defaults: CliDefaults) -> str | None:
    """Resolve model precedence as CLI flag > env var > config default."""
    explicit_model = _normalize_model(cli_model)
    if explicit_model is not None:
        return explicit_model

    env_model = _normalize_model(os.environ.get(MODEL_ENV_VAR))
    if env_model is not None:
        return env_model

    return _normalize_model(defaults.default_model)


def _parse_model(raw: str) -> str:
    """Parse and normalize a model argument value."""
    value = raw.strip()
    if not value:
        raise argparse.ArgumentTypeError("--model cannot be empty.")
    return value


def _parse_shell_allow_list(raw: str) -> list[str]:
    """Parse a comma-separated allow-list value into normalized command entries."""
    value = raw.strip()
    if not value:
        raise argparse.ArgumentTypeError("--shell-allow-list cannot be empty.")

    if value.lower() == "recommended":
        return ["recommended"]

    commands = [item.strip() for item in value.split(",") if item.strip()]
    if not commands:
        raise argparse.ArgumentTypeError(
            "--shell-allow-list must contain at least one command or 'recommended'."
        )

    return commands


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="adk-deepagents",
        description="ADK-native CLI for adk-deepagents.",
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=("list", "reset", "threads"),
        help="Profile command to run (list, reset, threads).",
    )
    parser.add_argument(
        "command_arg",
        nargs="?",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "command_arg2",
        nargs="?",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "-a",
        "--agent",
        help=(
            "Agent profile name. Defaults to the value from "
            "~/.adk-deepagents/config.toml (or 'agent' on first run)."
        ),
    )
    parser.add_argument(
        "--model",
        type=_parse_model,
        help=(
            "Override the model for this run and persist as the new default "
            f"(precedence: --model > {MODEL_ENV_VAR} > config default)."
        ),
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-n",
        "--non-interactive",
        dest="non_interactive_prompt",
        metavar="PROMPT",
        help="Run a single non-interactive task.",
    )
    mode_group.add_argument(
        "-m",
        "--message",
        dest="message_prompt",
        metavar="PROMPT",
        help="Start interactive mode and auto-submit PROMPT as the first turn.",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Emit only assistant text for non-interactive runs.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Buffer non-interactive assistant output and print after completion.",
    )

    parser.add_argument(
        "-r",
        "--resume",
        nargs="?",
        const="latest",
        metavar="THREAD_ID",
        help="Resume latest thread (or a specific THREAD_ID).",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve interactive confirmation prompts.",
    )
    parser.add_argument(
        "--shell-allow-list",
        type=_parse_shell_allow_list,
        metavar="CSV|recommended",
        help="Comma-separated shell commands allowed in non-interactive mode.",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Launch the full-screen terminal UI instead of the plain REPL.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Print installed adk-deepagents version.",
    )

    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate argument combinations that argparse can't express directly."""
    if args.shell_allow_list is not None and args.non_interactive_prompt is None:
        parser.error("--shell-allow-list requires -n/--non-interactive.")

    if args.command in {"list", "reset", "threads"} and (
        args.non_interactive_prompt is not None or args.message_prompt is not None
    ):
        parser.error(
            "list/reset/threads commands cannot be combined with "
            "-n/--non-interactive or -m/--message."
        )

    if args.command in {"list", "reset", "threads"} and (args.quiet or args.no_stream):
        parser.error("-q/--quiet and --no-stream are only valid for non-interactive runs.")

    if getattr(args, "tui", False) and args.non_interactive_prompt is not None:
        parser.error("--tui cannot be combined with -n/--non-interactive.")

    if args.message_prompt is not None and (args.quiet or args.no_stream):
        parser.error("-q/--quiet and --no-stream cannot be combined with -m/--message.")

    if args.command in {"list", "reset", "threads"} and args.resume is not None:
        parser.error("--resume cannot be combined with list/reset/threads commands.")

    if args.command != "threads" and (
        args.command_arg is not None or args.command_arg2 is not None
    ):
        parser.error("Unexpected extra command arguments.")

    if args.command == "reset" and args.agent is None:
        parser.error("reset requires --agent <name>.")

    if args.command == "threads":
        if args.command_arg not in {"list", "ls", "delete"}:
            parser.error("threads requires a subcommand: list, ls, or delete.")

        if args.command_arg in {"list", "ls"} and args.command_arg2 is not None:
            parser.error("threads list/ls does not accept a thread id.")

        if args.command_arg == "delete" and args.command_arg2 is None:
            parser.error("threads delete requires <thread_id>.")


def _initialize_cli_state() -> tuple[CliPaths, CliDefaults] | tuple[None, None]:
    """Bootstrap CLI directories and load persisted defaults."""
    paths = resolve_cli_paths()
    try:
        bootstrap_cli_home(paths)
        defaults = load_cli_defaults(paths)
        ensure_profile_memory(paths, defaults.default_agent)
    except (OSError, ValueError) as exc:
        print(f"error: failed to initialize CLI config: {exc}", file=sys.stderr)
        return None, None

    return paths, defaults


def _format_timestamp(timestamp: float | None) -> str:
    if timestamp is None:
        return "-"

    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat(timespec="seconds")


def _handle_threads_command(
    *,
    paths: CliPaths,
    agent_name: str,
    command: str,
    thread_id: str | None,
) -> int:
    try:
        if command in {"list", "ls"}:
            threads = list_threads(
                db_path=paths.sessions_db_path,
                user_id=CLI_USER_ID,
                agent_name=agent_name,
                limit=200,
            )

            if not threads:
                print(f"No threads found for profile '{agent_name}'.")
                return 0

            print(f"Threads for profile '{agent_name}':")
            print("THREAD_ID\tUPDATED_AT\tCREATED_AT\tMODEL")
            for thread in threads:
                model = thread.model or "-"
                print(
                    f"{thread.session_id}\t{_format_timestamp(thread.updated_at)}\t"
                    f"{_format_timestamp(thread.created_at)}\t{model}"
                )
            return 0

        assert command == "delete"
        assert thread_id is not None

        deleted = delete_thread(
            db_path=paths.sessions_db_path,
            user_id=CLI_USER_ID,
            session_id=thread_id,
        )
        if not deleted:
            print(
                f"error: thread '{thread_id}' was not found for profile '{agent_name}'.",
                file=sys.stderr,
            )
            return 1

        print(f"Deleted thread '{thread_id}' for profile '{agent_name}'.")
        return 0
    except ValueError as exc:
        print(f"error: failed to manage threads: {exc}", file=sys.stderr)
        return 1


def _resolve_resume_thread_id(paths: CliPaths, agent_name: str, resume_value: str) -> str:
    normalized_resume = resume_value.strip()
    if not normalized_resume:
        raise ValueError("--resume cannot be empty.")

    if normalized_resume == "latest":
        latest_thread = get_latest_thread(
            db_path=paths.sessions_db_path,
            user_id=CLI_USER_ID,
            agent_name=agent_name,
        )
        if latest_thread is None:
            raise ValueError(f"No threads found for profile '{agent_name}'.")
        return latest_thread.session_id

    thread = get_thread(
        db_path=paths.sessions_db_path,
        user_id=CLI_USER_ID,
        session_id=normalized_resume,
    )
    if thread is None or thread.agent_name != agent_name:
        raise ValueError(f"Thread '{normalized_resume}' was not found for profile '{agent_name}'.")

    return thread.session_id


def _ensure_active_thread(
    *,
    paths: CliPaths,
    agent_name: str,
    model: str | None,
    resume: str | None,
) -> str:
    if resume is not None:
        return _resolve_resume_thread_id(paths, agent_name, resume)

    thread = create_thread(
        db_path=paths.sessions_db_path,
        user_id=CLI_USER_ID,
        agent_name=agent_name,
        model=model,
    )
    return thread.session_id


def cli_main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by console scripts and module execution."""
    parser = build_parser()

    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        _validate_args(parser, args)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 1

    if args.version:
        print(f"adk-deepagents {__version__}")
        return 0

    paths, defaults = _initialize_cli_state()
    if paths is None or defaults is None:
        return 1

    if args.command == "list":
        for profile in list_profiles(paths):
            print(profile)
        return 0

    if args.command == "reset":
        assert args.agent is not None  # validated by _validate_args()
        try:
            reset_profile_memory(paths, args.agent)
        except (OSError, ValueError) as exc:
            print(f"error: failed to reset profile: {exc}", file=sys.stderr)
            return 1

        print(f"Reset profile '{args.agent}'.")
        return 0

    resolved_agent = args.agent or defaults.default_agent or DEFAULT_AGENT_NAME

    try:
        ensure_profile_memory(paths, resolved_agent)
    except (OSError, ValueError) as exc:
        print(f"error: failed to prepare profile '{resolved_agent}': {exc}", file=sys.stderr)
        return 1

    if args.command == "threads":
        assert args.command_arg is not None  # validated by _validate_args()
        return _handle_threads_command(
            paths=paths,
            agent_name=resolved_agent,
            command=args.command_arg,
            thread_id=args.command_arg2,
        )

    _load_workspace_env()
    resolved_model = resolve_model(args.model, defaults)

    piped_stdin = read_piped_stdin()
    piped_input_for_non_interactive = piped_stdin if args.message_prompt is None else None
    non_interactive_prompt = combine_non_interactive_prompt(
        args.non_interactive_prompt,
        piped_input_for_non_interactive,
    )

    if non_interactive_prompt is None and (args.quiet or args.no_stream):
        print(
            "-q/--quiet and --no-stream require -n/--non-interactive or piped stdin.",
            file=sys.stderr,
        )
        return 2

    should_save_defaults = False

    if args.agent is not None and resolved_agent != defaults.default_agent:
        defaults.default_agent = resolved_agent
        should_save_defaults = True

    if args.model is not None and resolved_model != defaults.default_model:
        defaults.default_model = resolved_model
        should_save_defaults = True

    if should_save_defaults:
        try:
            save_cli_defaults(paths, defaults)
        except (OSError, ValueError) as exc:
            print(f"error: failed to save CLI defaults: {exc}", file=sys.stderr)
            return 1

    try:
        active_thread_id = _ensure_active_thread(
            paths=paths,
            agent_name=resolved_agent,
            model=resolved_model,
            resume=args.resume,
        )
    except ValueError as exc:
        print(f"error: failed to resolve thread: {exc}", file=sys.stderr)
        return 1

    try:
        resources = discover_cli_agent_resources(
            paths=paths,
            agent_name=resolved_agent,
            cwd=Path.cwd(),
        )
    except (OSError, ValueError) as exc:
        print(f"error: failed to resolve memory/skills resources: {exc}", file=sys.stderr)
        return 1

    if non_interactive_prompt is not None:
        if not args.quiet:
            print(f"[thread {active_thread_id}] running non-interactive task", file=sys.stderr)

        return run_non_interactive(
            prompt=non_interactive_prompt,
            model=resolved_model,
            agent_name=resolved_agent,
            user_id=CLI_USER_ID,
            session_id=active_thread_id,
            db_path=paths.sessions_db_path,
            no_stream=args.no_stream,
            shell_allow_list=args.shell_allow_list,
            auto_approve=args.auto_approve,
            memory_sources=resources.memory_sources,
            memory_source_paths=resources.memory_source_paths,
            skills_dirs=resources.skills_dirs,
        )

    if args.tui:
        from adk_deepagents.cli.tui import run_tui

        return run_tui(
            first_prompt=args.message_prompt,
            model=resolved_model,
            agent_name=resolved_agent,
            user_id=CLI_USER_ID,
            session_id=active_thread_id,
            db_path=paths.sessions_db_path,
            auto_approve=args.auto_approve,
            memory_sources=resources.memory_sources,
            memory_source_paths=resources.memory_source_paths,
            skills_dirs=resources.skills_dirs,
        )

    return run_interactive(
        first_prompt=args.message_prompt,
        model=resolved_model,
        agent_name=resolved_agent,
        user_id=CLI_USER_ID,
        session_id=active_thread_id,
        db_path=paths.sessions_db_path,
        auto_approve=args.auto_approve,
        memory_sources=resources.memory_sources,
        memory_source_paths=resources.memory_source_paths,
        skills_dirs=resources.skills_dirs,
    )
