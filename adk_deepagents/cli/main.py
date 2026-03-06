"""Main command-line entrypoint for adk-deepagents."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
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

MODEL_ENV_VAR = "ADK_DEEPAGENTS_MODEL"


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
        choices=("list", "reset"),
        help="Profile command to run.",
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

    if args.command in {"list", "reset"} and (
        args.non_interactive_prompt is not None or args.message_prompt is not None
    ):
        parser.error(
            "list/reset commands cannot be combined with -n/--non-interactive or -m/--message."
        )

    if args.command == "reset" and args.agent is None:
        parser.error("reset requires --agent <name>.")


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

    _load_workspace_env()

    resolved_agent = args.agent or defaults.default_agent or DEFAULT_AGENT_NAME
    resolved_model = resolve_model(args.model, defaults)

    try:
        ensure_profile_memory(paths, resolved_agent)
    except (OSError, ValueError) as exc:
        print(f"error: failed to prepare profile '{resolved_agent}': {exc}", file=sys.stderr)
        return 1

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

    return 0
