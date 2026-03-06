"""Main command-line entrypoint for adk-deepagents."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from adk_deepagents import __version__

DEFAULT_AGENT_NAME = "agent"


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
        "-a",
        "--agent",
        default=DEFAULT_AGENT_NAME,
        help="Agent profile name.",
    )
    parser.add_argument(
        "--model",
        help="Override the model for this run.",
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

    return 0
