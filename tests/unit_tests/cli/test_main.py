"""Unit tests for CLI entrypoints and argument parsing."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from adk_deepagents import __version__
from adk_deepagents.cli.main import build_parser, cli_main


def test_build_parser_parses_non_interactive_surface_flags() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--agent",
            "demo",
            "--model",
            "gemini-2.5-pro",
            "-n",
            "ship it",
            "--resume",
            "--auto-approve",
            "--shell-allow-list",
            "git,ls",
        ]
    )

    assert args.agent == "demo"
    assert args.model == "gemini-2.5-pro"
    assert args.non_interactive_prompt == "ship it"
    assert args.message_prompt is None
    assert args.resume == "latest"
    assert args.auto_approve is True
    assert args.shell_allow_list == ["git", "ls"]


def test_build_parser_parses_message_mode_flag() -> None:
    parser = build_parser()

    args = parser.parse_args(["-m", "hello", "--resume", "thread-123"])

    assert args.message_prompt == "hello"
    assert args.non_interactive_prompt is None
    assert args.resume == "thread-123"


def test_cli_main_help_returns_exit_code_zero(capsys) -> None:
    exit_code = cli_main(["--help"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "adk-deepagents" in captured.out
    assert "--resume" in captured.out
    assert "--shell-allow-list" in captured.out


def test_cli_main_version_flag_prints_version(capsys) -> None:
    exit_code = cli_main(["--version"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert f"adk-deepagents {__version__}" in captured.out


def test_cli_main_usage_errors_return_exit_code_two(capsys) -> None:
    exit_code = cli_main(["--does-not-exist"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "unrecognized arguments: --does-not-exist" in captured.err


def test_cli_main_validation_errors_return_exit_code_two(capsys) -> None:
    exit_code = cli_main(["--shell-allow-list", "git"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "--shell-allow-list requires -n/--non-interactive" in captured.err


def test_python_module_entrypoint_help_works() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "adk_deepagents.cli", "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert "adk-deepagents" in result.stdout
    assert "--model" in result.stdout


def test_pyproject_declares_adk_deepagents_console_script() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["adk-deepagents"] == "adk_deepagents.cli.main:cli_main"
