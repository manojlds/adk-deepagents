"""Unit tests for CLI entrypoints, parser behavior, and profile config persistence."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from adk_deepagents import __version__
from adk_deepagents.cli.config import (
    CONFIG_FILENAME,
    PROFILE_MEMORY_FILENAME,
    PROFILES_DIRNAME,
    default_profile_template,
)
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

    assert args.command is None
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

    assert args.command is None
    assert args.agent is None
    assert args.message_prompt == "hello"
    assert args.non_interactive_prompt is None
    assert args.resume == "thread-123"


def test_build_parser_parses_profile_commands() -> None:
    parser = build_parser()

    args = parser.parse_args(["reset", "--agent", "demo"])

    assert args.command == "reset"
    assert args.agent == "demo"


def test_cli_main_help_returns_exit_code_zero(capsys) -> None:
    exit_code = cli_main(["--help"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "adk-deepagents" in captured.out
    assert "list" in captured.out
    assert "reset" in captured.out
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


def test_cli_main_reset_requires_agent(capsys) -> None:
    exit_code = cli_main(["reset"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "reset requires --agent <name>" in captured.err


def test_cli_bootstrap_creates_config_and_profile_paths(tmp_path, monkeypatch) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))

    exit_code = cli_main([])

    assert exit_code == 0
    assert home_dir.is_dir()
    assert (home_dir / CONFIG_FILENAME).is_file()
    assert (home_dir / PROFILES_DIRNAME / "agent" / PROFILE_MEMORY_FILENAME).is_file()


def test_cli_loads_and_saves_defaults_across_reruns(tmp_path, monkeypatch) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))

    first_exit = cli_main(["--agent", "demo", "--model", "gemini-2.5-flash"])
    assert first_exit == 0

    config_path = home_dir / CONFIG_FILENAME
    config_data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert config_data == {
        "default_agent": "demo",
        "default_model": "gemini-2.5-flash",
    }

    demo_profile = home_dir / PROFILES_DIRNAME / "demo"
    assert demo_profile.is_dir()

    # Re-run with no flags should load the persisted default profile name.
    shutil.rmtree(demo_profile)
    second_exit = cli_main([])
    assert second_exit == 0
    assert (demo_profile / PROFILE_MEMORY_FILENAME).is_file()


def test_list_command_shows_available_profiles(tmp_path, monkeypatch, capsys) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))

    assert cli_main(["--agent", "zeta"]) == 0
    assert cli_main(["--agent", "alpha"]) == 0

    capsys.readouterr()  # clear setup output
    exit_code = cli_main(["list"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip().splitlines() == ["agent", "alpha", "zeta"]


def test_reset_command_resets_profile_memory_template(tmp_path, monkeypatch, capsys) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))

    assert cli_main(["--agent", "demo"]) == 0
    memory_path = home_dir / PROFILES_DIRNAME / "demo" / PROFILE_MEMORY_FILENAME
    memory_path.write_text("custom instructions", encoding="utf-8")

    capsys.readouterr()  # clear setup output
    exit_code = cli_main(["reset", "--agent", "demo"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Reset profile 'demo'." in captured.out
    assert memory_path.read_text(encoding="utf-8") == default_profile_template("demo")


def test_reset_rejects_invalid_profile_name(tmp_path, monkeypatch, capsys) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))

    exit_code = cli_main(["reset", "--agent", "../oops"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to reset profile" in captured.err


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
