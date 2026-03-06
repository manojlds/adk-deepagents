"""Unit tests for CLI argument parsing and command dispatch."""

from __future__ import annotations

from pathlib import Path

import adk_deepagents.cli.main as cli_main_module
from adk_deepagents import __version__
from adk_deepagents.cli.main import build_parser, cli_main
from adk_deepagents.cli.paths import CLI_HOME_ENV


def test_build_parser_parses_list_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["list"])

    assert args.command == "list"
    assert args.agent == "agent"


def test_build_parser_parses_resume_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["--resume"])

    assert args.resume == "latest"


def test_build_parser_parses_non_interactive_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["-n", "hello", "-q", "--no-stream"])

    assert args.non_interactive_prompt == "hello"
    assert args.quiet is True
    assert args.no_stream is True


def test_cli_version_flag_prints_version(capsys) -> None:
    exit_code = cli_main(["--version"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert f"adk-deepagents {__version__}" in captured.out


def test_cli_list_handles_empty_profiles(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(CLI_HOME_ENV, str(tmp_path / "home"))

    exit_code = cli_main(["list"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No agent profiles configured yet." in captured.out


def test_cli_reset_creates_memory_file(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(CLI_HOME_ENV, str(tmp_path / "home"))

    exit_code = cli_main(["reset", "--agent", "demo"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Reset memory template" in captured.out
    assert (tmp_path / "home" / "demo" / "AGENTS.md").exists()


def test_cli_default_bootstraps_profile(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(CLI_HOME_ENV, str(tmp_path / "home"))

    exit_code = cli_main(["--agent", "demo"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "CLI foundation is ready" in captured.out
    assert "Created thread:" in captured.out
    assert (tmp_path / "home" / "demo" / "AGENTS.md").exists()


def test_cli_threads_list_shows_created_thread(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(CLI_HOME_ENV, str(tmp_path / "home"))

    cli_main(["--agent", "demo"])
    exit_code = cli_main(["threads", "list", "--agent", "demo"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "THREAD_ID" in captured.out
    assert "demo" in captured.out


def test_cli_resume_latest_reports_resumed_thread(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(CLI_HOME_ENV, str(tmp_path / "home"))

    cli_main(["--agent", "demo"])
    exit_code = cli_main(["--agent", "demo", "--resume"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Resumed thread:" in captured.out


def test_cli_threads_delete_removes_thread(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(CLI_HOME_ENV, str(tmp_path / "home"))

    cli_main(["--agent", "demo"])
    capsys.readouterr()

    cli_main(["threads", "list", "--agent", "demo"])
    listed = capsys.readouterr().out
    first_data_line = [
        line for line in listed.splitlines() if line and not line.startswith("THREAD_ID")
    ][0]
    thread_id = first_data_line.split()[0]

    exit_code = cli_main(["threads", "delete", thread_id])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert f"Deleted thread: {thread_id}" in captured.out


def test_cli_non_interactive_dispatches_with_flag_prompt(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv(CLI_HOME_ENV, str(tmp_path / "home"))
    monkeypatch.setattr(cli_main_module, "read_piped_stdin", lambda: None)

    captured_kwargs: dict = {}

    def fake_run_non_interactive(**kwargs):
        captured_kwargs.update(kwargs)
        return 0

    monkeypatch.setattr(cli_main_module, "run_non_interactive", fake_run_non_interactive)

    exit_code = cli_main(["--agent", "demo", "-n", "Say hello", "-q"])
    _ = capsys.readouterr()

    assert exit_code == 0
    assert captured_kwargs["prompt"] == "Say hello"
    assert captured_kwargs["agent_name"] == "demo"
    assert "session_id" in captured_kwargs


def test_cli_non_interactive_merges_piped_and_flag_prompt(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv(CLI_HOME_ENV, str(tmp_path / "home"))
    monkeypatch.setattr(cli_main_module, "read_piped_stdin", lambda: "piped content")

    captured_kwargs: dict = {}

    def fake_run_non_interactive(**kwargs):
        captured_kwargs.update(kwargs)
        return 0

    monkeypatch.setattr(cli_main_module, "run_non_interactive", fake_run_non_interactive)

    exit_code = cli_main(["--agent", "demo", "-n", "task prompt"])
    _ = capsys.readouterr()

    assert exit_code == 0
    assert captured_kwargs["prompt"] == "piped content\n\ntask prompt"


def test_cli_non_interactive_uses_pipe_only_prompt(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv(CLI_HOME_ENV, str(tmp_path / "home"))
    monkeypatch.setattr(cli_main_module, "read_piped_stdin", lambda: "pipe task")

    captured_kwargs: dict = {}

    def fake_run_non_interactive(**kwargs):
        captured_kwargs.update(kwargs)
        return 0

    monkeypatch.setattr(cli_main_module, "run_non_interactive", fake_run_non_interactive)

    exit_code = cli_main(["--agent", "demo"])
    _ = capsys.readouterr()

    assert exit_code == 0
    assert captured_kwargs["prompt"] == "pipe task"


def test_cli_non_interactive_forwards_no_stream_flag(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv(CLI_HOME_ENV, str(tmp_path / "home"))
    monkeypatch.setattr(cli_main_module, "read_piped_stdin", lambda: None)

    captured_kwargs: dict = {}

    def fake_run_non_interactive(**kwargs):
        captured_kwargs.update(kwargs)
        return 0

    monkeypatch.setattr(cli_main_module, "run_non_interactive", fake_run_non_interactive)

    exit_code = cli_main(["--agent", "demo", "-n", "hello", "--no-stream"])
    _ = capsys.readouterr()

    assert exit_code == 0
    assert captured_kwargs["no_stream"] is True


def test_cli_non_interactive_prints_status_when_not_quiet(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv(CLI_HOME_ENV, str(tmp_path / "home"))
    monkeypatch.setattr(cli_main_module, "read_piped_stdin", lambda: None)
    monkeypatch.setattr(cli_main_module, "run_non_interactive", lambda **_: 0)

    exit_code = cli_main(["--agent", "demo", "-n", "hello"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "running non-interactive task" in captured.err


def test_cli_non_interactive_returns_runner_exit_code(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv(CLI_HOME_ENV, str(tmp_path / "home"))
    monkeypatch.setattr(cli_main_module, "read_piped_stdin", lambda: None)
    monkeypatch.setattr(cli_main_module, "run_non_interactive", lambda **_: 1)

    exit_code = cli_main(["--agent", "demo", "-n", "hello", "-q"])
    _ = capsys.readouterr()

    assert exit_code == 1


def test_cli_non_interactive_resume_missing_thread_returns_error(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv(CLI_HOME_ENV, str(tmp_path / "home"))
    monkeypatch.setattr(cli_main_module, "read_piped_stdin", lambda: None)

    exit_code = cli_main(["--agent", "demo", "-n", "hello", "--resume", "missing-thread-id"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Thread not found: missing-thread-id" in captured.err


def test_cli_quiet_requires_non_interactive_or_pipe(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv(CLI_HOME_ENV, str(tmp_path / "home"))
    monkeypatch.setattr(cli_main_module, "read_piped_stdin", lambda: None)

    exit_code = cli_main(["--quiet"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "require -n/--non-interactive or piped stdin" in captured.err
