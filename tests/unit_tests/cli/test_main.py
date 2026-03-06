"""Unit tests for CLI entrypoints, parser behavior, and profile config persistence."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import adk_deepagents.cli.main as cli_main_module
from adk_deepagents import __version__
from adk_deepagents.cli.config import (
    CONFIG_FILENAME,
    PROFILE_MEMORY_FILENAME,
    PROFILES_DIRNAME,
    SESSIONS_DB_FILENAME,
    CliDefaults,
    default_profile_template,
)
from adk_deepagents.cli.main import (
    MODEL_ENV_VAR,
    _load_workspace_env,
    build_parser,
    cli_main,
    resolve_model,
)
from adk_deepagents.cli.session_store import get_latest_thread, list_threads


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
            "-q",
            "--no-stream",
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
    assert args.quiet is True
    assert args.no_stream is True
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
    assert args.command_arg is None
    assert args.command_arg2 is None
    assert args.agent == "demo"


def test_build_parser_parses_threads_subcommands() -> None:
    parser = build_parser()

    args = parser.parse_args(["threads", "delete", "thread-123", "--agent", "demo"])

    assert args.command == "threads"
    assert args.command_arg == "delete"
    assert args.command_arg2 == "thread-123"
    assert args.agent == "demo"


def test_cli_main_help_returns_exit_code_zero(capsys) -> None:
    exit_code = cli_main(["--help"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "adk-deepagents" in captured.out
    assert "list" in captured.out
    assert "reset" in captured.out
    assert "threads" in captured.out
    assert "--shell-allow-list" in captured.out
    assert "--quiet" in captured.out
    assert "--no-stream" in captured.out
    assert MODEL_ENV_VAR in captured.out


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


def test_cli_main_threads_requires_subcommand(capsys) -> None:
    exit_code = cli_main(["threads"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "threads requires a subcommand" in captured.err


def test_cli_main_threads_delete_requires_thread_id(capsys) -> None:
    exit_code = cli_main(["threads", "delete"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "threads delete requires <thread_id>" in captured.err


def test_cli_main_rejects_resume_with_threads_command(capsys) -> None:
    exit_code = cli_main(["threads", "list", "--resume"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "--resume cannot be combined with list/reset/threads commands" in captured.err


def test_cli_main_rejects_empty_model_flag(capsys) -> None:
    exit_code = cli_main(["--model", "   "])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "--model cannot be empty." in captured.err


def test_cli_main_rejects_quiet_with_message_mode(capsys) -> None:
    exit_code = cli_main(["-m", "hello", "-q"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "cannot be combined with -m/--message" in captured.err


def test_resolve_model_prefers_cli_over_env_and_config(monkeypatch) -> None:
    defaults = CliDefaults(default_agent="agent", default_model="config-model")
    monkeypatch.setenv(MODEL_ENV_VAR, "env-model")

    assert resolve_model("cli-model", defaults) == "cli-model"


def test_resolve_model_falls_back_to_env_then_config(monkeypatch) -> None:
    defaults = CliDefaults(default_agent="agent", default_model="config-model")

    monkeypatch.setenv(MODEL_ENV_VAR, "env-model")
    assert resolve_model(None, defaults) == "env-model"

    monkeypatch.setenv(MODEL_ENV_VAR, "   ")
    assert resolve_model(None, defaults) == "config-model"

    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    assert resolve_model(None, defaults) == "config-model"


def test_resolve_model_returns_none_when_all_sources_missing(monkeypatch) -> None:
    defaults = CliDefaults(default_agent="agent", default_model=None)
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)

    assert resolve_model(None, defaults) is None


def test_workspace_dotenv_does_not_override_exported_model(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(f"{MODEL_ENV_VAR}=dotenv-model\n", encoding="utf-8")
    monkeypatch.setenv(MODEL_ENV_VAR, "shell-model")

    _load_workspace_env()

    assert os.environ[MODEL_ENV_VAR] == "shell-model"


def test_workspace_dotenv_model_fallback_is_used_when_env_is_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(f"{MODEL_ENV_VAR}=dotenv-model\n", encoding="utf-8")
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)

    _load_workspace_env()
    defaults = CliDefaults(default_agent="agent", default_model="config-model")

    assert resolve_model(None, defaults) == "dotenv-model"


def test_cli_bootstrap_creates_config_and_profile_paths(tmp_path, monkeypatch) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))

    exit_code = cli_main([])

    assert exit_code == 0
    assert home_dir.is_dir()
    assert (home_dir / CONFIG_FILENAME).is_file()
    assert (home_dir / PROFILES_DIRNAME / "agent" / PROFILE_MEMORY_FILENAME).is_file()
    assert (home_dir / SESSIONS_DB_FILENAME).is_file()


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


def test_threads_list_and_ls_are_profile_aware_and_sorted(tmp_path, monkeypatch, capsys) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))

    assert cli_main(["--agent", "alpha"]) == 0
    assert cli_main(["--agent", "beta"]) == 0
    assert cli_main(["--agent", "alpha"]) == 0

    expected = list_threads(
        db_path=home_dir / SESSIONS_DB_FILENAME,
        user_id="local",
        agent_name="alpha",
        limit=20,
    )

    capsys.readouterr()
    assert cli_main(["threads", "list", "--agent", "alpha"]) == 0
    list_out = capsys.readouterr().out

    capsys.readouterr()
    assert cli_main(["threads", "ls", "--agent", "alpha"]) == 0
    ls_out = capsys.readouterr().out

    list_ids = [line.split("\t", maxsplit=1)[0] for line in list_out.strip().splitlines()[2:]]
    expected_ids = [thread.session_id for thread in expected]

    assert list_ids == expected_ids
    assert "beta" not in list_out
    assert ls_out == list_out


def test_threads_delete_removes_session(tmp_path, monkeypatch, capsys) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))

    assert cli_main(["--agent", "demo"]) == 0
    latest = get_latest_thread(
        db_path=home_dir / SESSIONS_DB_FILENAME,
        user_id="local",
        agent_name="demo",
    )
    assert latest is not None

    capsys.readouterr()
    delete_exit = cli_main(["threads", "delete", latest.session_id, "--agent", "demo"])
    delete_output = capsys.readouterr()

    assert delete_exit == 0
    assert "Deleted thread" in delete_output.out

    capsys.readouterr()
    missing_exit = cli_main(["threads", "delete", latest.session_id, "--agent", "demo"])
    missing_output = capsys.readouterr()

    assert missing_exit == 1
    assert "was not found" in missing_output.err


def test_resume_latest_requires_existing_thread(tmp_path, monkeypatch, capsys) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))

    exit_code = cli_main(["--agent", "demo", "--resume"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to resolve thread" in captured.err


def test_resume_supports_latest_and_explicit_ids(tmp_path, monkeypatch, capsys) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))

    assert cli_main(["--agent", "demo"]) == 0

    threads_before = list_threads(
        db_path=home_dir / SESSIONS_DB_FILENAME,
        user_id="local",
        agent_name="demo",
        limit=20,
    )
    assert len(threads_before) == 1

    thread_id = threads_before[0].session_id
    assert cli_main(["--agent", "demo", "--resume"]) == 0
    assert cli_main(["--agent", "demo", "--resume", thread_id]) == 0

    threads_after = list_threads(
        db_path=home_dir / SESSIONS_DB_FILENAME,
        user_id="local",
        agent_name="demo",
        limit=20,
    )
    assert [thread.session_id for thread in threads_after] == [thread_id]

    exit_code = cli_main(["--agent", "demo", "--resume", "missing-thread"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "failed to resolve thread" in captured.err


def test_cli_main_non_interactive_merges_piped_stdin_and_flag_prompt(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))
    monkeypatch.setattr(cli_main_module, "read_piped_stdin", lambda: "piped context")

    captured_kwargs: dict[str, object] = {}

    def _fake_run_non_interactive(**kwargs: object) -> int:
        captured_kwargs.update(kwargs)
        return 0

    monkeypatch.setattr(cli_main_module, "run_non_interactive", _fake_run_non_interactive)

    exit_code = cli_main(["--agent", "demo", "-n", "implement fix", "-q"])
    _ = capsys.readouterr()

    assert exit_code == 0
    assert captured_kwargs["prompt"] == "piped context\n\nimplement fix"
    assert captured_kwargs["agent_name"] == "demo"


def test_cli_main_non_interactive_forwards_no_stream_flag(tmp_path, monkeypatch, capsys) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))
    monkeypatch.setattr(cli_main_module, "read_piped_stdin", lambda: None)

    captured_kwargs: dict[str, object] = {}

    def _fake_run_non_interactive(**kwargs: object) -> int:
        captured_kwargs.update(kwargs)
        return 0

    monkeypatch.setattr(cli_main_module, "run_non_interactive", _fake_run_non_interactive)

    exit_code = cli_main(["--agent", "demo", "-n", "hello", "--no-stream", "-q"])
    _ = capsys.readouterr()

    assert exit_code == 0
    assert captured_kwargs["no_stream"] is True
    assert captured_kwargs["shell_allow_list"] is None
    assert captured_kwargs["auto_approve"] is False


def test_cli_main_non_interactive_forwards_shell_allow_list_and_auto_approve(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))
    monkeypatch.setattr(cli_main_module, "read_piped_stdin", lambda: None)

    captured_kwargs: dict[str, object] = {}

    def _fake_run_non_interactive(**kwargs: object) -> int:
        captured_kwargs.update(kwargs)
        return 0

    monkeypatch.setattr(cli_main_module, "run_non_interactive", _fake_run_non_interactive)

    exit_code = cli_main(
        [
            "--agent",
            "demo",
            "-n",
            "hello",
            "--shell-allow-list",
            "git,ls",
            "--auto-approve",
            "-q",
        ]
    )
    _ = capsys.readouterr()

    assert exit_code == 0
    assert captured_kwargs["shell_allow_list"] == ["git", "ls"]
    assert captured_kwargs["auto_approve"] is True


def test_cli_main_non_interactive_quiet_suppresses_status_line(
    tmp_path, monkeypatch, capsys
) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))
    monkeypatch.setattr(cli_main_module, "read_piped_stdin", lambda: None)
    monkeypatch.setattr(cli_main_module, "run_non_interactive", lambda **_: 0)

    exit_code = cli_main(["--agent", "demo", "-n", "hello", "-q"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "running non-interactive task" not in captured.err


def test_cli_main_non_interactive_prints_status_when_not_quiet(
    tmp_path, monkeypatch, capsys
) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))
    monkeypatch.setattr(cli_main_module, "read_piped_stdin", lambda: None)
    monkeypatch.setattr(cli_main_module, "run_non_interactive", lambda **_: 0)

    exit_code = cli_main(["--agent", "demo", "-n", "hello"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "running non-interactive task" in captured.err


def test_cli_main_non_interactive_returns_runner_exit_code(tmp_path, monkeypatch, capsys) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))
    monkeypatch.setattr(cli_main_module, "read_piped_stdin", lambda: None)
    monkeypatch.setattr(cli_main_module, "run_non_interactive", lambda **_: 1)

    exit_code = cli_main(["--agent", "demo", "-n", "hello", "-q"])
    _ = capsys.readouterr()

    assert exit_code == 1


def test_cli_main_interactive_mode_invokes_repl_runner(tmp_path, monkeypatch, capsys) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))
    monkeypatch.setattr(cli_main_module, "read_piped_stdin", lambda: None)

    captured_kwargs: dict[str, object] = {}

    def _fake_run_interactive(**kwargs: object) -> int:
        captured_kwargs.update(kwargs)
        return 0

    monkeypatch.setattr(cli_main_module, "run_interactive", _fake_run_interactive)

    exit_code = cli_main(["--agent", "demo"])
    _ = capsys.readouterr()

    assert exit_code == 0
    assert captured_kwargs["first_prompt"] is None
    assert captured_kwargs["agent_name"] == "demo"
    assert captured_kwargs["auto_approve"] is False


def test_cli_main_message_mode_passes_first_prompt_to_repl(tmp_path, monkeypatch, capsys) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))
    monkeypatch.setattr(cli_main_module, "read_piped_stdin", lambda: None)

    captured_kwargs: dict[str, object] = {}

    def _fake_run_interactive(**kwargs: object) -> int:
        captured_kwargs.update(kwargs)
        return 0

    monkeypatch.setattr(cli_main_module, "run_interactive", _fake_run_interactive)

    exit_code = cli_main(["--agent", "demo", "-m", "hello there"])
    _ = capsys.readouterr()

    assert exit_code == 0
    assert captured_kwargs["first_prompt"] == "hello there"
    assert captured_kwargs["session_id"]


def test_cli_main_interactive_forwards_auto_approve_flag(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))
    monkeypatch.setattr(cli_main_module, "read_piped_stdin", lambda: None)

    captured_kwargs: dict[str, object] = {}

    def _fake_run_interactive(**kwargs: object) -> int:
        captured_kwargs.update(kwargs)
        return 0

    monkeypatch.setattr(cli_main_module, "run_interactive", _fake_run_interactive)

    exit_code = cli_main(["--agent", "demo", "--auto-approve"])
    _ = capsys.readouterr()

    assert exit_code == 0
    assert captured_kwargs["auto_approve"] is True


def test_cli_main_interactive_forwards_resolved_model_precedence(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    monkeypatch.setattr(cli_main_module, "read_piped_stdin", lambda: None)

    seen_models: list[str | None] = []

    def _fake_run_interactive(**kwargs: object) -> int:
        model = kwargs.get("model")
        assert model is None or isinstance(model, str)
        seen_models.append(model)
        return 0

    monkeypatch.setattr(cli_main_module, "run_interactive", _fake_run_interactive)

    assert cli_main(["--agent", "demo", "--model", "config-model"]) == 0
    _ = capsys.readouterr()

    monkeypatch.setenv(MODEL_ENV_VAR, "env-model")
    assert cli_main(["--agent", "demo"]) == 0
    _ = capsys.readouterr()

    assert cli_main(["--agent", "demo", "--model", "cli-model"]) == 0
    _ = capsys.readouterr()

    assert seen_models == ["config-model", "env-model", "cli-model"]


def test_cli_main_quiet_requires_non_interactive_prompt_or_piped_stdin(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))
    monkeypatch.setattr(cli_main_module, "read_piped_stdin", lambda: None)

    exit_code = cli_main(["--quiet"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "require -n/--non-interactive or piped stdin" in captured.err


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
