"""Unit tests for CLI path and profile helpers."""

from __future__ import annotations

from pathlib import Path

from adk_deepagents.cli import paths


def test_get_cli_home_uses_env_override(tmp_path: Path, monkeypatch) -> None:
    custom_home = tmp_path / "cli-home"
    monkeypatch.setenv(paths.CLI_HOME_ENV, str(custom_home))

    assert paths.get_cli_home() == custom_home.resolve()


def test_bootstrap_agent_profile_creates_agents_memory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(paths.CLI_HOME_ENV, str(tmp_path / "state"))

    profile_dir = paths.bootstrap_agent_profile("demo")
    memory_path = profile_dir / paths.AGENT_MEMORY_FILENAME

    assert profile_dir.exists()
    assert memory_path.exists()
    assert "persistent preferences" in memory_path.read_text(encoding="utf-8")


def test_reset_agent_memory_overwrites_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(paths.CLI_HOME_ENV, str(tmp_path / "state"))

    memory_path = paths.reset_agent_memory("demo", template="# custom")

    assert memory_path.read_text(encoding="utf-8") == "# custom\n"


def test_list_agent_profiles_returns_sorted_profiles(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(paths.CLI_HOME_ENV, str(tmp_path / "state"))

    paths.ensure_agent_profile("zeta")
    paths.ensure_agent_profile("alpha")

    assert paths.list_agent_profiles() == ["alpha", "zeta"]


def test_get_sessions_db_path_is_under_cli_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(paths.CLI_HOME_ENV, str(tmp_path / "state"))

    assert paths.get_sessions_db_path() == (tmp_path / "state" / paths.SESSIONS_DB_FILENAME)
