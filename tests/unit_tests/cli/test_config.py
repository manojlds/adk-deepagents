"""Unit tests for CLI TOML configuration helpers."""

from __future__ import annotations

import tomllib

import pytest

from adk_deepagents.cli.config import (
    CliDefaults,
    load_cli_defaults,
    resolve_cli_paths,
    save_cli_defaults,
)


def test_load_cli_defaults_parses_dynamic_task_table(tmp_path) -> None:
    paths = resolve_cli_paths(tmp_path)
    paths.root_dir.mkdir(parents=True, exist_ok=True)
    paths.config_path.write_text(
        (
            'default_agent = "demo"\n'
            'default_model = "openai/glm-5"\n'
            "\n"
            "[dynamic_task]\n"
            "max_parallel = 6\n"
            'concurrency_policy = "wait"\n'
            "queue_timeout_seconds = 45.5\n"
        ),
        encoding="utf-8",
    )

    defaults = load_cli_defaults(paths)

    assert defaults.default_agent == "demo"
    assert defaults.default_model == "openai/glm-5"
    assert defaults.dynamic_task_max_parallel == 6
    assert defaults.dynamic_task_concurrency_policy == "wait"
    assert defaults.dynamic_task_queue_timeout_seconds == 45.5


def test_load_cli_defaults_rejects_invalid_dynamic_task_values(tmp_path) -> None:
    paths = resolve_cli_paths(tmp_path)
    paths.root_dir.mkdir(parents=True, exist_ok=True)
    paths.config_path.write_text(
        ('default_agent = "demo"\n\n[dynamic_task]\nmax_parallel = 0\n'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_parallel"):
        load_cli_defaults(paths)


def test_save_cli_defaults_writes_dynamic_task_table(tmp_path) -> None:
    paths = resolve_cli_paths(tmp_path)
    paths.root_dir.mkdir(parents=True, exist_ok=True)

    save_cli_defaults(
        paths,
        CliDefaults(
            default_agent="demo",
            default_model="openai/glm-5",
            dynamic_task_max_parallel=8,
            dynamic_task_concurrency_policy="error",
            dynamic_task_queue_timeout_seconds=9.25,
        ),
    )

    data = tomllib.loads(paths.config_path.read_text(encoding="utf-8"))
    assert data["default_agent"] == "demo"
    assert data["default_model"] == "openai/glm-5"
    assert data["dynamic_task"]["max_parallel"] == 8
    assert data["dynamic_task"]["concurrency_policy"] == "error"
    assert data["dynamic_task"]["queue_timeout_seconds"] == 9.25


def test_load_cli_defaults_parses_tui_keybinds(tmp_path) -> None:
    paths = resolve_cli_paths(tmp_path)
    paths.root_dir.mkdir(parents=True, exist_ok=True)
    paths.config_path.write_text(
        (
            'default_agent = "agent"\n'
            "\n"
            "[tui.keybinds]\n"
            'app_quit = "ctrl+q"\n'
            'session_new = "ctrl+x n"\n'
        ),
        encoding="utf-8",
    )

    defaults = load_cli_defaults(paths)
    assert defaults.tui_keybinds == {"app_quit": "ctrl+q", "session_new": "ctrl+x n"}


def test_load_cli_defaults_no_tui_section(tmp_path) -> None:
    paths = resolve_cli_paths(tmp_path)
    paths.root_dir.mkdir(parents=True, exist_ok=True)
    paths.config_path.write_text(
        'default_agent = "agent"\n',
        encoding="utf-8",
    )

    defaults = load_cli_defaults(paths)
    assert defaults.tui_keybinds is None


def test_load_cli_defaults_rejects_invalid_tui_keybinds(tmp_path) -> None:
    paths = resolve_cli_paths(tmp_path)
    paths.root_dir.mkdir(parents=True, exist_ok=True)
    paths.config_path.write_text(
        'default_agent = "agent"\n\n[tui]\nkeybinds = "bad"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[tui\.keybinds\]"):
        load_cli_defaults(paths)


def test_load_cli_defaults_rejects_invalid_tui_table(tmp_path) -> None:
    paths = resolve_cli_paths(tmp_path)
    paths.root_dir.mkdir(parents=True, exist_ok=True)
    paths.config_path.write_text(
        'default_agent = "agent"\ntui = "bad"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[tui\]"):
        load_cli_defaults(paths)


def test_save_cli_defaults_writes_tui_keybinds(tmp_path) -> None:
    paths = resolve_cli_paths(tmp_path)
    paths.root_dir.mkdir(parents=True, exist_ok=True)

    save_cli_defaults(
        paths,
        CliDefaults(
            default_agent="agent",
            tui_keybinds={"app_quit": "ctrl+q", "session_new": "ctrl+x n"},
        ),
    )

    data = tomllib.loads(paths.config_path.read_text(encoding="utf-8"))
    assert data["tui"]["keybinds"]["app_quit"] == "ctrl+q"
    assert data["tui"]["keybinds"]["session_new"] == "ctrl+x n"


def test_save_cli_defaults_omits_tui_section_when_none(tmp_path) -> None:
    paths = resolve_cli_paths(tmp_path)
    paths.root_dir.mkdir(parents=True, exist_ok=True)

    save_cli_defaults(paths, CliDefaults(default_agent="agent"))

    data = tomllib.loads(paths.config_path.read_text(encoding="utf-8"))
    assert "tui" not in data
