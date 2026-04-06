"""Unit tests for non-interactive CLI helpers."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TextIO, cast
from unittest.mock import MagicMock

import pytest

import adk_deepagents.cli.non_interactive as ni
from adk_deepagents.types import DeepAgentConfig


class _FakeStdin(io.StringIO):
    def __init__(self, data: str, *, isatty_value: bool):
        super().__init__(data)
        self._isatty_value = isatty_value

    def isatty(self) -> bool:
        return self._isatty_value


class _UnreadableStdin:
    def isatty(self) -> bool:
        return False

    def read(self) -> str:
        raise OSError("stdin not readable")


class _FakePart:
    def __init__(self, text: str | None):
        self.text = text


class _FakeContent:
    def __init__(self, parts: list[_FakePart]):
        self.parts = parts


class _FakeEvent:
    def __init__(self, author: str, parts: list[_FakePart] | None):
        self.author = author
        self.content = _FakeContent(parts) if parts is not None else None


class _FakeRunner:
    def __init__(self, *, app_name, agent, session_service):
        del app_name, agent, session_service

    async def run_async(self, *, user_id, session_id, new_message):
        del user_id, session_id, new_message
        yield _FakeEvent("user", [_FakePart("ignore me")])
        yield _FakeEvent("assistant", [_FakePart("hello")])
        yield _FakeEvent("assistant", [_FakePart(" world")])


class _FakeTool:
    def __init__(self, name: str):
        self.name = name


def test_read_piped_stdin_returns_none_for_tty() -> None:
    stdin = _FakeStdin("hello", isatty_value=True)

    assert ni.read_piped_stdin(stdin) is None


def test_read_piped_stdin_returns_data_for_pipe() -> None:
    stdin = _FakeStdin("hello", isatty_value=False)

    assert ni.read_piped_stdin(stdin) == "hello"


def test_read_piped_stdin_returns_none_for_unreadable_stream() -> None:
    assert ni.read_piped_stdin(cast(TextIO, _UnreadableStdin())) is None


def test_combine_non_interactive_prompt_merges_inputs() -> None:
    combined = ni.combine_non_interactive_prompt("task", "pipe")
    assert combined == "pipe\n\ntask"


def test_combine_non_interactive_prompt_uses_single_source() -> None:
    assert ni.combine_non_interactive_prompt("task", None) == "task"
    assert ni.combine_non_interactive_prompt(None, "pipe") == "pipe"
    assert ni.combine_non_interactive_prompt(None, None) is None


def test_normalize_shell_allow_list_expands_recommended() -> None:
    allow_list = ni.normalize_shell_allow_list(["recommended", "git"])

    assert "git" in allow_list
    assert "ls" in allow_list


def test_non_interactive_policy_blocks_shell_by_default() -> None:
    callback = ni.build_non_interactive_before_tool_callback(
        shell_allow_list=None,
        auto_approve=False,
    )

    with pytest.raises(ni.NonInteractivePolicyError, match="Shell execution is blocked"):
        callback(_FakeTool("execute"), {"command": "git status"}, MagicMock())


def test_non_interactive_policy_allows_configured_shell_command() -> None:
    callback = ni.build_non_interactive_before_tool_callback(
        shell_allow_list=["git"],
        auto_approve=False,
    )

    result = callback(_FakeTool("execute"), {"command": "git status"}, MagicMock())

    assert result is None


def test_non_interactive_policy_rejects_unlisted_shell_command() -> None:
    callback = ni.build_non_interactive_before_tool_callback(
        shell_allow_list=["git"],
        auto_approve=False,
    )

    with pytest.raises(ni.NonInteractivePolicyError, match="not in --shell-allow-list"):
        callback(_FakeTool("execute"), {"command": "python -m pytest"}, MagicMock())


def test_non_interactive_policy_rejects_shell_control_operators() -> None:
    callback = ni.build_non_interactive_before_tool_callback(
        shell_allow_list=["git"],
        auto_approve=False,
    )

    with pytest.raises(ni.NonInteractivePolicyError, match="control operators"):
        callback(_FakeTool("execute"), {"command": "git status && rm -rf /"}, MagicMock())


def test_non_interactive_policy_rejects_confirmation_required_tool() -> None:
    callback = ni.build_non_interactive_before_tool_callback(
        shell_allow_list=["git"],
        auto_approve=False,
    )

    with pytest.raises(ni.NonInteractivePolicyError, match="requires confirmation"):
        callback(
            _FakeTool("write_file"),
            {"path": "/tmp/demo.txt", "content": "data"},
            MagicMock(),
        )


def test_non_interactive_policy_allows_confirmation_tool_with_auto_approve() -> None:
    callback = ni.build_non_interactive_before_tool_callback(
        shell_allow_list=["git"],
        auto_approve=True,
    )

    result = callback(
        _FakeTool("write_file"),
        {"path": "/tmp/demo.txt", "content": "data"},
        MagicMock(),
    )

    assert result is None


def test_build_cli_agent_missing_skills_dependency_is_actionable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def _raise(**kwargs: object):
        del kwargs
        raise ImportError(
            "adk-skills-agent is required for skills support. "
            "Install it with: pip install adk-skills-agent"
        )

    monkeypatch.setattr(ni, "create_deep_agent", _raise)
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    with pytest.raises(RuntimeError, match="Install optional support"):
        ni._build_cli_agent(
            agent_name="demo",
            model=None,
            cwd=tmp_path,
            shell_allow_list=None,
            auto_approve=False,
            skills_dirs=[str(skills_dir)],
        )


def test_build_cli_agent_enables_dynamic_delegation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ADK_DYNAMIC_TASK_MAX_PARALLEL", raising=False)
    monkeypatch.delenv("ADK_DYNAMIC_TASK_CONCURRENCY_POLICY", raising=False)
    monkeypatch.delenv("ADK_DYNAMIC_TASK_QUEUE_TIMEOUT_SECONDS", raising=False)

    captured: dict[str, object] = {}

    def _fake_create_deep_agent(**kwargs: object):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(ni, "create_deep_agent", _fake_create_deep_agent)

    ni._build_cli_agent(
        agent_name="demo",
        model=None,
        cwd=tmp_path,
        shell_allow_list=None,
        auto_approve=False,
    )

    assert captured["name"] == "demo_cli"
    assert captured["execution"] == "local"
    cfg = cast(DeepAgentConfig, captured["config"])
    assert cfg.delegation_mode == "dynamic"
    dynamic_cfg = cfg.dynamic_task_config
    assert dynamic_cfg is not None
    assert dynamic_cfg.concurrency_policy == "wait"


def test_run_non_interactive_async_streams_chunks(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(ni, "_build_cli_agent", lambda **_: object())
    monkeypatch.setattr(ni, "SqliteSessionService", lambda *_: object())
    monkeypatch.setattr(ni, "Runner", _FakeRunner)

    output = ni.asyncio.run(
        ni._run_non_interactive_async(
            prompt="hello",
            model="gemini-2.5-flash",
            agent_name="demo",
            user_id="u1",
            session_id="s1",
            db_path=tmp_path / "sessions.db",
            no_stream=False,
            shell_allow_list=None,
            auto_approve=False,
        )
    )
    captured = capsys.readouterr()

    assert output == "hello world"
    assert captured.out == "hello world\n"


def test_run_non_interactive_async_buffers_chunks_when_no_stream(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(ni, "_build_cli_agent", lambda **_: object())
    monkeypatch.setattr(ni, "SqliteSessionService", lambda *_: object())
    monkeypatch.setattr(ni, "Runner", _FakeRunner)

    output = ni.asyncio.run(
        ni._run_non_interactive_async(
            prompt="hello",
            model="gemini-2.5-flash",
            agent_name="demo",
            user_id="u1",
            session_id="s1",
            db_path=tmp_path / "sessions.db",
            no_stream=True,
            shell_allow_list=None,
            auto_approve=False,
        )
    )
    captured = capsys.readouterr()

    assert output == "hello world"
    assert captured.out == ""


def test_run_non_interactive_prints_buffered_output(monkeypatch, capsys) -> None:
    def _fake_run(coro):
        coro.close()
        return "buffered output"

    monkeypatch.setattr(ni.asyncio, "run", _fake_run)

    exit_code = ni.run_non_interactive(
        prompt="hello",
        model="gemini-2.5-flash",
        agent_name="demo",
        user_id="u1",
        session_id="s1",
        db_path=Path("/tmp/dummy.db"),
        no_stream=True,
        shell_allow_list=None,
        auto_approve=False,
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "buffered output" in captured.out


def test_run_non_interactive_returns_error_code_on_exception(monkeypatch, capsys) -> None:
    def _raise(coro):
        coro.close()
        raise RuntimeError("boom")

    monkeypatch.setattr(ni.asyncio, "run", _raise)

    exit_code = ni.run_non_interactive(
        prompt="hello",
        model="gemini-2.5-flash",
        agent_name="demo",
        user_id="u1",
        session_id="s1",
        db_path=Path("/tmp/dummy.db"),
        no_stream=False,
        shell_allow_list=None,
        auto_approve=False,
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error: boom" in captured.err
