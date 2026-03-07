"""Integration coverage for CLI release hardening workflows."""

from __future__ import annotations

import io
from typing import Any

import pytest

import adk_deepagents.cli.interactive as repl
import adk_deepagents.cli.main as cli_main_module
import adk_deepagents.cli.non_interactive as non_interactive
from adk_deepagents.cli.main import cli_main

pytestmark = pytest.mark.integration


class _FakePart:
    def __init__(
        self,
        text: str | None = None,
        *,
        function_call: Any | None = None,
        function_response: Any | None = None,
    ) -> None:
        self.text = text
        self.function_call = function_call
        self.function_response = function_response


class _FakeContent:
    def __init__(self, parts: list[_FakePart]) -> None:
        self.parts = parts


class _FakeEvent:
    def __init__(self, author: str, parts: list[_FakePart]) -> None:
        self.author = author
        self.content = _FakeContent(parts)


class _FakeFunctionCall:
    def __init__(
        self,
        name: str,
        *,
        call_id: str | None = None,
        args: dict[str, object] | None = None,
    ) -> None:
        self.name = name
        self.id = call_id
        self.args = args if args is not None else {}


class _FakeStdin:
    def __init__(self, *, isatty_value: bool) -> None:
        self._isatty_value = isatty_value

    def isatty(self) -> bool:
        return self._isatty_value


class _NonInteractiveRunner:
    calls: list[tuple[str, str]] = []

    def __init__(self, *, app_name: str, agent: object, session_service: object) -> None:
        del app_name, agent, session_service

    async def run_async(self, *, user_id: str, session_id: str, new_message: Any):
        del user_id
        prompt = new_message.parts[0].text
        assert isinstance(prompt, str)

        _NonInteractiveRunner.calls.append((session_id, prompt))
        yield _FakeEvent("assistant", [_FakePart(text=f"echo:{prompt}")])


class _HitlRunner:
    messages: list[Any] = []

    def __init__(self, *, app_name: str, agent: object, session_service: object) -> None:
        del app_name, agent, session_service

    async def run_async(self, *, user_id: str, session_id: str, new_message: Any):
        del user_id, session_id
        _HitlRunner.messages.append(new_message)

        first_part = new_message.parts[0]
        function_response = getattr(first_part, "function_response", None)

        if function_response is None:
            yield _FakeEvent(
                "assistant",
                [
                    _FakePart(
                        function_call=_FakeFunctionCall(
                            repl.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                            call_id="req-1",
                            args={
                                "originalFunctionCall": {
                                    "id": "tool-1",
                                    "name": "write_file",
                                    "args": {
                                        "path": "README.md",
                                        "content": "updated",
                                    },
                                },
                                "toolConfirmation": {
                                    "hint": "Tool 'write_file' requires approval.",
                                    "confirmed": False,
                                },
                            },
                        )
                    )
                ],
            )
            return

        response_payload = getattr(function_response, "response", None)
        approved = isinstance(response_payload, dict) and bool(response_payload.get("confirmed"))
        text = "approved and resumed" if approved else "rejected"
        yield _FakeEvent("assistant", [_FakePart(text=text)])


def test_cli_non_interactive_supports_piped_stdin_and_resume(monkeypatch, tmp_path, capsys) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    monkeypatch.setenv("ADK_DEEPAGENTS_HOME", str(home_dir))

    monkeypatch.setattr(non_interactive, "_build_cli_agent", lambda **_: object())
    monkeypatch.setattr(non_interactive, "Runner", _NonInteractiveRunner)

    piped_inputs = iter(["first piped prompt", "second piped prompt"])
    monkeypatch.setattr(cli_main_module, "read_piped_stdin", lambda: next(piped_inputs))

    _NonInteractiveRunner.calls.clear()

    first_exit = cli_main(["--agent", "release", "-q"])
    first_output = capsys.readouterr()

    second_exit = cli_main(["--agent", "release", "--resume", "-q"])
    second_output = capsys.readouterr()

    assert first_exit == 0
    assert second_exit == 0
    assert "echo:first piped prompt" in first_output.out
    assert "echo:second piped prompt" in second_output.out

    assert len(_NonInteractiveRunner.calls) == 2
    first_session_id, first_prompt = _NonInteractiveRunner.calls[0]
    second_session_id, second_prompt = _NonInteractiveRunner.calls[1]

    assert first_prompt == "first piped prompt"
    assert second_prompt == "second piped prompt"
    assert second_session_id == first_session_id


def test_interactive_hitl_confirmation_flow_is_resumed(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(repl, "_build_cli_agent", lambda **_: object())
    monkeypatch.setattr(repl, "Runner", _HitlRunner)

    _HitlRunner.messages.clear()

    responses = iter(["approve", "/q"])

    def _input_reader(_prompt: str) -> str:
        return next(responses)

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = repl.asyncio.run(
        repl._run_interactive_async(
            first_prompt="please update README",
            model="gemini-2.5-flash",
            agent_name="release",
            user_id="local",
            session_id="thread-1",
            db_path=tmp_path / "sessions.db",
            auto_approve=False,
            input_reader=_input_reader,
            stdin=_FakeStdin(isatty_value=True),
            stdout=stdout,
            stderr=stderr,
        )
    )

    assert exit_code == 0
    assert "assistant> approved and resumed" in stdout.getvalue()
    assert "requested confirmation" in stderr.getvalue()
    assert "approved tool 'write_file'" in stderr.getvalue()

    assert len(_HitlRunner.messages) == 2
    confirmation_message = _HitlRunner.messages[1]
    function_response = confirmation_message.parts[0].function_response
    assert function_response is not None
    assert function_response.id == "req-1"
    assert function_response.response.get("confirmed") is True
