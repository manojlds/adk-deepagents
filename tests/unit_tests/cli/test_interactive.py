"""Unit tests for interactive CLI REPL helpers."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import adk_deepagents.cli.interactive as repl
from adk_deepagents.cli.session_store import ThreadRecord


class _FakeStdin:
    def __init__(self, *, isatty_value: bool):
        self._isatty_value = isatty_value

    def isatty(self) -> bool:
        return self._isatty_value


class _FakeFunctionCall:
    def __init__(
        self,
        name: str,
        *,
        call_id: str | None = None,
        args: dict[str, object] | None = None,
    ):
        self.name = name
        self.id = call_id
        self.args = args if args is not None else {}


class _FakeFunctionResponse:
    def __init__(
        self,
        name: str,
        response: dict[str, object],
        *,
        call_id: str | None = None,
    ):
        self.name = name
        self.response = response
        self.id = call_id


class _FakePart:
    def __init__(
        self,
        text: str | None = None,
        *,
        function_call: _FakeFunctionCall | None = None,
        function_response: _FakeFunctionResponse | None = None,
    ):
        self.text = text
        self.function_call = function_call
        self.function_response = function_response


class _FakeContent:
    def __init__(self, parts: list[_FakePart]):
        self.parts = parts


class _FakeEvent:
    def __init__(self, author: str, parts: list[_FakePart] | None):
        self.author = author
        self.content = _FakeContent(parts) if parts is not None else None


class _TurnFakeRunner:
    async def run_async(self, *, user_id, session_id, new_message):
        del user_id, session_id, new_message
        yield _FakeEvent("user", [_FakePart("ignore me")])
        yield _FakeEvent("assistant", [_FakePart("hello"), _FakePart(" world")])
        yield _FakeEvent("assistant", [_FakePart(function_call=_FakeFunctionCall("ls"))])
        yield _FakeEvent(
            "assistant",
            [
                _FakePart(
                    function_response=_FakeFunctionResponse(
                        "ls",
                        {"error": "permission denied"},
                    )
                )
            ],
        )


def _confirmation_request_args(
    *,
    tool_name: str = "write_file",
    tool_args: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "originalFunctionCall": {
            "id": "fc-tool-1",
            "name": tool_name,
            "args": tool_args if tool_args is not None else {"file_path": "README.md"},
        },
        "toolConfirmation": {
            "hint": f"Tool '{tool_name}' requires approval.",
            "confirmed": False,
        },
    }


class _ApprovalFlowRunner:
    def __init__(self):
        self.messages: list[Any] = []

    async def run_async(self, *, user_id, session_id, new_message):
        del user_id, session_id
        self.messages.append(new_message)

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
                            args=_confirmation_request_args(),
                        )
                    )
                ],
            )
            return

        decision = bool(function_response.response.get("confirmed"))
        if decision:
            yield _FakeEvent("assistant", [_FakePart("approved and resumed")])
            return

        yield _FakeEvent(
            "assistant",
            [
                _FakePart(
                    function_response=_FakeFunctionResponse(
                        "write_file",
                        {
                            "status": "rejected",
                            "message": "Tool 'write_file' was rejected.",
                        },
                    )
                )
            ],
        )


def _thread_record(
    session_id: str,
    *,
    user_id: str = "u1",
    agent_name: str = "demo",
    model: str | None = "gemini-2.5-flash",
    created_at: float | None = 1.0,
    updated_at: float = 1.0,
) -> ThreadRecord:
    return ThreadRecord(
        session_id=session_id,
        user_id=user_id,
        agent_name=agent_name,
        model=model,
        created_at=created_at,
        updated_at=updated_at,
    )


def test_handle_slash_command_help_prints_usage() -> None:
    out = io.StringIO()
    err = io.StringIO()

    result = repl.handle_slash_command("/help", stdout=out, stderr=err)

    assert result == "handled"
    assert "Interactive commands" in out.getvalue()
    assert "/threads" in out.getvalue()
    assert "/clear" in out.getvalue()
    assert "/model" in out.getvalue()
    assert err.getvalue() == ""


def test_handle_slash_command_quit_variants_exit_loop() -> None:
    out = io.StringIO()
    err = io.StringIO()

    assert repl.handle_slash_command("/quit", stdout=out, stderr=err) == "exit"
    assert repl.handle_slash_command("/q", stdout=out, stderr=err) == "exit"


def test_handle_slash_command_unknown_reports_error() -> None:
    out = io.StringIO()
    err = io.StringIO()

    result = repl.handle_slash_command("/unknown", stdout=out, stderr=err)

    assert result == "handled"
    assert out.getvalue() == ""
    assert "Unknown slash command" in err.getvalue()


def test_handle_slash_command_threads_lists_and_switches_by_index(monkeypatch) -> None:
    thread_a = _thread_record("thread-a", created_at=1.0, updated_at=1.0)
    thread_b = _thread_record("thread-b", created_at=2.0, updated_at=2.0)

    monkeypatch.setattr(repl, "list_threads", lambda **_: [thread_b, thread_a])

    thread_context = repl._ThreadCommandContext(
        db_path=Path("/tmp/sessions.db"),
        user_id="u1",
        agent_name="demo",
        model="gemini-2.5-flash",
        active_session_id="thread-b",
    )
    out = io.StringIO()
    err = io.StringIO()

    list_result = repl.handle_slash_command(
        "/threads",
        stdout=out,
        stderr=err,
        thread_context=thread_context,
    )

    assert list_result == "handled"
    rendered = out.getvalue()
    assert "Threads for profile 'demo'" in rendered
    assert "*\t1\tthread-b" in rendered
    assert "-\t2\tthread-a" in rendered
    assert "Use /threads <index|thread_id|latest> to switch active thread." in rendered

    switch_result = repl.handle_slash_command(
        "/threads 2",
        stdout=out,
        stderr=err,
        thread_context=thread_context,
    )

    assert switch_result == "handled"
    assert thread_context.active_session_id == "thread-a"
    assert "switched active thread" in err.getvalue()


def test_handle_slash_command_clear_creates_new_thread(monkeypatch) -> None:
    monkeypatch.setattr(repl, "create_thread", lambda **_: _thread_record("thread-new"))

    thread_context = repl._ThreadCommandContext(
        db_path=Path("/tmp/sessions.db"),
        user_id="u1",
        agent_name="demo",
        model="gemini-2.5-flash",
        active_session_id="thread-a",
    )
    out = io.StringIO()
    err = io.StringIO()

    result = repl.handle_slash_command(
        "/clear",
        stdout=out,
        stderr=err,
        thread_context=thread_context,
    )

    assert result == "handled"
    assert thread_context.active_session_id == "thread-new"
    assert "started a new thread" in err.getvalue()


def test_handle_slash_command_model_queries_and_switches() -> None:
    switched_models: list[str | None] = []

    thread_context = repl._ThreadCommandContext(
        db_path=Path("/tmp/sessions.db"),
        user_id="u1",
        agent_name="demo",
        model="gemini-2.5-flash",
        active_session_id="thread-a",
    )
    model_context = repl._ModelCommandContext(
        model="gemini-2.5-flash",
        switch_model=lambda model: switched_models.append(model),
    )

    out = io.StringIO()
    err = io.StringIO()

    query_result = repl.handle_slash_command(
        "/model",
        stdout=out,
        stderr=err,
        thread_context=thread_context,
        model_context=model_context,
    )

    assert query_result == "handled"
    assert "Active model: gemini-2.5-flash" in out.getvalue()

    switch_result = repl.handle_slash_command(
        "/model gemini-2.5-pro",
        stdout=out,
        stderr=err,
        thread_context=thread_context,
        model_context=model_context,
    )

    assert switch_result == "handled"
    assert switched_models == ["gemini-2.5-pro"]
    assert model_context.model == "gemini-2.5-pro"
    assert thread_context.model == "gemini-2.5-pro"
    assert "switched active model" in err.getvalue()

    reset_result = repl.handle_slash_command(
        "/model default",
        stdout=out,
        stderr=err,
        thread_context=thread_context,
        model_context=model_context,
    )

    assert reset_result == "handled"
    assert switched_models == ["gemini-2.5-pro", None]
    assert model_context.model is None
    assert thread_context.model is None


def test_build_cli_agent_enables_hitl_interrupts(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_create_deep_agent(**kwargs: object):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(repl, "create_deep_agent", _fake_create_deep_agent)

    repl._build_cli_agent(agent_name="demo", model=None, cwd=Path("/tmp/workspace"))

    assert captured["name"] == "demo_cli"
    assert captured["execution"] == "local"
    assert captured["interrupt_on"] == repl.INTERACTIVE_INTERRUPT_ON


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

    monkeypatch.setattr(repl, "create_deep_agent", _raise)
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    try:
        repl._build_cli_agent(
            agent_name="demo",
            model=None,
            cwd=tmp_path,
            skills_dirs=[str(skills_dir)],
        )
    except RuntimeError as exc:
        assert "Install optional support" in str(exc)
    else:
        raise AssertionError("Expected _build_cli_agent to raise an actionable skills error")


def test_run_interactive_turn_streams_text_and_tool_events() -> None:
    out = io.StringIO()
    err = io.StringIO()

    repl.asyncio.run(
        repl._run_interactive_turn(
            runner=_TurnFakeRunner(),
            prompt="hello",
            user_id="u1",
            session_id="s1",
            stdout=out,
            stderr=err,
        )
    )

    assert out.getvalue() == "assistant> hello world\n[tool] ls\n"
    assert "[error] ls: permission denied" in err.getvalue()


def test_run_interactive_turn_hitl_approve_path() -> None:
    runner = _ApprovalFlowRunner()

    out = io.StringIO()
    err = io.StringIO()

    responses = iter(["approve"])

    def _input_reader(_prompt: str) -> str:
        return next(responses)

    repl.asyncio.run(
        repl._run_interactive_turn(
            runner=runner,
            prompt="please update README",
            user_id="u1",
            session_id="s1",
            stdout=out,
            stderr=err,
            input_reader=_input_reader,
            approval_context=repl._InteractiveApprovalContext(auto_approve=False),
        )
    )

    assert len(runner.messages) == 2
    first_message = runner.messages[0]
    assert first_message.parts[0].text == "please update README"

    confirmation_message = runner.messages[1]
    function_response = confirmation_message.parts[0].function_response
    assert function_response is not None
    assert function_response.name == repl.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
    assert function_response.id == "req-1"
    assert function_response.response["confirmed"] is True

    assert "approved and resumed" in out.getvalue()
    assert "requested confirmation" in err.getvalue()
    assert "approved tool 'write_file'" in err.getvalue()


def test_run_interactive_turn_hitl_reject_path() -> None:
    runner = _ApprovalFlowRunner()

    out = io.StringIO()
    err = io.StringIO()

    responses = iter(["reject"])

    def _input_reader(_prompt: str) -> str:
        return next(responses)

    repl.asyncio.run(
        repl._run_interactive_turn(
            runner=runner,
            prompt="please update README",
            user_id="u1",
            session_id="s1",
            stdout=out,
            stderr=err,
            input_reader=_input_reader,
            approval_context=repl._InteractiveApprovalContext(auto_approve=False),
        )
    )

    assert len(runner.messages) == 2
    confirmation_message = runner.messages[1]
    function_response = confirmation_message.parts[0].function_response
    assert function_response is not None
    assert function_response.response["confirmed"] is False

    assert "rejected tool 'write_file'" in err.getvalue()


def test_run_interactive_turn_hitl_auto_approve_path() -> None:
    runner = _ApprovalFlowRunner()

    out = io.StringIO()
    err = io.StringIO()

    def _input_reader(_prompt: str) -> str:
        raise AssertionError("input reader should not be called when auto-approve is enabled")

    repl.asyncio.run(
        repl._run_interactive_turn(
            runner=runner,
            prompt="please update README",
            user_id="u1",
            session_id="s1",
            stdout=out,
            stderr=err,
            input_reader=_input_reader,
            approval_context=repl._InteractiveApprovalContext(auto_approve=True),
        )
    )

    assert len(runner.messages) == 2
    confirmation_message = runner.messages[1]
    function_response = confirmation_message.parts[0].function_response
    assert function_response is not None
    assert function_response.response["confirmed"] is True

    assert "auto-approved tool 'write_file'" in err.getvalue()


def test_run_interactive_async_auto_submits_first_message(monkeypatch) -> None:
    monkeypatch.setattr(repl, "_build_cli_agent", lambda **_: object())
    monkeypatch.setattr(repl, "SqliteSessionService", lambda *_: object())

    prompts: list[str] = []

    class _EchoRunner:
        def __init__(self, *, app_name, agent, session_service):
            del app_name, agent, session_service

        async def run_async(self, *, user_id, session_id, new_message):
            del user_id, session_id
            prompt = new_message.parts[0].text
            prompts.append(prompt)
            yield _FakeEvent("assistant", [_FakePart(f"echo:{prompt}")])

    monkeypatch.setattr(repl, "Runner", _EchoRunner)

    user_inputs = iter(["/q"])

    def _input_reader(_prompt: str) -> str:
        return next(user_inputs)

    out = io.StringIO()
    err = io.StringIO()

    exit_code = repl.asyncio.run(
        repl._run_interactive_async(
            first_prompt="hello",
            model="gemini-2.5-flash",
            agent_name="demo",
            user_id="u1",
            session_id="s1",
            db_path=Path("/tmp/sessions.db"),
            auto_approve=False,
            input_reader=_input_reader,
            stdin=_FakeStdin(isatty_value=True),
            stdout=out,
            stderr=err,
        )
    )

    assert exit_code == 0
    assert prompts == ["hello"]
    assert "assistant> echo:hello" in out.getvalue()
    assert "interactive mode" in err.getvalue()


def test_run_interactive_async_thread_commands_switch_session_context(monkeypatch) -> None:
    monkeypatch.setattr(repl, "_build_cli_agent", lambda **_: object())
    monkeypatch.setattr(repl, "SqliteSessionService", lambda *_: object())

    turn_calls: list[tuple[str, str]] = []

    class _EchoRunner:
        def __init__(self, *, app_name, agent, session_service):
            del app_name, agent, session_service

        async def run_async(self, *, user_id, session_id, new_message):
            del user_id
            prompt = new_message.parts[0].text
            turn_calls.append((session_id, prompt))
            yield _FakeEvent("assistant", [_FakePart(f"echo:{session_id}:{prompt}")])

    monkeypatch.setattr(repl, "Runner", _EchoRunner)

    threads: dict[str, ThreadRecord] = {
        "thread-a": _thread_record("thread-a", created_at=1.0, updated_at=1.0),
    }

    def _create_thread(*, db_path, user_id, agent_name, model):
        del db_path, model
        thread = _thread_record(
            "thread-b",
            user_id=user_id,
            agent_name=agent_name,
            created_at=2.0,
            updated_at=2.0,
        )
        threads[thread.session_id] = thread
        return thread

    def _list_threads(*, db_path, user_id, agent_name, limit):
        del db_path, user_id
        sorted_threads = sorted(
            (thread for thread in threads.values() if thread.agent_name == agent_name),
            key=lambda thread: thread.updated_at,
            reverse=True,
        )
        return sorted_threads[:limit]

    def _get_thread(*, db_path, user_id, session_id):
        del db_path, user_id
        return threads.get(session_id)

    monkeypatch.setattr(repl, "create_thread", _create_thread)
    monkeypatch.setattr(repl, "list_threads", _list_threads)
    monkeypatch.setattr(repl, "get_thread", _get_thread)

    user_inputs = iter(["/clear", "turn one", "/threads thread-a", "turn two", "/q"])

    def _input_reader(_prompt: str) -> str:
        return next(user_inputs)

    out = io.StringIO()
    err = io.StringIO()

    exit_code = repl.asyncio.run(
        repl._run_interactive_async(
            first_prompt=None,
            model="gemini-2.5-flash",
            agent_name="demo",
            user_id="u1",
            session_id="thread-a",
            db_path=Path("/tmp/sessions.db"),
            auto_approve=False,
            input_reader=_input_reader,
            stdin=_FakeStdin(isatty_value=True),
            stdout=out,
            stderr=err,
        )
    )

    assert exit_code == 0
    assert turn_calls == [("thread-b", "turn one"), ("thread-a", "turn two")]
    assert "[thread thread-b] started a new thread." in err.getvalue()
    assert "[thread thread-a] switched active thread." in err.getvalue()


def test_run_interactive_async_model_command_switches_model_and_preserves_thread(
    monkeypatch,
) -> None:
    class _FakeAgent:
        def __init__(self, model: str | None):
            self.model = model

    def _fake_build_cli_agent(*, agent_name, model, cwd, **kwargs):
        del agent_name, cwd, kwargs
        return _FakeAgent(model)

    monkeypatch.setattr(repl, "_build_cli_agent", _fake_build_cli_agent)
    monkeypatch.setattr(repl, "SqliteSessionService", lambda *_: object())

    turn_calls: list[tuple[str, str, str | None]] = []

    class _EchoRunner:
        def __init__(self, *, app_name, agent, session_service):
            del app_name, session_service
            self.model = agent.model

        async def run_async(self, *, user_id, session_id, new_message):
            del user_id
            prompt = new_message.parts[0].text
            turn_calls.append((session_id, prompt, self.model))
            model_label = self.model or "default"
            yield _FakeEvent("assistant", [_FakePart(f"echo:{model_label}:{prompt}")])

    monkeypatch.setattr(repl, "Runner", _EchoRunner)

    user_inputs = iter(
        [
            "/model gemini-2.5-pro",
            "turn one",
            "/model default",
            "turn two",
            "/q",
        ]
    )

    def _input_reader(_prompt: str) -> str:
        return next(user_inputs)

    out = io.StringIO()
    err = io.StringIO()

    exit_code = repl.asyncio.run(
        repl._run_interactive_async(
            first_prompt=None,
            model="gemini-2.5-flash",
            agent_name="demo",
            user_id="u1",
            session_id="thread-a",
            db_path=Path("/tmp/sessions.db"),
            auto_approve=False,
            input_reader=_input_reader,
            stdin=_FakeStdin(isatty_value=True),
            stdout=out,
            stderr=err,
        )
    )

    assert exit_code == 0
    assert turn_calls == [
        ("thread-a", "turn one", "gemini-2.5-pro"),
        ("thread-a", "turn two", None),
    ]
    assert "[model gemini-2.5-pro] switched active model." in err.getvalue()
    assert "[model default] switched active model." in err.getvalue()


def test_run_interactive_async_returns_zero_when_no_tty_and_no_first_prompt(monkeypatch) -> None:
    monkeypatch.setattr(
        repl, "_build_cli_agent", lambda **_: (_ for _ in ()).throw(AssertionError())
    )

    exit_code = repl.asyncio.run(
        repl._run_interactive_async(
            first_prompt=None,
            model="gemini-2.5-flash",
            agent_name="demo",
            user_id="u1",
            session_id="s1",
            db_path=Path("/tmp/sessions.db"),
            auto_approve=False,
            stdin=_FakeStdin(isatty_value=False),
        )
    )

    assert exit_code == 0


def test_run_interactive_returns_error_code_on_exception(monkeypatch, capsys) -> None:
    def _raise(coro):
        coro.close()
        raise RuntimeError("boom")

    monkeypatch.setattr(repl.asyncio, "run", _raise)

    exit_code = repl.run_interactive(
        first_prompt="hello",
        model="gemini-2.5-flash",
        agent_name="demo",
        user_id="u1",
        session_id="s1",
        db_path=Path("/tmp/sessions.db"),
        auto_approve=False,
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error: boom" in captured.err
