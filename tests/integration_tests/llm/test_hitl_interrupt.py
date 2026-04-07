"""Integration tests — HITL interrupt flow with a real LLM.

Verifies ``interrupt_on`` triggers ADK confirmation and both approved/rejected
resume paths behave correctly for ``write_file``.

Run with:
    uv run pytest -m llm tests/integration_tests/llm/test_hitl_interrupt.py
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from google.adk.runners import InMemoryRunner
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.genai import types

from adk_deepagents import DeepAgentConfig, create_deep_agent
from tests.integration_tests.conftest import backend_factory, get_file_content, make_litellm_model

pytestmark = [pytest.mark.integration, pytest.mark.llm]

_CONFIRMATION_FUNCTION_NAME = "adk_request_confirmation"


async def _run_turn_collect_events(
    *,
    runner: InMemoryRunner,
    session_id: str,
    user_id: str,
    message: types.Content,
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    texts: list[str] = []
    function_calls: list[str] = []
    function_responses: list[str] = []
    confirmation_ids: list[str] = []
    confirmation_tools: list[str] = []

    async for event in runner.run_async(
        session_id=session_id,
        user_id=user_id,
        new_message=message,
    ):
        if not event.content or not event.content.parts:
            continue

        for part in event.content.parts:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                texts.append(text)

            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                call_name = getattr(function_call, "name", None)
                if isinstance(call_name, str) and call_name:
                    function_calls.append(call_name)

                    if call_name == _CONFIRMATION_FUNCTION_NAME:
                        request_id = getattr(function_call, "id", None)
                        if isinstance(request_id, str) and request_id:
                            confirmation_ids.append(request_id)

                        raw_args = getattr(function_call, "args", None)
                        if isinstance(raw_args, dict):
                            original_call = raw_args.get("originalFunctionCall")
                            if isinstance(original_call, dict):
                                original_name = original_call.get("name")
                                if isinstance(original_name, str) and original_name:
                                    confirmation_tools.append(original_name)

            function_response = getattr(part, "function_response", None)
            if function_response is not None:
                response_name = getattr(function_response, "name", None)
                if isinstance(response_name, str) and response_name:
                    function_responses.append(response_name)

    return texts, function_calls, function_responses, confirmation_ids, confirmation_tools


def _build_confirmation_message(*, request_id: str, approved: bool) -> types.Content:
    confirmation = ToolConfirmation(confirmed=approved, payload=None)
    response_payload: dict[str, Any] = confirmation.model_dump(by_alias=True, exclude_none=True)
    part = types.Part.from_function_response(
        name=_CONFIRMATION_FUNCTION_NAME,
        response=response_payload,
    )
    if part.function_response is not None:
        part.function_response.id = request_id

    return types.Content(role="user", parts=[part])


async def _resolve_confirmations(
    *,
    runner: InMemoryRunner,
    session_id: str,
    user_id: str,
    confirmation_ids: list[str],
    approved: bool,
) -> None:
    pending_ids = list(dict.fromkeys(confirmation_ids))
    processed_ids: set[str] = set()
    approval_rounds = 0

    while pending_ids:
        approval_rounds += 1
        assert approval_rounds <= 5, "Exceeded confirmation approval rounds"

        request_id = pending_ids.pop(0)
        if request_id in processed_ids:
            continue
        processed_ids.add(request_id)

        (
            _texts_resume,
            _function_calls_resume,
            _function_responses_resume,
            new_confirmation_ids,
            _new_confirmation_tools,
        ) = await _run_turn_collect_events(
            runner=runner,
            session_id=session_id,
            user_id=user_id,
            message=_build_confirmation_message(request_id=request_id, approved=approved),
        )

        for new_id in new_confirmation_ids:
            if new_id not in processed_ids and new_id not in pending_ids:
                pending_ids.append(new_id)


@pytest.mark.timeout(180)
async def test_llm_interrupt_on_write_file_requires_approval_then_resumes():
    """LLM flow pauses on approval request, then proceeds after confirmation."""
    if os.environ.get("ADK_DEEPAGENTS_LLM_TRANSPORT", "inmemory").strip().lower() == "a2a":
        pytest.skip("HITL approval resume test currently requires in-memory LLM transport")

    model = make_litellm_model()
    target_path = "/hitl_llm_test.txt"
    target_content = "HITL_APPROVAL_TOKEN_314159"

    agent = create_deep_agent(
        model=model,
        name="llm_hitl_interrupt_test",
        instruction=(
            "You are a test agent. When asked to create a file, call write_file exactly once "
            "with the exact path and content provided by the user."
        ),
        config=DeepAgentConfig(interrupt_on={"write_file": True}),
    )

    runner = InMemoryRunner(agent=agent, app_name="integration_test")
    session = await runner.session_service.create_session(
        app_name="integration_test",
        user_id="test_user",
        state={
            "files": {},
            "_backend_factory": backend_factory,
        },
    )

    prompt = (
        f"Use write_file to create {target_path} with exact content '{target_content}'. "
        "Do not ask follow-up questions."
    )
    (
        _texts_first,
        function_calls_first,
        _function_responses_first,
        confirmation_ids,
        confirmation_tools,
    ) = await _run_turn_collect_events(
        runner=runner,
        session_id=session.id,
        user_id="test_user",
        message=types.Content(role="user", parts=[types.Part(text=prompt)]),
    )

    assert "write_file" in function_calls_first, (
        f"Expected write_file tool call before approval gate, got: {function_calls_first}"
    )
    assert _CONFIRMATION_FUNCTION_NAME in function_calls_first, (
        "Expected ADK confirmation function call when write_file is interrupted, "
        f"got: {function_calls_first}"
    )
    assert confirmation_ids, "Expected at least one confirmation request id"
    if confirmation_tools:
        assert "write_file" in confirmation_tools, (
            f"Expected confirmation target to include write_file, got: {confirmation_tools}"
        )

    files_before_approval = await get_file_content(runner, session)
    assert target_path not in files_before_approval, (
        "File should not be created before confirmation is approved"
    )

    await _resolve_confirmations(
        runner=runner,
        session_id=session.id,
        user_id="test_user",
        confirmation_ids=confirmation_ids,
        approved=True,
    )

    files_after_approval = await get_file_content(runner, session)
    assert target_path in files_after_approval, (
        f"Expected file {target_path} to be created after approval, got: {list(files_after_approval)}"
    )
    assert target_content in files_after_approval[target_path], (
        "Expected approved write_file execution to persist requested content"
    )


@pytest.mark.timeout(180)
async def test_llm_interrupt_on_write_file_rejected_does_not_write():
    """LLM flow pauses on approval request; rejected resume keeps files unchanged."""
    if os.environ.get("ADK_DEEPAGENTS_LLM_TRANSPORT", "inmemory").strip().lower() == "a2a":
        pytest.skip("HITL approval resume test currently requires in-memory LLM transport")

    model = make_litellm_model()
    target_path = "/hitl_llm_reject_test.txt"
    target_content = "HITL_REJECT_TOKEN_271828"

    agent = create_deep_agent(
        model=model,
        name="llm_hitl_interrupt_reject_test",
        instruction=(
            "You are a test agent. When asked to create a file, call write_file exactly once "
            "with the exact path and content provided by the user."
        ),
        config=DeepAgentConfig(interrupt_on={"write_file": True}),
    )

    runner = InMemoryRunner(agent=agent, app_name="integration_test")
    session = await runner.session_service.create_session(
        app_name="integration_test",
        user_id="test_user",
        state={
            "files": {},
            "_backend_factory": backend_factory,
        },
    )

    prompt = (
        f"Use write_file to create {target_path} with exact content '{target_content}'. "
        "Do not ask follow-up questions."
    )
    (
        _texts_first,
        function_calls_first,
        _function_responses_first,
        confirmation_ids,
        confirmation_tools,
    ) = await _run_turn_collect_events(
        runner=runner,
        session_id=session.id,
        user_id="test_user",
        message=types.Content(role="user", parts=[types.Part(text=prompt)]),
    )

    assert "write_file" in function_calls_first, (
        f"Expected write_file tool call before approval gate, got: {function_calls_first}"
    )
    assert _CONFIRMATION_FUNCTION_NAME in function_calls_first, (
        "Expected ADK confirmation function call when write_file is interrupted, "
        f"got: {function_calls_first}"
    )
    assert confirmation_ids, "Expected at least one confirmation request id"
    if confirmation_tools:
        assert "write_file" in confirmation_tools, (
            f"Expected confirmation target to include write_file, got: {confirmation_tools}"
        )

    await _resolve_confirmations(
        runner=runner,
        session_id=session.id,
        user_id="test_user",
        confirmation_ids=confirmation_ids,
        approved=False,
    )

    files_after_rejection = await get_file_content(runner, session)
    assert target_path not in files_after_rejection, (
        "File should not be created when confirmation is rejected"
    )
