"""Integration tests — dynamic task tool delegation with a real LLM.

Run with: uv run pytest -m llm tests/integration_tests/llm/test_dynamic_task_delegation.py
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from adk_deepagents.types import SubAgentSpec
from tests.integration_tests.conftest import (
    make_litellm_model,
    run_agent_with_events,
    send_followup_with_events,
)

pytestmark = [pytest.mark.integration, pytest.mark.llm]


async def _get_integration_session_state(runner, session) -> dict:
    updated = await runner.session_service.get_session(
        app_name="integration_test",
        user_id="test_user",
        session_id=session.id,
    )
    return updated.state


@pytest.mark.timeout(120)
async def test_dynamic_task_tool_delegates_to_named_subagent():
    """Dynamic mode delegates through ``task`` to the requested sub-agent type."""
    model = make_litellm_model()

    math_subagent: SubAgentSpec = SubAgentSpec(
        name="math_expert",
        description="Solves arithmetic problems.",
    )

    agent = create_deep_agent(
        model=model,
        name="dynamic_task_routing_test",
        instruction=(
            "You must delegate using the task tool, not direct answers. "
            "For arithmetic requests, call task with subagent_type='math_expert' "
            "and return the delegated result."
        ),
        subagents=[math_subagent],
        delegation_mode="dynamic",
    )

    texts, function_calls, function_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Use task with subagent_type math_expert to solve: what is 144 divided by 12?",
    )

    response_text = " ".join(texts)
    assert "task" in function_calls, f"Expected task tool call, got: {function_calls}"
    assert "task" in function_responses, f"Expected task tool response, got: {function_responses}"
    assert "12" in response_text, f"Expected delegated result 12, got: {response_text}"


@pytest.mark.timeout(120)
async def test_dynamic_task_tool_reuses_task_id_session():
    """Dynamic task delegation can continue a child session via task_id."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="dynamic_task_resume_test",
        instruction=(
            "Always use the task tool for requests. "
            "If the user provides a task_id, pass that same task_id to task."
        ),
        delegation_mode="dynamic",
    )

    texts1, calls1, responses1, runner, session = await run_agent_with_events(
        agent,
        "Use task to tell the delegated worker: remember this codeword exactly: ORBIT. "
        "Acknowledge when done.",
    )
    response1 = " ".join(texts1).lower()
    assert "task" in calls1, f"Expected task tool call on turn 1, got: {calls1}"
    assert "task" in responses1, f"Expected task tool response on turn 1, got: {responses1}"
    assert any(word in response1 for word in ("orbit", "remember", "done", "acknowledged")), (
        f"Expected acknowledgement for codeword setup, got: {response1}"
    )

    texts2, calls2, responses2 = await send_followup_with_events(
        runner,
        session,
        "Now call task again with task_id task_1 and ask: what codeword did I ask you "
        "to remember earlier? Return the exact codeword only.",
    )
    response2 = " ".join(texts2).lower()
    assert "task" in calls2, f"Expected task tool call on turn 2, got: {calls2}"
    assert "task" in responses2, f"Expected task tool response on turn 2, got: {responses2}"
    assert "orbit" in response2, f"Expected resumed task to recall ORBIT, got: {response2}"


@pytest.mark.timeout(120)
async def test_dynamic_task_register_subagent_then_delegate():
    """Dynamic mode can register a runtime sub-agent before task delegation."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="dynamic_task_register_subagent_test",
        instruction=(
            "For every arithmetic request: first call register_subagent with "
            "name='math_expert', description='Arithmetic specialist', "
            "system_prompt='You are a math specialist. Return only the numeric answer.', "
            "then delegate with task using subagent_type='math_expert'."
        ),
        delegation_mode="dynamic",
    )

    texts, calls, responses, runner, session = await run_agent_with_events(
        agent,
        "Use runtime registration, then delegate with task: what is 144 divided by 12?",
    )

    response_text = " ".join(texts).lower()
    assert "register_subagent" in calls, f"Expected register_subagent call, got: {calls}"
    assert "register_subagent" in responses, (
        f"Expected register_subagent response, got: {responses}"
    )
    assert "task" in calls, f"Expected task call, got: {calls}"
    assert "task" in responses, f"Expected task response, got: {responses}"
    assert "12" in response_text, f"Expected delegated arithmetic result 12, got: {response_text}"

    state = await _get_integration_session_state(runner, session)
    runtime_specs = state.get("_dynamic_subagent_specs", {})
    assert isinstance(runtime_specs, dict), "Expected runtime sub-agent store in session state"
    assert "math_expert" in runtime_specs, (
        f"Expected math_expert in runtime sub-agent store, got: {list(runtime_specs)}"
    )


@pytest.mark.timeout(120)
async def test_dynamic_task_unknown_type_auto_creates_runtime_specialist():
    """Unknown ``subagent_type`` auto-creates a runtime specialist in dynamic mode."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="dynamic_task_auto_create_test",
        instruction=(
            "Do not call register_subagent. Always delegate through task with "
            "subagent_type='auto_math' for arithmetic requests, then return the delegated result."
        ),
        delegation_mode="dynamic",
    )

    texts, calls, responses, runner, session = await run_agent_with_events(
        agent,
        "Use task with subagent_type auto_math to solve: what is 81 divided by 9?",
    )

    response_text = " ".join(texts).lower()
    assert "task" in calls, f"Expected task call, got: {calls}"
    assert "task" in responses, f"Expected task response, got: {responses}"
    assert "register_subagent" not in calls, (
        "Expected auto-create path without explicit register_subagent call"
    )
    assert "9" in response_text, f"Expected delegated arithmetic result 9, got: {response_text}"

    state = await _get_integration_session_state(runner, session)
    runtime_specs = state.get("_dynamic_subagent_specs", {})
    assert isinstance(runtime_specs, dict), "Expected runtime sub-agent store in session state"
    assert "auto_math" in runtime_specs, (
        f"Expected auto_math in runtime sub-agent store, got: {list(runtime_specs)}"
    )
