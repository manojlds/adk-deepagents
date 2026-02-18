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
