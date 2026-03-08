"""Integration test — sub-agent delegation with a real LLM.

Scenario: Main agent delegates a task to a sub-agent, verifies the sub-agent
ran and returned a result.

Run with: uv run pytest -m llm
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from adk_deepagents.types import SubAgentSpec
from tests.integration_tests.conftest import (
    get_file_content,
    make_litellm_model,
    run_agent_with_events,
)

pytestmark = [pytest.mark.integration, pytest.mark.llm]


@pytest.mark.timeout(120)
async def test_subagent_delegation():
    """Main agent delegates a task to a named sub-agent and gets a result."""
    model = make_litellm_model()

    math_subagent: SubAgentSpec = SubAgentSpec(
        name="math_expert",
        description="A sub-agent that solves math problems. Delegate math questions to this agent.",
    )

    agent = create_deep_agent(
        model=model,
        name="delegation_test_agent",
        instruction=(
            "You are a test agent. You have a sub-agent called 'math_expert' that is "
            "specialized in solving math problems. When the user asks a math question, "
            "you MUST delegate to the math_expert sub-agent using the math_expert tool. "
            "After receiving the result, report it back to the user."
        ),
        subagents=[math_subagent],
    )

    texts, function_calls, function_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Please delegate this math question to the math_expert sub-agent: "
        "What is 15 multiplied by 7? Report the answer back to me.",
    )

    response_text = " ".join(texts)
    assert "math_expert" in function_calls, (
        f"Expected a call to math_expert tool, got calls: {function_calls}"
    )
    assert "math_expert" in function_responses, (
        f"Expected a function response from math_expert tool, got responses: {function_responses}"
    )
    # The sub-agent should compute 15 * 7 = 105 and the main agent should relay it
    assert "105" in response_text, (
        f"Expected '105' (15*7) in delegated response, got: {response_text}"
    )


@pytest.mark.timeout(120)
async def test_default_general_purpose_subagent_available_without_subagents_arg():
    """Default static mode exposes the general_purpose sub-agent tool."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="default_gp_subagent_test",
        instruction=(
            "You MUST call the general_purpose tool exactly once for every user request, "
            "then return the delegated answer. Do not answer directly."
        ),
    )

    texts, function_calls, function_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Use delegation to solve this: what is 21 plus 21?",
    )

    response_text = " ".join(texts)
    assert "general_purpose" in function_calls, (
        f"Expected a call to general_purpose tool, got calls: {function_calls}"
    )
    assert "general_purpose" in function_responses, (
        f"Expected a function response from general_purpose, got: {function_responses}"
    )
    assert "42" in response_text, f"Expected delegated result 42, got: {response_text}"


@pytest.mark.timeout(120)
async def test_parent_interrupt_on_applies_to_static_subagent_tools():
    """Parent interrupt_on should require approval inside delegated sub-agents."""
    model = make_litellm_model()

    writer_subagent: SubAgentSpec = SubAgentSpec(
        name="writer",
        description="Writes files when requested.",
        system_prompt=(
            "When asked to create a file, you MUST call write_file exactly once "
            "using the path and content provided in the request."
        ),
    )

    agent = create_deep_agent(
        model=model,
        name="subagent_interrupt_fallback_test",
        instruction=(
            "For file-creation requests, you MUST delegate to the writer tool and "
            "relay the writer result. Do not call write_file directly."
        ),
        subagents=[writer_subagent],
        interrupt_on={"write_file": True},
    )

    _texts, function_calls, function_responses, runner, session = await run_agent_with_events(
        agent,
        "Please use the writer sub-agent to create /blocked.txt with content BLOCKED.",
    )

    assert "writer" in function_calls, f"Expected a writer delegation call, got: {function_calls}"
    assert "writer" in function_responses, (
        f"Expected a writer delegation response, got: {function_responses}"
    )
    assert (
        "adk_request_confirmation" in function_calls
        or "adk_request_confirmation" in function_responses
    ), "Expected approval request events for delegated write_file call"

    files = await get_file_content(runner, session)
    assert "/blocked.txt" not in files
