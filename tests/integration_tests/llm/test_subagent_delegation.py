"""Integration test — sub-agent delegation with a real LLM.

Scenario: Main agent delegates a task to a sub-agent, verifies the sub-agent
ran and returned a result.

Run with: uv run pytest -m llm
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from adk_deepagents.types import SubAgentSpec
from tests.integration_tests.conftest import make_litellm_model, run_agent_with_events

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
