"""Integration test — sub-agent delegation with a real LLM.

Scenario: Main agent delegates a task to a sub-agent, verifies the sub-agent
tool was actually called and returned the correct result.

Run with: uv run pytest -m integration
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from adk_deepagents.types import SubAgentSpec

from .conftest import make_model, run_agent

pytestmark = pytest.mark.integration


@pytest.mark.timeout(120)
async def test_subagent_delegation():
    """Main agent delegates a task to a named sub-agent and gets a result."""
    math_subagent: SubAgentSpec = SubAgentSpec(
        name="math_expert",
        description="A sub-agent that solves math problems. Delegate math questions to this agent.",
    )

    agent = create_deep_agent(
        model=make_model(),
        name="delegation_test_agent",
        instruction=(
            "You are a test agent. You have a sub-agent called 'math_expert' that is "
            "specialized in solving math problems. When the user asks a math question, "
            "you MUST delegate to the math_expert sub-agent using the math_expert tool. "
            "After receiving the result, report it back to the user."
        ),
        subagents=[math_subagent],
    )

    texts, _runner, _session, tool_calls = await run_agent(
        agent,
        "Please delegate this math question to the math_expert sub-agent: "
        "What is 15 multiplied by 7? Report the answer back to me.",
    )

    # Verify the sub-agent tool was actually invoked
    assert "math_expert" in tool_calls, f"Expected 'math_expert' in tool calls, got: {tool_calls}"

    # Verify the correct answer came back
    response_text = " ".join(texts)
    assert "105" in response_text, (
        f"Expected '105' (15*7) in delegated response, got: {response_text}"
    )
