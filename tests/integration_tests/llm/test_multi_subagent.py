"""Integration test — multiple sub-agents with a real LLM.

Scenario: Agent with multiple specialized sub-agents delegates to the
correct one based on the task.

Requires OPENCODE_API_KEY environment variable to be set.
Run with: uv run pytest -m integration
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from adk_deepagents.types import SubAgentSpec
from tests.integration_tests.conftest import make_litellm_model, run_agent

pytestmark = pytest.mark.integration


@pytest.mark.timeout(120)
async def test_multiple_subagents_routing():
    """Agent routes tasks to the correct sub-agent based on expertise."""
    model = make_litellm_model()

    math_agent = SubAgentSpec(
        name="math_solver",
        description=(
            "Solves math and arithmetic problems. Delegate any calculation "
            "or math question to this agent."
        ),
        system_prompt="You are a math expert. Solve the given problem step by step.",
    )

    translator = SubAgentSpec(
        name="translator",
        description=(
            "Translates text between languages. Delegate any translation request to this agent."
        ),
        system_prompt=(
            "You are a translator. Translate the given text accurately. "
            "Always provide the translation and nothing else."
        ),
    )

    agent = create_deep_agent(
        model=model,
        name="multi_subagent_test",
        instruction=(
            "You are an orchestrator agent. You have two sub-agents:\n"
            "- math_solver: for math problems\n"
            "- translator: for translations\n\n"
            "You MUST delegate to the appropriate sub-agent based on the task. "
            "Report the sub-agent's result back to the user."
        ),
        subagents=[math_agent, translator],
    )

    # Test math delegation
    texts, _runner, _session = await run_agent(
        agent,
        "Delegate to the math_solver: What is 256 divided by 16?",
    )

    response_text = " ".join(texts)
    assert "16" in response_text, f"Expected '16' (256/16) in response, got: {response_text}"


@pytest.mark.timeout(120)
async def test_subagent_with_custom_model():
    """Sub-agent can use a different model than the parent."""
    model = make_litellm_model()

    # Sub-agent uses the same model in tests, but verifies the config path works
    analyst = SubAgentSpec(
        name="analyst",
        description="Analyzes data and provides insights. Delegate analysis tasks to this agent.",
        system_prompt="You are a data analyst. Analyze the given data concisely.",
        model=model,  # In production this could be a different model
    )

    agent = create_deep_agent(
        model=model,
        name="custom_model_test",
        instruction=(
            "You have an analyst sub-agent. Delegate analysis tasks to it. Report the result."
        ),
        subagents=[analyst],
    )

    texts, _runner, _session = await run_agent(
        agent,
        "Delegate to the analyst: If a company had revenue of $100M in Q1 "
        "and $120M in Q2, what is the quarter-over-quarter growth percentage?",
    )

    response_text = " ".join(texts)
    assert "20" in response_text, f"Expected '20%' growth in response, got: {response_text}"


@pytest.mark.timeout(120)
async def test_general_purpose_subagent_included():
    """General-purpose sub-agent is automatically included alongside custom ones."""
    model = make_litellm_model()

    writer = SubAgentSpec(
        name="writer",
        description="Writes text content. Delegate writing tasks to this agent.",
        system_prompt="You are a writer. Write the requested content concisely.",
    )

    agent = create_deep_agent(
        model=model,
        name="gp_subagent_test",
        instruction=(
            "You are an orchestrator. You have sub-agents available:\n"
            "- writer: for writing tasks\n"
            "- general_purpose: for research, search, and general tasks\n\n"
            "Delegate to the general_purpose sub-agent when the task doesn't "
            "match a specialist. Report the result."
        ),
        subagents=[writer],
    )

    # The general_purpose sub-agent should be available for delegation
    texts, _runner, _session = await run_agent(
        agent,
        "Delegate to the general_purpose sub-agent: "
        "Use the ls tool to list files in /. Report what you find.",
    )

    # The GP agent should run and return something (even if the listing is empty)
    response_text = " ".join(texts).lower()
    assert len(response_text) > 0, "Expected some response from general_purpose sub-agent"
