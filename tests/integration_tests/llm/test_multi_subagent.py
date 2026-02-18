"""Integration test — multiple sub-agents with a real LLM.

Scenario: Agent with multiple specialized sub-agents delegates to the
correct one based on the task.

Run with: uv run pytest -m llm
"""

from __future__ import annotations

import os

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
    texts, function_calls, function_responses, runner, session = await run_agent_with_events(
        agent,
        "Delegate to the math_solver: What is 256 divided by 16?",
    )

    response_text = " ".join(texts)
    assert "math_solver" in function_calls, f"Expected math_solver tool call, got: {function_calls}"
    assert "math_solver" in function_responses, (
        f"Expected math_solver tool response, got: {function_responses}"
    )
    assert "translator" not in function_calls, (
        f"Did not expect translator for a math-only task, got: {function_calls}"
    )
    assert "16" in response_text, f"Expected '16' (256/16) in response, got: {response_text}"

    # Test translation delegation in the same session
    texts2, function_calls2, function_responses2 = await send_followup_with_events(
        runner,
        session,
        "Delegate to the translator: Translate 'Good morning' to Spanish.",
    )

    response_text2 = " ".join(texts2).lower()
    assert "translator" in function_calls2, f"Expected translator tool call, got: {function_calls2}"
    assert "translator" in function_responses2, (
        f"Expected translator tool response, got: {function_responses2}"
    )
    assert "math_solver" not in function_calls2, (
        f"Did not expect math_solver for a translation task, got: {function_calls2}"
    )
    assert any(word in response_text2 for word in ("buenos", "dias", "morning", "spanish")), (
        f"Expected translation content in response, got: {response_text2}"
    )


@pytest.mark.timeout(120)
async def test_multi_subagent_pipeline_delegates_twice():
    """Agent can orchestrate a task that requires multiple sub-agent delegations."""
    model = make_litellm_model()

    math_agent = SubAgentSpec(
        name="math_solver",
        description="Solves math and arithmetic problems.",
        system_prompt="You are a math expert. Solve the given problem step by step.",
    )
    translator = SubAgentSpec(
        name="translator",
        description="Translates text between languages.",
        system_prompt="You are a translator. Translate the given text accurately.",
    )

    agent = create_deep_agent(
        model=model,
        name="multi_subagent_pipeline_test",
        instruction=(
            "You are an orchestrator agent. Use translator for translation tasks and "
            "math_solver for arithmetic tasks. If a request asks for both, delegate to "
            "both sub-agents and report both outputs."
        ),
        subagents=[math_agent, translator],
    )

    texts, function_calls, function_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Do two things: (1) delegate to translator to translate 'Good evening' to French; "
        "(2) delegate to math_solver to compute 9 + 6. Return both results.",
    )

    response_text = " ".join(texts).lower()
    assert "translator" in function_calls, f"Expected translator tool call, got: {function_calls}"
    assert "math_solver" in function_calls, f"Expected math_solver tool call, got: {function_calls}"
    assert "translator" in function_responses, (
        f"Expected translator tool response, got: {function_responses}"
    )
    assert "math_solver" in function_responses, (
        f"Expected math_solver tool response, got: {function_responses}"
    )
    assert "15" in response_text, f"Expected 15 in response, got: {response_text}"


@pytest.mark.timeout(120)
async def test_subagent_with_custom_model():
    """Sub-agent can use a different model than the parent."""
    model = make_litellm_model()
    subagent_model = os.environ.get("LITELLM_MODEL", "openai/gpt-4o-mini")

    # Sub-agent uses the same model in tests, but verifies the config path works
    analyst = SubAgentSpec(
        name="analyst",
        description="Analyzes data and provides insights. Delegate analysis tasks to this agent.",
        system_prompt="You are a data analyst. Analyze the given data concisely.",
        model=subagent_model,
    )

    agent = create_deep_agent(
        model=model,
        name="custom_model_test",
        instruction=(
            "You have an analyst sub-agent. Delegate analysis tasks to it. Report the result."
        ),
        subagents=[analyst],
    )

    texts, function_calls, function_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Delegate to the analyst: If a company had revenue of $100M in Q1 "
        "and $120M in Q2, what is the quarter-over-quarter growth percentage?",
    )

    response_text = " ".join(texts)
    assert "analyst" in function_calls, f"Expected analyst tool call, got: {function_calls}"
    assert "analyst" in function_responses, (
        f"Expected analyst tool response, got: {function_responses}"
    )
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
    texts, function_calls, function_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Delegate to the general_purpose sub-agent: "
        "Use the ls tool to list files in /. Report what you find.",
    )

    # The GP agent should run and return something (even if the listing is empty)
    response_text = " ".join(texts).lower()
    assert "general_purpose" in function_calls, (
        f"Expected general_purpose tool call, got: {function_calls}"
    )
    assert "general_purpose" in function_responses, (
        f"Expected general_purpose tool response, got: {function_responses}"
    )
    assert len(response_text) > 0, "Expected some response from general_purpose sub-agent"
