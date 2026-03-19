"""Integration test — tool error handler with a real LLM.

Verifies that when a user-provided tool raises an exception, the error
handler catches it and returns a structured error dict, allowing the
LLM to self-correct rather than crashing the agent loop.

Run with: uv run pytest -m llm
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from tests.integration_tests.conftest import (
    make_litellm_model,
    run_agent,
    run_agent_with_events,
)

pytestmark = [pytest.mark.integration, pytest.mark.llm]


def flaky_calculator(expression: str) -> dict:
    """Evaluate a math expression. Raises on division by zero.

    Args:
        expression: A Python math expression to evaluate.
    """
    result = eval(expression)  # noqa: S307
    return {"status": "success", "result": float(result)}


def always_fails(query: str) -> dict:
    """A tool that always raises an error. Used for testing.

    Args:
        query: The input query.
    """
    raise RuntimeError(f"Service unavailable for query: {query}")


@pytest.mark.timeout(120)
async def test_error_handler_catches_tool_exception():
    """Agent recovers when a tool raises an exception via error handler."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="error_handler_test",
        tools=[flaky_calculator],
        instruction=(
            "You are a test agent with a flaky_calculator tool. "
            "If the tool returns an error, explain what went wrong. "
            "Try to compute the user's request."
        ),
        error_handling=True,
    )

    # Division by zero will raise inside eval
    texts, _runner, _session = await run_agent(
        agent,
        "Use the flaky_calculator tool to evaluate '1 / 0'. "
        "If it fails, tell me what error occurred.",
    )

    response_text = " ".join(texts).lower()
    # The agent should see the error and report it rather than crashing
    assert any(
        kw in response_text for kw in ["error", "division", "zero", "fail", "cannot", "undefined"]
    ), f"Expected error-related response, got: {response_text}"


@pytest.mark.timeout(120)
async def test_error_handler_agent_completes_despite_failure():
    """Agent loop completes even when a tool always raises."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="always_fails_test",
        tools=[always_fails],
        instruction=(
            "You are a test agent with an always_fails tool. "
            "Try calling it once. If it errors, report the error and stop."
        ),
        error_handling=True,
    )

    texts, function_calls, function_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Call the always_fails tool with query 'test'. Report what happens.",
    )

    # The agent should have attempted the tool call
    assert "always_fails" in function_calls, (
        f"Expected always_fails in function_calls, got: {function_calls}"
    )

    # And the agent should have produced a text response (not crashed)
    assert len(texts) > 0, "Expected at least one text response"
    response_text = " ".join(texts).lower()
    assert any(kw in response_text for kw in ["error", "unavailable", "fail", "service"]), (
        f"Expected error report, got: {response_text}"
    )
