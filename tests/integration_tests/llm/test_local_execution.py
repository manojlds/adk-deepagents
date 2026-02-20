"""Integration tests — local shell execution with a real LLM.

Scenario: Agent uses the execute tool and combines shell output with
filesystem tools. Assertions verify both tool usage and persisted state.

Run with: uv run pytest -m llm tests/integration_tests/llm/test_local_execution.py
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from tests.integration_tests.conftest import (
    get_file_content,
    make_litellm_model,
    run_agent_with_events,
)

pytestmark = [pytest.mark.integration, pytest.mark.llm]


@pytest.mark.timeout(120)
async def test_execute_simple_command():
    """Agent invokes execute and returns deterministic shell output."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="exec_test_agent",
        instruction=(
            "You are a test agent with shell access. Use the execute tool to run "
            "shell commands. Report the output exactly."
        ),
        execution="local",
    )

    texts, function_calls, function_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Use the execute tool to run: printf 'HELLO_FROM_TEST_42'. Show me the exact output.",
    )

    response_text = " ".join(texts)
    assert "execute" in function_calls, f"Expected execute call, got: {function_calls}"
    assert "execute" in function_responses, f"Expected execute response, got: {function_responses}"
    assert "HELLO_FROM_TEST_42" in response_text, (
        f"Expected echo output in response, got: {response_text}"
    )


@pytest.mark.timeout(120)
async def test_execute_python_command():
    """Agent invokes execute for Python one-liner computation."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="exec_python_agent",
        instruction=(
            "You are a test agent. Use the execute tool to run shell commands. "
            "Report results accurately."
        ),
        execution="local",
    )

    texts, function_calls, function_responses, _runner, _session = await run_agent_with_events(
        agent,
        'Use the execute tool to run: python3 -c "print(7 * 13)". What is the result?',
    )

    response_text = " ".join(texts)
    assert "execute" in function_calls, f"Expected execute call, got: {function_calls}"
    assert "execute" in function_responses, f"Expected execute response, got: {function_responses}"
    assert "91" in response_text, f"Expected '91' (7*13) in response, got: {response_text}"


@pytest.mark.timeout(120)
async def test_execute_and_write_file():
    """Agent combines execute and write_file and persists exact output."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="exec_write_agent",
        instruction=(
            "You are a test agent. You can run shell commands with execute and "
            "work with files using filesystem tools. Follow instructions exactly."
        ),
        execution="local",
    )

    token = "YEAR_TEST_2042"

    texts, function_calls, function_responses, runner, session = await run_agent_with_events(
        agent,
        f"Use the execute tool to run: printf '{token}'. "
        "Then use write_file to save the exact output to /year.txt. "
        "Confirm when done.",
    )

    response_text = " ".join(texts)
    assert "execute" in function_calls, f"Expected execute call, got: {function_calls}"
    assert "execute" in function_responses, f"Expected execute response, got: {function_responses}"
    assert "write_file" in function_calls, f"Expected write_file call, got: {function_calls}"
    assert "write_file" in function_responses, (
        f"Expected write_file response, got: {function_responses}"
    )
    assert token in response_text, f"Expected token in response, got: {response_text}"

    files = await get_file_content(runner, session)
    assert "/year.txt" in files, f"Expected /year.txt in backend files, got: {list(files.keys())}"
    file_content = files["/year.txt"].strip()
    assert token == file_content, f"Expected exact token in /year.txt, got: {file_content}"

    lower_response = response_text.lower()
    assert any(
        word in lower_response for word in ("done", "saved", "written", "created", "file")
    ), f"Expected confirmation, got: {lower_response}"
