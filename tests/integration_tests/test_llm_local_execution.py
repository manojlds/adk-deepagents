"""Integration test — local shell execution with a real LLM.

Scenario: Agent uses the execute tool to run shell commands and process
the results.

Requires OPENCODE_API_KEY environment variable to be set.
Run with: uv run pytest -m integration
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent

from .conftest import make_litellm_model, run_agent, send_followup

pytestmark = pytest.mark.integration


@pytest.mark.timeout(120)
async def test_execute_simple_command():
    """Agent runs a simple shell command and reports the output."""
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

    texts, _runner, _session = await run_agent(
        agent,
        "Use the execute tool to run: echo 'HELLO_FROM_TEST_42'. "
        "Show me the exact output.",
    )

    response_text = " ".join(texts)
    assert "HELLO_FROM_TEST_42" in response_text, (
        f"Expected echo output in response, got: {response_text}"
    )


@pytest.mark.timeout(120)
async def test_execute_python_command():
    """Agent runs a Python one-liner via shell and reports the result."""
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

    texts, _runner, _session = await run_agent(
        agent,
        "Use the execute tool to run: python3 -c \"print(7 * 13)\". "
        "What is the result?",
    )

    response_text = " ".join(texts)
    assert "91" in response_text, (
        f"Expected '91' (7*13) in response, got: {response_text}"
    )


@pytest.mark.timeout(120)
async def test_execute_and_write_file():
    """Agent runs a command, then writes the output to a file."""
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

    texts, runner, session = await run_agent(
        agent,
        "Use the execute tool to run: date +%Y. "
        "Then use write_file to save the year to /year.txt. "
        "Confirm when done.",
    )

    response_text = " ".join(texts).lower()
    assert any(
        word in response_text for word in ("done", "saved", "written", "created", "file")
    ), f"Expected confirmation, got: {response_text}"

    # Verify the file was created by reading it
    read_texts = await send_followup(
        runner, session, "Read the file /year.txt and show me the content."
    )
    read_response = " ".join(read_texts)
    assert "202" in read_response, (
        f"Expected year (202x) in file content, got: {read_response}"
    )
