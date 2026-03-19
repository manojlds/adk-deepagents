"""Integration test — HTTP tools with a real LLM.

Verifies that the agent can use ``fetch_url`` and ``http_request`` tools
with a real LLM backend.

Run with: uv run pytest -m llm
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from tests.integration_tests.conftest import (
    make_litellm_model,
    run_agent_with_events,
)

pytestmark = [pytest.mark.integration, pytest.mark.llm]


@pytest.mark.timeout(120)
async def test_fetch_url_tool_invocation():
    """Agent uses fetch_url to retrieve a web page."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="fetch_url_test",
        http_tools=True,
        instruction=(
            "You are a test agent with HTTP tools. "
            "Use fetch_url when asked to fetch a web page. "
            "Report a brief summary of what you fetched."
        ),
    )

    texts, function_calls, function_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Use the fetch_url tool to fetch https://httpbin.org/html "
        "and tell me what the page contains.",
    )

    # The agent should have called fetch_url
    assert "fetch_url" in function_calls, (
        f"Expected fetch_url in function_calls, got: {function_calls}"
    )

    # And produced a text response about the content
    response_text = " ".join(texts).lower()
    assert len(response_text) > 0, "Expected a text response"


@pytest.mark.timeout(120)
async def test_http_request_get():
    """Agent uses http_request to make a GET request."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="http_request_test",
        http_tools=True,
        instruction=(
            "You are a test agent with HTTP tools. "
            "Use http_request for API calls. Report the response."
        ),
    )

    texts, function_calls, function_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Use the http_request tool to make a GET request to "
        "https://httpbin.org/get and tell me what IP address it shows.",
    )

    assert "http_request" in function_calls, (
        f"Expected http_request in function_calls, got: {function_calls}"
    )

    response_text = " ".join(texts)
    assert len(response_text) > 0, "Expected a text response about the API result"


@pytest.mark.timeout(120)
async def test_ssrf_protection_blocks_private_url():
    """Agent sees an error when trying to fetch a private IP."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="ssrf_block_test",
        http_tools=True,
        instruction=(
            "You are a test agent. Use the fetch_url tool when asked. "
            "If the tool returns an error, report the error message."
        ),
    )

    texts, function_calls, function_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Use fetch_url to fetch http://192.168.1.1/admin. Report exactly what the tool returned.",
    )

    assert "fetch_url" in function_calls, (
        f"Expected fetch_url in function_calls, got: {function_calls}"
    )

    response_text = " ".join(texts).lower()
    assert any(
        kw in response_text for kw in ["block", "error", "private", "denied", "not allowed"]
    ), f"Expected SSRF block message, got: {response_text}"
