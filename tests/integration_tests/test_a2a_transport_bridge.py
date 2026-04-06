"""Integration tests for the optional A2A LLM transport bridge.

These tests validate that the shared integration helpers can execute a full
turn against an in-process A2A app and still expose the same helper contract
used by existing LLM integration tests.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from adk_deepagents import create_deep_agent
from tests.integration_tests.conftest import (
    get_file_content,
    make_litellm_model,
    run_agent,
    run_agent_with_events,
    send_followup,
)

pytestmark = [pytest.mark.integration, pytest.mark.llm]


@contextmanager
def _llm_transport_mode(mode: str):
    import os

    original = os.environ.get("ADK_DEEPAGENTS_LLM_TRANSPORT")
    os.environ["ADK_DEEPAGENTS_LLM_TRANSPORT"] = mode
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("ADK_DEEPAGENTS_LLM_TRANSPORT", None)
        else:
            os.environ["ADK_DEEPAGENTS_LLM_TRANSPORT"] = original


async def test_a2a_transport_bridge_supports_run_and_followup():
    model = make_litellm_model()
    agent = create_deep_agent(
        model=model,
        name="a2a_transport_bridge_test",
        instruction=(
            "Use write_file to create /bridge.txt with content 'bridge-ok' and then confirm done."
        ),
    )

    with _llm_transport_mode("a2a"):
        texts, runner, session = await run_agent(
            agent,
            "Create /bridge.txt with content bridge-ok, then confirm.",
        )
        assert texts, "Expected at least one text response in A2A mode"

        files = await get_file_content(runner, session)
        assert "/bridge.txt" in files
        assert "bridge-ok" in files["/bridge.txt"]

        followup_texts = await send_followup(
            runner,
            session,
            "Read /bridge.txt and reply with its content exactly.",
        )
        response = " ".join(followup_texts).lower()
        assert "bridge-ok" in response


async def test_a2a_transport_bridge_reports_function_calls():
    model = make_litellm_model()
    agent = create_deep_agent(
        model=model,
        name="a2a_transport_bridge_events_test",
        instruction=(
            "Use write_todos to create a single todo named 'Ship bridge test' with pending status, "
            "then confirm."
        ),
    )

    with _llm_transport_mode("a2a"):
        texts, function_calls, function_responses, _runner, _session = await run_agent_with_events(
            agent,
            "Create one pending todo called Ship bridge test and confirm completion.",
        )

    assert texts
    assert "write_todos" in function_calls
    assert "write_todos" in function_responses
