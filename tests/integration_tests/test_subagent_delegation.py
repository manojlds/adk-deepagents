"""Integration test — sub-agent delegation with a real LLM.

Scenario: Main agent delegates a task to a sub-agent, verifies the sub-agent
ran and returned a result.

Requires OPENCODE_API_KEY environment variable to be set.
Run with: uv run pytest -m integration
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from adk_deepagents import create_deep_agent
from adk_deepagents.backends.state import StateBackend
from adk_deepagents.types import SubAgentSpec

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OPENCODE_API_KEY = os.environ.get("OPENCODE_API_KEY", "")
OPENCODE_API_BASE = "https://opencode.ai/zen/v1/chat/completions"


def _make_litellm_model():
    """Create a LiteLlm model pointing at the OpenCode Zen endpoint."""
    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(
        model="openai/gpt-4o-mini",
        api_key=OPENCODE_API_KEY,
        api_base=OPENCODE_API_BASE,
    )


def _backend_factory(state: dict[str, Any]) -> StateBackend:
    return StateBackend(state)


async def _run_agent(agent, prompt: str, *, state: dict[str, Any] | None = None):
    """Run *agent* with a single user prompt and return all text responses."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=agent, app_name="integration_test")

    initial_state: dict[str, Any] = {
        "files": {},
        "_backend_factory": _backend_factory,
    }
    if state:
        initial_state.update(state)

    session = await runner.session_service.create_session(
        app_name="integration_test",
        user_id="test_user",
        state=initial_state,
    )

    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    texts: list[str] = []

    async for event in runner.run_async(
        session_id=session.id,
        user_id="test_user",
        new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    texts.append(part.text)

    return texts, runner, session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
async def test_subagent_delegation():
    """Main agent delegates a task to a named sub-agent and gets a result."""
    model = _make_litellm_model()

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

    texts, _runner, _session = await _run_agent(
        agent,
        "Please delegate this math question to the math_expert sub-agent: "
        "What is 15 multiplied by 7? Report the answer back to me.",
    )

    response_text = " ".join(texts)
    # The sub-agent should compute 15 * 7 = 105 and the main agent should relay it
    assert "105" in response_text, (
        f"Expected '105' (15*7) in delegated response, got: {response_text}"
    )
