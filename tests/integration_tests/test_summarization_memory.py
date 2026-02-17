"""Integration tests — summarization trigger and memory loading.

Scenario 1 (Summarization trigger): Send enough messages to trigger
summarization, verify conversation history is preserved (not lost).

Scenario 2 (Memory loading): Agent loads an AGENTS.md file via memory
config, verify the content appears in the agent's system prompt.

Requires OPENCODE_API_KEY environment variable to be set.
Run with: uv run pytest -m integration
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from adk_deepagents import create_deep_agent
from adk_deepagents.backends.state import StateBackend
from adk_deepagents.types import SummarizationConfig

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
async def test_summarization_trigger():
    """Send enough messages to trigger summarization, verify history is preserved.

    We configure a very small context window so summarization fires after
    a few exchanges. Then we verify that the agent still knows about earlier
    context (i.e., the summary preserved it).
    """
    model = _make_litellm_model()

    # Use a tiny context window so summarization triggers quickly.
    # With 500 tokens (~2000 chars), even 2-3 exchanges will trigger.
    summarization = SummarizationConfig(
        model="openai/gpt-4o-mini",
        context_window=500,
        trigger=("fraction", 0.5),
        keep=("messages", 2),
        use_llm_summary=False,  # Use inline summary to avoid extra LLM call
    )

    agent = create_deep_agent(
        model=model,
        name="summarization_test_agent",
        instruction=(
            "You are a test agent. Remember all facts the user tells you. "
            "When asked to recall facts, list them all. "
            "Always respond concisely."
        ),
        summarization=summarization,
    )

    # Turn 1: Establish a memorable fact
    texts, runner, session = await _run_agent(
        agent,
        "Remember this important fact: The secret code is ALPHA-7. Confirm you understand.",
    )
    response_text = " ".join(texts).lower()
    assert any(
        word in response_text
        for word in ("alpha", "understood", "noted", "remember", "got it", "secret")
    ), f"Expected acknowledgment, got: {response_text}"

    from google.genai import types

    # Turn 2: Add more context to push toward the trigger threshold
    msg2 = types.Content(
        role="user",
        parts=[
            types.Part(
                text=(
                    "Here is another fact to remember: The project name is Phoenix. "
                    "Also remember that the deadline is March 15th. "
                    "And the team lead is Alice. Confirm you have all these facts."
                )
            )
        ],
    )
    texts2: list[str] = []
    async for event in runner.run_async(
        session_id=session.id,
        user_id="test_user",
        new_message=msg2,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    texts2.append(part.text)

    # Turn 3: Ask the agent to recall — by now summarization should have
    # triggered, but the agent should still know about the facts
    msg3 = types.Content(
        role="user",
        parts=[
            types.Part(
                text=(
                    "Please recall all the facts I told you. "
                    "What is the secret code? What is the project name?"
                )
            )
        ],
    )
    texts3: list[str] = []
    async for event in runner.run_async(
        session_id=session.id,
        user_id="test_user",
        new_message=msg3,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    texts3.append(part.text)

    recall_response = " ".join(texts3)
    # The agent should still know at least some facts — either from the kept
    # messages or from the summary that was injected
    has_any_fact = any(
        fact in recall_response.lower() for fact in ("alpha", "phoenix", "march", "alice")
    )
    assert has_any_fact, (
        f"Expected agent to recall at least one fact after summarization, got: {recall_response}"
    )


@pytest.mark.timeout(120)
async def test_memory_loading():
    """Agent loads an AGENTS.md file via memory config.

    The AGENTS.md content should appear in the agent's system prompt and
    influence the agent's behavior/responses.
    """
    model = _make_litellm_model()

    # Pre-populate an AGENTS.md file in the backend's files
    agents_md_content = (
        "# Project Guidelines\n\n"
        "- The project mascot is a golden retriever named Buddy.\n"
        "- All responses must end with the phrase 'Stay golden!'\n"
        "- The team motto is: 'Code with kindness.'\n"
    )

    agent = create_deep_agent(
        model=model,
        name="memory_test_agent",
        instruction=(
            "You are a helpful test agent. Follow the guidelines loaded from "
            "your AGENTS.md memory exactly. Always follow project guidelines."
        ),
        memory=["/AGENTS.md"],
    )

    # Pre-populate the AGENTS.md file in state so the backend can load it
    initial_files = {
        "/AGENTS.md": agents_md_content,
    }

    texts, _runner, _session = await _run_agent(
        agent,
        "What is the project mascot's name? And what is the team motto?",
        state={"files": initial_files},
    )

    response_text = " ".join(texts)
    # The agent should know the mascot name and/or motto from the loaded memory
    has_mascot = "buddy" in response_text.lower()
    has_motto = "kindness" in response_text.lower()
    assert has_mascot or has_motto, (
        f"Expected agent to reference memory content (Buddy or kindness), got: {response_text}"
    )
