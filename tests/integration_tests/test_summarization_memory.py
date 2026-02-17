"""Integration tests — summarization trigger and memory loading.

Scenario 1 (Summarization trigger): Send enough messages to trigger
summarization, verify conversation history is preserved (not lost).

Scenario 2 (Memory loading): Agent loads an AGENTS.md file via memory
config, verify the content appears in the agent's system prompt.

Run with: uv run pytest -m integration
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from adk_deepagents.types import SummarizationConfig

from .conftest import LITELLM_MODEL, make_model, run_agent

pytestmark = pytest.mark.integration


@pytest.mark.timeout(120)
async def test_summarization_trigger():
    """Send enough messages to trigger summarization, verify history is preserved.

    We configure a very small context window so summarization fires after
    a few exchanges. Then we verify that the agent still knows about earlier
    context (i.e., the LLM-generated summary preserved it).
    """
    # Use a tiny context window (200 tokens ≈ 800 chars) so summarization
    # triggers after just 1-2 exchanges, and a low trigger fraction.
    summarization = SummarizationConfig(
        model=LITELLM_MODEL,
        context_window=200,
        trigger=("fraction", 0.3),
        keep=("messages", 2),
        use_llm_summary=True,
    )

    agent = create_deep_agent(
        model=make_model(),
        name="summarization_test_agent",
        instruction=(
            "You are a test agent. Remember all facts the user tells you. "
            "When asked to recall facts, list them all. "
            "Always respond concisely."
        ),
        summarization=summarization,
    )

    # Turn 1: Establish a memorable fact
    texts, runner, session, _tool_calls = await run_agent(
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
    has_any_fact = any(
        fact in recall_response.lower() for fact in ("alpha", "phoenix", "march", "alice")
    )
    assert has_any_fact, (
        f"Expected agent to recall at least one fact after summarization, got: {recall_response}"
    )

    # Verify summarization actually fired by checking session state
    updated = await runner.session_service.get_session(
        app_name="integration_test",
        user_id="test_user",
        session_id=session.id,
    )
    summ_state = updated.state.get("_summarization_state")
    assert summ_state is not None, "Expected _summarization_state in session state"
    assert summ_state["summaries_performed"] > 0, (
        f"Expected at least one summarization, got: {summ_state}"
    )


@pytest.mark.timeout(120)
async def test_memory_loading():
    """Agent loads an AGENTS.md file via memory config.

    The AGENTS.md content should appear in the agent's system prompt and
    influence the agent's behavior/responses.
    """
    agents_md_content = (
        "# Project Guidelines\n\n"
        "- The project mascot is a golden retriever named Buddy.\n"
        "- All responses must end with the phrase 'Stay golden!'\n"
        "- The team motto is: 'Code with kindness.'\n"
    )

    agent = create_deep_agent(
        model=make_model(),
        name="memory_test_agent",
        instruction=(
            "You are a helpful test agent. Follow the guidelines loaded from "
            "your AGENTS.md memory exactly. Always follow project guidelines."
        ),
        memory=["/AGENTS.md"],
    )

    initial_files = {
        "/AGENTS.md": agents_md_content,
    }

    texts, _runner, _session, _tool_calls = await run_agent(
        agent,
        "What is the project mascot's name? And what is the team motto?",
        state={"files": initial_files},
    )

    response_text = " ".join(texts)
    has_mascot = "buddy" in response_text.lower()
    has_motto = "kindness" in response_text.lower()
    assert has_mascot or has_motto, (
        f"Expected agent to reference memory content (Buddy or kindness), got: {response_text}"
    )
