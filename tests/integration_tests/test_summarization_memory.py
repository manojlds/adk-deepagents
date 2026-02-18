"""Integration tests — summarization trigger and memory loading with a real LLM.

Scenario 1 (Summarization trigger): Send enough messages to trigger
summarization, verify conversation history is preserved (not lost).

Scenario 2 (Memory loading): Agent loads an AGENTS.md file via memory
config, verify the content appears in the agent's system prompt.

Requires OPENCODE_API_KEY environment variable to be set.
Run with: uv run pytest -m integration
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from adk_deepagents.backends.utils import create_file_data
from adk_deepagents.types import SummarizationConfig

from .conftest import make_litellm_model, run_agent, send_followup

pytestmark = pytest.mark.integration


@pytest.mark.timeout(120)
async def test_summarization_trigger():
    """Send enough messages to trigger summarization, verify history is preserved.

    We configure a very small context window so summarization fires after
    a few exchanges. Then we verify that the agent still knows about earlier
    context (i.e., the LLM-generated summary preserved it).
    """
    model = make_litellm_model()

    # Use a tiny context window so summarization triggers quickly.
    # With 500 tokens (~2000 chars), even 2-3 exchanges will trigger.
    summarization = SummarizationConfig(
        model="openai/gpt-4o-mini",
        context_window=500,
        trigger=("fraction", 0.5),
        keep=("messages", 2),
        use_llm_summary=True,
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
    texts, runner, session = await run_agent(
        agent,
        "Remember this important fact: The secret code is ALPHA-7. Confirm you understand.",
    )
    response_text = " ".join(texts).lower()
    assert any(
        word in response_text
        for word in ("alpha", "understood", "noted", "remember", "got it", "secret")
    ), f"Expected acknowledgment, got: {response_text}"

    # Turn 2: Add more context to push toward the trigger threshold
    await send_followup(
        runner,
        session,
        "Here is another fact to remember: The project name is Phoenix. "
        "Also remember that the deadline is March 15th. "
        "And the team lead is Alice. Confirm you have all these facts.",
    )

    # Turn 3: Ask the agent to recall — by now summarization should have
    # triggered, but the agent should still know about the facts
    texts3 = await send_followup(
        runner,
        session,
        "Please recall all the facts I told you. "
        "What is the secret code? What is the project name?",
    )

    recall_response = " ".join(texts3)
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
    model = make_litellm_model()

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

    initial_files = {
        "/AGENTS.md": create_file_data(agents_md_content),
    }

    texts, _runner, _session = await run_agent(
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
