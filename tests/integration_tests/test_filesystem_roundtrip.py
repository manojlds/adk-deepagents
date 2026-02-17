"""Integration test — filesystem round-trip with a real LLM.

Scenario: Agent creates a file via write_file, reads it back via read_file,
edits it via edit_file, then verifies the final content.

Requires OPENCODE_API_KEY environment variable to be set.
Run with: uv run pytest -m integration
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from adk_deepagents import create_deep_agent
from adk_deepagents.backends.state import StateBackend

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
async def test_filesystem_roundtrip():
    """Agent writes, reads, edits, and verifies a file end-to-end."""
    model = _make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="fs_roundtrip_agent",
        instruction=(
            "You are a test agent. Follow the user's instructions exactly. "
            "Use the filesystem tools (write_file, read_file, edit_file) as directed. "
            "After each step, confirm what you did."
        ),
    )

    # Step 1: Write a file
    texts, runner, session = await _run_agent(
        agent,
        'Use write_file to create a file at /test.txt with the content "Hello World". '
        "Confirm when done.",
    )

    response_text = " ".join(texts).lower()
    assert any(
        word in response_text for word in ("done", "created", "written", "success", "file")
    ), f"Expected confirmation of file creation, got: {response_text}"

    # Step 2: Read it back (same session)
    from google.genai import types

    read_msg = types.Content(
        role="user",
        parts=[types.Part(text="Now use read_file to read /test.txt. Show me the content.")],
    )
    read_texts: list[str] = []
    async for event in runner.run_async(
        session_id=session.id,
        user_id="test_user",
        new_message=read_msg,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    read_texts.append(part.text)

    read_response = " ".join(read_texts)
    assert "Hello World" in read_response, (
        f"Expected file content in response, got: {read_response}"
    )

    # Step 3: Edit the file
    edit_msg = types.Content(
        role="user",
        parts=[
            types.Part(
                text=(
                    'Use edit_file to replace "Hello World" with "Hello Universe" '
                    "in /test.txt. Confirm when done."
                )
            )
        ],
    )
    edit_texts: list[str] = []
    async for event in runner.run_async(
        session_id=session.id,
        user_id="test_user",
        new_message=edit_msg,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    edit_texts.append(part.text)

    edit_response = " ".join(edit_texts).lower()
    assert any(
        word in edit_response for word in ("done", "edited", "replaced", "success", "updated")
    ), f"Expected confirmation of edit, got: {edit_response}"

    # Step 4: Read again to verify the edit
    verify_msg = types.Content(
        role="user",
        parts=[types.Part(text="Read /test.txt again and show me the current content.")],
    )
    verify_texts: list[str] = []
    async for event in runner.run_async(
        session_id=session.id,
        user_id="test_user",
        new_message=verify_msg,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    verify_texts.append(part.text)

    verify_response = " ".join(verify_texts)
    assert "Hello Universe" in verify_response, (
        f"Expected edited content in response, got: {verify_response}"
    )
