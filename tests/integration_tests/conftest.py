"""Shared fixtures and helpers for integration tests."""

from __future__ import annotations

import os
from typing import Any

import pytest

from adk_deepagents.backends.state import StateBackend

# ---------------------------------------------------------------------------
# Skip gate
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items):
    """Skip integration tests if no LLM API key is available."""
    has_key = os.environ.get("OPENCODE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if has_key:
        return
    skip_marker = pytest.mark.skip(
        reason="No API key set (OPENCODE_API_KEY or OPENAI_API_KEY) — skipping integration tests"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_litellm_model():
    """Create a LiteLlm model using available environment variables.

    Reads from OPENAI_API_BASE / LITELLM_MODEL / OPENAI_API_KEY if set,
    falling back to OPENCODE_API_KEY with default endpoint.
    """
    from google.adk.models.lite_llm import LiteLlm

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENCODE_API_KEY", "")
    api_base = os.environ.get("OPENAI_API_BASE", "https://opencode.ai/zen/v1")
    model = os.environ.get("LITELLM_MODEL", "openai/gpt-4o-mini")

    return LiteLlm(
        model=model,
        api_key=api_key,
        api_base=api_base,
    )


def backend_factory(state: dict[str, Any]) -> StateBackend:
    """Default backend factory for integration tests."""
    return StateBackend(state)


async def run_agent(agent, prompt: str, *, state: dict[str, Any] | None = None):
    """Run *agent* with a single user prompt and return (texts, runner, session).

    Returns all text responses, the runner instance, and the session object
    so callers can send follow-up messages on the same session.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=agent, app_name="integration_test")

    initial_state: dict[str, Any] = {
        "files": {},
        "_backend_factory": backend_factory,
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


async def get_file_content(runner, session) -> dict[str, str]:
    """Return a dict of {path: content_str} from the session's file state."""
    updated = await runner.session_service.get_session(
        app_name="integration_test",
        user_id="test_user",
        session_id=session.id,
    )
    files = updated.state.get("files", {})
    result: dict[str, str] = {}
    for path, file_data in files.items():
        if isinstance(file_data, dict) and "content" in file_data:
            result[path] = "\n".join(file_data["content"])
        elif isinstance(file_data, str):
            result[path] = file_data
    return result


async def send_followup(runner, session, prompt: str) -> list[str]:
    """Send a follow-up message on an existing session and return text responses."""
    from google.genai import types

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

    return texts
