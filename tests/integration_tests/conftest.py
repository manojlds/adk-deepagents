"""Shared fixtures for integration tests."""

from __future__ import annotations

import os
from typing import Any

import pytest
from dotenv import load_dotenv

from adk_deepagents.backends.state import StateBackend

# Load .env from project root so env vars are available
load_dotenv()

LITELLM_MODEL = os.environ.get("LITELLM_MODEL", "openai/glm-5-free")


def pytest_collection_modifyitems(config, items):
    """Skip integration tests if OPENAI_API_KEY is not set."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    skip_marker = pytest.mark.skip(reason="OPENAI_API_KEY not set — skipping integration tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


def _backend_factory(state: dict[str, Any]) -> StateBackend:
    return StateBackend(state)


def make_model():
    """Create a LiteLlm model using standard env vars."""
    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(model=LITELLM_MODEL)


async def run_agent(agent, prompt: str, *, state: dict[str, Any] | None = None):
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
    tool_calls: list[str] = []

    async for event in runner.run_async(
        session_id=session.id,
        user_id="test_user",
        new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    texts.append(part.text)
                if hasattr(part, "function_call") and part.function_call:
                    tool_calls.append(part.function_call.name)

    return texts, runner, session, tool_calls


async def get_file_content(runner, session) -> dict[str, str]:
    """Return a dict of {path: content_str} from the session's file state."""
    updated = await runner.session_service.get_session(
        app_name="integration_test",
        user_id="test_user",
        session_id=session.id,
    )
    files = updated.state.get("files", {})
    result = {}
    for path, file_data in files.items():
        if isinstance(file_data, dict) and "content" in file_data:
            result[path] = "\n".join(file_data["content"])
        elif isinstance(file_data, str):
            result[path] = file_data
    return result
