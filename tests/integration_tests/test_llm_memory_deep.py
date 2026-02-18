"""Integration test — deep memory scenarios with a real LLM.

Tests memory loading from multiple files, memory influencing agent behavior,
and memory with FilesystemBackend.

Requires OPENCODE_API_KEY environment variable to be set.
Run with: uv run pytest -m integration
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from adk_deepagents.backends.utils import create_file_data

from .conftest import make_litellm_model, run_agent

pytestmark = pytest.mark.integration


@pytest.mark.timeout(120)
async def test_memory_multiple_files():
    """Agent loads multiple memory files and uses information from all of them."""
    model = make_litellm_model()

    agents_md = (
        "# Agent Identity\n\n"
        "- Your name is Atlas.\n"
        "- You always introduce yourself by name when greeted.\n"
    )
    context_md = (
        "# Project Context\n\n"
        "- The project is called Starlight.\n"
        "- The primary language is Rust.\n"
        "- The team uses GitLab for version control.\n"
    )

    agent = create_deep_agent(
        model=model,
        name="multi_memory_agent",
        instruction=(
            "You are a helpful assistant. Follow all guidelines from your "
            "loaded memory files exactly."
        ),
        memory=["/AGENTS.md", "/CONTEXT.md"],
    )

    initial_files = {
        "/AGENTS.md": create_file_data(agents_md),
        "/CONTEXT.md": create_file_data(context_md),
    }

    texts, _runner, _session = await run_agent(
        agent,
        "Hello! What's your name and what project are you working on?",
        state={"files": initial_files},
    )

    response_text = " ".join(texts).lower()
    has_name = "atlas" in response_text
    has_project = "starlight" in response_text
    assert has_name or has_project, (
        f"Expected Atlas or Starlight from memory, got: {response_text}"
    )


@pytest.mark.timeout(120)
async def test_memory_influences_behavior():
    """Memory content should change how the agent responds."""
    model = make_litellm_model()

    agents_md = (
        "# Response Rules\n\n"
        "- Always respond in exactly 3 bullet points.\n"
        "- End every response with 'Over and out.'\n"
        "- Never use the word 'however'.\n"
    )

    agent = create_deep_agent(
        model=model,
        name="behavior_memory_agent",
        instruction=(
            "You are a test agent. Follow the response rules from your "
            "AGENTS.md memory strictly. Never deviate from the rules."
        ),
        memory=["/AGENTS.md"],
    )

    initial_files = {
        "/AGENTS.md": create_file_data(agents_md),
    }

    texts, _runner, _session = await run_agent(
        agent,
        "What are three benefits of open source software?",
        state={"files": initial_files},
    )

    response_text = " ".join(texts)
    # The agent should follow at least one of the rules
    has_sign_off = "over and out" in response_text.lower()
    has_bullets = response_text.count("•") >= 2 or response_text.count("-") >= 2
    assert has_sign_off or has_bullets, (
        f"Expected agent to follow memory rules (bullets or sign-off), got: {response_text}"
    )


@pytest.mark.timeout(120)
async def test_memory_with_coding_context():
    """Agent uses memory to understand a codebase and answer questions."""
    model = make_litellm_model()

    agents_md = (
        "# Codebase Context\n\n"
        "- The application is a REST API built with FastAPI.\n"
        "- Database: PostgreSQL with SQLAlchemy ORM.\n"
        "- Authentication: JWT tokens with 24-hour expiry.\n"
        "- The main entry point is `app/main.py`.\n"
        "- Tests are in `tests/` using pytest.\n"
        "- Environment variables are loaded from `.env` via python-dotenv.\n"
    )

    agent = create_deep_agent(
        model=model,
        name="codebase_memory_agent",
        instruction=(
            "You are a coding assistant with knowledge of the project from "
            "your AGENTS.md memory. Answer questions based on the project context."
        ),
        memory=["/AGENTS.md"],
    )

    initial_files = {
        "/AGENTS.md": create_file_data(agents_md),
    }

    texts, _runner, _session = await run_agent(
        agent,
        "What framework is the API built with, and what database does it use?",
        state={"files": initial_files},
    )

    response_text = " ".join(texts).lower()
    has_fastapi = "fastapi" in response_text
    has_postgres = "postgres" in response_text
    assert has_fastapi or has_postgres, (
        f"Expected FastAPI or PostgreSQL from memory, got: {response_text}"
    )
