"""Integration test — multi-turn conversations with a real LLM.

Tests that the agent maintains context across multiple turns, including
file state, todo state, and conversation history.

Requires OPENCODE_API_KEY environment variable to be set.
Run with: uv run pytest -m integration
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent

from .conftest import make_litellm_model, run_agent, send_followup

pytestmark = pytest.mark.integration


@pytest.mark.timeout(180)
async def test_multi_turn_file_operations():
    """Agent performs a sequence of file operations across multiple turns."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="multi_turn_fs_agent",
        instruction=(
            "You are a coding assistant. Help the user build a project step by step. "
            "Use filesystem tools to create and modify files. Confirm each action."
        ),
    )

    # Turn 1: Create a Python file
    texts, runner, session = await run_agent(
        agent,
        "Use write_file to create /app.py with content:\n"
        "def greet(name):\n"
        '    return f"Hello, {name}!"\n\n'
        "Confirm when done.",
    )
    response = " ".join(texts).lower()
    assert any(w in response for w in ("created", "written", "done", "success")), (
        f"Expected file creation confirmation, got: {response}"
    )

    # Turn 2: Create another file
    await send_followup(
        runner,
        session,
        "Now use write_file to create /test_app.py with content:\n"
        "from app import greet\n\n"
        "def test_greet():\n"
        '    assert greet("World") == "Hello, World!"\n',
    )

    # Turn 3: List the files
    ls_texts = await send_followup(
        runner, session, "Use ls to list files in /. How many files are there?"
    )
    ls_response = " ".join(ls_texts).lower()
    assert "app" in ls_response, f"Expected app.py in listing, got: {ls_response}"

    # Turn 4: Read a file the agent created earlier
    read_texts = await send_followup(
        runner, session, "Read /app.py and show me the content."
    )
    read_response = " ".join(read_texts)
    assert "greet" in read_response, (
        f"Expected greet function in file content, got: {read_response}"
    )


@pytest.mark.timeout(180)
async def test_multi_turn_todo_and_files():
    """Agent uses both todo tools and filesystem tools across turns."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="multi_turn_combo_agent",
        instruction=(
            "You are a project assistant. Use todo tools to track tasks and "
            "filesystem tools to create deliverables. Follow instructions step by step."
        ),
    )

    # Turn 1: Create a plan
    texts, runner, session = await run_agent(
        agent,
        "Create a todo list with write_todos:\n"
        "1. 'Write README' (status: pending)\n"
        "2. 'Create config' (status: pending)\n",
    )

    # Turn 2: Complete the first task
    await send_followup(
        runner,
        session,
        "Now complete the first todo: use write_file to create /README.md "
        "with content '# My Project\\nA test project.'. "
        "Then update the todo list to mark 'Write README' as completed.",
    )

    # Turn 3: Verify both
    verify_texts = await send_followup(
        runner,
        session,
        "Read the todos with read_todos and read /README.md. "
        "Tell me the status of each todo and what's in the README.",
    )
    verify_response = " ".join(verify_texts).lower()
    # Should have some reference to completed or the README content
    has_readme = "my project" in verify_response or "readme" in verify_response
    has_status = "completed" in verify_response or "done" in verify_response
    assert has_readme or has_status, (
        f"Expected task progress and file content, got: {verify_response}"
    )


@pytest.mark.timeout(120)
async def test_conversation_context_preserved():
    """Agent remembers information from earlier turns without memory/summarization."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="context_agent",
        instruction=(
            "You are a helpful assistant. Remember everything the user tells you "
            "in this conversation. Answer questions based on earlier context."
        ),
    )

    # Turn 1: Tell the agent something
    texts, runner, session = await run_agent(
        agent,
        "My favorite programming language is Elixir and I work at a company "
        "called NovaTech. Remember this.",
    )

    # Turn 2: Ask about something unrelated
    await send_followup(
        runner, session, "What is 2 + 2?"
    )

    # Turn 3: Ask about the earlier context
    recall_texts = await send_followup(
        runner,
        session,
        "What is my favorite programming language and where do I work?",
    )
    recall_response = " ".join(recall_texts).lower()
    has_elixir = "elixir" in recall_response
    has_company = "novatech" in recall_response
    assert has_elixir or has_company, (
        f"Expected Elixir or NovaTech from earlier context, got: {recall_response}"
    )
