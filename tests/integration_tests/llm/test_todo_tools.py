"""Integration test — todo tools with a real LLM.

Scenario: Agent uses write_todos to create a task list, then reads it back
with read_todos to verify the data round-trips through the LLM correctly.

Requires OPENCODE_API_KEY environment variable to be set.
Run with: uv run pytest -m integration
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from tests.integration_tests.conftest import make_litellm_model, run_agent, send_followup

pytestmark = [pytest.mark.integration, pytest.mark.llm]


@pytest.mark.timeout(120)
async def test_todo_write_and_read():
    """Agent writes a todo list, then reads it back."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="todo_test_agent",
        instruction=(
            "You are a test agent. When asked to create todos, use the write_todos "
            "tool with a list of items. Each item must have 'content' and 'status' keys. "
            "Valid statuses are: 'pending', 'in_progress', 'completed'. "
            "When asked to check todos, use the read_todos tool."
        ),
    )

    # Step 1: Write todos
    texts, runner, session = await run_agent(
        agent,
        "Create a todo list with these 3 items:\n"
        "1. Write documentation (status: in_progress)\n"
        "2. Fix bug in auth module (status: pending)\n"
        "3. Deploy to staging (status: completed)\n"
        "Use the write_todos tool.",
    )
    response_text = " ".join(texts).lower()
    assert any(
        word in response_text
        for word in ("created", "written", "todo", "success", "done", "list", "items")
    ), f"Expected confirmation of todo creation, got: {response_text}"

    # Step 2: Read todos back
    read_texts = await send_followup(
        runner,
        session,
        "Now use read_todos to show me the current todo list. List each item with its status.",
    )
    read_response = " ".join(read_texts).lower()

    # At least some of the todo content should appear
    has_doc = "documentation" in read_response
    has_bug = "bug" in read_response or "auth" in read_response
    has_deploy = "deploy" in read_response or "staging" in read_response
    assert has_doc or has_bug or has_deploy, (
        f"Expected todo items in response, got: {read_response}"
    )


@pytest.mark.timeout(120)
async def test_todo_update_status():
    """Agent creates todos, then updates them to reflect progress."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="todo_update_agent",
        instruction=(
            "You are a task management agent. Use write_todos and read_todos to manage "
            "a todo list. Each item has 'content' and 'status' fields. "
            "When asked to update a todo, rewrite the full list with the updated status."
        ),
    )

    # Step 1: Create initial todos
    texts, runner, session = await run_agent(
        agent,
        "Create todos: 'Setup database' (pending), 'Write API' (pending). Use write_todos.",
    )

    # Step 2: Update the first item to completed
    await send_followup(
        runner,
        session,
        "Update the todo list: mark 'Setup database' as completed, "
        "keep 'Write API' as pending. Use write_todos with the updated list.",
    )

    # Step 3: Read to verify the update
    verify_texts = await send_followup(
        runner,
        session,
        "Read the todos with read_todos. What is the status of 'Setup database'?",
    )
    verify_response = " ".join(verify_texts).lower()
    assert "completed" in verify_response or "done" in verify_response, (
        f"Expected 'Setup database' to be completed, got: {verify_response}"
    )
