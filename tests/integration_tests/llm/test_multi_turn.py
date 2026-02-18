"""Integration test — multi-turn conversations with a real LLM.

Tests that the agent maintains context across multiple turns, including
file state, todo state, and conversation history.

Run with: uv run pytest -m llm
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from tests.integration_tests.conftest import (
    get_file_content,
    make_litellm_model,
    run_agent_with_events,
    send_followup,
    send_followup_with_events,
)

pytestmark = [pytest.mark.integration, pytest.mark.llm]


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
    texts, function_calls, function_responses, runner, session = await run_agent_with_events(
        agent,
        "Use write_file to create /app.py with content:\n"
        "def greet(name):\n"
        '    return f"Hello, {name}!"\n\n'
        "Confirm when done.",
    )
    assert "write_file" in function_calls, f"Expected write_file call, got: {function_calls}"
    assert "write_file" in function_responses, (
        f"Expected write_file response, got: {function_responses}"
    )
    files = await get_file_content(runner, session)
    assert "/app.py" in files, f"Expected /app.py in backend files, got: {list(files.keys())}"
    assert "def greet(name):" in files["/app.py"], (
        f"Expected greet function in /app.py, got: {files['/app.py']}"
    )
    response = " ".join(texts).lower()
    assert any(w in response for w in ("created", "written", "done", "success")), (
        f"Expected file creation confirmation, got: {response}"
    )

    # Turn 2: Create another file
    _, function_calls2, function_responses2 = await send_followup_with_events(
        runner,
        session,
        "Now use write_file to create /test_app.py with content:\n"
        "from app import greet\n\n"
        "def test_greet():\n"
        '    assert greet("World") == "Hello, World!"\n',
    )
    assert "write_file" in function_calls2, f"Expected write_file call, got: {function_calls2}"
    assert "write_file" in function_responses2, (
        f"Expected write_file response, got: {function_responses2}"
    )
    files_after = await get_file_content(runner, session)
    assert "/test_app.py" in files_after, (
        f"Expected /test_app.py in backend files, got: {list(files_after.keys())}"
    )
    assert 'assert greet("World") == "Hello, World!"' in files_after["/test_app.py"], (
        f"Expected test assertion in /test_app.py, got: {files_after['/test_app.py']}"
    )

    # Turn 3: List the files
    ls_texts = await send_followup(
        runner, session, "Use ls to list files in /. How many files are there?"
    )
    ls_response = " ".join(ls_texts).lower()
    assert "app" in ls_response, f"Expected app.py in listing, got: {ls_response}"

    # Turn 4: Read a file the agent created earlier
    read_texts = await send_followup(runner, session, "Read /app.py and show me the content.")
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
    _, function_calls, function_responses, runner, session = await run_agent_with_events(
        agent,
        "Create a todo list with write_todos:\n"
        "1. 'Write README' (status: pending)\n"
        "2. 'Create config' (status: pending)\n",
    )
    assert "write_todos" in function_calls, f"Expected write_todos call, got: {function_calls}"
    assert "write_todos" in function_responses, (
        f"Expected write_todos response, got: {function_responses}"
    )
    updated = await runner.session_service.get_session(
        app_name="integration_test",
        user_id="test_user",
        session_id=session.id,
    )
    todos = updated.state.get("todos", [])
    assert len(todos) == 2, f"Expected 2 todos after initialization, got: {todos}"

    # Turn 2: Complete the first task
    _, function_calls2, function_responses2 = await send_followup_with_events(
        runner,
        session,
        "Now complete the first todo: use write_file to create /README.md "
        "with content '# My Project\\nA test project.'. "
        "Then update the todo list to mark 'Write README' as completed.",
    )
    assert "write_file" in function_calls2, f"Expected write_file call, got: {function_calls2}"
    assert "write_todos" in function_calls2, f"Expected write_todos call, got: {function_calls2}"
    assert "write_file" in function_responses2, (
        f"Expected write_file response, got: {function_responses2}"
    )
    assert "write_todos" in function_responses2, (
        f"Expected write_todos response, got: {function_responses2}"
    )
    files = await get_file_content(runner, session)
    assert "/README.md" in files, f"Expected /README.md in backend files, got: {list(files.keys())}"
    assert "# My Project" in files["/README.md"], (
        f"Expected README heading in /README.md, got: {files['/README.md']}"
    )
    updated = await runner.session_service.get_session(
        app_name="integration_test",
        user_id="test_user",
        session_id=session.id,
    )
    todos_after = updated.state.get("todos", [])
    write_readme = next(
        (todo for todo in todos_after if todo.get("content") == "Write README"), None
    )
    assert write_readme is not None, f"Expected 'Write README' todo to exist, got: {todos_after}"
    assert write_readme.get("status") == "completed", (
        f"Expected 'Write README' to be completed, got: {todos_after}"
    )

    # Turn 3: Verify both
    verify_texts, function_calls3, function_responses3 = await send_followup_with_events(
        runner,
        session,
        "Read the todos with read_todos and read /README.md. "
        "Tell me the status of each todo and what's in the README.",
    )
    assert "read_todos" in function_calls3, f"Expected read_todos call, got: {function_calls3}"
    assert "read_file" in function_calls3, f"Expected read_file call, got: {function_calls3}"
    assert "read_todos" in function_responses3, (
        f"Expected read_todos response, got: {function_responses3}"
    )
    assert "read_file" in function_responses3, (
        f"Expected read_file response, got: {function_responses3}"
    )
    verify_response = " ".join(verify_texts).lower()
    has_readme = "my project" in verify_response or "readme" in verify_response
    has_status = "completed" in verify_response or "done" in verify_response
    assert has_readme and has_status, (
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
    _texts, _function_calls, _function_responses, runner, session = await run_agent_with_events(
        agent,
        "My favorite programming language is Elixir and I work at a company "
        "called NovaTech. Remember this.",
    )

    # Turn 2: Ask about something unrelated
    await send_followup(runner, session, "What is 2 + 2?")

    # Turn 3: Ask about the earlier context
    recall_texts = await send_followup(
        runner,
        session,
        "What is my favorite programming language and where do I work?",
    )
    recall_response = " ".join(recall_texts).lower()
    has_elixir = "elixir" in recall_response
    has_company = "novatech" in recall_response
    assert has_elixir and has_company, (
        f"Expected Elixir and NovaTech from earlier context, got: {recall_response}"
    )
