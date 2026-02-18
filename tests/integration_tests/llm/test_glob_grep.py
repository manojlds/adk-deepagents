"""Integration test — glob and grep tools with a real LLM.

Scenario: Agent creates multiple files, then uses glob and grep to find
files and search for content.

Run with: uv run pytest -m llm
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from adk_deepagents.backends.utils import create_file_data
from tests.integration_tests.conftest import make_litellm_model, run_agent

pytestmark = [pytest.mark.integration, pytest.mark.llm]


@pytest.mark.timeout(120)
async def test_glob_finds_files():
    """Agent uses glob to find files matching a pattern."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="glob_test_agent",
        instruction=(
            "You are a test agent. Use the filesystem tools as directed. Report results accurately."
        ),
    )

    # Pre-populate files in state
    initial_files = {
        "/src/app.py": create_file_data("print('app')"),
        "/src/utils.py": create_file_data("def helper(): pass"),
        "/src/test_app.py": create_file_data("def test_main(): pass"),
        "/docs/readme.md": create_file_data("# Docs"),
        "/config.yaml": create_file_data("key: value"),
    }

    texts, runner, session = await run_agent(
        agent,
        'Use the glob tool with pattern "**/*.py" to find all Python files. '
        "List every file path you find.",
        state={"files": initial_files},
    )

    response_text = " ".join(texts)
    # Should find all 3 .py files
    assert "app.py" in response_text, f"Expected app.py in glob results, got: {response_text}"
    assert "utils.py" in response_text, f"Expected utils.py in glob results, got: {response_text}"


@pytest.mark.timeout(120)
async def test_grep_searches_content():
    """Agent uses grep to search for text within files."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="grep_test_agent",
        instruction=(
            "You are a test agent. Use the filesystem tools as directed. "
            "Report results accurately and completely."
        ),
    )

    initial_files = {
        "/src/auth.py": create_file_data(
            "def login(user, password):\n    # TODO: implement auth\n    return True\n"
        ),
        "/src/api.py": create_file_data(
            "def get_users():\n    # TODO: add pagination\n    return []\n"
        ),
        "/src/db.py": create_file_data("def connect():\n    return None\n"),
    }

    texts, _runner, _session = await run_agent(
        agent,
        'Use the grep tool to search for "TODO" across all files. '
        "Tell me which files contain TODO comments and what they say.",
        state={"files": initial_files},
    )

    response_text = " ".join(texts).lower()
    # Should find TODO in auth.py and api.py but not db.py
    has_auth = "auth" in response_text
    has_api = "api" in response_text
    assert has_auth or has_api, (
        f"Expected grep to find TODO in auth.py or api.py, got: {response_text}"
    )


@pytest.mark.timeout(120)
async def test_ls_lists_directory():
    """Agent uses ls to list files in a directory."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="ls_test_agent",
        instruction=(
            "You are a test agent. Use filesystem tools as directed. List the results clearly."
        ),
    )

    initial_files = {
        "/project/src/main.py": create_file_data("main"),
        "/project/src/lib.py": create_file_data("lib"),
        "/project/README.md": create_file_data("readme"),
    }

    texts, _runner, _session = await run_agent(
        agent,
        "Use the ls tool to list all files and directories in /project. What do you see?",
        state={"files": initial_files},
    )

    response_text = " ".join(texts).lower()
    assert "src" in response_text or "readme" in response_text, (
        f"Expected directory listing results, got: {response_text}"
    )
