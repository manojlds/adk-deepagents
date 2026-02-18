"""Integration test — filesystem round-trip with a real LLM.

Scenario: Agent creates a file via write_file, reads it back via read_file,
edits it via edit_file, then verifies the final content.

Run with: uv run pytest -m integration
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from tests.integration_tests.conftest import (
    get_file_content,
    make_litellm_model,
    run_agent,
    send_followup,
)

pytestmark = pytest.mark.integration


@pytest.mark.timeout(120)
async def test_filesystem_roundtrip():
    """Agent writes, reads, edits, and verifies a file end-to-end."""
    model = make_litellm_model()

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
    texts, runner, session = await run_agent(
        agent,
        'Use write_file to create a file at /test.txt with the content "Hello World". '
        "Confirm when done.",
    )

    # Verify file was actually created in backend
    files = await get_file_content(runner, session)
    assert "/test.txt" in files, f"Expected /test.txt in backend, got: {list(files.keys())}"
    assert files["/test.txt"] == "Hello World", (
        f"Expected 'Hello World' in file, got: {files['/test.txt']}"
    )

    # Step 2: Edit the file (same session)
    await send_followup(
        runner,
        session,
        'Use edit_file to replace "Hello World" with "Hello Universe" '
        "in /test.txt. Confirm when done.",
    )

    # Verify the edit in the backend
    files = await get_file_content(runner, session)
    assert files["/test.txt"] == "Hello Universe", (
        f"Expected 'Hello Universe' after edit, got: {files['/test.txt']}"
    )
