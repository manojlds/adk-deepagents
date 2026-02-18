"""Integration test — filesystem round-trip with a real LLM.

Scenario: Agent creates a file via write_file, reads it back via read_file,
edits it via edit_file, then verifies the final content.

Requires OPENCODE_API_KEY environment variable to be set.
Run with: uv run pytest -m integration
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent

from .conftest import make_litellm_model, run_agent, send_followup

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

    response_text = " ".join(texts).lower()
    assert any(
        word in response_text for word in ("done", "created", "written", "success", "file")
    ), f"Expected confirmation of file creation, got: {response_text}"

    # Step 2: Read it back (same session)
    read_texts = await send_followup(
        runner, session, "Now use read_file to read /test.txt. Show me the content."
    )
    read_response = " ".join(read_texts)
    assert "Hello World" in read_response, (
        f"Expected file content in response, got: {read_response}"
    )

    # Step 3: Edit the file
    edit_texts = await send_followup(
        runner,
        session,
        'Use edit_file to replace "Hello World" with "Hello Universe" '
        "in /test.txt. Confirm when done.",
    )
    edit_response = " ".join(edit_texts).lower()
    assert any(
        word in edit_response for word in ("done", "edited", "replaced", "success", "updated")
    ), f"Expected confirmation of edit, got: {edit_response}"

    # Step 4: Read again to verify the edit
    verify_texts = await send_followup(
        runner, session, "Read /test.txt again and show me the current content."
    )
    verify_response = " ".join(verify_texts)
    assert "Hello Universe" in verify_response, (
        f"Expected edited content in response, got: {verify_response}"
    )
