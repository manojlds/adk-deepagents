"""Integration test — FilesystemBackend with a real LLM.

Tests that the agent can read and write real files on disk via FilesystemBackend.

Requires OPENCODE_API_KEY environment variable to be set.
Run with: uv run pytest -m integration
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from adk_deepagents import create_deep_agent
from adk_deepagents.backends import FilesystemBackend

from .conftest import make_litellm_model, run_agent, send_followup

pytestmark = pytest.mark.integration


def _fs_backend_factory(tmp_path: Path):
    """Create a backend factory that returns a FilesystemBackend for the given tmp_path."""

    def factory(state: dict[str, Any]) -> FilesystemBackend:
        return FilesystemBackend(root_dir=tmp_path, virtual_mode=True)

    return factory


@pytest.mark.timeout(120)
async def test_filesystem_backend_write_and_read(tmp_path):
    """Agent writes a file to disk and reads it back via FilesystemBackend."""
    model = make_litellm_model()

    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)

    agent = create_deep_agent(
        model=model,
        name="fs_backend_agent",
        backend=backend,
        instruction=(
            "You are a test agent. Use filesystem tools to create and read files. "
            "Follow instructions exactly."
        ),
    )

    texts, runner, session = await run_agent(
        agent,
        'Use write_file to create /hello.txt with content "Written to disk!". Confirm when done.',
    )

    response = " ".join(texts).lower()
    assert any(w in response for w in ("done", "created", "written", "success")), (
        f"Expected confirmation, got: {response}"
    )

    # Verify the file actually exists on disk
    expected_path = tmp_path / "hello.txt"
    assert expected_path.exists(), f"Expected {expected_path} to exist on disk"
    assert "Written to disk!" in expected_path.read_text()

    # Read it back via the agent
    read_texts = await send_followup(
        runner, session, "Read the file /hello.txt and show me the content."
    )
    read_response = " ".join(read_texts)
    assert "Written to disk" in read_response, (
        f"Expected file content in response, got: {read_response}"
    )


@pytest.mark.timeout(120)
async def test_filesystem_backend_reads_existing_files(tmp_path):
    """Agent reads files that already exist on disk."""
    model = make_litellm_model()

    # Create a file on disk before the agent runs
    (tmp_path / "existing.txt").write_text("This file was here before the agent.\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello from main')\n")

    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)

    agent = create_deep_agent(
        model=model,
        name="fs_existing_agent",
        backend=backend,
        instruction=(
            "You are a test agent. Use filesystem tools as directed. Report results accurately."
        ),
    )

    texts, _runner, _session = await run_agent(
        agent,
        "Use ls to list files in /. Then read /existing.txt. "
        "Tell me what files exist and what's in existing.txt.",
    )

    response_text = " ".join(texts).lower()
    has_listing = "existing" in response_text or "src" in response_text
    has_content = "before the agent" in response_text
    assert has_listing or has_content, f"Expected listing and/or file content, got: {response_text}"


@pytest.mark.timeout(120)
async def test_filesystem_backend_glob_and_grep(tmp_path):
    """Agent uses glob and grep on real filesystem files."""
    model = make_litellm_model()

    # Create some files
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text("# TODO: implement login\ndef login(): pass\n")
    (tmp_path / "src" / "api.py").write_text("# TODO: add rate limiting\ndef get(): pass\n")
    (tmp_path / "src" / "models.py").write_text("class User: pass\n")

    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)

    agent = create_deep_agent(
        model=model,
        name="fs_search_agent",
        backend=backend,
        instruction=(
            "You are a test agent. Use glob and grep tools as directed. Report all findings."
        ),
    )

    texts, _runner, _session = await run_agent(
        agent,
        'First use glob with pattern "**/*.py" to find all Python files. '
        'Then use grep to search for "TODO" in all files. '
        "Report which files have TODOs.",
    )

    response_text = " ".join(texts).lower()
    has_auth = "auth" in response_text
    has_api = "api" in response_text
    assert has_auth or has_api, (
        f"Expected auth.py or api.py in grep/glob results, got: {response_text}"
    )
