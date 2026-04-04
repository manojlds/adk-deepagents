"""Integration tests — browser skill with Playwright MCP + LLM.

Validates that the agent-browser skill integration works correctly with
Playwright MCP tools, including prompt injection, navigation, multi-step
workflows, and data extraction.

Run with: uv run pytest -m browser tests/integration_tests/llm/test_browser_skill.py
"""

from __future__ import annotations

import pytest

from adk_deepagents.browser.prompts import BROWSER_SYSTEM_PROMPT
from tests.integration_tests.conftest import get_file_content, run_agent_with_events

pytestmark = [pytest.mark.integration, pytest.mark.llm, pytest.mark.browser]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(180)
async def test_browser_skill_prompt_present(browser_agent_factory):
    """Browser system prompt is injected into the agent's instruction."""
    agent = browser_agent_factory(
        "browser_skill_prompt_test",
        "You are a helpful assistant with browser capabilities.",
    )

    assert BROWSER_SYSTEM_PROMPT in agent.instruction


@pytest.mark.timeout(180)
async def test_browser_skill_navigate_and_extract(browser_agent_factory):
    """Agent navigates to a URL, snapshots, and extracts information."""
    agent = browser_agent_factory(
        "browser_skill_nav_extract_test",
        (
            "You are a browser automation agent. "
            "Navigate to the requested URL, snapshot the page, "
            "and extract the requested information."
        ),
    )

    texts, fn_calls, fn_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Navigate to https://example.com and tell me the main heading text.",
    )

    assert "browser_navigate" in fn_calls, f"Expected browser_navigate call, got: {fn_calls}"
    assert "browser_snapshot" in fn_calls, f"Expected browser_snapshot call, got: {fn_calls}"

    response_text = " ".join(texts).lower()
    assert "example" in response_text, f"Expected 'example' in response, got: {response_text}"


@pytest.mark.timeout(180)
async def test_browser_skill_multi_step_workflow(browser_agent_factory):
    """Agent follows the snapshot-ref-action workflow across multiple steps."""
    agent = browser_agent_factory(
        "browser_skill_multi_step_test",
        (
            "You are a browser automation agent. "
            "Follow the snapshot → ref → action workflow. "
            "After every page change, take a fresh snapshot to get new refs."
        ),
    )

    texts, fn_calls, fn_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Go to https://example.com, take a snapshot, then click the 'More information' link "
        "and tell me what you see.",
    )

    assert "browser_navigate" in fn_calls, f"Expected browser_navigate, got: {fn_calls}"
    assert "browser_click" in fn_calls, f"Expected browser_click, got: {fn_calls}"

    snapshot_count = fn_calls.count("browser_snapshot")
    assert snapshot_count >= 2, (
        f"Expected at least 2 snapshots (before and after click), got {snapshot_count}"
    )


@pytest.mark.timeout(180)
async def test_browser_skill_save_extracted_data(browser_agent_factory):
    """Agent extracts page data and saves it to a file via write_file."""
    agent = browser_agent_factory(
        "browser_skill_save_data_test",
        (
            "You are a browser automation agent with filesystem tools. "
            "When asked to extract data, navigate to the page, take a snapshot, "
            "extract the relevant data, and save it to a file using write_file."
        ),
    )

    texts, fn_calls, fn_responses, runner, session = await run_agent_with_events(
        agent,
        "Navigate to https://example.com, extract the page title and all paragraph text, "
        "and save it to /extracted.txt using write_file.",
    )

    assert "browser_navigate" in fn_calls, f"Expected browser_navigate, got: {fn_calls}"
    assert "write_file" in fn_calls, f"Expected write_file, got: {fn_calls}"

    files = await get_file_content(runner, session)
    assert "/extracted.txt" in files, f"Expected /extracted.txt, got: {list(files.keys())}"
    content = files["/extracted.txt"].lower()
    assert "example" in content, f"Expected 'example' in file, got: {files['/extracted.txt']}"
