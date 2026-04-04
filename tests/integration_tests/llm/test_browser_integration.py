"""Integration tests — browser automation with real Playwright MCP + LLM.

Spins up the actual ``@playwright/mcp`` server via stdio, connects through
ADK's ``McpToolset``, and verifies the agent uses browser tools against
real web pages (example.com).

Requirements:
- Node.js >= 18 with npx
- Chromium installed (``npx playwright install chromium``)
- LLM API key (OPENAI_API_KEY or OPENCODE_API_KEY)

Run with: uv run pytest -m browser tests/integration_tests/llm/test_browser_integration.py
"""

from __future__ import annotations

import pytest

from tests.integration_tests.conftest import (
    get_file_content,
    run_agent_with_events,
)

pytestmark = [pytest.mark.integration, pytest.mark.llm, pytest.mark.browser]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(180)
async def test_browser_navigate_and_snapshot(browser_agent_factory):
    """Agent navigates to example.com and takes a snapshot to read the page."""
    agent = browser_agent_factory(
        "browser_nav_test",
        (
            "You are a browser automation agent. "
            "Always use browser_navigate first, then browser_snapshot to see the page."
        ),
    )

    texts, fn_calls, fn_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Navigate to https://example.com and tell me the page title.",
    )

    assert "browser_navigate" in fn_calls, f"Expected browser_navigate call, got: {fn_calls}"

    response_text = " ".join(texts).lower()
    assert "example" in response_text, f"Expected 'example' in response, got: {response_text}"


@pytest.mark.timeout(180)
async def test_browser_extract_data_to_file(browser_agent_factory):
    """Agent navigates, snapshots, and saves extracted data to a file."""
    agent = browser_agent_factory(
        "browser_extract_test",
        (
            "You are a browser automation agent with filesystem tools. "
            "When asked to extract data, navigate to the page, take a snapshot, "
            "extract the relevant data, and save it to a file using write_file."
        ),
    )

    texts, fn_calls, fn_responses, runner, session = await run_agent_with_events(
        agent,
        "Go to https://example.com, take a snapshot, and save the page title "
        "to /page_info.txt using write_file.",
    )

    assert "browser_navigate" in fn_calls, f"Expected browser_navigate, got: {fn_calls}"
    assert "write_file" in fn_calls, f"Expected write_file, got: {fn_calls}"

    files = await get_file_content(runner, session)
    assert "/page_info.txt" in files, f"Expected /page_info.txt, got: {list(files.keys())}"
    content = files["/page_info.txt"].lower()
    assert "example" in content, f"Expected 'example' in file, got: {files['/page_info.txt']}"


@pytest.mark.timeout(180)
async def test_browser_click_link(browser_agent_factory):
    """Agent navigates, snapshots, clicks a link, and reports the new page."""
    agent = browser_agent_factory(
        "browser_click_test",
        (
            "You are a browser automation agent. "
            "Follow the snapshot → ref → action workflow. "
            "After clicking a link, take another snapshot or report what happened."
        ),
    )

    texts, fn_calls, fn_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Go to https://example.com, take a snapshot, click the 'More information' link, "
        "and tell me what the new page says.",
    )

    assert "browser_navigate" in fn_calls, f"Expected browser_navigate, got: {fn_calls}"
    assert "browser_click" in fn_calls, f"Expected browser_click, got: {fn_calls}"


@pytest.mark.timeout(180)
async def test_browser_prompt_guides_workflow(browser_agent_factory):
    """The browser system prompt guides the agent to follow navigate→snapshot→interact."""
    agent = browser_agent_factory(
        "browser_prompt_test",
        # Minimal instruction — the browser prompt should guide behavior
        "You are a helpful assistant with browser capabilities.",
    )

    from adk_deepagents.browser.prompts import BROWSER_SYSTEM_PROMPT

    assert BROWSER_SYSTEM_PROMPT in agent.instruction

    texts, fn_calls, fn_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Open https://example.com and read the page content for me.",
    )

    assert "browser_navigate" in fn_calls, f"Expected browser_navigate, got: {fn_calls}"

    response_text = " ".join(texts).lower()
    assert "example" in response_text, f"Expected page content in response, got: {response_text}"
