"""Integration tests — browser automation with a real LLM.

Tests that the agent correctly uses browser tools following the
snapshot → ref → action workflow. Uses mock browser tool functions
(simulating Playwright MCP tools) since the actual MCP server
isn't available in CI.

Run with: uv run pytest -m llm tests/integration_tests/llm/test_browser_integration.py
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from tests.integration_tests.conftest import (
    get_file_content,
    make_litellm_model,
    run_agent_with_events,
)

pytestmark = [pytest.mark.integration, pytest.mark.llm]

# ---------------------------------------------------------------------------
# Mock browser tools — simulate Playwright MCP tool responses
# ---------------------------------------------------------------------------

# Tracks navigation state so snapshot returns context-appropriate results.
_browser_state: dict[str, str] = {"url": "", "title": ""}

# Simple page database keyed by URL prefix.
_MOCK_PAGES: dict[str, dict] = {
    "https://example.com": {
        "title": "Example Domain",
        "snapshot": (
            "- document 'Example Domain'\n"
            "  - heading 'Example Domain' [ref=e1]\n"
            "  - paragraph 'This domain is for use in illustrative examples.' [ref=e2]\n"
            "  - link 'More information...' [ref=e3]"
        ),
    },
    "https://news.ycombinator.com": {
        "title": "Hacker News",
        "snapshot": (
            "- document 'Hacker News'\n"
            "  - heading 'Hacker News' [ref=e1]\n"
            "  - list 'stories'\n"
            "    - listitem '1. Show HN: A new database engine' [ref=e2]\n"
            "    - listitem '2. Understanding Large Language Models' [ref=e3]\n"
            "    - listitem '3. Rust for Python developers' [ref=e4]\n"
            "    - listitem '4. The future of web browsers' [ref=e5]\n"
            "    - listitem '5. Open source AI tools roundup' [ref=e6]\n"
            "  - link 'More' [ref=e7]"
        ),
    },
    "https://example.com/form": {
        "title": "Contact Form",
        "snapshot": (
            "- document 'Contact Form'\n"
            "  - heading 'Contact Us' [ref=e1]\n"
            "  - textbox 'Name' [ref=e2]\n"
            "  - textbox 'Email' [ref=e3]\n"
            "  - textbox 'Message' [ref=e4]\n"
            "  - button 'Submit' [ref=e5]"
        ),
    },
}


def _resolve_page(url: str) -> dict:
    """Find the best matching mock page for a URL."""
    for prefix, page in _MOCK_PAGES.items():
        if url.startswith(prefix):
            return page
    return {
        "title": "Page",
        "snapshot": (
            "- document 'Page'\n  - heading 'Page' [ref=e1]\n  - paragraph 'Page content' [ref=e2]"
        ),
    }


def browser_navigate(url: str) -> dict:
    """Navigate the browser to a URL.

    Args:
        url: The URL to navigate to.
    """
    page = _resolve_page(url)
    _browser_state["url"] = url
    _browser_state["title"] = page["title"]
    return {"status": "success", "url": url, "title": page["title"]}


def browser_snapshot() -> dict:
    """Take an accessibility tree snapshot of the current page.

    Returns the page's accessibility tree with element refs for interaction.
    """
    url = _browser_state.get("url", "")
    page = _resolve_page(url)
    return {
        "status": "success",
        "url": url,
        "title": page["title"],
        "snapshot": page["snapshot"],
    }


def browser_click(element: str, ref: str) -> dict:
    """Click an element on the page by its ref.

    Args:
        element: Description of the element to click.
        ref: The element ref from the snapshot (e.g., 'e3').
    """
    return {"status": "success", "action": "click", "element": element, "ref": ref}


def browser_type(element: str, ref: str, text: str) -> dict:
    """Type text into an element on the page.

    Args:
        element: Description of the element to type into.
        ref: The element ref from the snapshot (e.g., 'e2').
        text: The text to type.
    """
    return {"status": "success", "action": "type", "element": element, "ref": ref, "text": text}


def browser_close() -> dict:
    """Close the browser."""
    _browser_state["url"] = ""
    _browser_state["title"] = ""
    return {"status": "success", "action": "closed"}


MOCK_BROWSER_TOOLS = [
    browser_navigate,
    browser_snapshot,
    browser_click,
    browser_type,
    browser_close,
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_browser_state():
    """Reset mock browser state between tests."""
    _browser_state.clear()
    _browser_state.update({"url": "", "title": ""})


@pytest.mark.timeout(120)
async def test_browser_navigate_and_snapshot():
    """Agent navigates to a URL and takes a snapshot to read the page."""
    agent = create_deep_agent(
        model=make_litellm_model(),
        name="browser_nav_test",
        tools=MOCK_BROWSER_TOOLS,
        instruction=(
            "You are a browser automation agent. You have browser tools: "
            "browser_navigate, browser_snapshot, browser_click, browser_type, browser_close. "
            "Always navigate first, then take a snapshot to see the page content."
        ),
        browser="_resolved",
    )

    texts, fn_calls, fn_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Navigate to https://example.com and tell me what's on the page.",
    )

    assert "browser_navigate" in fn_calls, f"Expected browser_navigate call, got: {fn_calls}"
    assert "browser_snapshot" in fn_calls, f"Expected browser_snapshot call, got: {fn_calls}"
    assert "browser_navigate" in fn_responses
    assert "browser_snapshot" in fn_responses

    response_text = " ".join(texts).lower()
    assert "example" in response_text, f"Expected page content in response, got: {response_text}"


@pytest.mark.timeout(120)
async def test_browser_extract_data_to_file():
    """Agent extracts data from a page and saves it to a file."""
    agent = create_deep_agent(
        model=make_litellm_model(),
        name="browser_extract_test",
        tools=MOCK_BROWSER_TOOLS,
        instruction=(
            "You are a browser automation agent. You have browser tools "
            "(browser_navigate, browser_snapshot, browser_click, browser_type, browser_close) "
            "and filesystem tools (write_file, read_file). "
            "When asked to extract data, navigate to the page, take a snapshot, "
            "extract the data, and save it to a file."
        ),
        browser="_resolved",
    )

    texts, fn_calls, fn_responses, runner, session = await run_agent_with_events(
        agent,
        "Go to https://news.ycombinator.com, take a snapshot, and save the list "
        "of story titles you find to /stories.txt using write_file.",
    )

    assert "browser_navigate" in fn_calls, f"Expected browser_navigate, got: {fn_calls}"
    assert "browser_snapshot" in fn_calls, f"Expected browser_snapshot, got: {fn_calls}"
    assert "write_file" in fn_calls, f"Expected write_file, got: {fn_calls}"

    files = await get_file_content(runner, session)
    assert "/stories.txt" in files, f"Expected /stories.txt, got: {list(files.keys())}"
    content = files["/stories.txt"].lower()
    # Should contain at least some of the story titles from the mock snapshot
    has_stories = any(
        term in content for term in ["database", "large language", "rust", "browser", "open source"]
    )
    assert has_stories, f"Expected story titles in file, got: {files['/stories.txt']}"


@pytest.mark.timeout(120)
async def test_browser_form_interaction():
    """Agent navigates to a form, snapshots it, and fills fields."""
    agent = create_deep_agent(
        model=make_litellm_model(),
        name="browser_form_test",
        tools=MOCK_BROWSER_TOOLS,
        instruction=(
            "You are a browser automation agent. You have browser tools: "
            "browser_navigate, browser_snapshot, browser_click, browser_type, browser_close. "
            "Follow the snapshot → ref → action workflow: "
            "1) Navigate to the URL, 2) Take a snapshot, 3) Use refs to interact."
        ),
        browser="_resolved",
    )

    texts, fn_calls, fn_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Go to https://example.com/form, take a snapshot, then type 'John Doe' "
        "into the Name field and 'john@example.com' into the Email field.",
    )

    assert "browser_navigate" in fn_calls, f"Expected browser_navigate, got: {fn_calls}"
    assert "browser_snapshot" in fn_calls, f"Expected browser_snapshot, got: {fn_calls}"
    assert "browser_type" in fn_calls, f"Expected browser_type, got: {fn_calls}"

    # Should have called browser_type at least twice (Name + Email)
    type_count = fn_calls.count("browser_type")
    assert type_count >= 2, f"Expected ≥2 browser_type calls, got {type_count}: {fn_calls}"


@pytest.mark.timeout(120)
async def test_browser_prompt_enables_workflow():
    """The browser system prompt is injected and guides the agent's behavior.

    Verifies that when browser='_resolved' is set, the BROWSER_SYSTEM_PROMPT
    is included in the agent's instruction, causing the agent to follow the
    navigate → snapshot → interact pattern without explicit instruction.
    """
    agent = create_deep_agent(
        model=make_litellm_model(),
        name="browser_prompt_test",
        tools=MOCK_BROWSER_TOOLS,
        # Minimal instruction — the browser prompt should guide behavior
        instruction="You are a helpful assistant with browser capabilities.",
        browser="_resolved",
    )

    from adk_deepagents.browser.prompts import BROWSER_SYSTEM_PROMPT

    assert BROWSER_SYSTEM_PROMPT in agent.instruction

    texts, fn_calls, fn_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Open https://example.com and read the page content.",
    )

    assert "browser_navigate" in fn_calls, f"Expected browser_navigate, got: {fn_calls}"
    # The browser prompt instructs to snapshot before interacting
    assert "browser_snapshot" in fn_calls, f"Expected browser_snapshot, got: {fn_calls}"
