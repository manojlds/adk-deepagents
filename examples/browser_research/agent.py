"""Browser research example — deep research with browser-based data extraction.

Demonstrates:
- Hybrid research: web search APIs + browser automation
- Browser sub-agent for JavaScript-heavy pages and interactive content
- Dynamic task delegation to specialist sub-agents
- Playwright MCP for browser tools

Usage:
    # Interactive runner
    python -m examples.browser_research.agent

    # Requires:
    # - GOOGLE_API_KEY (or LITELLM_MODEL + provider key)
    # - SERPER_API_KEY or TAVILY_API_KEY (for web search)
    # - Node.js >= 18 (for @playwright/mcp via npx)
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from google.genai import types

from adk_deepagents import (
    BrowserConfig,
    DynamicTaskConfig,
    SubAgentSpec,
    SummarizationConfig,
    create_deep_agent_async,
)

from ..deep_research.tools import think, web_search
from .prompts import (
    BROWSER_RESEARCH_INSTRUCTIONS,
    BROWSER_RESEARCHER_INSTRUCTIONS,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Sub-agent definitions
# ---------------------------------------------------------------------------

browser_researcher_subagent = SubAgentSpec(
    name="browser_researcher",
    description=(
        "Browser research specialist that navigates complex web pages, "
        "interacts with JavaScript-heavy sites, extracts data from "
        "interactive elements, and handles pages that regular search "
        "APIs cannot access."
    ),
    system_prompt=BROWSER_RESEARCHER_INSTRUCTIONS,
    tools=[think],
)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


async def build_agent_async(
    model: str = DEFAULT_MODEL,
    *,
    headless: bool = True,
):
    """Create the browser research agent with Playwright MCP tools resolved.

    Returns ``(agent, cleanup)`` where ``cleanup`` is an async function
    that must be called to close MCP connections.

    Parameters
    ----------
    model:
        Model string.
    headless:
        Run browser in headless mode (default True).
    """
    resolved_model = os.environ.get("LITELLM_MODEL", model)
    return await create_deep_agent_async(
        name="browser_research",
        model=resolved_model,
        instruction=BROWSER_RESEARCH_INSTRUCTIONS,
        tools=[web_search, think],
        subagents=[browser_researcher_subagent],
        browser=BrowserConfig(headless=headless),
        delegation_mode="dynamic",
        dynamic_task_config=DynamicTaskConfig(
            max_parallel=2,
            max_depth=2,
            timeout_seconds=180.0,
        ),
        summarization=SummarizationConfig(
            model=resolved_model,
            trigger=("fraction", 0.75),
            keep=("messages", 8),
        ),
    )


# ---------------------------------------------------------------------------
# Interactive runner
# ---------------------------------------------------------------------------


async def main():
    """Run the browser research agent interactively."""
    from google.adk.runners import InMemoryRunner

    model = os.environ.get("LITELLM_MODEL", DEFAULT_MODEL)
    agent, cleanup = await build_agent_async(model=model)

    try:
        runner = InMemoryRunner(agent=agent, app_name="browser_research")
        session = await runner.session_service.create_session(
            app_name="browser_research",
            user_id="user",
        )

        print(f"Browser Research Agent ready (model: {model}).")
        print("Combines web search + browser automation for deep research.")
        print("Type 'quit' to exit.\n")
        print(
            "Example: Research the pricing pages of the top 3 cloud providers "
            "and compare their free tiers\n"
        )

        while True:
            user_input = input("You: ").strip()
            if not user_input or user_input.lower() in ("quit", "exit"):
                break

            async for event in runner.run_async(
                session_id=session.id,
                user_id="user",
                new_message=types.Content(
                    role="user", parts=[types.Part(text=user_input)]
                ),
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            print(f"Agent: {part.text}")
    finally:
        if cleanup:
            await cleanup()

    print("\nGoodbye.")


if __name__ == "__main__":
    asyncio.run(main())
