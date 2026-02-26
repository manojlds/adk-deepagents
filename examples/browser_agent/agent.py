"""Browser agent example — autonomous web interaction via Playwright MCP.

Demonstrates:
- Browser automation via @playwright/mcp (Playwright MCP server)
- Accessibility tree snapshots for page understanding
- Form filling, clicking, data extraction
- Multi-step browser workflows
- Todo-based planning for complex tasks

Usage:
    # Interactive runner (requires @playwright/mcp via npx)
    python -m examples.browser_agent.agent

    # Or use with ADK CLI (sync mode — browser tools not available):
    adk run examples/browser_agent/
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from google.genai import types

from adk_deepagents import BrowserConfig, create_deep_agent, create_deep_agent_async

from .prompts import BROWSER_WORKFLOW_INSTRUCTIONS, DATA_EXTRACTION_INSTRUCTIONS

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Build prompt
# ---------------------------------------------------------------------------


def _build_prompt() -> str:
    """Build the browser agent system prompt from templates."""
    return "\n\n".join(
        [
            BROWSER_WORKFLOW_INSTRUCTIONS,
            DATA_EXTRACTION_INSTRUCTIONS,
        ]
    )


# ---------------------------------------------------------------------------
# Agent factory (async — for Playwright MCP)
# ---------------------------------------------------------------------------


async def build_agent_async(
    model: str = DEFAULT_MODEL,
    *,
    headless: bool = True,
    browser: str = "chromium",
):
    """Create the browser agent with Playwright MCP tools resolved.

    Returns ``(agent, cleanup)`` where ``cleanup`` is an async function
    that must be called to close the MCP connection and browser.

    Parameters
    ----------
    model:
        Model string. Supports any ADK-compatible model:
        - ``"gemini-2.5-flash"`` (default)
        - ``"gemini-2.5-pro"``
        - ``"openai/gpt-4o"`` (requires OPENAI_API_KEY + litellm)
    headless:
        Run browser in headless mode (default True).
    browser:
        Browser engine: ``"chromium"``, ``"firefox"``, or ``"webkit"``.
    """
    resolved_model = os.environ.get("LITELLM_MODEL", model)
    return await create_deep_agent_async(
        name="browser_agent",
        model=resolved_model,
        instruction=_build_prompt(),
        browser=BrowserConfig(
            headless=headless,
            browser=browser,
        ),
    )


# Default agent for ADK CLI (adk run examples/browser_agent/)
# Note: browser tools require async resolution, so the CLI agent
# won't have browser tools. Use the interactive runner instead.
root_agent = create_deep_agent(
    name="browser_agent",
    model=DEFAULT_MODEL,
    instruction=_build_prompt(),
)


# ---------------------------------------------------------------------------
# Interactive runner
# ---------------------------------------------------------------------------


async def main():
    """Run the browser agent interactively with Playwright MCP."""
    from google.adk.runners import InMemoryRunner

    model = os.environ.get("LITELLM_MODEL", DEFAULT_MODEL)
    agent, cleanup = await build_agent_async(model=model)

    try:
        runner = InMemoryRunner(agent=agent, app_name="browser_agent")
        session = await runner.session_service.create_session(
            app_name="browser_agent",
            user_id="user",
        )

        print(f"Browser Agent ready (model: {model}).")
        print("Navigate websites, fill forms, extract data. Type 'quit' to exit.\n")
        print("Example: Go to https://news.ycombinator.com and get the top 5 stories\n")

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
