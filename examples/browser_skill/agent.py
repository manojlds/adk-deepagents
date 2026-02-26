"""Browser skill example — agent-browser CLI via adk-skills integration.

Demonstrates:
- Agent Skills integration for browser automation
- agent-browser CLI as a skill (SKILL.md discovery)
- Shell execution to run agent-browser commands
- Progressive disclosure: agent learns browser commands on demand

This approach uses the agent-browser CLI tool through shell execution,
guided by an Agent Skill (SKILL.md) that teaches the agent the
snapshot → ref → action workflow.

Usage:
    # Requires agent-browser CLI (npm i -g agent-browser)
    python -m examples.browser_skill.agent

    # Or with ADK CLI:
    adk run examples/browser_skill/
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from google.genai import types

from adk_deepagents import create_deep_agent

from .prompts import BROWSER_SKILL_INSTRUCTIONS

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-2.5-flash"
SKILLS_DIR = os.path.join(os.path.dirname(__file__), "..", "skills")


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def build_agent(model: str = DEFAULT_MODEL):
    """Create the browser skill agent.

    Uses local shell execution + adk-skills for agent-browser CLI.

    Parameters
    ----------
    model:
        Model string.
    """
    resolved_model = os.environ.get("LITELLM_MODEL", model)
    return create_deep_agent(
        name="browser_skill",
        model=resolved_model,
        instruction=BROWSER_SKILL_INSTRUCTIONS,
        skills=[SKILLS_DIR],
        execution="local",
    )


# Default agent for ADK CLI (adk run examples/browser_skill/)
root_agent = build_agent()


# ---------------------------------------------------------------------------
# Interactive runner
# ---------------------------------------------------------------------------


async def main():
    """Run the browser skill agent interactively."""
    from google.adk.runners import InMemoryRunner

    model = os.environ.get("LITELLM_MODEL", DEFAULT_MODEL)
    agent = build_agent(model=model)

    runner = InMemoryRunner(agent=agent, app_name="browser_skill")
    session = await runner.session_service.create_session(
        app_name="browser_skill",
        user_id="user",
    )

    print(f"Browser Skill Agent ready (model: {model}).")
    print("Uses agent-browser CLI via skill activation. Type 'quit' to exit.\n")
    print("Example: Open https://example.com and tell me what's on the page\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit"):
            break

        async for event in runner.run_async(
            session_id=session.id,
            user_id="user",
            new_message=types.Content(role="user", parts=[types.Part(text=user_input)]),
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        print(f"Agent: {part.text}")

    print("\nGoodbye.")


if __name__ == "__main__":
    asyncio.run(main())
