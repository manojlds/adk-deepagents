"""Sandboxed coder example — Heimdall MCP execution with skills.

Demonstrates:
- Sandboxed code execution via Heimdall MCP (Pyodide WebAssembly + just-bash)
- Agent Skills for code review guidelines
- Writing and testing code in a sandbox
- Cross-language workflows (Bash → Python)
- Package installation in the sandbox

Usage:
    # Requires Heimdall MCP server (npm i -g @heimdall-ai/heimdall)
    python -m examples.sandboxed_coder.agent

    # Or use with ADK CLI:
    adk run examples/sandboxed_coder/
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from adk_deepagents import create_deep_agent, create_deep_agent_async

from .prompts import (
    CODE_QUALITY_INSTRUCTIONS,
    CODING_WORKFLOW_INSTRUCTIONS,
    EXECUTION_INSTRUCTIONS,
    TESTING_INSTRUCTIONS,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-2.5-flash"
SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")


# ---------------------------------------------------------------------------
# Build prompt
# ---------------------------------------------------------------------------


def _build_prompt() -> str:
    """Build the sandboxed coder system prompt from templates."""
    return "\n\n".join(
        [
            CODING_WORKFLOW_INSTRUCTIONS,
            EXECUTION_INSTRUCTIONS,
            TESTING_INSTRUCTIONS,
            CODE_QUALITY_INSTRUCTIONS,
        ]
    )


# ---------------------------------------------------------------------------
# Agent factory (sync — for ADK CLI)
# ---------------------------------------------------------------------------


def build_agent(model: str = DEFAULT_MODEL):
    """Create the sandboxed coder agent (sync variant).

    Uses local subprocess execution. For sandboxed Heimdall execution,
    use ``build_agent_async()`` instead.

    Parameters
    ----------
    model:
        Model string. Supports any ADK-compatible model:
        - ``"gemini-2.5-flash"`` (default)
        - ``"gemini-2.5-pro"``
        - ``"openai/gpt-4o"`` (requires OPENAI_API_KEY + litellm)
        - ``"anthropic/claude-sonnet-4-20250514"`` (requires ANTHROPIC_API_KEY + litellm)
    """
    return create_deep_agent(
        name="sandboxed_coder",
        model=model,
        instruction=_build_prompt(),
        skills=[SKILLS_DIR],
        execution="local",
    )


# ---------------------------------------------------------------------------
# Agent factory (async — for Heimdall MCP)
# ---------------------------------------------------------------------------


async def build_agent_async(model: str = DEFAULT_MODEL):
    """Create the sandboxed coder agent with Heimdall MCP tools resolved.

    Returns ``(agent, cleanup)`` where ``cleanup`` is an async function
    that must be called to close the MCP connection.

    Parameters
    ----------
    model:
        Model string (same options as ``build_agent``).
    """
    return await create_deep_agent_async(
        name="sandboxed_coder",
        model=model,
        instruction=_build_prompt(),
        skills=[SKILLS_DIR],
        execution="heimdall",
    )


# Default agent for ADK CLI (adk run examples/sandboxed_coder/)
# Uses local execution. For sandboxed Heimdall MCP, use the async runner below.
root_agent = build_agent()


# ---------------------------------------------------------------------------
# Interactive runner
# ---------------------------------------------------------------------------


async def main():
    """Run the sandboxed coder interactively with Heimdall MCP."""
    from google.adk.runners import InMemoryRunner

    agent, cleanup = await build_agent_async()

    try:
        runner = InMemoryRunner(agent=agent, app_name="sandboxed_coder")
        session = await runner.session_service.create_session(
            app_name="sandboxed_coder",
            user_id="user",
        )

        print("Sandboxed Coder ready (Heimdall MCP).")
        print("Write code, test it, and iterate. Type 'quit' to exit.\n")

        while True:
            user_input = input("You: ").strip()
            if not user_input or user_input.lower() in ("quit", "exit"):
                break

            async for event in runner.run_async(
                session_id=session.id,
                user_id="user",
                new_message=user_input,
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
