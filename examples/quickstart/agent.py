"""Quickstart example — minimal working deep agent.

Usage:
    # Requires GOOGLE_API_KEY or Vertex AI configuration
    python examples/quickstart/agent.py

    # Or use with ADK CLI:
    adk run examples/quickstart/
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from google.genai import types

from adk_deepagents import create_deep_agent

load_dotenv()

# Create a deep agent with default settings:
# - Gemini 2.5 Flash model
# - Filesystem tools (ls, read, write, edit, glob, grep)
# - Todo tools (write_todos, read_todos)
# - StateBackend (in-memory file storage)
root_agent = create_deep_agent(
    name="quickstart_agent",
    instruction=(
        "You are a helpful coding assistant. Use your filesystem and todo tools "
        "to help the user organize and manage their work."
    ),
)


async def main():
    """Run the agent interactively."""
    from google.adk.runners import InMemoryRunner

    runner = InMemoryRunner(agent=root_agent, app_name="quickstart")
    session = await runner.session_service.create_session(
        app_name="quickstart",
        user_id="user",
    )

    print("Deep Agent ready. Type 'quit' to exit.\n")

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


if __name__ == "__main__":
    asyncio.run(main())
