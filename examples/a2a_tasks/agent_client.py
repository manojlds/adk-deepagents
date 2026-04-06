"""A2A client-orchestrator example using dynamic task delegation.

This agent exposes the dynamic `task` tool and dispatches delegated turns to
an external A2A worker endpoint.

Usage:
    export A2A_AGENT_URL=http://127.0.0.1:8000
    uv run python -m examples.a2a_tasks.agent_client
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from google.genai import types

from adk_deepagents import A2ATaskConfig, DeepAgentConfig, DynamicTaskConfig, create_deep_agent

load_dotenv()

DEFAULT_MODEL = "gemini-2.5-flash"


def build_agent():
    """Build an orchestrator agent that delegates via A2A-backed task()."""
    model = os.environ.get("LITELLM_MODEL", DEFAULT_MODEL)
    agent_url = os.environ.get("A2A_AGENT_URL", "http://127.0.0.1:8000")

    return create_deep_agent(
        name="a2a_orchestrator",
        model=model,
        instruction=(
            "Always use the task tool for user requests. "
            "Reuse task_id when the user asks to continue earlier delegated work."
        ),
        config=DeepAgentConfig(
            delegation_mode="dynamic",
            dynamic_task_config=DynamicTaskConfig(
                a2a=A2ATaskConfig(agent_url=agent_url, timeout_seconds=120.0),
                timeout_seconds=120.0,
            ),
        ),
    )


root_agent = build_agent()


async def main() -> None:
    """Run the orchestrator interactively."""
    from google.adk.runners import InMemoryRunner

    agent = build_agent()
    runner = InMemoryRunner(agent=agent, app_name="a2a_tasks_client")
    session = await runner.session_service.create_session(
        app_name="a2a_tasks_client",
        user_id="user",
    )

    print("A2A task client ready.")
    print("Type 'quit' to exit.\n")
    print("Try:")
    print("- Use task to write /notes.txt with 'hello from a2a'.")
    print("- Then call task again with task_id task_1 and read /notes.txt.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in {"quit", "exit"}:
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
