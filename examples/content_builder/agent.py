"""Content builder example — skills, memory, and sub-agents.

Demonstrates:
- Agent Skills via adk-skills (blog-writing, social-media)
- Memory via AGENTS.md
- Sub-agent delegation (researcher)
- Local filesystem backend for output

Port of deepagents ``examples/content-builder-agent/``.

Usage:
    python examples/content_builder/agent.py
"""

from __future__ import annotations

import asyncio

from adk_deepagents import SubAgentSpec, create_deep_agent
from adk_deepagents.backends import FilesystemBackend

# Sub-agent: a researcher that can search and synthesize
researcher = SubAgentSpec(
    name="researcher",
    description=(
        "Research agent for gathering information on topics. "
        "Use this when you need to find facts, statistics, or "
        "background information before writing content."
    ),
    system_prompt=(
        "You are a research assistant. Gather relevant information "
        "on the given topic. Provide structured findings with sources "
        "and key points that can be used for content creation."
    ),
)

# Create the content builder agent
root_agent = create_deep_agent(
    name="content_builder",
    model="gemini-2.5-flash",
    instruction=(
        "You are a content creation assistant. You help users create "
        "blog posts, social media content, and other written materials. "
        "Use your skills for writing guidelines, delegate research to "
        "sub-agents, and write output files to the local filesystem."
    ),
    memory=["./AGENTS.md"],
    skills=["./skills/"],
    subagents=[researcher],
    backend=FilesystemBackend(root_dir=".", virtual_mode=True),
)


async def main():
    """Run the content builder interactively."""
    from google.adk.runners import InMemoryRunner

    runner = InMemoryRunner(agent=root_agent, app_name="content_builder")
    session = await runner.session_service.create_session(
        app_name="content_builder",
        user_id="user",
    )

    print("Content Builder ready. Type 'quit' to exit.\n")

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


if __name__ == "__main__":
    asyncio.run(main())
