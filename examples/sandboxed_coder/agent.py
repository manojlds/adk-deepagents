"""Sandboxed coder example — Heimdall MCP execution with skills.

Demonstrates:
- Sandboxed code execution via Heimdall MCP
- Agent Skills for code review guidelines
- Writing and testing code in a sandbox
- Cross-language workflows (Bash → Python)

Usage:
    # Requires Heimdall MCP server (npx @heimdall-ai/heimdall)
    python examples/sandboxed_coder/agent.py
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from adk_deepagents import create_deep_agent_async

load_dotenv()


# Create the sandboxed coder agent (async for Heimdall MCP)
async def create_agent():
    """Create the agent with Heimdall execution backend."""
    agent, cleanup = await create_deep_agent_async(
        name="sandboxed_coder",
        model="gemini-2.5-flash",
        instruction=(
            "You are a coding assistant with sandboxed execution. "
            "Write code, test it in the sandbox, and iterate until "
            "it works correctly.\n\n"
            "Available execution tools:\n"
            "- execute_python: Run Python code in a WebAssembly sandbox\n"
            "- execute_bash: Run Bash commands in a simulated shell\n"
            "- install_packages: Install Python packages (numpy, pandas, etc.)\n\n"
            "Workflow:\n"
            "1. Write code to a file in the workspace\n"
            "2. Execute and verify the output\n"
            "3. Fix any issues and re-run\n"
            "4. Activate code-review skill for quality checks"
        ),
        skills=["./skills/"],
        execution="heimdall",
    )
    return agent, cleanup


async def main():
    """Run the sandboxed coder interactively."""
    from google.adk.runners import InMemoryRunner

    agent, cleanup = await create_agent()

    try:
        runner = InMemoryRunner(agent=agent, app_name="sandboxed_coder")
        session = await runner.session_service.create_session(
            app_name="sandboxed_coder",
            user_id="user",
        )

        print("Sandboxed Coder ready. Type 'quit' to exit.\n")

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


if __name__ == "__main__":
    asyncio.run(main())
