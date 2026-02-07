"""Deep research example — parallel sub-agent delegation.

Demonstrates:
- Multiple specialized sub-agents
- Parallel delegation for research tasks
- Summarization for long conversations
- Research → synthesis workflow

Port of deepagents ``examples/deep_research/``.

Usage:
    python examples/deep_research/agent.py
"""

from __future__ import annotations

import asyncio

from adk_deepagents import SubAgentSpec, SummarizationConfig, create_deep_agent

# Specialized sub-agents for different research tasks
web_researcher = SubAgentSpec(
    name="web_researcher",
    description=(
        "Searches the web for current information on a topic. "
        "Use for finding recent data, news, and online sources."
    ),
    system_prompt=(
        "You are a web research specialist. Search for and synthesize "
        "information from online sources. Always note your sources and "
        "provide structured findings with key facts and data points."
    ),
)

analyst = SubAgentSpec(
    name="analyst",
    description=(
        "Analyzes data, identifies patterns, and draws conclusions. "
        "Use for interpreting research findings and creating insights."
    ),
    system_prompt=(
        "You are a data analyst. Take research findings and identify "
        "key patterns, trends, and insights. Present your analysis "
        "in a structured format with supporting evidence."
    ),
)

writer = SubAgentSpec(
    name="writer",
    description=(
        "Synthesizes research and analysis into polished reports. "
        "Use for producing the final written deliverable."
    ),
    system_prompt=(
        "You are a research writer. Take research findings and analysis "
        "and synthesize them into a well-structured, clear report. "
        "Use proper citations and maintain academic rigor."
    ),
)

# Create the deep research agent with summarization for long sessions
root_agent = create_deep_agent(
    name="deep_research",
    model="gemini-2.5-flash",
    instruction=(
        "You are a deep research coordinator. When given a research topic:\n"
        "1. Break it down into specific research questions\n"
        "2. Delegate research tasks to sub-agents in parallel\n"
        "3. Have the analyst identify patterns across findings\n"
        "4. Have the writer produce the final report\n"
        "5. Save the report to a file\n\n"
        "Use your todo list to track research progress."
    ),
    subagents=[web_researcher, analyst, writer],
    summarization=SummarizationConfig(
        trigger=("fraction", 0.75),
        keep=("messages", 8),
    ),
)


async def main():
    """Run the deep research agent interactively."""
    from google.adk.runners import InMemoryRunner

    runner = InMemoryRunner(agent=root_agent, app_name="deep_research")
    session = await runner.session_service.create_session(
        app_name="deep_research",
        user_id="user",
    )

    print("Deep Research Agent ready. Type 'quit' to exit.\n")

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
