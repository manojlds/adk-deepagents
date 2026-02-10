"""Deep research agent — parallel sub-agent research with web search.

Demonstrates:
- Multi-model support (Gemini, OpenAI, Anthropic, etc.)
- Specialized sub-agents with parallel delegation
- Web search with full page content extraction
- Strategic thinking/reflection tool
- Todo-based planning and tracking
- Conversation summarization for long sessions
- Citation consolidation across sub-agents
- Final report generation

Port of langchain deepagents ``examples/deep_research/``.

Usage:
    # Default (Gemini)
    python examples/deep_research/agent.py

    # With OpenAI (requires OPENAI_API_KEY, pip install litellm)
    python examples/deep_research/agent.py --model openai/gpt-4o

    # With Anthropic (requires ANTHROPIC_API_KEY, pip install litellm)
    python examples/deep_research/agent.py --model anthropic/claude-sonnet-4-20250514

    # With Tavily web search (optional, better results)
    export TAVILY_API_KEY=your-key
    python examples/deep_research/agent.py

    # Or use with ADK CLI:
    adk run examples/deep_research/
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

from dotenv import load_dotenv

from adk_deepagents import SubAgentSpec, SummarizationConfig, create_deep_agent

from .prompts import (
    RESEARCH_WORKFLOW_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)
from .tools import think, web_search

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CONCURRENT_RESEARCH_UNITS = 3
MAX_RESEARCHER_ITERATIONS = 3
DEFAULT_MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Build prompts with configuration
# ---------------------------------------------------------------------------


def _build_orchestrator_prompt() -> str:
    """Build the orchestrator system prompt from templates."""
    delegation = SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_research_units=MAX_CONCURRENT_RESEARCH_UNITS,
        max_researcher_iterations=MAX_RESEARCHER_ITERATIONS,
    )
    return RESEARCH_WORKFLOW_INSTRUCTIONS + "\n\n" + delegation


def _build_researcher_prompt() -> str:
    """Build the researcher sub-agent prompt from template."""
    current_date = datetime.now(UTC).strftime("%Y-%m-%d")
    return RESEARCHER_INSTRUCTIONS.format(date=current_date)


# ---------------------------------------------------------------------------
# Sub-agent definitions
# ---------------------------------------------------------------------------

research_subagent = SubAgentSpec(
    name="research_agent",
    description=(
        "Delegate a research task to this sub-agent. Give it ONE focused "
        "research topic at a time. It will search the web, analyze sources, "
        "and return structured findings with citations. For comparisons, "
        "launch multiple research agents in parallel (one per element)."
    ),
    system_prompt=_build_researcher_prompt(),
    tools=[web_search, think],
)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def build_agent(model: str = DEFAULT_MODEL):
    """Create the deep research agent.

    Parameters
    ----------
    model:
        Model string. Supports any ADK-compatible model:
        - ``"gemini-2.5-flash"`` (default, requires GOOGLE_API_KEY)
        - ``"gemini-2.5-pro"`` (requires GOOGLE_API_KEY)
        - ``"openai/gpt-4o"`` (requires OPENAI_API_KEY + litellm)
        - ``"anthropic/claude-sonnet-4-20250514"`` (requires ANTHROPIC_API_KEY + litellm)
        - ``"groq/llama3-70b-8192"`` (requires GROQ_API_KEY + litellm)
    """
    return create_deep_agent(
        name="deep_research",
        model=model,
        instruction=_build_orchestrator_prompt(),
        tools=[web_search, think],
        subagents=[research_subagent],
        summarization=SummarizationConfig(
            model=model if model.startswith("gemini") else "gemini-2.5-flash",
            trigger=("fraction", 0.75),
            keep=("messages", 8),
        ),
    )


# Default agent for ADK CLI (adk run examples/deep_research/)
root_agent = build_agent()


# ---------------------------------------------------------------------------
# Interactive runner
# ---------------------------------------------------------------------------


async def main():
    """Run the deep research agent interactively."""
    parser = argparse.ArgumentParser(description="Deep Research Agent")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Model to use. Examples: gemini-2.5-flash, gemini-2.5-pro, "
            "openai/gpt-4o, anthropic/claude-sonnet-4-20250514"
        ),
    )
    args = parser.parse_args()

    agent = build_agent(args.model)

    from google.adk.runners import InMemoryRunner

    runner = InMemoryRunner(agent=agent, app_name="deep_research")
    session = await runner.session_service.create_session(
        app_name="deep_research",
        user_id="user",
    )

    print(f"Deep Research Agent ready (model: {args.model}).")
    print("Type 'quit' to exit.\n")

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

    print("\nGoodbye.")


if __name__ == "__main__":
    asyncio.run(main())
