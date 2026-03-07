"""Deep research agent — dynamic task delegation with web search.

Demonstrates:
- Dynamic task delegation with runtime-defined specialist sub-agents
- Runtime sub-agent registration via ``register_subagent``
- Search provider routing (Serper-first auto mode)
- Strategic thinking/reflection tool
- Todo-based planning and tracking
- Conversation summarization for long sessions
- Final report generation with citations

Port of langchain deepagents ``examples/deep_research/``.

Usage:
    # Interactive runner (model from LITELLM_MODEL or default)
    python examples/deep_research/agent.py

    # With Serper search (recommended)
    export SERPER_API_KEY=your-key
    python examples/deep_research/agent.py

    # Or use with ADK CLI:
    adk run examples/deep_research/
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from typing import TypedDict

from dotenv import load_dotenv
from google.genai import types

from adk_deepagents import (
    DynamicTaskConfig,
    SummarizationConfig,
    create_deep_agent,
)

from .prompts import (
    GRADER_INSTRUCTIONS,
    PLANNER_INSTRUCTIONS,
    REPORTER_INSTRUCTIONS,
    RESEARCH_WORKFLOW_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    RUNTIME_SUBAGENT_REGISTRATION_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)
from .tools import think, web_search

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CONCURRENT_RESEARCH_UNITS = 4
MAX_RESEARCHER_ITERATIONS = 3
DEFAULT_MODEL = "openai/gpt-4o-mini"


class RuntimeSubagentProfile(TypedDict):
    """Runtime profile consumed by ``register_subagent``."""

    name: str
    description: str
    system_prompt: str
    tool_names: list[str]


def build_runtime_subagent_profiles() -> list[RuntimeSubagentProfile]:
    """Return runtime specialist profiles for deep research delegation."""
    return [
        RuntimeSubagentProfile(
            name="planner",
            description=(
                "Planning specialist that breaks large research requests into concise, "
                "high-impact tasks."
            ),
            system_prompt=PLANNER_INSTRUCTIONS,
            tool_names=["think"],
        ),
        RuntimeSubagentProfile(
            name="researcher",
            description=(
                "Research specialist that gathers evidence from web search and returns "
                "structured findings with citations."
            ),
            system_prompt=_build_researcher_prompt(),
            tool_names=["web_search", "think"],
        ),
        RuntimeSubagentProfile(
            name="reporter",
            description=(
                "Reporting specialist that synthesizes findings into a polished report "
                "with citations."
            ),
            system_prompt=REPORTER_INSTRUCTIONS,
            tool_names=["write_file", "think"],
        ),
        RuntimeSubagentProfile(
            name="grader",
            description="Quality specialist that grades report completeness and citation quality.",
            system_prompt=GRADER_INSTRUCTIONS,
            tool_names=["read_file", "think"],
        ),
    ]


# ---------------------------------------------------------------------------
# Build prompts with configuration
# ---------------------------------------------------------------------------


def _build_orchestrator_prompt() -> str:
    """Build the orchestrator system prompt from templates."""
    registration_payload = json.dumps(build_runtime_subagent_profiles(), indent=2)
    registration = RUNTIME_SUBAGENT_REGISTRATION_INSTRUCTIONS.format(
        registration_payload=registration_payload,
    )
    delegation = SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_research_units=MAX_CONCURRENT_RESEARCH_UNITS,
        max_researcher_iterations=MAX_RESEARCHER_ITERATIONS,
    )
    return RESEARCH_WORKFLOW_INSTRUCTIONS + "\n\n" + registration + "\n\n" + delegation


def _build_researcher_prompt() -> str:
    """Build the researcher sub-agent prompt from template."""
    current_date = datetime.now(UTC).strftime("%Y-%m-%d")
    return RESEARCHER_INSTRUCTIONS.format(date=current_date)


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
    resolved_model = (
        os.environ.get("LITELLM_MODEL") or os.environ.get("ADK_DEEPAGENTS_MODEL") or model
    )
    return create_deep_agent(
        name="deep_research",
        model=resolved_model,
        instruction=_build_orchestrator_prompt(),
        tools=[web_search, think],
        delegation_mode="dynamic",
        dynamic_task_config=DynamicTaskConfig(
            max_parallel=MAX_CONCURRENT_RESEARCH_UNITS,
            max_depth=2,
            timeout_seconds=240.0,
            allow_model_override=False,
        ),
        summarization=SummarizationConfig(
            model=resolved_model,
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
    agent = build_agent()

    from google.adk.runners import InMemoryRunner

    runner = InMemoryRunner(agent=agent, app_name="deep_research")
    session = await runner.session_service.create_session(
        app_name="deep_research",
        user_id="user",
    )

    active_model = (
        os.environ.get("LITELLM_MODEL") or os.environ.get("ADK_DEEPAGENTS_MODEL") or DEFAULT_MODEL
    )
    print(f"Deep Research Agent ready (model: {active_model}).")
    print("Type 'quit' to exit.\n")

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
