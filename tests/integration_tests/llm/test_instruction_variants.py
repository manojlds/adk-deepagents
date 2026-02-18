"""Integration test — instruction and configuration variants with a real LLM.

Tests different create_deep_agent configurations to verify correct behavior.

Run with: uv run pytest -m llm
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from adk_deepagents.backends.utils import create_file_data
from tests.integration_tests.conftest import make_litellm_model, run_agent

pytestmark = [pytest.mark.integration, pytest.mark.llm]


@pytest.mark.timeout(120)
async def test_custom_instruction():
    """Agent follows a custom system instruction."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="custom_instr_agent",
        instruction=(
            "You are a pirate assistant. You MUST speak like a pirate in every "
            "response. Use 'Arr!', 'matey', 'ye', 'aye', and other pirate speech. "
            "Never break character."
        ),
    )

    texts, _runner, _session = await run_agent(
        agent,
        "Tell me about programming.",
    )

    response_text = " ".join(texts).lower()
    pirate_words = ("arr", "matey", "ye", "aye", "captain", "ship", "sail", "treasure")
    has_pirate = any(word in response_text for word in pirate_words)
    assert has_pirate, f"Expected pirate speech in response, got: {response_text}"


@pytest.mark.timeout(120)
async def test_no_instruction_uses_default():
    """Agent with no custom instruction uses the default base prompt."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="default_instr_agent",
    )

    texts, _runner, _session = await run_agent(
        agent,
        "What tools do you have available?",
    )

    response_text = " ".join(texts).lower()
    # Agent should know about its tools from the base prompt
    has_tool_ref = any(
        word in response_text for word in ("tool", "file", "read", "write", "todo", "grep", "glob")
    )
    assert has_tool_ref, f"Expected agent to reference its tools, got: {response_text}"


@pytest.mark.timeout(120)
async def test_agent_with_prepopulated_files():
    """Agent works with pre-populated files in state."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="prepopulated_agent",
        instruction=(
            "You are a code review assistant. Use filesystem tools to read "
            "files and provide feedback."
        ),
    )

    initial_files = {
        "/src/calculator.py": create_file_data(
            "def add(a, b):\n"
            "    return a + b\n\n"
            "def divide(a, b):\n"
            "    return a / b  # BUG: no zero check\n"
        ),
    }

    texts, _runner, _session = await run_agent(
        agent,
        "Read /src/calculator.py and tell me if there are any bugs.",
        state={"files": initial_files},
    )

    response_text = " ".join(texts).lower()
    has_bug_ref = any(
        word in response_text for word in ("zero", "division", "bug", "error", "check", "divide")
    )
    assert has_bug_ref, f"Expected agent to find the division bug, got: {response_text}"
