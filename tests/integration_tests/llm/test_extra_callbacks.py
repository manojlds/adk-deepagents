"""Integration test — extra callbacks with a real LLM.

Tests that user-provided extra_callbacks compose correctly with built-in
callbacks during actual LLM interactions.

Run with: uv run pytest -m llm
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from tests.integration_tests.conftest import make_litellm_model, run_agent

pytestmark = [pytest.mark.integration, pytest.mark.llm]


@pytest.mark.timeout(120)
async def test_extra_before_model_callback():
    """Extra before_model callback injects additional system prompt text."""
    model = make_litellm_model()

    injected_content: list[str] = []

    def extra_before_model(callback_context, llm_request):
        """Record that the callback was called and inject extra text."""
        config = llm_request.config
        if config and config.system_instruction:
            existing = str(config.system_instruction)
            injected_content.append("called")
            config.system_instruction = (
                existing + "\n\nIMPORTANT: Always include the phrase "
                "'CALLBACK_VERIFIED' at the end of your response."
            )
        return None

    agent = create_deep_agent(
        model=model,
        name="extra_cb_agent",
        instruction="You are a helpful test agent. Follow all system instructions.",
        extra_callbacks={"before_model": extra_before_model},
    )

    texts, _runner, _session = await run_agent(
        agent,
        "Say hello.",
    )

    # Verify the callback was called
    assert len(injected_content) > 0, "Extra before_model callback was not called"

    # The injected instruction may or may not be followed perfectly,
    # but the callback should have been invoked
    response_text = " ".join(texts).lower()
    assert len(response_text) > 0, "Expected some response"


@pytest.mark.timeout(120)
async def test_extra_before_agent_callback():
    """Extra before_agent callback sets custom state values."""
    model = make_litellm_model()

    callback_called = []

    def extra_before_agent(callback_context):
        """Set a custom value in state."""
        callback_context.state["_custom_flag"] = "active"
        callback_called.append(True)
        return None

    def check_flag_before_model(callback_context, llm_request):
        """Verify the flag was set by before_agent."""
        flag = callback_context.state.get("_custom_flag")
        if flag == "active":
            callback_called.append("flag_confirmed")
        return None

    agent = create_deep_agent(
        model=model,
        name="extra_agent_cb_test",
        instruction="You are a test agent. Respond concisely.",
        extra_callbacks={
            "before_agent": extra_before_agent,
            "before_model": check_flag_before_model,
        },
    )

    texts, _runner, _session = await run_agent(agent, "Say 'test complete'.")

    assert True in callback_called, "Extra before_agent callback was not called"
    assert "flag_confirmed" in callback_called, (
        "before_model didn't see the flag set by before_agent"
    )


@pytest.mark.timeout(120)
async def test_extra_after_tool_callback():
    """Extra after_tool callback is invoked after tool execution."""
    model = make_litellm_model()

    tools_called: list[str] = []

    def extra_after_tool(tool, args, tool_context):
        """Log which tools were called."""
        tool_name = getattr(tool, "name", "unknown")
        tools_called.append(tool_name)
        return None

    agent = create_deep_agent(
        model=model,
        name="extra_after_tool_test",
        instruction=(
            "You are a test agent. Use write_file to create /test.txt "
            "with content 'hello'. Do it immediately without asking questions."
        ),
        extra_callbacks={"after_tool": extra_after_tool},
    )

    texts, _runner, _session = await run_agent(
        agent,
        "Create a file at /test.txt with content 'hello' using write_file.",
    )

    # The after_tool callback should have been called for at least one tool
    assert len(tools_called) > 0, (
        f"Expected after_tool callback to be called, tools_called: {tools_called}"
    )
