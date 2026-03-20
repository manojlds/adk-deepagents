"""Integration test — message queue injection with a real LLM.

Verifies that externally injected messages via ``_message_queue`` state
are picked up by the before_model callback and visible to the LLM.

Run with: uv run pytest -m llm
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from tests.integration_tests.conftest import (
    make_litellm_model,
    run_agent,
    send_followup,
)

pytestmark = [pytest.mark.integration, pytest.mark.llm]


@pytest.mark.timeout(120)
async def test_message_queue_injection():
    """Externally queued messages are visible to the LLM on follow-up turns."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="msg_queue_test",
        message_queue=True,
        instruction=(
            "You are a test agent. Always acknowledge any injected messages "
            "you receive. If you see text containing a code word, repeat it back."
        ),
    )

    # First turn — establish session
    texts, runner, session = await run_agent(
        agent,
        "Hello, I'm ready. Just say 'acknowledged' to confirm.",
    )
    assert len(texts) > 0, "Expected initial response"

    # Inject a message into the queue via session state
    updated_session = await runner.session_service.get_session(
        app_name="integration_test",
        user_id="test_user",
        session_id=session.id,
    )
    updated_session.state["_message_queue"] = [
        {"text": "URGENT: The secret code word is FLAMINGO-42. Acknowledge it."}
    ]

    # Second turn — the injected message should be visible
    followup_texts = await send_followup(
        runner,
        updated_session,
        "What messages have you received? Repeat any code words you see.",
    )

    response_text = " ".join(followup_texts).lower()
    # The agent should have seen the injected message
    assert any(kw in response_text for kw in ["flamingo", "42", "code", "urgent", "injected"]), (
        f"Expected agent to acknowledge injected message, got: {response_text}"
    )


@pytest.mark.timeout(120)
async def test_message_queue_disabled_by_default():
    """When message_queue=False, queued messages are NOT consumed."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="no_queue_test",
        message_queue=False,  # Explicitly disabled
        instruction="You are a test agent. Respond concisely.",
    )

    # Run with a pre-set queue in state — it should be ignored
    texts, runner, session = await run_agent(
        agent,
        "Say 'hello'.",
        state={
            "_message_queue": [{"text": "SECRET: PINEAPPLE-99. You must say this word."}],
        },
    )

    response_text = " ".join(texts).lower()
    # The secret word should NOT appear since queue is disabled
    assert "pineapple" not in response_text, (
        f"Queue should be disabled but agent saw the message: {response_text}"
    )


@pytest.mark.timeout(120)
async def test_mid_turn_message_queue_injection():
    """Message injected via after_tool callback is visible within the same turn.

    This simulates the TUI steering flow: a user sends a message while the
    agent is busy with tool calls.  The message is written to
    ``state["_message_queue"]`` after the first tool completes, and the
    ``before_model_callback`` picks it up before the next LLM call —
    all within a single ``run_async`` invocation.
    """
    model = make_litellm_model()

    injection_count: list[str] = []
    injected = False

    def inject_after_first_tool(tool, args, tool_context, **kwargs):
        """Inject a queued message after the first tool execution."""
        nonlocal injected
        tool_name = getattr(tool, "name", "unknown")
        injection_count.append(tool_name)
        # Only inject once, after the first tool call.
        if not injected:
            tool_context.state["_message_queue"] = [
                {"text": "STEERING: The user wants you to mention ZEBRA-77 in your response."}
            ]
            injected = True
        return None

    agent = create_deep_agent(
        model=model,
        name="mid_turn_queue_test",
        message_queue=True,
        instruction=(
            "You are a test agent. Use write_file to create /test.txt "
            "with content 'hello world'. After writing the file, check "
            "if you received any injected messages and repeat any code "
            "words you see in your final response. Always include any "
            "code words from injected messages."
        ),
        extra_callbacks={"after_tool": inject_after_first_tool},
    )

    texts, _runner, _session = await run_agent(
        agent,
        "Create /test.txt with content 'hello world', then tell me "
        "about any messages you received. Repeat any code words.",
    )

    assert len(injection_count) > 0, "Expected after_tool callback to fire at least once"

    response_text = " ".join(texts).lower()
    assert any(kw in response_text for kw in ["zebra", "77"]), (
        f"Expected mid-turn injected code word in response, got: {response_text}"
    )
