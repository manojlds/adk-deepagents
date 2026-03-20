"""Integration test — message queue injection with a real LLM.

Verifies that externally injected messages via ``_message_queue`` state
and ``message_queue_provider`` callable are picked up by the
before_model callback and visible to the LLM.

Run with: uv run pytest -m llm
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from adk_deepagents.cli.tui.agent_service import _SharedMessageQueue
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


# ---------------------------------------------------------------------------
# message_queue_provider tests (the mechanism the TUI uses)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
async def test_message_queue_provider_pre_loaded():
    """Messages pre-loaded into the provider are visible to the LLM on the first turn.

    This is the simplest provider test: push a message into
    ``_SharedMessageQueue`` *before* ``run_async``, and verify the
    ``before_model_callback`` drains it and the LLM sees it.
    """
    model = make_litellm_model()
    queue = _SharedMessageQueue()

    agent = create_deep_agent(
        model=model,
        name="provider_preload_test",
        message_queue=True,
        message_queue_provider=queue.drain,
        instruction=(
            "You are a test agent.  If you see any injected messages "
            "containing a code word, you MUST repeat the code word in "
            "your response.  Always repeat code words exactly."
        ),
    )

    # Pre-load a message before the turn starts.
    queue.push("INJECTED: The code word is PARROT-55. Repeat it.")

    texts, _runner, _session = await run_agent(
        agent,
        "What code words have you received? Repeat them exactly.",
    )

    response_text = " ".join(texts).lower()
    assert any(kw in response_text for kw in ["parrot", "55"]), (
        f"Expected provider-injected code word in response, got: {response_text}"
    )


@pytest.mark.timeout(120)
async def test_message_queue_provider_mid_turn():
    """Messages pushed into the provider mid-turn are visible within the same turn.

    This simulates the real TUI flow: the user queues a message while the
    agent is executing tool calls.  We use ``after_tool`` to push a message
    into the ``_SharedMessageQueue``.  The ``before_model_callback`` drains
    it before the next LLM call — all within a single ``run_async``.
    """
    model = make_litellm_model()
    queue = _SharedMessageQueue()
    injected = False

    def inject_via_provider(tool, args, tool_context, **kwargs):
        """Push a message into the shared queue after the first tool call."""
        nonlocal injected
        if not injected:
            queue.push("STEERING: The user wants you to say DOLPHIN-88 in your answer.")
            injected = True
        return None

    agent = create_deep_agent(
        model=model,
        name="provider_mid_turn_test",
        message_queue=True,
        message_queue_provider=queue.drain,
        instruction=(
            "You are a test agent. Use write_file to create /test.txt "
            "with content 'hello world'. After writing the file, check "
            "if you received any injected messages and repeat any code "
            "words you see in your final response. Always include any "
            "code words from injected messages."
        ),
        extra_callbacks={"after_tool": inject_via_provider},
    )

    texts, _runner, _session = await run_agent(
        agent,
        "Create /test.txt with content 'hello world', then tell me "
        "about any messages you received. Repeat any code words.",
    )

    assert injected, "Expected after_tool callback to fire and inject a message"

    response_text = " ".join(texts).lower()
    assert any(kw in response_text for kw in ["dolphin", "88"]), (
        f"Expected mid-turn provider code word in response, got: {response_text}"
    )


@pytest.mark.timeout(120)
async def test_message_queue_provider_coexists_with_state_queue():
    """Provider-based and state-based message queues work together.

    This test pre-loads a message in the provider AND injects a
    second message via ``state["_message_queue"]`` during a tool call.
    Both code words should appear in the response.
    """
    model = make_litellm_model()
    queue = _SharedMessageQueue()
    injected_via_state = False

    def inject_state_queue(tool, args, tool_context, **kwargs):
        """Inject a message via session state after the first tool call."""
        nonlocal injected_via_state
        if not injected_via_state:
            tool_context.state["_message_queue"] = [
                {"text": "STATE-INJECTED: The second code word is TIGER-33."}
            ]
            injected_via_state = True
        return None

    agent = create_deep_agent(
        model=model,
        name="provider_state_coexist_test",
        message_queue=True,
        message_queue_provider=queue.drain,
        instruction=(
            "You are a test agent. Use write_file to create /test.txt "
            "with content 'hello'. After writing the file, check "
            "if you received any injected messages and repeat ALL "
            "code words you see. Always include every code word."
        ),
        extra_callbacks={"after_tool": inject_state_queue},
    )

    # Pre-load a provider message.
    queue.push("PROVIDER-INJECTED: The first code word is EAGLE-11.")

    texts, _runner, _session = await run_agent(
        agent,
        "Create /test.txt with 'hello', then repeat all code words you received.",
    )

    response_text = " ".join(texts).lower()
    # Provider message should have been picked up on the first LLM call.
    assert any(kw in response_text for kw in ["eagle", "11"]), (
        f"Expected provider code word EAGLE-11 in response, got: {response_text}"
    )
    # State-based message should have been picked up after tool call.
    assert any(kw in response_text for kw in ["tiger", "33"]), (
        f"Expected state code word TIGER-33 in response, got: {response_text}"
    )
