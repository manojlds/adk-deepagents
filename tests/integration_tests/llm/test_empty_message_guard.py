"""Integration test — empty message guard (after_model callback).

Verifies that the after_model_callback is correctly wired and the agent
completes normally. Direct empty-response testing requires specific
model behavior that's hard to trigger reliably, so we test:
1. The callback is wired and doesn't interfere with normal operation.
2. An extra after_model callback composes correctly with the built-in one.

Run with: uv run pytest -m llm
"""

from __future__ import annotations

import pytest

from adk_deepagents import CallbackHooks, DeepAgentConfig, create_deep_agent
from tests.integration_tests.conftest import make_litellm_model, run_agent

pytestmark = [pytest.mark.integration, pytest.mark.llm]


@pytest.mark.timeout(120)
async def test_after_model_callback_wired():
    """The built-in after_model callback doesn't interfere with normal responses."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="after_model_test",
        instruction="You are a test agent. Respond with exactly: 'GUARD_OK'.",
    )

    texts, _runner, _session = await run_agent(
        agent,
        "Say 'GUARD_OK'.",
    )

    response_text = " ".join(texts)
    assert "GUARD_OK" in response_text, f"Expected 'GUARD_OK' in response, got: {response_text}"


@pytest.mark.timeout(120)
async def test_extra_after_model_callback_composes():
    """Extra after_model callback composes with the built-in empty guard."""
    model = make_litellm_model()

    after_model_called = []

    def extra_after_model(callback_context, llm_response):
        """Track that the extra callback was invoked."""
        after_model_called.append(True)
        return None  # Don't modify the response

    agent = create_deep_agent(
        model=model,
        name="extra_after_model_test",
        instruction="You are a test agent. Respond concisely.",
        config=DeepAgentConfig(callbacks=CallbackHooks(after_model=extra_after_model)),
    )

    texts, _runner, _session = await run_agent(
        agent,
        "Say 'hello'.",
    )

    assert len(texts) > 0, "Expected at least one text response"
    assert len(after_model_called) > 0, "Expected extra after_model callback to be called"
