"""Integration test — basic deep agent tool usage.

Verifies that ``create_deep_agent`` can use filesystem tools end-to-end
with a real LLM backend.

Run with: uv run pytest tests/integration_tests/test_basic_agent.py -m integration
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent
from tests.integration_tests.conftest import make_litellm_model, run_agent

pytestmark = pytest.mark.integration


@pytest.mark.timeout(120)
async def test_write_and_read_file():
    """Agent writes a file and reads it back using filesystem tools."""
    agent = create_deep_agent(
        model=make_litellm_model(),
        name="basic_test_agent",
        instruction=(
            "You are a test agent. Follow the user's instructions exactly. "
            "Use the filesystem tools (write_file, read_file) as directed. "
            "After each step, confirm what you did."
        ),
    )

    texts, _runner, _session = await run_agent(
        agent,
        'Use write_file to create a file at /hello.txt with the content "Hello from deep agent". '
        "Then use read_file to read it back and show me the content.",
    )

    response_text = " ".join(texts)
    assert "Hello from deep agent" in response_text, (
        f"Expected file content in response, got: {response_text}"
    )
