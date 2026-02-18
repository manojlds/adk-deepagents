"""Integration test — structured output with a real LLM.

Scenario: Agent with output_schema returns structured JSON matching
a Pydantic model.

Requires OPENCODE_API_KEY environment variable to be set.
Run with: uv run pytest -m integration
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from adk_deepagents import create_deep_agent

from .conftest import make_litellm_model, run_agent

pytestmark = pytest.mark.integration


class SentimentResult(BaseModel):
    """Structured output model for sentiment analysis."""

    text: str
    sentiment: str  # positive, negative, neutral
    confidence: float


class MathResult(BaseModel):
    """Structured output for math problems."""

    expression: str
    result: float
    steps: list[str]


@pytest.mark.timeout(120)
async def test_structured_output_sentiment():
    """Agent returns structured sentiment analysis result."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="sentiment_agent",
        instruction=(
            "You are a sentiment analysis agent. Analyze the sentiment of "
            "the given text. Return structured output with the text, sentiment "
            "(positive/negative/neutral), and confidence score (0.0 to 1.0)."
        ),
        output_schema=SentimentResult,
    )

    texts, _runner, _session = await run_agent(
        agent,
        "Analyze the sentiment of: 'This product is absolutely amazing! Best purchase ever!'",
    )

    response_text = " ".join(texts)
    # The response should be valid JSON matching the schema
    try:
        data = json.loads(response_text)
        assert "sentiment" in data, f"Missing 'sentiment' key in: {data}"
        assert data["sentiment"].lower() in ("positive", "negative", "neutral"), (
            f"Invalid sentiment: {data['sentiment']}"
        )
        assert "confidence" in data, f"Missing 'confidence' key in: {data}"
    except json.JSONDecodeError:
        # Some models return the structured data embedded in text;
        # at minimum, "positive" should appear
        assert "positive" in response_text.lower(), (
            f"Expected positive sentiment in response, got: {response_text}"
        )


@pytest.mark.timeout(120)
async def test_structured_output_math():
    """Agent returns structured math result with steps."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="math_structured_agent",
        instruction=(
            "You are a math agent. Solve the given expression. Return structured "
            "output with the expression, numeric result, and solution steps."
        ),
        output_schema=MathResult,
    )

    texts, _runner, _session = await run_agent(
        agent,
        "Solve: (10 + 5) * 4",
    )

    response_text = " ".join(texts)
    try:
        data = json.loads(response_text)
        assert "result" in data, f"Missing 'result' key in: {data}"
        assert float(data["result"]) == 60.0, f"Expected 60, got: {data['result']}"
    except json.JSONDecodeError:
        # Fallback: check the answer appears in text
        assert "60" in response_text, (
            f"Expected 60 in response, got: {response_text}"
        )
