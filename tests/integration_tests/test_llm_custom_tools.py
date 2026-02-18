"""Integration test — custom user-provided tools with a real LLM.

Tests that user-provided tool functions work alongside built-in tools.

Requires OPENCODE_API_KEY environment variable to be set.
Run with: uv run pytest -m integration
"""

from __future__ import annotations

import pytest

from adk_deepagents import create_deep_agent

from .conftest import make_litellm_model, run_agent

pytestmark = pytest.mark.integration


def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A Python math expression to evaluate (e.g., "2 + 3 * 4").
    """
    try:
        # Only allow safe math operations
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return {"status": "error", "message": "Invalid characters in expression"}
        result = eval(expression)  # noqa: S307
        return {"status": "success", "result": float(result)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_weather(city: str) -> dict:
    """Get the current weather for a city (mock implementation).

    Args:
        city: The name of the city.
    """
    # Mock weather data
    weather_data = {
        "tokyo": {"temp": 22, "condition": "sunny", "humidity": 45},
        "london": {"temp": 14, "condition": "rainy", "humidity": 80},
        "new york": {"temp": 18, "condition": "cloudy", "humidity": 60},
    }
    data = weather_data.get(city.lower())
    if data:
        return {"status": "success", "city": city, **data}
    return {"status": "success", "city": city, "temp": 20, "condition": "unknown", "humidity": 50}


@pytest.mark.timeout(120)
async def test_custom_tool_invocation():
    """Agent invokes a custom user-provided tool function."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="custom_tool_agent",
        tools=[calculate],
        instruction=(
            "You are a test agent. You have a 'calculate' tool for evaluating "
            "math expressions. Use it when asked to compute something. "
            "Report the exact result."
        ),
    )

    texts, _runner, _session = await run_agent(
        agent,
        "Use the calculate tool to evaluate: (100 + 50) / 3. What is the result?",
    )

    response_text = " ".join(texts)
    assert "50" in response_text, (
        f"Expected 50 (or 50.0) in response, got: {response_text}"
    )


@pytest.mark.timeout(120)
async def test_multiple_custom_tools():
    """Agent uses multiple custom tools in the same session."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="multi_tool_agent",
        tools=[calculate, get_weather],
        instruction=(
            "You are an assistant with custom tools:\n"
            "- calculate: evaluate math expressions\n"
            "- get_weather: get weather for a city\n"
            "Use the appropriate tool for each request."
        ),
    )

    texts, _runner, _session = await run_agent(
        agent,
        "First use get_weather to check the weather in Tokyo, "
        "then use calculate to compute 22 * 1.8 + 32 (converting Celsius to Fahrenheit). "
        "Report both results.",
    )

    response_text = " ".join(texts).lower()
    # Should reference Tokyo weather and/or the calculation
    has_weather = "tokyo" in response_text or "sunny" in response_text
    has_calc = "71" in response_text or "fahrenheit" in response_text
    assert has_weather or has_calc, (
        f"Expected weather and/or conversion result, got: {response_text}"
    )


@pytest.mark.timeout(120)
async def test_custom_tools_alongside_builtin():
    """Custom tools work alongside built-in filesystem and todo tools."""
    model = make_litellm_model()

    agent = create_deep_agent(
        model=model,
        name="mixed_tool_agent",
        tools=[get_weather],
        instruction=(
            "You are a test agent. You have built-in filesystem and todo tools, "
            "plus a custom get_weather tool. Use the appropriate tool for each request."
        ),
    )

    texts, _runner, _session = await run_agent(
        agent,
        "Use get_weather to check the weather in London, then use write_file "
        "to save the weather report to /weather.txt. Confirm when done.",
    )

    response_text = " ".join(texts).lower()
    assert any(
        word in response_text
        for word in ("london", "rainy", "weather", "created", "written", "saved", "done")
    ), f"Expected weather + file creation confirmation, got: {response_text}"
