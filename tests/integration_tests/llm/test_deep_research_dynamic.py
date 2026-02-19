from __future__ import annotations

import pytest

from examples.deep_research.agent import build_agent
from tests.integration_tests.conftest import (
    get_file_content,
    run_agent_with_events,
)

pytestmark = [pytest.mark.integration, pytest.mark.llm]


@pytest.mark.timeout(300)
async def test_deep_research_uses_dynamic_task_and_writes_report():
    agent = build_agent()

    prompt = (
        "Run a minimal deep-research flow. Use dynamic task delegation at least once (planner is fine), "
        "skip external web research, and write a tiny report to /final_report.md with one inline citation "
        "and a Sources section."
    )
    texts, function_calls, function_responses, runner, session = await run_agent_with_events(
        agent,
        prompt,
    )

    assert "task" in function_calls, f"Expected dynamic task delegation call, got: {function_calls}"
    assert "task" in function_responses, (
        f"Expected dynamic task delegation response, got: {function_responses}"
    )

    files = await get_file_content(runner, session)
    assert "/final_report.md" in files, f"Expected /final_report.md in files, got: {list(files)}"

    report = files["/final_report.md"]
    assert "sources" in report.lower(), f"Expected Sources section in report, got: {report}"
    assert "[" in report and "]" in report, f"Expected inline citations in report, got: {report}"

    response = " ".join(texts).lower()
    assert "final_report.md" in response or "report" in response, (
        f"Expected completion mention of report output, got: {response}"
    )
