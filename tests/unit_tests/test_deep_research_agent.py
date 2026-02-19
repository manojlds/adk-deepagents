from __future__ import annotations

from examples.deep_research.agent import build_agent, grader_subagent, reporter_subagent


def test_reporter_has_write_file_tool():
    tools = reporter_subagent.get("tools", [])
    names = [getattr(tool, "__name__", "") for tool in tools]
    assert "write_file" in names


def test_grader_has_read_file_tool():
    tools = grader_subagent.get("tools", [])
    names = [getattr(tool, "__name__", "") for tool in tools]
    assert "read_file" in names


def test_build_agent_includes_task_tool():
    agent = build_agent()
    names = [getattr(tool, "__name__", getattr(tool, "name", "")) for tool in agent.tools]
    assert "task" in names
