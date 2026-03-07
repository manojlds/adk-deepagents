from __future__ import annotations

from examples.deep_research.agent import build_agent, build_runtime_subagent_profiles


def test_runtime_profiles_include_reporter_write_file_tool_name():
    profiles = build_runtime_subagent_profiles()
    reporter = next(profile for profile in profiles if profile["name"] == "reporter")
    assert "write_file" in reporter["tool_names"]


def test_runtime_profiles_include_grader_read_file_tool_name():
    profiles = build_runtime_subagent_profiles()
    grader = next(profile for profile in profiles if profile["name"] == "grader")
    assert "read_file" in grader["tool_names"]


def test_build_agent_includes_dynamic_delegation_tools():
    agent = build_agent()
    names = [getattr(tool, "__name__", getattr(tool, "name", "")) for tool in agent.tools]
    assert "register_subagent" in names
    assert "task" in names
