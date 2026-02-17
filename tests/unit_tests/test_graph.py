"""Tests for create_deep_agent factory."""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest
from google.adk.agents import LlmAgent

from adk_deepagents.backends.state import StateBackend
from adk_deepagents.graph import create_deep_agent
from adk_deepagents.prompts import BASE_AGENT_PROMPT
from adk_deepagents.types import SummarizationConfig


class TestCreateDeepAgent:
    def test_default_agent(self):
        agent = create_deep_agent()
        assert isinstance(agent, LlmAgent)
        assert agent.name == "deep_agent"
        assert agent.model == "gemini-2.5-flash"

    def test_custom_name(self):
        agent = create_deep_agent(name="my_agent")
        assert agent.name == "my_agent"

    def test_custom_model(self):
        agent = create_deep_agent(model="gemini-2.5-pro")
        assert agent.model == "gemini-2.5-pro"

    def test_instruction_prepended(self):
        agent = create_deep_agent(instruction="Custom instruction.")
        assert "Custom instruction." in agent.instruction
        assert BASE_AGENT_PROMPT in agent.instruction

    def test_default_instruction(self):
        agent = create_deep_agent()
        assert agent.instruction == BASE_AGENT_PROMPT

    def test_core_tools_included(self):
        agent = create_deep_agent()
        tool_funcs = [t for t in agent.tools if callable(t)]
        # At least the 8 core tools
        assert len(tool_funcs) >= 8

    def test_custom_tools_added(self):
        def my_tool(x: str) -> str:
            """My custom tool."""
            return x

        agent = create_deep_agent(tools=[my_tool])
        assert any(getattr(t, "__name__", None) == "my_tool" or t is my_tool for t in agent.tools)

    def test_subagents_create_agent_tools(self):
        agent = create_deep_agent(
            subagents=[
                {
                    "name": "researcher",
                    "description": "Research agent",
                }
            ]
        )
        # Should have AgentTool instances in tools
        from google.adk.tools import AgentTool

        agent_tools = [t for t in agent.tools if isinstance(t, AgentTool)]
        # Should have general_purpose + researcher
        assert len(agent_tools) >= 2

    def test_local_execution(self):
        agent = create_deep_agent(execution="local")
        # Should have the execute tool
        tool_names = []
        for t in agent.tools:
            name = getattr(t, "__name__", getattr(t, "name", None))
            if name:
                tool_names.append(name)
        assert "execute" in tool_names

    def test_heimdall_execution_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            create_deep_agent(execution="heimdall")
            assert len(w) == 1
            assert "async" in str(w[0].message).lower()

    def test_dict_execution_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            create_deep_agent(execution={"uri": "http://localhost:3000/sse"})
            assert len(w) == 1

    def test_backend_factory(self):
        called = False

        def my_factory(state):
            nonlocal called
            called = True
            return StateBackend(state)

        agent = create_deep_agent(backend=my_factory)
        assert isinstance(agent, LlmAgent)

    def test_backend_instance(self):
        state = {"files": {}}
        backend = StateBackend(state)
        agent = create_deep_agent(backend=backend)
        assert isinstance(agent, LlmAgent)

    def test_interrupt_on(self):
        agent = create_deep_agent(interrupt_on={"write_file": True})
        assert agent.before_tool_callback is not None

    def test_no_interrupt_on(self):
        agent = create_deep_agent()
        # before_tool_callback is None when no interrupt_on is configured
        assert agent.before_tool_callback is None

    def test_summarization_config(self):
        config = SummarizationConfig(model="gemini-2.5-flash")
        agent = create_deep_agent(summarization=config)
        assert isinstance(agent, LlmAgent)

    def test_memory_sources(self):
        agent = create_deep_agent(memory=["/AGENTS.md"])
        assert isinstance(agent, LlmAgent)

    def test_callbacks_set(self):
        agent = create_deep_agent()
        assert agent.before_agent_callback is not None
        assert agent.before_model_callback is not None
        assert agent.after_tool_callback is not None

    def test_output_schema(self):
        from pydantic import BaseModel

        class MyOutput(BaseModel):
            result: str

        agent = create_deep_agent(output_schema=MyOutput)
        assert agent.output_schema is MyOutput

    def test_skills_raises_import_error_when_not_installed(self):
        """When skills are requested but adk-skills-agent is missing, raise ImportError."""
        with (
            patch.dict("sys.modules", {"adk_skills_agent": None}),
            pytest.raises(ImportError, match="adk-skills-agent is required"),
        ):
            create_deep_agent(skills=["/skills"])

    def test_no_error_when_skills_not_requested_and_import_fails(self):
        """When skills are NOT requested, import failure is irrelevant."""
        with patch.dict("sys.modules", {"adk_skills_agent": None}):
            agent = create_deep_agent()
            assert isinstance(agent, LlmAgent)
