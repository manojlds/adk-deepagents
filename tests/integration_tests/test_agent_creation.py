"""Integration tests — create_deep_agent factory configuration.

Verifies that the agent factory wires together tools, callbacks, sub-agents,
backend, and other configuration correctly.  No LLM calls — we only inspect
the returned LlmAgent's attributes.
No API key required.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool

from adk_deepagents import SubAgentSpec, SummarizationConfig, create_deep_agent
from adk_deepagents.backends import FilesystemBackend, StateBackend
from adk_deepagents.prompts import BASE_AGENT_PROMPT

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CORE_TOOL_NAMES = frozenset(
    {
        "write_todos",
        "read_todos",
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
    }
)


def _tool_names(agent: LlmAgent) -> set[str]:
    """Extract tool names (functions only, not AgentTools) from an agent."""
    names: set[str] = set()
    for t in agent.tools:
        name = getattr(t, "__name__", getattr(t, "name", None))
        if name and not isinstance(t, AgentTool):
            names.add(name)
    return names


def _agent_tools(agent: LlmAgent) -> list[AgentTool]:
    """Extract AgentTool instances from an agent."""
    return [t for t in agent.tools if isinstance(t, AgentTool)]


def _agent_tool_names(agent: LlmAgent) -> set[str]:
    """Extract sub-agent names from AgentTool instances."""
    return {t.agent.name for t in _agent_tools(agent)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDefaultAgent:
    def test_default_agent(self):
        agent = create_deep_agent()
        assert isinstance(agent, LlmAgent)
        assert agent.name == "deep_agent"
        assert agent.model == "gemini-2.5-flash"
        names = _tool_names(agent)
        assert CORE_TOOL_NAMES.issubset(names)
        # No sub-agent tools by default (subagents=None)
        assert len(_agent_tools(agent)) == 0


class TestCustomInstruction:
    def test_custom_instruction(self):
        agent = create_deep_agent(instruction="Be helpful.")
        assert isinstance(agent.instruction, str)
        assert agent.instruction.startswith("Be helpful.")
        assert BASE_AGENT_PROMPT in agent.instruction


class TestCustomModel:
    def test_custom_model(self):
        agent = create_deep_agent(model="gemini-2.5-pro")
        assert agent.model == "gemini-2.5-pro"


class TestCustomTools:
    def test_custom_tools(self):
        def my_tool(x: str) -> str:
            """Custom tool."""
            return x

        agent = create_deep_agent(tools=[my_tool])
        names = _tool_names(agent)
        assert "my_tool" in names
        assert CORE_TOOL_NAMES.issubset(names)


class TestBackendInstance:
    def test_backend_instance(self, tmp_path):
        backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
        agent = create_deep_agent(backend=backend)
        assert isinstance(agent, LlmAgent)


class TestBackendFactory:
    def test_backend_factory(self):
        factory_calls: list[dict] = []

        def my_factory(state):
            factory_calls.append(state)
            return StateBackend(state)

        agent = create_deep_agent(backend=my_factory)
        assert isinstance(agent, LlmAgent)
        # Factory is stored for later use, not called immediately
        assert len(factory_calls) == 0


class TestLocalExecution:
    def test_local_execution(self):
        agent = create_deep_agent(execution="local")
        names = _tool_names(agent)
        assert "execute" in names


class TestSubagentsCreatesAgentTools:
    def test_subagents_creates_agent_tools(self):
        agent = create_deep_agent(
            subagents=[
                SubAgentSpec(name="researcher", description="Research agent"),
            ]
        )
        at = _agent_tools(agent)
        assert len(at) >= 2  # researcher + general_purpose
        names = _agent_tool_names(agent)
        assert "researcher" in names


class TestDelegationModes:
    def test_dynamic_mode_adds_task_without_static_agent_tools(self):
        agent = create_deep_agent(
            subagents=[SubAgentSpec(name="researcher", description="Research agent")],
            delegation_mode="dynamic",
        )
        names = _tool_names(agent)
        assert "task" in names
        assert len(_agent_tools(agent)) == 0

    def test_both_mode_adds_task_and_static_agent_tools(self):
        agent = create_deep_agent(
            subagents=[SubAgentSpec(name="researcher", description="Research agent")],
            delegation_mode="both",
        )
        names = _tool_names(agent)
        assert "task" in names
        assert "researcher" in _agent_tool_names(agent)

    def test_invalid_delegation_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid delegation_mode"):
            create_deep_agent(delegation_mode=cast(Any, "invalid"))


class TestSubagentsGeneralPurposeIncluded:
    def test_subagents_general_purpose_included(self):
        agent = create_deep_agent(
            subagents=[
                SubAgentSpec(name="helper", description="Helper agent"),
            ]
        )
        names = _agent_tool_names(agent)
        assert "general_purpose" in names
        assert "helper" in names


class TestSubagentsNoDuplicateGP:
    def test_subagents_no_duplicate_gp(self):
        agent = create_deep_agent(
            subagents=[
                SubAgentSpec(name="general_purpose", description="My custom GP"),
            ]
        )
        names_list = [t.agent.name for t in _agent_tools(agent)]
        assert names_list.count("general_purpose") == 1


class TestOutputSchema:
    def test_output_schema(self):
        from pydantic import BaseModel

        class Result(BaseModel):
            answer: str

        agent = create_deep_agent(output_schema=Result)
        assert agent.output_schema is Result


class TestInterruptOnCreatesCallback:
    def test_interrupt_on_creates_callback(self):
        agent = create_deep_agent(interrupt_on={"write_file": True})
        assert agent.before_tool_callback is not None

    def test_no_interrupt_on_no_callback(self):
        agent = create_deep_agent()
        assert agent.before_tool_callback is None


class TestSummarizationConfig:
    def test_summarization_config(self):
        config = SummarizationConfig(model="gemini-2.5-flash")
        agent = create_deep_agent(summarization=config)
        assert isinstance(agent, LlmAgent)
        assert agent.before_model_callback is not None


class TestMemoryConfig:
    def test_memory_config(self):
        agent = create_deep_agent(memory=["/AGENTS.md"])
        assert isinstance(agent, LlmAgent)
        assert agent.before_agent_callback is not None
        assert agent.before_model_callback is not None


class TestExtraCallbacks:
    def test_extra_callbacks(self):
        def extra_before_agent(ctx):
            return None

        def extra_before_model(ctx, req):
            return None

        def extra_after_tool(tool, args, ctx):
            return None

        def extra_before_tool(tool, args, ctx):
            return None

        agent = create_deep_agent(
            extra_callbacks={
                "before_agent": extra_before_agent,
                "before_model": extra_before_model,
                "after_tool": extra_after_tool,
                "before_tool": extra_before_tool,
            }
        )
        assert agent.before_agent_callback is not None
        assert agent.before_model_callback is not None
        assert agent.after_tool_callback is not None
        # With no interrupt_on, builtin before_tool is None → extra used directly
        assert agent.before_tool_callback is extra_before_tool


class TestPrebuiltLlmAgentSubagent:
    def test_prebuilt_llmagent_subagent(self):
        prebuilt = LlmAgent(
            name="prebuilt_worker",
            model="gemini-2.5-flash",
            description="A pre-built worker agent",
            instruction="Do work.",
        )
        agent = create_deep_agent(subagents=[prebuilt])
        at = _agent_tools(agent)
        names = {t.agent.name for t in at}
        assert "prebuilt_worker" in names
        # general_purpose should be auto-added
        assert "general_purpose" in names

        # The prebuilt agent instance should be preserved as-is
        prebuilt_tools = [t for t in at if t.agent.name == "prebuilt_worker"]
        assert prebuilt_tools[0].agent is prebuilt
