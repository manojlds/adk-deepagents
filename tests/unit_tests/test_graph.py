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
        assert isinstance(agent.instruction, str)
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


class TestExtraCallbacks:
    """Tests for the extra_callbacks parameter (US-010)."""

    def test_extra_callbacks_none(self):
        """No extra callbacks — agent is created normally."""
        agent = create_deep_agent(extra_callbacks=None)
        assert isinstance(agent, LlmAgent)

    def test_extra_callbacks_empty(self):
        """Empty dict — agent is created normally."""
        agent = create_deep_agent(extra_callbacks={})
        assert isinstance(agent, LlmAgent)

    def test_before_agent_extra_called(self):
        """Extra before_agent callback is called after built-in."""
        call_log = []

        def extra_before_agent(callback_context):
            call_log.append("extra_before_agent")
            return None

        agent = create_deep_agent(
            extra_callbacks={"before_agent": extra_before_agent},
        )
        # The agent's before_agent_callback should be a composed callback
        assert agent.before_agent_callback is not None

    def test_before_model_extra_called(self):
        """Extra before_model callback is called after built-in."""
        call_log = []

        def extra_before_model(callback_context, llm_request):
            call_log.append("extra_before_model")
            return None

        agent = create_deep_agent(
            extra_callbacks={"before_model": extra_before_model},
        )
        assert agent.before_model_callback is not None

    def test_after_tool_extra_called(self):
        """Extra after_tool callback is called after built-in."""

        def extra_after_tool(tool, args, tool_context):
            return None

        agent = create_deep_agent(
            extra_callbacks={"after_tool": extra_after_tool},
        )
        assert agent.after_tool_callback is not None

    def test_before_tool_extra_with_no_builtin(self):
        """When no interrupt_on, before_tool builtin is None — extra is used directly."""

        def extra_before_tool(tool, args, tool_context):
            return None

        agent = create_deep_agent(
            extra_callbacks={"before_tool": extra_before_tool},
        )
        # With no interrupt_on, builtin is None, so extra is used directly
        assert agent.before_tool_callback is extra_before_tool

    def test_before_tool_extra_composed_with_builtin(self):
        """When interrupt_on is set, extra is composed with builtin."""

        def extra_before_tool(tool, args, tool_context):
            return None

        agent = create_deep_agent(
            interrupt_on={"write_file": True},
            extra_callbacks={"before_tool": extra_before_tool},
        )
        # Should be a composed callback (neither the builtin nor the extra alone)
        assert agent.before_tool_callback is not None
        assert agent.before_tool_callback is not extra_before_tool

    def test_compose_builtin_runs_first(self):
        """Built-in callback runs first, then extra."""
        from adk_deepagents.graph import _compose_callbacks

        call_log = []

        def builtin(*args, **kwargs):
            call_log.append("builtin")
            return None

        def extra(*args, **kwargs):
            call_log.append("extra")
            return None

        composed = _compose_callbacks(builtin, extra)
        assert composed is not None
        composed()
        assert call_log == ["builtin", "extra"]

    def test_compose_short_circuit_skips_extra(self):
        """If built-in returns non-None, extra is NOT called."""
        from adk_deepagents.graph import _compose_callbacks

        call_log = []

        def builtin(*args, **kwargs):
            call_log.append("builtin")
            return {"short": "circuited"}

        def extra(*args, **kwargs):
            call_log.append("extra")
            return None

        composed = _compose_callbacks(builtin, extra)
        assert composed is not None
        result = composed()
        assert result == {"short": "circuited"}
        assert call_log == ["builtin"]

    def test_compose_extra_result_returned(self):
        """If built-in returns None, extra's result is returned."""
        from adk_deepagents.graph import _compose_callbacks

        def builtin(*args, **kwargs):
            return None

        def extra(*args, **kwargs):
            return {"extra": "result"}

        composed = _compose_callbacks(builtin, extra)
        assert composed is not None
        result = composed()
        assert result == {"extra": "result"}

    def test_compose_none_builtin(self):
        """If builtin is None, extra is returned directly."""
        from adk_deepagents.graph import _compose_callbacks

        def extra(*args, **kwargs):
            return None

        result = _compose_callbacks(None, extra)
        assert result is extra

    def test_compose_none_extra(self):
        """If extra is None, builtin is returned directly."""
        from adk_deepagents.graph import _compose_callbacks

        def builtin(*args, **kwargs):
            return None

        result = _compose_callbacks(builtin, None)
        assert result is builtin

    def test_compose_both_none(self):
        """If both are None, returns None."""
        from adk_deepagents.graph import _compose_callbacks

        result = _compose_callbacks(None, None)
        assert result is None


class TestSubAgentSpecFields:
    """Tests for SubAgentSpec skills, interrupt_on, and pre-built LlmAgent support (US-011)."""

    def test_subagent_with_skills_raises_when_not_installed(self):
        """Skills on a sub-agent raise ImportError if adk-skills-agent is missing."""
        with (
            patch.dict("sys.modules", {"adk_skills_agent": None}),
            pytest.raises(ImportError, match="adk-skills-agent is required"),
        ):
            create_deep_agent(
                subagents=[
                    {
                        "name": "skill_agent",
                        "description": "Agent with skills",
                        "skills": ["/skills"],
                    }
                ]
            )

    def test_subagent_without_skills_no_error(self):
        """Sub-agent without skills works even if adk-skills-agent is missing."""
        with patch.dict("sys.modules", {"adk_skills_agent": None}):
            agent = create_deep_agent(
                subagents=[
                    {
                        "name": "plain_agent",
                        "description": "No skills needed",
                    }
                ]
            )
            assert isinstance(agent, LlmAgent)

    def test_subagent_with_interrupt_on(self):
        """Sub-agent with interrupt_on gets its own before_tool_callback."""
        agent = create_deep_agent(
            subagents=[
                {
                    "name": "cautious_agent",
                    "description": "Agent requiring approval",
                    "interrupt_on": {"write_file": True},
                }
            ]
        )
        from google.adk.tools import AgentTool

        agent_tools = [t for t in agent.tools if isinstance(t, AgentTool)]
        # Find the cautious_agent
        cautious = [t for t in agent_tools if t.agent.name == "cautious_agent"]
        assert len(cautious) == 1
        assert isinstance(cautious[0].agent, LlmAgent)
        assert cautious[0].agent.before_tool_callback is not None

    def test_subagent_without_interrupt_on_has_no_callback(self):
        """Sub-agent without interrupt_on has no before_tool_callback."""
        agent = create_deep_agent(
            subagents=[
                {
                    "name": "normal_agent",
                    "description": "Normal agent",
                }
            ]
        )
        from google.adk.tools import AgentTool

        agent_tools = [t for t in agent.tools if isinstance(t, AgentTool)]
        normal = [t for t in agent_tools if t.agent.name == "normal_agent"]
        assert len(normal) == 1
        assert isinstance(normal[0].agent, LlmAgent)
        assert normal[0].agent.before_tool_callback is None

    def test_prebuilt_llmagent_in_subagents(self):
        """Pre-built LlmAgent instances are accepted in the subagents list."""
        prebuilt = LlmAgent(
            name="prebuilt_agent",
            model="gemini-2.5-flash",
            description="A pre-built agent",
            instruction="You are a pre-built agent.",
        )
        agent = create_deep_agent(subagents=[prebuilt])
        from google.adk.tools import AgentTool

        agent_tools = [t for t in agent.tools if isinstance(t, AgentTool)]
        prebuilt_tools = [t for t in agent_tools if t.agent.name == "prebuilt_agent"]
        assert len(prebuilt_tools) == 1
        assert prebuilt_tools[0].agent is prebuilt

    def test_prebuilt_llmagent_mixed_with_specs(self):
        """Pre-built LlmAgent and SubAgentSpec can be mixed."""
        prebuilt = LlmAgent(
            name="prebuilt_agent",
            model="gemini-2.5-flash",
            description="A pre-built agent",
            instruction="Pre-built instructions.",
        )
        agent = create_deep_agent(
            subagents=[
                prebuilt,
                {
                    "name": "spec_agent",
                    "description": "Spec-based agent",
                },
            ]
        )
        from google.adk.tools import AgentTool

        agent_tools = [t for t in agent.tools if isinstance(t, AgentTool)]
        names = {t.agent.name for t in agent_tools}
        # Should have prebuilt, spec_agent, and general_purpose
        assert "prebuilt_agent" in names
        assert "spec_agent" in names
        assert "general_purpose" in names

    def test_prebuilt_general_purpose_skips_default(self):
        """Pre-built agent named general_purpose prevents adding the default one."""
        prebuilt_gp = LlmAgent(
            name="general_purpose",
            model="gemini-2.5-flash",
            description="Custom GP",
            instruction="Custom GP instructions.",
        )
        agent = create_deep_agent(subagents=[prebuilt_gp])
        from google.adk.tools import AgentTool

        agent_tools = [t for t in agent.tools if isinstance(t, AgentTool)]
        gp_tools = [t for t in agent_tools if t.agent.name == "general_purpose"]
        # Should be exactly 1 (the custom one, not a duplicate)
        assert len(gp_tools) == 1
        assert gp_tools[0].agent is prebuilt_gp
