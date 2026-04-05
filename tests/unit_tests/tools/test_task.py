"""Tests for static sub-agent builder (tools/task.py)."""

from __future__ import annotations

from typing import cast
from unittest.mock import patch

import pytest
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool

from adk_deepagents.prompts import (
    DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    DEFAULT_SUBAGENT_PROMPT,
)
from adk_deepagents.tools.task import (
    GENERAL_PURPOSE_SUBAGENT,
    _sanitize_agent_name,
    build_subagent_tools,
)
from adk_deepagents.types import SubAgentSpec


def _agent_tool_agent(tools: list[AgentTool], index: int = 0) -> LlmAgent:
    return cast(LlmAgent, tools[index].agent)


# ---------------------------------------------------------------------------
# _sanitize_agent_name
# ---------------------------------------------------------------------------


class TestSanitizeAgentName:
    def test_plain_name(self):
        assert _sanitize_agent_name("researcher") == "researcher"

    def test_hyphens_replaced(self):
        assert _sanitize_agent_name("code-reviewer") == "code_reviewer"

    def test_spaces_replaced(self):
        assert _sanitize_agent_name("my agent") == "my_agent"

    def test_leading_digit_prefixed(self):
        assert _sanitize_agent_name("3d_renderer") == "_3d_renderer"

    def test_special_chars_stripped(self):
        assert _sanitize_agent_name("agent@v2!") == "agent_v2_"

    def test_empty_string_fallback(self):
        assert _sanitize_agent_name("") == "agent"

    def test_all_invalid_chars_become_underscores(self):
        assert _sanitize_agent_name("@#$") == "___"

    def test_mixed_invalid_and_valid(self):
        result = _sanitize_agent_name("my-agent v2.0")
        assert result == "my_agent_v2_0"

    def test_underscores_preserved(self):
        assert _sanitize_agent_name("my_agent") == "my_agent"


# ---------------------------------------------------------------------------
# GENERAL_PURPOSE_SUBAGENT constant
# ---------------------------------------------------------------------------


class TestGeneralPurposeSubagent:
    def test_is_subagent_spec(self):
        assert isinstance(GENERAL_PURPOSE_SUBAGENT, dict)
        assert "name" in GENERAL_PURPOSE_SUBAGENT
        assert "description" in GENERAL_PURPOSE_SUBAGENT

    def test_name(self):
        assert GENERAL_PURPOSE_SUBAGENT["name"] == "general_purpose"

    def test_description_matches_prompt(self):
        assert GENERAL_PURPOSE_SUBAGENT["description"] == DEFAULT_GENERAL_PURPOSE_DESCRIPTION

    def test_system_prompt_matches_default(self):
        assert GENERAL_PURPOSE_SUBAGENT["system_prompt"] == DEFAULT_SUBAGENT_PROMPT


# ---------------------------------------------------------------------------
# build_subagent_tools
# ---------------------------------------------------------------------------


class TestBuildSubagentTools:
    """Core tests for build_subagent_tools."""

    def test_empty_subagents_with_general_purpose(self):
        """Empty subagents list should produce one general-purpose AgentTool."""
        tools = build_subagent_tools(
            subagents=[],
            default_model="gemini-2.5-flash",
            default_tools=[],
        )
        assert len(tools) == 1
        assert isinstance(tools[0], AgentTool)
        assert tools[0].agent.name == "general_purpose"

    def test_empty_subagents_without_general_purpose(self):
        """Disabling include_general_purpose with empty list produces no tools."""
        tools = build_subagent_tools(
            subagents=[],
            default_model="gemini-2.5-flash",
            default_tools=[],
            include_general_purpose=False,
        )
        assert len(tools) == 0

    def test_single_spec_produces_two_tools(self):
        """One spec + general_purpose = 2 AgentTools."""
        spec = SubAgentSpec(name="researcher", description="Researches topics")
        tools = build_subagent_tools(
            subagents=[spec],
            default_model="gemini-2.5-flash",
            default_tools=[],
        )
        assert len(tools) == 2
        names = {t.agent.name for t in tools}
        assert "general_purpose" in names
        assert "researcher" in names

    def test_spec_uses_default_model(self):
        """Sub-agent without model uses the default_model."""
        spec = SubAgentSpec(name="helper", description="Helps")
        tools = build_subagent_tools(
            subagents=[spec],
            default_model="gemini-2.5-pro",
            default_tools=[],
            include_general_purpose=False,
        )
        assert len(tools) == 1
        assert _agent_tool_agent(tools).model == "gemini-2.5-pro"

    def test_spec_custom_model(self):
        """Sub-agent with model= overrides the default."""
        spec = SubAgentSpec(name="helper", description="Helps", model="gpt-4o")
        tools = build_subagent_tools(
            subagents=[spec],
            default_model="gemini-2.5-flash",
            default_tools=[],
            include_general_purpose=False,
        )
        assert _agent_tool_agent(tools).model == "gpt-4o"

    def test_spec_custom_system_prompt(self):
        """Sub-agent with system_prompt uses it as instruction."""
        spec = SubAgentSpec(
            name="coder",
            description="Codes stuff",
            system_prompt="You are a coder.",
        )
        tools = build_subagent_tools(
            subagents=[spec],
            default_model="gemini-2.5-flash",
            default_tools=[],
            include_general_purpose=False,
        )
        assert _agent_tool_agent(tools).instruction == "You are a coder."

    def test_spec_default_system_prompt(self):
        """Sub-agent without system_prompt falls back to DEFAULT_SUBAGENT_PROMPT."""
        spec = SubAgentSpec(name="helper", description="Helps")
        tools = build_subagent_tools(
            subagents=[spec],
            default_model="gemini-2.5-flash",
            default_tools=[],
            include_general_purpose=False,
        )
        assert _agent_tool_agent(tools).instruction == DEFAULT_SUBAGENT_PROMPT

    def test_spec_inherits_default_tools(self):
        """Sub-agent without tools= inherits default_tools."""

        def my_tool(x: str) -> str:
            """My custom tool."""
            return x

        spec = SubAgentSpec(name="helper", description="Helps")
        tools = build_subagent_tools(
            subagents=[spec],
            default_model="gemini-2.5-flash",
            default_tools=[my_tool],
            include_general_purpose=False,
        )
        assert my_tool in _agent_tool_agent(tools).tools

    def test_spec_custom_tools_override_defaults(self):
        """Sub-agent with tools= uses those instead of default_tools."""

        def default_tool(x: str) -> str:
            """Default tool."""
            return x

        def custom_tool(x: str) -> str:
            """Custom tool."""
            return x

        spec = SubAgentSpec(
            name="helper",
            description="Helps",
            tools=[custom_tool],
        )
        tools = build_subagent_tools(
            subagents=[spec],
            default_model="gemini-2.5-flash",
            default_tools=[default_tool],
            include_general_purpose=False,
        )
        assert custom_tool in _agent_tool_agent(tools).tools
        assert default_tool not in _agent_tool_agent(tools).tools

    def test_name_sanitization(self):
        """Agent names with hyphens are sanitized."""
        spec = SubAgentSpec(name="code-reviewer", description="Reviews code")
        tools = build_subagent_tools(
            subagents=[spec],
            default_model="gemini-2.5-flash",
            default_tools=[],
            include_general_purpose=False,
        )
        assert tools[0].agent.name == "code_reviewer"

    def test_description_passed_through(self):
        """Agent description is set on the LlmAgent."""
        spec = SubAgentSpec(name="helper", description="Helps with tasks")
        tools = build_subagent_tools(
            subagents=[spec],
            default_model="gemini-2.5-flash",
            default_tools=[],
            include_general_purpose=False,
        )
        assert tools[0].agent.description == "Helps with tasks"


class TestBuildSubagentToolsPrebuilt:
    """Tests for pre-built LlmAgent support."""

    def test_prebuilt_agent_wrapped(self):
        """Pre-built LlmAgent is wrapped in AgentTool directly."""
        prebuilt = LlmAgent(
            name="prebuilt",
            model="gemini-2.5-flash",
            description="Pre-built agent",
            instruction="Pre-built instructions.",
        )
        tools = build_subagent_tools(
            subagents=[prebuilt],
            default_model="gemini-2.5-flash",
            default_tools=[],
            include_general_purpose=False,
        )
        assert len(tools) == 1
        assert tools[0].agent is prebuilt

    def test_prebuilt_mixed_with_specs(self):
        """Pre-built agents and specs can be mixed."""
        prebuilt = LlmAgent(
            name="prebuilt",
            model="gemini-2.5-flash",
            description="Pre-built",
            instruction="Pre-built.",
        )
        spec = SubAgentSpec(name="from_spec", description="From spec")
        tools = build_subagent_tools(
            subagents=[prebuilt, spec],
            default_model="gemini-2.5-flash",
            default_tools=[],
            include_general_purpose=False,
        )
        assert len(tools) == 2
        assert tools[0].agent is prebuilt
        assert tools[1].agent.name == "from_spec"

    def test_prebuilt_general_purpose_suppresses_default(self):
        """Pre-built agent named general_purpose suppresses the default."""
        prebuilt = LlmAgent(
            name="general_purpose",
            model="gemini-2.5-flash",
            description="Custom GP",
            instruction="Custom GP.",
        )
        tools = build_subagent_tools(
            subagents=[prebuilt],
            default_model="gemini-2.5-flash",
            default_tools=[],
            include_general_purpose=True,
        )
        gp = [t for t in tools if t.agent.name == "general_purpose"]
        assert len(gp) == 1
        assert gp[0].agent is prebuilt

    def test_prebuilt_agents_come_first(self):
        """Pre-built agents appear before spec-built ones."""
        prebuilt = LlmAgent(
            name="prebuilt",
            model="gemini-2.5-flash",
            description="Pre-built",
            instruction="Pre-built.",
        )
        spec = SubAgentSpec(name="from_spec", description="From spec")
        tools = build_subagent_tools(
            subagents=[prebuilt, spec],
            default_model="gemini-2.5-flash",
            default_tools=[],
            include_general_purpose=False,
        )
        assert tools[0].agent is prebuilt
        assert tools[1].agent.name == "from_spec"


class TestBuildSubagentToolsCallbacks:
    """Tests for callback injection."""

    def test_callbacks_injected(self):
        """before_agent, before_model, after_tool callbacks are injected."""

        def ba_cb(ctx):
            return None

        def bm_cb(ctx, req):
            return None

        def at_cb(tool, args, ctx):
            return None

        spec = SubAgentSpec(name="helper", description="Helps")
        tools = build_subagent_tools(
            subagents=[spec],
            default_model="gemini-2.5-flash",
            default_tools=[],
            include_general_purpose=False,
            before_agent_callback=ba_cb,
            before_model_callback=bm_cb,
            after_tool_callback=at_cb,
        )
        agent = tools[0].agent
        agent = cast(LlmAgent, agent)
        assert agent.before_agent_callback is ba_cb
        assert agent.before_model_callback is bm_cb
        assert agent.after_tool_callback is at_cb

    def test_interrupt_on_creates_before_tool_callback(self):
        """Sub-agent with interrupt_on gets a before_tool_callback."""
        spec = SubAgentSpec(
            name="cautious",
            description="Cautious agent",
            interrupt_on={"write_file": True},
        )
        tools = build_subagent_tools(
            subagents=[spec],
            default_model="gemini-2.5-flash",
            default_tools=[],
            include_general_purpose=False,
        )
        assert _agent_tool_agent(tools).before_tool_callback is not None

    def test_no_interrupt_on_no_before_tool_callback(self):
        """Sub-agent without interrupt_on gets no before_tool_callback (None)."""
        spec = SubAgentSpec(name="helper", description="Helps")
        tools = build_subagent_tools(
            subagents=[spec],
            default_model="gemini-2.5-flash",
            default_tools=[],
            include_general_purpose=False,
        )
        assert _agent_tool_agent(tools).before_tool_callback is None

    def test_default_interrupt_on_fallback(self):
        """Sub-agent without interrupt_on inherits default_interrupt_on."""
        spec = SubAgentSpec(name="helper", description="Helps")
        tools = build_subagent_tools(
            subagents=[spec],
            default_model="gemini-2.5-flash",
            default_tools=[],
            include_general_purpose=False,
            default_interrupt_on={"execute": True},
        )
        assert _agent_tool_agent(tools).before_tool_callback is not None

    def test_spec_interrupt_on_overrides_default(self):
        """Sub-agent's own interrupt_on takes precedence over default."""
        spec = SubAgentSpec(
            name="helper",
            description="Helps",
            interrupt_on={"write_file": True},
        )
        tools = build_subagent_tools(
            subagents=[spec],
            default_model="gemini-2.5-flash",
            default_tools=[],
            include_general_purpose=False,
            default_interrupt_on={"execute": True},
        )
        assert _agent_tool_agent(tools).before_tool_callback is not None


class TestBuildSubagentToolsSkills:
    """Tests for skills integration in sub-agents."""

    def test_skills_raises_import_error(self):
        """Skills on a sub-agent raise ImportError if adk-skills-agent is missing."""
        spec = SubAgentSpec(
            name="skill_agent",
            description="Agent with skills",
            skills=["/skills"],
        )
        with (
            patch.dict("sys.modules", {"adk_skills_agent": None}),
            pytest.raises(ImportError, match="adk-skills-agent is required"),
        ):
            build_subagent_tools(
                subagents=[spec],
                default_model="gemini-2.5-flash",
                default_tools=[],
                include_general_purpose=False,
            )

    def test_no_skills_no_error(self):
        """Sub-agent without skills works fine."""
        spec = SubAgentSpec(name="plain", description="Plain agent")
        with patch.dict("sys.modules", {"adk_skills_agent": None}):
            tools = build_subagent_tools(
                subagents=[spec],
                default_model="gemini-2.5-flash",
                default_tools=[],
                include_general_purpose=False,
            )
            assert len(tools) == 1


class TestBuildSubagentToolsGeneralPurpose:
    """Tests for general-purpose sub-agent deduplication."""

    def test_general_purpose_not_duplicated(self):
        """If a spec named general_purpose exists, default is not added."""
        spec = SubAgentSpec(
            name="general_purpose",
            description="Custom GP",
            system_prompt="Custom GP prompt.",
        )
        tools = build_subagent_tools(
            subagents=[spec],
            default_model="gemini-2.5-flash",
            default_tools=[],
            include_general_purpose=True,
        )
        gp = [t for t in tools if t.agent.name == "general_purpose"]
        assert len(gp) == 1
        assert cast(LlmAgent, gp[0].agent).instruction == "Custom GP prompt."

    def test_general_purpose_with_hyphenated_name(self):
        """Hyphenated general-purpose is sanitized and deduplication still works."""
        spec = SubAgentSpec(
            name="general-purpose",
            description="Custom GP",
        )
        tools = build_subagent_tools(
            subagents=[spec],
            default_model="gemini-2.5-flash",
            default_tools=[],
            include_general_purpose=True,
        )
        gp = [t for t in tools if t.agent.name == "general_purpose"]
        assert len(gp) == 1

    def test_multiple_specs_with_general_purpose(self):
        """Multiple specs plus general-purpose produces correct count."""
        specs = [
            SubAgentSpec(name="researcher", description="Researches"),
            SubAgentSpec(name="writer", description="Writes"),
        ]
        tools = build_subagent_tools(
            subagents=specs,
            default_model="gemini-2.5-flash",
            default_tools=[],
            include_general_purpose=True,
        )
        assert len(tools) == 3
        names = {t.agent.name for t in tools}
        assert names == {"general_purpose", "researcher", "writer"}
