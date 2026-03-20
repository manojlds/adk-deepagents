"""Unit tests for TUI data models (cli/tui/models.py)."""

from __future__ import annotations

from adk_deepagents.cli.tui.models import (
    BUILTIN_AGENTS,
    DEFAULT_AGENT_NAME,
    AgentProfile,
    AgentRegistry,
    ConversationLog,
    MessageRecord,
)

# ---------------------------------------------------------------------------
# MessageRecord
# ---------------------------------------------------------------------------


class TestMessageRecord:
    def test_user_record(self):
        rec = MessageRecord(role="user", text="hello")
        assert rec.role == "user"
        assert rec.text == "hello"
        assert rec.timestamp > 0
        assert rec.tool_name is None
        assert rec.metadata == {}

    def test_tool_call_record(self):
        rec = MessageRecord(role="tool_call", text="running", tool_name="bash")
        assert rec.tool_name == "bash"

    def test_metadata(self):
        rec = MessageRecord(role="assistant", text="hi", metadata={"key": "val"})
        assert rec.metadata["key"] == "val"


# ---------------------------------------------------------------------------
# ConversationLog
# ---------------------------------------------------------------------------


class TestConversationLog:
    def test_append_and_len(self):
        log = ConversationLog()
        assert len(log.records) == 0
        log.append(MessageRecord(role="user", text="a"))
        assert len(log.records) == 1

    def test_clear(self):
        log = ConversationLog()
        log.append(MessageRecord(role="user", text="a"))
        log.clear()
        assert len(log.records) == 0

    def test_to_markdown_user(self):
        log = ConversationLog()
        log.append(MessageRecord(role="user", text="hello"))
        md = log.to_markdown()
        assert "**User:** hello" in md

    def test_to_markdown_assistant(self):
        log = ConversationLog()
        log.append(MessageRecord(role="assistant", text="world"))
        md = log.to_markdown()
        assert "**Assistant:** world" in md

    def test_to_markdown_system(self):
        log = ConversationLog()
        log.append(MessageRecord(role="system", text="info"))
        md = log.to_markdown()
        assert "*System: info*" in md

    def test_to_markdown_error(self):
        log = ConversationLog()
        log.append(MessageRecord(role="error", text="oops"))
        md = log.to_markdown()
        assert "*Error: oops*" in md

    def test_to_markdown_tool_call(self):
        log = ConversationLog()
        log.append(MessageRecord(role="tool_call", text="ls .", tool_name="bash"))
        md = log.to_markdown()
        assert "$ **bash**" in md
        assert "ls ." in md

    def test_to_markdown_tool_result(self):
        log = ConversationLog()
        log.append(MessageRecord(role="tool_result", text="done", tool_name="bash"))
        md = log.to_markdown()
        assert "-> bash" in md
        assert "done" in md

    def test_to_markdown_diff(self):
        log = ConversationLog()
        log.append(MessageRecord(role="diff", text="+added\n-removed"))
        md = log.to_markdown()
        assert "```diff" in md
        assert "+added" in md
        assert "-removed" in md

    def test_to_markdown_queued(self):
        log = ConversationLog()
        log.append(MessageRecord(role="queued", text="later"))
        md = log.to_markdown()
        assert "*[queued] later*" in md

    def test_to_markdown_approval(self):
        log = ConversationLog()
        log.append(MessageRecord(role="approval", text="tool xyz"))
        md = log.to_markdown()
        assert "*Approval: tool xyz*" in md

    def test_to_markdown_empty_log(self):
        log = ConversationLog()
        md = log.to_markdown()
        assert md == ""

    def test_to_markdown_multi_message(self):
        log = ConversationLog()
        log.append(MessageRecord(role="user", text="question"))
        log.append(MessageRecord(role="assistant", text="answer"))
        md = log.to_markdown()
        assert "**User:** question" in md
        assert "**Assistant:** answer" in md
        # User comes before assistant.
        assert md.index("User") < md.index("Assistant")


# ---------------------------------------------------------------------------
# AgentProfile
# ---------------------------------------------------------------------------


class TestAgentProfile:
    def test_defaults(self):
        p = AgentProfile(name="test")
        assert p.name == "test"
        assert p.description == ""
        assert p.mode == "primary"
        assert p.model is None
        assert p.prompt is None
        assert p.color is None
        assert p.hidden is False

    def test_custom_fields(self):
        p = AgentProfile(
            name="plan",
            description="Planner",
            mode="subagent",
            model="gpt-4",
            prompt="Be helpful",
            color="#abc",
            hidden=True,
        )
        assert p.mode == "subagent"
        assert p.model == "gpt-4"
        assert p.hidden is True


# ---------------------------------------------------------------------------
# BUILTIN_AGENTS
# ---------------------------------------------------------------------------


class TestBuiltinAgents:
    def test_has_at_least_two(self):
        assert len(BUILTIN_AGENTS) >= 2

    def test_build_agent_exists(self):
        names = [a.name for a in BUILTIN_AGENTS]
        assert "build" in names

    def test_plan_agent_exists(self):
        names = [a.name for a in BUILTIN_AGENTS]
        assert "plan" in names

    def test_all_primary(self):
        for a in BUILTIN_AGENTS:
            assert a.mode == "primary"

    def test_default_agent_name(self):
        assert DEFAULT_AGENT_NAME == "build"


# ---------------------------------------------------------------------------
# AgentRegistry
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    def test_default_has_builtins(self):
        reg = AgentRegistry()
        assert reg.get("build") is not None
        assert reg.get("plan") is not None

    def test_get_unknown_returns_none(self):
        reg = AgentRegistry()
        assert reg.get("nonexistent") is None

    def test_primary_agents(self):
        reg = AgentRegistry()
        primaries = reg.primary_agents()
        names = [p.name for p in primaries]
        assert "build" in names
        assert "plan" in names

    def test_subagents_empty_by_default(self):
        reg = AgentRegistry()
        assert reg.subagents() == []

    def test_all_visible(self):
        reg = AgentRegistry()
        visible = reg.all_visible()
        assert len(visible) >= 2

    def test_hidden_excluded(self):
        reg = AgentRegistry()
        reg.add(AgentProfile(name="secret", hidden=True))
        visible = reg.all_visible()
        names = [p.name for p in visible]
        assert "secret" not in names

    def test_add_new_profile(self):
        reg = AgentRegistry()
        reg.add(AgentProfile(name="new_agent", mode="subagent"))
        profile = reg.get("new_agent")
        assert profile is not None
        assert profile.mode == "subagent"

    def test_add_replaces_existing(self):
        reg = AgentRegistry()
        reg.add(AgentProfile(name="build", description="Updated"))
        profile = reg.get("build")
        assert profile is not None
        assert profile.description == "Updated"

    def test_cycle_next_wraps(self):
        reg = AgentRegistry()
        # build -> plan
        nxt = reg.cycle_next("build")
        assert nxt is not None
        assert nxt.name == "plan"
        # plan -> build (wraps)
        nxt2 = reg.cycle_next("plan")
        assert nxt2 is not None
        assert nxt2.name == "build"

    def test_cycle_prev_wraps(self):
        reg = AgentRegistry()
        # build -> plan (wraps backward)
        prev = reg.cycle_prev("build")
        assert prev is not None
        assert prev.name == "plan"
        # plan -> build
        prev2 = reg.cycle_prev("plan")
        assert prev2 is not None
        assert prev2.name == "build"

    def test_cycle_next_single_agent_returns_none(self):
        reg = AgentRegistry(profiles=[AgentProfile(name="solo")])
        assert reg.cycle_next("solo") is None

    def test_cycle_prev_single_agent_returns_none(self):
        reg = AgentRegistry(profiles=[AgentProfile(name="solo")])
        assert reg.cycle_prev("solo") is None

    def test_cycle_next_unknown_name_returns_first(self):
        reg = AgentRegistry()
        nxt = reg.cycle_next("unknown")
        assert nxt is not None
        assert nxt.name == reg.primary_agents()[0].name

    def test_cycle_prev_unknown_name_returns_last(self):
        reg = AgentRegistry()
        prev = reg.cycle_prev("unknown")
        assert prev is not None
        assert prev.name == reg.primary_agents()[-1].name

    def test_custom_profiles(self):
        profiles = [
            AgentProfile(name="a", mode="primary"),
            AgentProfile(name="b", mode="primary"),
            AgentProfile(name="c", mode="subagent"),
        ]
        reg = AgentRegistry(profiles=profiles)
        assert len(reg.primary_agents()) == 2
        assert len(reg.subagents()) == 1
        nxt = reg.cycle_next("a")
        assert nxt is not None
        assert nxt.name == "b"
