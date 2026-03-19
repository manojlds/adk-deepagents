"""Tests for shared type definitions (types.py)."""

from __future__ import annotations

from adk_deepagents.types import (
    BrowserConfig,
    DynamicTaskConfig,
    SkillsConfig,
    SubAgentSpec,
    SummarizationConfig,
    TemporalTaskConfig,
    TruncateArgsConfig,
)

# ---------------------------------------------------------------------------
# SubAgentSpec (TypedDict)
# ---------------------------------------------------------------------------


class TestSubAgentSpec:
    def test_required_keys(self):
        assert SubAgentSpec.__required_keys__ == frozenset({"name", "description"})

    def test_minimal_creation(self):
        spec = SubAgentSpec(name="helper", description="Helps")
        assert spec["name"] == "helper"
        assert spec["description"] == "Helps"

    def test_optional_fields(self):
        spec = SubAgentSpec(
            name="helper",
            description="Helps",
            system_prompt="Be helpful.",
            model="gpt-4o",
            skills=["/skills"],
            interrupt_on={"write_file": True},
        )
        assert spec["system_prompt"] == "Be helpful."
        assert spec["model"] == "gpt-4o"
        assert spec["skills"] == ["/skills"]
        assert spec["interrupt_on"] == {"write_file": True}

    def test_is_dict(self):
        spec = SubAgentSpec(name="x", description="y")
        assert isinstance(spec, dict)

    def test_get_missing_optional_returns_none(self):
        spec = SubAgentSpec(name="x", description="y")
        assert spec.get("model") is None
        assert spec.get("system_prompt") is None


# ---------------------------------------------------------------------------
# TemporalTaskConfig
# ---------------------------------------------------------------------------


class TestTemporalTaskConfig:
    def test_defaults(self):
        cfg = TemporalTaskConfig()
        assert cfg.target_host == "localhost:7233"
        assert cfg.namespace == "default"
        assert cfg.task_queue == "adk-deepagents-tasks"
        assert cfg.workflow_id_prefix == "dynamic-task"
        assert cfg.activity_timeout_seconds is None
        assert cfg.retry_max_attempts == 1
        assert cfg.idle_timeout_seconds == 600.0

    def test_custom_values(self):
        cfg = TemporalTaskConfig(
            target_host="temporal.example.com:7233",
            namespace="production",
            task_queue="my-tasks",
            workflow_id_prefix="custom",
            activity_timeout_seconds=300.0,
            retry_max_attempts=3,
            idle_timeout_seconds=1200.0,
        )
        assert cfg.target_host == "temporal.example.com:7233"
        assert cfg.namespace == "production"
        assert cfg.task_queue == "my-tasks"
        assert cfg.workflow_id_prefix == "custom"
        assert cfg.activity_timeout_seconds == 300.0
        assert cfg.retry_max_attempts == 3
        assert cfg.idle_timeout_seconds == 1200.0


# ---------------------------------------------------------------------------
# DynamicTaskConfig
# ---------------------------------------------------------------------------


class TestDynamicTaskConfig:
    def test_defaults(self):
        cfg = DynamicTaskConfig()
        assert cfg.max_parallel == 4
        assert cfg.concurrency_policy == "error"
        assert cfg.queue_timeout_seconds == 30.0
        assert cfg.max_depth == 2
        assert cfg.timeout_seconds == 120.0
        assert cfg.allow_model_override is False
        assert cfg.temporal is None

    def test_custom_values(self):
        temporal = TemporalTaskConfig(target_host="example:7233")
        cfg = DynamicTaskConfig(
            max_parallel=8,
            concurrency_policy="wait",
            queue_timeout_seconds=60.0,
            max_depth=5,
            timeout_seconds=300.0,
            allow_model_override=True,
            temporal=temporal,
        )
        assert cfg.max_parallel == 8
        assert cfg.concurrency_policy == "wait"
        assert cfg.queue_timeout_seconds == 60.0
        assert cfg.max_depth == 5
        assert cfg.timeout_seconds == 300.0
        assert cfg.allow_model_override is True
        assert cfg.temporal is temporal

    def test_concurrency_policy_values(self):
        """Both valid concurrency policies are accepted."""
        cfg_error = DynamicTaskConfig(concurrency_policy="error")
        assert cfg_error.concurrency_policy == "error"

        cfg_wait = DynamicTaskConfig(concurrency_policy="wait")
        assert cfg_wait.concurrency_policy == "wait"


# ---------------------------------------------------------------------------
# TruncateArgsConfig
# ---------------------------------------------------------------------------


class TestTruncateArgsConfig:
    def test_defaults(self):
        cfg = TruncateArgsConfig()
        assert cfg.trigger is None
        assert cfg.keep == ("messages", 20)
        assert cfg.max_length == 2000
        assert cfg.truncation_text == "...(argument truncated)"

    def test_custom_values(self):
        cfg = TruncateArgsConfig(
            trigger=("fraction", 0.7),
            keep=("messages", 10),
            max_length=500,
            truncation_text="[truncated]",
        )
        assert cfg.trigger == ("fraction", 0.7)
        assert cfg.keep == ("messages", 10)
        assert cfg.max_length == 500
        assert cfg.truncation_text == "[truncated]"


# ---------------------------------------------------------------------------
# SummarizationConfig
# ---------------------------------------------------------------------------


class TestSummarizationConfig:
    def test_defaults(self):
        cfg = SummarizationConfig()
        assert cfg.model == "gemini-2.5-flash"
        assert cfg.trigger == ("fraction", 0.85)
        assert cfg.keep == ("messages", 6)
        assert cfg.history_path_prefix == "/conversation_history"
        assert cfg.use_llm_summary is True
        assert cfg.truncate_args is None
        assert cfg.context_window is None

    def test_custom_values(self):
        trunc = TruncateArgsConfig(max_length=1000)
        cfg = SummarizationConfig(
            model="gpt-4o",
            trigger=("fraction", 0.5),
            keep=("messages", 10),
            history_path_prefix="/history",
            use_llm_summary=False,
            truncate_args=trunc,
            context_window=128000,
        )
        assert cfg.model == "gpt-4o"
        assert cfg.trigger == ("fraction", 0.5)
        assert cfg.keep == ("messages", 10)
        assert cfg.history_path_prefix == "/history"
        assert cfg.use_llm_summary is False
        assert cfg.truncate_args is trunc
        assert cfg.context_window == 128000


# ---------------------------------------------------------------------------
# BrowserConfig
# ---------------------------------------------------------------------------


class TestBrowserConfig:
    def test_defaults(self):
        cfg = BrowserConfig()
        assert cfg.provider == "playwright"
        assert cfg.headless is True
        assert cfg.browser == "chromium"
        assert cfg.viewport == (1280, 720)
        assert cfg.caps == []
        assert cfg.cdp_endpoint is None
        assert cfg.storage_state is None

    def test_custom_values(self):
        cfg = BrowserConfig(
            headless=False,
            browser="firefox",
            viewport=(1920, 1080),
            caps=["vision", "pdf"],
            cdp_endpoint="ws://localhost:9222",
            storage_state="/tmp/auth.json",
        )
        assert cfg.headless is False
        assert cfg.browser == "firefox"
        assert cfg.viewport == (1920, 1080)
        assert cfg.caps == ["vision", "pdf"]
        assert cfg.cdp_endpoint == "ws://localhost:9222"
        assert cfg.storage_state == "/tmp/auth.json"

    def test_caps_default_factory_independence(self):
        """Default caps lists are independent per instance."""
        cfg1 = BrowserConfig()
        cfg2 = BrowserConfig()
        cfg1.caps.append("vision")
        assert cfg2.caps == []


# ---------------------------------------------------------------------------
# SkillsConfig
# ---------------------------------------------------------------------------


class TestSkillsConfig:
    def test_defaults(self):
        cfg = SkillsConfig()
        assert cfg.extra == {}

    def test_custom_extra(self):
        cfg = SkillsConfig(extra={"key": "value"})
        assert cfg.extra == {"key": "value"}

    def test_extra_default_factory_independence(self):
        """Default extra dicts are independent per instance."""
        cfg1 = SkillsConfig()
        cfg2 = SkillsConfig()
        cfg1.extra["key"] = "value"
        assert cfg2.extra == {}
