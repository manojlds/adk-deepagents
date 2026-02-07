"""Tests for skills integration."""

from unittest.mock import MagicMock, patch

from adk_deepagents.skills.integration import add_skills_tools, inject_skills_into_prompt


class TestAddSkillsTools:
    def test_returns_tools_unchanged_when_import_fails(self):
        """When adk-skills-agent is not installed, tools are returned as-is."""
        original_tools = [lambda: None]
        with patch.dict("sys.modules", {"adk_skills_agent": None}):
            result = add_skills_tools(list(original_tools), skills_dirs=["/skills"])
        # Should return the original tools unchanged (ImportError caught)
        assert len(result) == len(original_tools)

    def test_stores_registry_in_state(self):
        """When state is provided, registry and metadata are stored."""
        mock_registry = MagicMock()
        mock_registry.list_metadata.return_value = [
            {"name": "test-skill", "description": "A test skill"}
        ]
        mock_registry.create_use_skill_tool.return_value = lambda: "use_skill"
        mock_registry.create_run_script_tool.return_value = lambda: "run_script"
        mock_registry.create_read_reference_tool.return_value = lambda: "read_reference"

        mock_module = MagicMock()
        mock_module.SkillsRegistry.return_value = mock_registry

        state: dict = {}
        tools: list = []

        with patch.dict("sys.modules", {"adk_skills_agent": mock_module}):
            result = add_skills_tools(
                tools,
                skills_dirs=["/skills"],
                state=state,
            )

        assert state.get("_skills_registry") is mock_registry
        assert state.get("skills_metadata") == [
            {"name": "test-skill", "description": "A test skill"}
        ]
        # Should have added 3 tools
        assert len(result) == 3

    def test_passes_config_to_registry(self):
        """SkillsConfig.extra is forwarded to SkillsRegistry constructor."""
        from adk_deepagents.types import SkillsConfig

        mock_registry = MagicMock()
        mock_registry.create_use_skill_tool.return_value = lambda: None
        mock_registry.create_run_script_tool.return_value = lambda: None
        mock_registry.create_read_reference_tool.return_value = lambda: None

        mock_module = MagicMock()
        mock_module.SkillsRegistry.return_value = mock_registry

        config = SkillsConfig(extra={"db_url": "sqlite:///skills.db"})

        with patch.dict("sys.modules", {"adk_skills_agent": mock_module}):
            add_skills_tools(
                [],
                skills_dirs=["/skills"],
                skills_config=config,
            )

        mock_module.SkillsRegistry.assert_called_once_with(db_url="sqlite:///skills.db")

    def test_handles_discovery_error_gracefully(self):
        """If one directory fails, others are still tried."""
        mock_registry = MagicMock()
        call_count = 0

        def mock_discover(directory):
            nonlocal call_count
            call_count += 1
            if directory == "/bad":
                raise RuntimeError("Discovery failed")

        mock_registry.discover = mock_discover
        mock_registry.create_use_skill_tool.return_value = lambda: None
        mock_registry.create_run_script_tool.return_value = lambda: None
        mock_registry.create_read_reference_tool.return_value = lambda: None

        mock_module = MagicMock()
        mock_module.SkillsRegistry.return_value = mock_registry

        with patch.dict("sys.modules", {"adk_skills_agent": mock_module}):
            result = add_skills_tools(
                [],
                skills_dirs=["/bad", "/good"],
            )

        assert call_count == 2
        # /good succeeded, so tools should be added
        assert len(result) == 3


class TestInjectSkillsIntoPrompt:
    def test_returns_original_when_no_registry(self):
        result = inject_skills_into_prompt("Hello", {})
        assert result == "Hello"

    def test_calls_registry_inject(self):
        mock_registry = MagicMock()
        mock_registry.inject_skills_prompt.return_value = (
            "Hello\n<available_skills>...</available_skills>"
        )

        state = {"_skills_registry": mock_registry}
        result = inject_skills_into_prompt("Hello", state, format="xml")

        mock_registry.inject_skills_prompt.assert_called_once_with("Hello", format="xml")
        assert "<available_skills>" in result

    def test_handles_error_gracefully(self):
        mock_registry = MagicMock()
        mock_registry.inject_skills_prompt.side_effect = RuntimeError("fail")

        state = {"_skills_registry": mock_registry}
        result = inject_skills_into_prompt("Hello", state)
        assert result == "Hello"
