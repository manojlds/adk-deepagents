"""Tests for browser automation integration."""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from adk_deepagents.browser.playwright_mcp import (
    PLAYWRIGHT_CORE_TOOL_NAMES,
    PLAYWRIGHT_VISION_TOOL_NAMES,
    _allowed_tool_names,
    _build_server_args,
)
from adk_deepagents.browser.prompts import BROWSER_SYSTEM_PROMPT
from adk_deepagents.graph import create_deep_agent
from adk_deepagents.types import BrowserConfig


class TestBrowserConfig:
    def test_default_config(self):
        config = BrowserConfig()
        assert config.provider == "playwright"
        assert config.headless is True
        assert config.browser == "chromium"
        assert config.viewport == (1280, 720)
        assert config.caps == []
        assert config.cdp_endpoint is None
        assert config.storage_state is None

    def test_custom_config(self):
        config = BrowserConfig(
            headless=False,
            browser="firefox",
            viewport=(1920, 1080),
            caps=["vision", "pdf"],
            cdp_endpoint="ws://localhost:9222",
            storage_state="./auth.json",
        )
        assert config.headless is False
        assert config.browser == "firefox"
        assert config.viewport == (1920, 1080)
        assert config.caps == ["vision", "pdf"]
        assert config.cdp_endpoint == "ws://localhost:9222"
        assert config.storage_state == "./auth.json"


class TestBuildServerArgs:
    def test_default_args(self):
        config = BrowserConfig()
        args = _build_server_args(config)
        assert args[0] == "@playwright/mcp@latest"
        assert "--headless" in args
        assert "--viewport-size" in args
        assert "1280x720" in args

    def test_non_default_browser(self):
        config = BrowserConfig(browser="firefox")
        args = _build_server_args(config)
        assert "--browser" in args
        idx = args.index("--browser")
        assert args[idx + 1] == "firefox"

    def test_chromium_not_in_args(self):
        config = BrowserConfig(browser="chromium")
        args = _build_server_args(config)
        assert "--browser" not in args

    def test_headed_mode(self):
        config = BrowserConfig(headless=False)
        args = _build_server_args(config)
        assert "--headless" not in args

    def test_caps_included(self):
        config = BrowserConfig(caps=["vision", "pdf"])
        args = _build_server_args(config)
        assert "--caps" in args
        idx = args.index("--caps")
        assert args[idx + 1] == "vision,pdf"

    def test_cdp_endpoint(self):
        config = BrowserConfig(cdp_endpoint="ws://localhost:9222")
        args = _build_server_args(config)
        assert "--cdp-endpoint" in args
        idx = args.index("--cdp-endpoint")
        assert args[idx + 1] == "ws://localhost:9222"

    def test_storage_state(self):
        config = BrowserConfig(storage_state="./auth.json")
        args = _build_server_args(config)
        assert "--storage-state" in args
        idx = args.index("--storage-state")
        assert args[idx + 1] == "./auth.json"


class TestAllowedToolNames:
    def test_default_core_only(self):
        config = BrowserConfig()
        allowed = _allowed_tool_names(config)
        assert allowed == PLAYWRIGHT_CORE_TOOL_NAMES

    def test_vision_caps(self):
        config = BrowserConfig(caps=["vision"])
        allowed = _allowed_tool_names(config)
        assert PLAYWRIGHT_CORE_TOOL_NAMES.issubset(allowed)
        assert PLAYWRIGHT_VISION_TOOL_NAMES.issubset(allowed)

    def test_testing_caps(self):
        config = BrowserConfig(caps=["testing"])
        allowed = _allowed_tool_names(config)
        assert "browser_verify_element_visible" in allowed
        assert "browser_generate_locator" in allowed


class TestCreateDeepAgentBrowser:
    def test_browser_warns_sync(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            create_deep_agent(browser="playwright")
            assert len(w) == 1
            assert "async MCP tool resolution" in str(w[0].message)

    def test_browser_config_warns_sync(self):
        config = BrowserConfig(headless=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            create_deep_agent(browser=config)
            assert len(w) == 1
            assert "async MCP tool resolution" in str(w[0].message)

    def test_browser_prompt_injected(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            agent = create_deep_agent(browser="playwright")
        assert BROWSER_SYSTEM_PROMPT in agent.instruction

    def test_no_browser_no_prompt(self):
        agent = create_deep_agent()
        assert BROWSER_SYSTEM_PROMPT not in agent.instruction

    def test_resolved_browser_injects_prompt(self):
        """When browser='_resolved', prompt is injected but no warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agent = create_deep_agent(browser="_resolved")
            # _resolved should not warn
            browser_warnings = [x for x in w if "browser" in str(x.message).lower()]
            assert len(browser_warnings) == 0
        # But prompt should still be injected
        assert BROWSER_SYSTEM_PROMPT in agent.instruction


class TestGetPlaywrightBrowserTools:
    @pytest.mark.asyncio
    async def test_import_error_without_mcp(self):
        with patch.dict("sys.modules", {"google.adk.tools.mcp_tool.mcp_toolset": None}):
            from adk_deepagents.browser.playwright_mcp import get_playwright_browser_tools

            with pytest.raises(ImportError, match="MCP toolset not available"):
                await get_playwright_browser_tools()

    def test_filter_logic(self):
        """Verify that the tool filter correctly includes/excludes tools."""
        config = BrowserConfig()
        allowed = _allowed_tool_names(config)
        assert "browser_navigate" in allowed
        assert "browser_snapshot" in allowed
        assert "unknown_tool" not in allowed
