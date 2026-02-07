"""Tests for Heimdall MCP integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adk_deepagents.execution.heimdall import (
    HEIMDALL_TOOL_NAMES,
    HEIMDALL_WORKSPACE_TOOL_NAMES,
    get_heimdall_tools,
    get_heimdall_tools_from_config,
)


def _make_mock_tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    return tool


@pytest.fixture
def mock_mcp_toolset():
    """Mock the MCPToolset.from_server to avoid actual MCP connections."""
    tools = [
        _make_mock_tool("execute_python"),
        _make_mock_tool("execute_bash"),
        _make_mock_tool("install_packages"),
        _make_mock_tool("write_file"),
        _make_mock_tool("read_file"),
        _make_mock_tool("list_files"),
        _make_mock_tool("delete_file"),
        _make_mock_tool("some_other_tool"),
    ]
    exit_stack = AsyncMock()
    with patch(
        "adk_deepagents.execution.heimdall.MCPToolset",
        create=True,
    ) as mock:
        mock.from_server = AsyncMock(return_value=(tools, exit_stack))
        yield mock, tools, exit_stack


class TestGetHeimdallTools:
    async def test_returns_filtered_tools(self, mock_mcp_toolset):
        _mock, _tools, _exit_stack = mock_mcp_toolset

        with (
            patch(
                "adk_deepagents.execution.heimdall.MCPToolset",
                new=_mock,
            ),
            patch(
                "adk_deepagents.execution.heimdall.StdioServerParameters",
                create=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "google.adk.tools.mcp_tool.mcp_toolset": MagicMock(
                        MCPToolset=_mock,
                        StdioServerParameters=MagicMock,
                    ),
                },
            ),
        ):
            tools, cleanup = await get_heimdall_tools()

        # Should filter to only allowed tools
        tool_names = {getattr(t, "name", "") for t in tools}
        all_allowed = HEIMDALL_TOOL_NAMES | HEIMDALL_WORKSPACE_TOOL_NAMES
        assert tool_names <= all_allowed
        assert "some_other_tool" not in tool_names

    async def test_no_filter(self, mock_mcp_toolset):
        _mock, _tools, _exit_stack = mock_mcp_toolset

        with patch.dict(
            "sys.modules",
            {
                "google.adk.tools.mcp_tool.mcp_toolset": MagicMock(
                    MCPToolset=_mock,
                    StdioServerParameters=MagicMock,
                ),
            },
        ):
            tools, cleanup = await get_heimdall_tools(filter_tools=False)

        # Should include all tools
        assert len(tools) == 8


class TestGetHeimdallToolsFromConfig:
    async def test_stdio_config(self, mock_mcp_toolset):
        _mock, _tools, _exit_stack = mock_mcp_toolset

        with patch.dict(
            "sys.modules",
            {
                "google.adk.tools.mcp_tool.mcp_toolset": MagicMock(
                    MCPToolset=_mock,
                    StdioServerParameters=MagicMock,
                    SseServerParams=MagicMock,
                ),
            },
        ):
            tools, cleanup = await get_heimdall_tools_from_config(
                {
                    "command": "npx",
                    "args": ["@heimdall-ai/heimdall"],
                }
            )

        assert len(tools) > 0

    async def test_sse_config(self, mock_mcp_toolset):
        _mock, _tools, _exit_stack = mock_mcp_toolset

        with patch.dict(
            "sys.modules",
            {
                "google.adk.tools.mcp_tool.mcp_toolset": MagicMock(
                    MCPToolset=_mock,
                    StdioServerParameters=MagicMock,
                    SseServerParams=MagicMock,
                ),
            },
        ):
            tools, cleanup = await get_heimdall_tools_from_config(
                {
                    "uri": "http://localhost:3000/sse",
                }
            )

        assert len(tools) > 0


class TestToolNameConstants:
    def test_heimdall_tools_are_frozenset(self):
        assert isinstance(HEIMDALL_TOOL_NAMES, frozenset)
        assert "execute_python" in HEIMDALL_TOOL_NAMES
        assert "execute_bash" in HEIMDALL_TOOL_NAMES

    def test_workspace_tools_are_frozenset(self):
        assert isinstance(HEIMDALL_WORKSPACE_TOOL_NAMES, frozenset)
        assert "write_file" in HEIMDALL_WORKSPACE_TOOL_NAMES
