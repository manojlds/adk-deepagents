"""Integration tests — local execution tool.

Verifies local shell execution: echo, failing commands, timeouts,
output truncation, and tool creation.  No API key required.
"""

from __future__ import annotations

import pytest

from adk_deepagents.execution.local import _execute_local, create_local_execute_tool

pytestmark = pytest.mark.integration


class TestExecuteLocal:
    def test_execute_echo(self):
        result = _execute_local("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.output

    def test_execute_failing_command(self):
        result = _execute_local("exit 1")
        assert result.exit_code != 0

    def test_execute_timeout(self):
        result = _execute_local("sleep 10", timeout=0.1)
        assert result.exit_code == -1
        assert "timed out" in result.output.lower()

    def test_execute_output_truncation(self):
        # Generate output larger than max_output_bytes
        result = _execute_local(
            "python3 -c \"print('x' * 200_000)\"",
            max_output_bytes=1000,
        )
        assert result.truncated is True
        assert len(result.output) <= 1000


class TestCreateLocalExecuteTool:
    def test_create_local_execute_tool_returns_callable(self):
        tool = create_local_execute_tool()
        assert callable(tool)
        assert getattr(tool, "__name__", None) == "execute"

    def test_tool_invocation(self):
        tool = create_local_execute_tool()
        result = tool("echo integration_test")
        assert result["status"] == "success"
        assert result["exit_code"] == 0
        assert "integration_test" in result["output"]
