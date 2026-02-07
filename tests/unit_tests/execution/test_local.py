"""Tests for local execution module."""

from __future__ import annotations

from adk_deepagents.execution.local import _execute_local, create_local_execute_tool


class TestExecuteLocal:
    def test_simple_echo(self):
        result = _execute_local("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.output
        assert result.truncated is False

    def test_exit_code_nonzero(self):
        result = _execute_local("exit 1")
        assert result.exit_code == 1

    def test_stderr_captured(self):
        result = _execute_local("echo error >&2")
        assert "error" in result.output

    def test_timeout(self):
        result = _execute_local("sleep 10", timeout=0.1)
        assert result.exit_code == -1
        assert "timed out" in result.output

    def test_truncation(self):
        # Generate output larger than max
        result = _execute_local("python3 -c \"print('x' * 200_000)\"", max_output_bytes=100)
        assert result.truncated is True
        assert len(result.output) <= 100


class TestCreateLocalExecuteTool:
    def test_creates_callable(self):
        tool = create_local_execute_tool()
        assert callable(tool)
        assert tool.__name__ == "execute"

    def test_tool_returns_dict(self):
        tool = create_local_execute_tool()
        result = tool("echo test")
        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert "test" in result["output"]
        assert result["exit_code"] == 0

    def test_tool_error_status(self):
        tool = create_local_execute_tool()
        result = tool("exit 42")
        assert result["status"] == "error"
        assert result["exit_code"] == 42
