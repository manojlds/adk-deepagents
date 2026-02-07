"""Tests for HeimdallScriptExecutor bridge."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from adk_deepagents.execution.bridge import HeimdallScriptExecutor, _normalize_result


def _make_mock_tool(name: str, return_value: dict | None = None) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.run_async = AsyncMock(return_value=return_value or {"output": "ok", "exit_code": 0})
    return tool


# ---------------------------------------------------------------------------
# HeimdallScriptExecutor
# ---------------------------------------------------------------------------


class TestHeimdallScriptExecutor:
    def test_init_finds_tools(self):
        tools = [
            _make_mock_tool("execute_python"),
            _make_mock_tool("execute_bash"),
            _make_mock_tool("other_tool"),
        ]
        executor = HeimdallScriptExecutor(tools)
        assert executor.has_python
        assert executor.has_bash

    def test_init_missing_tools(self):
        executor = HeimdallScriptExecutor([])
        assert not executor.has_python
        assert not executor.has_bash

    def test_detect_language_python(self):
        executor = HeimdallScriptExecutor([])
        assert executor._detect_language("script.py") == "python"
        assert executor._detect_language("path/to/SCRIPT.PY") == "python"

    def test_detect_language_bash(self):
        executor = HeimdallScriptExecutor([])
        assert executor._detect_language("run.sh") == "bash"
        assert executor._detect_language("run.bash") == "bash"

    def test_detect_language_unknown(self):
        executor = HeimdallScriptExecutor([])
        assert executor._detect_language("script.rb") == "bash"  # default

    async def test_execute_python(self):
        py_tool = _make_mock_tool("execute_python", {"output": "42", "exit_code": 0})
        executor = HeimdallScriptExecutor([py_tool])
        result = await executor.execute("calc.py", "print(42)")
        assert result["status"] == "success"
        assert result["output"] == "42"
        py_tool.run_async.assert_awaited_once()

    async def test_execute_bash(self):
        bash_tool = _make_mock_tool("execute_bash", {"output": "hello", "exit_code": 0})
        executor = HeimdallScriptExecutor([bash_tool])
        result = await executor.execute("run.sh", "echo hello")
        assert result["status"] == "success"
        bash_tool.run_async.assert_awaited_once()

    async def test_execute_python_not_available(self):
        executor = HeimdallScriptExecutor([])
        result = await executor.execute("script.py", "print(1)")
        assert result["status"] == "error"
        assert "not available" in result["output"]

    async def test_execute_bash_not_available(self):
        executor = HeimdallScriptExecutor([])
        result = await executor.execute("run.sh", "echo hi")
        assert result["status"] == "error"
        assert "not available" in result["output"]

    async def test_execute_python_with_timeout(self):
        py_tool = _make_mock_tool("execute_python", {"output": "", "exit_code": 0})
        executor = HeimdallScriptExecutor([py_tool])
        await executor.execute("calc.py", "import time; time.sleep(1)", timeout=5)
        call_kwargs = py_tool.run_async.call_args.kwargs
        assert call_kwargs["timeout"] == 5

    async def test_execute_python_exception(self):
        py_tool = _make_mock_tool("execute_python")
        py_tool.run_async.side_effect = RuntimeError("sandbox error")
        executor = HeimdallScriptExecutor([py_tool])
        result = await executor.execute("calc.py", "print(1)")
        assert result["status"] == "error"
        assert "sandbox error" in result["output"]

    async def test_execute_error_exit_code(self):
        py_tool = _make_mock_tool("execute_python", {"output": "error", "exit_code": 1})
        executor = HeimdallScriptExecutor([py_tool])
        result = await executor.execute("calc.py", "raise Exception()")
        assert result["status"] == "error"
        assert result["exit_code"] == 1


# ---------------------------------------------------------------------------
# _normalize_result
# ---------------------------------------------------------------------------


class TestNormalizeResult:
    def test_dict_success(self):
        result = _normalize_result({"output": "ok", "exit_code": 0})
        assert result["status"] == "success"
        assert result["output"] == "ok"

    def test_dict_error(self):
        result = _normalize_result({"output": "fail", "exit_code": 1})
        assert result["status"] == "error"

    def test_dict_with_stdout(self):
        result = _normalize_result({"stdout": "output data", "exit_code": 0})
        assert result["output"] == "output data"

    def test_string_result(self):
        result = _normalize_result("plain text output")
        assert result["status"] == "success"
        assert result["output"] == "plain text output"

    def test_none_result(self):
        result = _normalize_result(None)
        assert result["status"] == "success"
        assert result["output"] == "None"
