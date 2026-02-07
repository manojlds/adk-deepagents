"""Bridge adk-skills run_script to Heimdall executor.

Routes adk-skills script execution through Heimdall MCP for sandboxed
execution. Python scripts are routed to ``execute_python`` and Bash
scripts to ``execute_bash``.

Usage::

    from adk_deepagents.execution.bridge import HeimdallScriptExecutor

    # After connecting to Heimdall
    tools, cleanup = await get_heimdall_tools()
    executor = HeimdallScriptExecutor(tools)

    # Execute a Python script
    result = await executor.execute("script.py", "print('hello')")
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class HeimdallScriptExecutor:
    """Executor that routes skill scripts through Heimdall MCP tools.

    This class bridges the adk-skills ``run_script`` tool with Heimdall's
    sandboxed execution. It looks up ``execute_python`` and ``execute_bash``
    tools from the Heimdall toolset and delegates script execution to them.

    Parameters
    ----------
    heimdall_tools:
        List of MCP tools from ``get_heimdall_tools()``.
    """

    def __init__(self, heimdall_tools: list[Any]) -> None:
        self._tools: dict[str, Any] = {}
        for tool in heimdall_tools:
            name = getattr(tool, "name", "")
            if name in ("execute_python", "execute_bash"):
                self._tools[name] = tool

        if "execute_python" not in self._tools:
            logger.warning("Heimdall execute_python tool not found")
        if "execute_bash" not in self._tools:
            logger.warning("Heimdall execute_bash tool not found")

    @property
    def has_python(self) -> bool:
        """Whether Python execution is available."""
        return "execute_python" in self._tools

    @property
    def has_bash(self) -> bool:
        """Whether Bash execution is available."""
        return "execute_bash" in self._tools

    def _detect_language(self, script_path: str) -> str:
        """Detect script language from file extension.

        Returns ``"python"`` or ``"bash"``. Defaults to ``"bash"``
        for unknown extensions.
        """
        lower = script_path.lower()
        if lower.endswith(".py"):
            return "python"
        if lower.endswith((".sh", ".bash")):
            return "bash"
        # Default to bash for unknown extensions
        return "bash"

    async def execute(
        self,
        script_path: str,
        script_content: str,
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Execute a script via Heimdall.

        Parameters
        ----------
        script_path:
            Path to the script file (used for language detection).
        script_content:
            The script source code to execute.
        timeout:
            Optional execution timeout in seconds.

        Returns
        -------
        dict
            Execution result with ``status``, ``output``, etc.
        """
        language = self._detect_language(script_path)

        if language == "python":
            return await self._execute_python(script_content, timeout=timeout)
        return await self._execute_bash(script_content, timeout=timeout)

    async def _execute_python(
        self,
        code: str,
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Execute Python code via Heimdall's execute_python tool."""
        tool = self._tools.get("execute_python")
        if tool is None:
            return {
                "status": "error",
                "output": "Heimdall execute_python tool not available",
                "exit_code": -1,
            }

        try:
            kwargs: dict[str, Any] = {"code": code}
            if timeout is not None:
                kwargs["timeout"] = timeout
            result = await tool.run_async(**kwargs)
            return _normalize_result(result)
        except Exception as e:
            logger.exception("Heimdall Python execution failed")
            return {
                "status": "error",
                "output": f"Execution failed: {e}",
                "exit_code": -1,
            }

    async def _execute_bash(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Execute Bash command via Heimdall's execute_bash tool."""
        tool = self._tools.get("execute_bash")
        if tool is None:
            return {
                "status": "error",
                "output": "Heimdall execute_bash tool not available",
                "exit_code": -1,
            }

        try:
            kwargs: dict[str, Any] = {"command": command}
            if timeout is not None:
                kwargs["timeout"] = timeout
            result = await tool.run_async(**kwargs)
            return _normalize_result(result)
        except Exception as e:
            logger.exception("Heimdall Bash execution failed")
            return {
                "status": "error",
                "output": f"Execution failed: {e}",
                "exit_code": -1,
            }


def _normalize_result(result: Any) -> dict[str, Any]:
    """Normalize an MCP tool result into a standard dict format."""
    if isinstance(result, dict):
        return {
            "status": "success" if result.get("exit_code", 0) == 0 else "error",
            "output": result.get("output", result.get("stdout", "")),
            "exit_code": result.get("exit_code", 0),
        }
    # Handle string or other result types
    return {
        "status": "success",
        "output": str(result),
        "exit_code": 0,
    }
