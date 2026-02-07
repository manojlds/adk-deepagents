"""Local shell execution fallback.

Provides a local ``execute`` tool using ``subprocess.run()``.
Less secure than Heimdall MCP — intended for development/testing only.
"""

from __future__ import annotations

import subprocess
from typing import Callable

from adk_deepagents.backends.protocol import ExecuteResponse


def _execute_local(
    command: str,
    *,
    timeout: float = 120.0,
    max_output_bytes: int = 100_000,
) -> ExecuteResponse:
    """Run a shell command locally and return the result."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        truncated = False
        if len(output) > max_output_bytes:
            output = output[:max_output_bytes]
            truncated = True
        return ExecuteResponse(
            output=output,
            exit_code=result.returncode,
            truncated=truncated,
        )
    except subprocess.TimeoutExpired:
        return ExecuteResponse(
            output=f"Command timed out after {timeout}s",
            exit_code=-1,
            truncated=False,
        )
    except Exception as e:
        return ExecuteResponse(
            output=f"Error executing command: {e}",
            exit_code=-1,
            truncated=False,
        )


def create_local_execute_tool() -> Callable:
    """Create a local ``execute`` tool function for use with ADK."""

    def execute(command: str) -> dict:
        """Execute a shell command locally and return the output.

        WARNING: This runs commands directly on the host system with no
        sandboxing. Use Heimdall MCP for production workloads.

        Args:
            command: The shell command to execute.
        """
        result = _execute_local(command)
        return {
            "status": "success" if result.exit_code == 0 else "error",
            "output": result.output,
            "exit_code": result.exit_code,
            "truncated": result.truncated,
        }

    return execute
