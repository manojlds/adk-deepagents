"""Harbor execution tool — proxies shell commands into the Harbor task container."""

from __future__ import annotations

from collections.abc import Callable

from harbor.environments.base import BaseEnvironment


def create_harbor_execute_tool(environment: BaseEnvironment) -> Callable:
    """Create an ``execute`` tool that runs commands in the Harbor container.

    Mirrors ``create_local_execute_tool()`` but routes through
    ``environment.exec()`` instead of a local subprocess.
    """

    async def execute(command: str) -> dict:
        """Execute a shell command in the task environment.

        Use this for running scripts, installing packages, or any shell
        operation that produces output or modifies the environment.

        Args:
            command: Shell command to run in the task container.
        """
        try:
            result = await environment.exec(command=command, timeout_sec=120)
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}" if output else f"STDERR:\n{result.stderr}"
            return {
                "status": "success",
                "output": output or "(no output)",
                "exit_code": 0,
            }
        except Exception as exc:
            return {
                "status": "error",
                "output": f"ERROR: {exc}",
                "exit_code": -1,
            }

    return execute
