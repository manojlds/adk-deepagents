# Code Execution

## Overview

adk-deepagents supports two code execution backends:

- **Local** — Runs shell commands via `subprocess.run()` on the host system. Simple but unsandboxed.
- **Heimdall MCP** — Sandboxed execution via the Heimdall MCP server. Python runs in Pyodide (WebAssembly) and Bash runs in just-bash.

The execution backend is selected via the `execution` parameter on `create_deep_agent`.

## Local Execution

### create_local_execute_tool

Creates a local `execute` tool function using `subprocess.run()` with `shell=True`:

```python
from adk_deepagents.execution.local import create_local_execute_tool

execute = create_local_execute_tool()
result = execute("echo hello")
# {"status": "success", "output": "hello\n", "exit_code": 0, "truncated": False}
```

**Defaults:**

| Setting | Value |
|---|---|
| Timeout | 120 seconds |
| Max output | 100,000 bytes (~100 KB) |
| Shell | `True` (runs through system shell) |

If output exceeds `max_output_bytes`, it is truncated and `truncated` is set to `True`. Timed-out commands return `exit_code: -1`.

### Security Warning

> ⚠️ **Local execution runs commands directly on the host system with no sandboxing.** Use Heimdall MCP for production workloads or any scenario involving untrusted input.

### Example

```python
from adk_deepagents import create_deep_agent

agent = create_deep_agent(
    instruction="You are a coding assistant. Use execute to run commands.",
    execution="local",
)
```

## Heimdall MCP Execution

### What Heimdall Provides

[Heimdall](https://www.npmjs.com/package/@heimdall-ai/heimdall) is an MCP server that provides sandboxed execution:

| Tool | Description |
|---|---|
| `execute_python` | Sandboxed Python via Pyodide WebAssembly |
| `execute_bash` | Bash command simulation via just-bash |
| `install_packages` | Python package installation via micropip |
| `write_file` | Write to the virtual workspace filesystem |
| `read_file` | Read from the virtual workspace filesystem |
| `list_files` | List files in the virtual workspace |
| `delete_file` | Delete files from the virtual workspace |

### get_heimdall_tools

Connects to a Heimdall MCP server via stdio transport and returns its tools:

```python
from adk_deepagents.execution.heimdall import get_heimdall_tools

tools, cleanup = await get_heimdall_tools(
    workspace_path="/workspace",
    command="npx",
    args=["@heimdall-ai/heimdall"],
    env=None,
    filter_tools=True,
)

# Use tools with create_deep_agent...

# When done, clean up the MCP connection:
await cleanup()
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `workspace_path` | `"/workspace"` | Path to the Heimdall workspace directory |
| `command` | `"npx"` | Command to start the MCP server |
| `args` | `["@heimdall-ai/heimdall"]` | Arguments for the MCP server command |
| `env` | `None` | Additional environment variables |
| `filter_tools` | `True` | Only include execution and workspace tools |

Returns `(tools, cleanup)` where `cleanup` is an async function.

### get_heimdall_tools_from_config

Connects using a custom configuration dict, supporting both stdio and SSE transports:

```python
from adk_deepagents.execution.heimdall import get_heimdall_tools_from_config

# Stdio transport
tools, cleanup = await get_heimdall_tools_from_config({
    "command": "npx",
    "args": ["@heimdall-ai/heimdall"],
    "env": {"HEIMDALL_WORKSPACE": "/my/workspace"},
})

# SSE transport
tools, cleanup = await get_heimdall_tools_from_config({
    "uri": "http://localhost:3000/sse",
})
```

### Tool Filtering

By default, `get_heimdall_tools` filters tools to only include known execution and workspace tools:

```python
HEIMDALL_TOOL_NAMES = {"execute_python", "execute_bash", "install_packages"}
HEIMDALL_WORKSPACE_TOOL_NAMES = {"write_file", "read_file", "list_files", "delete_file"}
```

Set `filter_tools=False` to include all tools from the server.

### create_deep_agent_async

A convenience wrapper that resolves MCP tools and creates the agent in one step:

```python
from adk_deepagents import create_deep_agent_async

agent, cleanup = await create_deep_agent_async(
    instruction="You are a sandboxed coding assistant.",
    execution="heimdall",
)

# Use the agent...

# Clean up when done
if cleanup:
    await cleanup()
```

Returns `(agent, cleanup_fn)` where `cleanup_fn` must be awaited when the agent is no longer needed.

### Cleanup Lifecycle

The MCP connection cleanup must be called to properly close the server connection:

```python
agent, cleanup = await create_deep_agent_async(execution="heimdall")

try:
    # Run agent sessions...
    pass
finally:
    if cleanup:
        await cleanup()
```

### Examples

#### Basic Heimdall Usage

```python
from adk_deepagents import create_deep_agent_async

async def main():
    agent, cleanup = await create_deep_agent_async(
        instruction="Write and execute Python code.",
        execution="heimdall",
    )
    try:
        # Use agent with a runner...
        pass
    finally:
        if cleanup:
            await cleanup()
```

#### Custom Workspace Path

```python
from adk_deepagents.execution.heimdall import get_heimdall_tools
from adk_deepagents import create_deep_agent

async def main():
    tools, cleanup = await get_heimdall_tools(
        workspace_path="/my/project",
    )
    try:
        agent = create_deep_agent(
            tools=tools,
            execution="_resolved",  # Signal that execution tools are already resolved
        )
        # Use agent...
    finally:
        await cleanup()
```

#### SSE Transport

```python
from adk_deepagents import create_deep_agent_async

async def main():
    agent, cleanup = await create_deep_agent_async(
        execution={"uri": "http://localhost:3000/sse"},
    )
    try:
        # Use agent...
        pass
    finally:
        if cleanup:
            await cleanup()
```

## HeimdallScriptExecutor Bridge

The `HeimdallScriptExecutor` bridges adk-skills `run_script` execution through Heimdall for sandboxed script execution.

### How It Works

1. Takes a list of Heimdall MCP tools and extracts `execute_python` and `execute_bash`
2. Detects the script language from the file extension:
   - `.py` → Python (routes to `execute_python`)
   - `.sh`, `.bash` → Bash (routes to `execute_bash`)
   - Unknown → defaults to Bash
3. Delegates execution to the appropriate Heimdall tool

### Usage

```python
from adk_deepagents.execution.bridge import HeimdallScriptExecutor
from adk_deepagents.execution.heimdall import get_heimdall_tools

# Connect to Heimdall
tools, cleanup = await get_heimdall_tools()

# Create the executor
executor = HeimdallScriptExecutor(tools)

# Execute a Python script
result = await executor.execute("script.py", "print('hello world')")
# {"status": "success", "output": "hello world\n", "exit_code": 0}

# Execute a Bash script
result = await executor.execute("setup.sh", "echo 'setting up...'")
# {"status": "success", "output": "setting up...\n", "exit_code": 0}

# Check availability
executor.has_python  # True
executor.has_bash    # True
```

## Execution Parameter Values

The `execution` parameter on `create_deep_agent` accepts these values:

| Value | Description |
|---|---|
| `"local"` | Local subprocess execution via `subprocess.run()` |
| `"heimdall"` | Heimdall MCP (requires `create_deep_agent_async()`) |
| `dict` | Custom MCP config (requires `create_deep_agent_async()`) |
| `"_resolved"` | Internal: signals execution tools are already in the tools list |
| `None` | No execution tools (default) |

Using `"heimdall"` or a dict config with the synchronous `create_deep_agent()` will emit a warning directing you to use `create_deep_agent_async()` or pre-resolve MCP tools.
