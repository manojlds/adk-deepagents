# Tools

Every agent created with `create_deep_agent()` includes a set of built-in tools for filesystem operations and task management. Additional tools are added when you enable execution or sub-agents.

## How Tools Work in ADK

Tools in Google ADK are plain Python functions registered on an `LlmAgent`. When the LLM decides to call a tool, ADK invokes the function with the LLM-provided arguments and an injected `ToolContext` object.

The `ToolContext` provides:

- **`tool_context.state`** — A dict backed by ADK's session state. This is how tools access the backend, persist files, and communicate with callbacks.
- **`tool_context.actions`** — Control flow actions (e.g., `skip_summarization`).
- **`tool_context.function_call_id`** — Unique ID for this tool invocation.
- **`tool_context.request_confirmation()`** — Pause execution for human approval (used by HITL).

## How Tools Resolve the Backend

All filesystem tools resolve the backend from `tool_context.state` using the `_get_backend()` helper:

```python
def _get_backend(tool_context: ToolContext) -> Backend:
    backend = tool_context.state.get("_backend")
    if backend is None:
        factory = tool_context.state.get("_backend_factory")
        if factory is not None:
            backend = factory(tool_context.state)
            tool_context.state["_backend"] = backend
    if backend is None:
        raise RuntimeError(
            "No backend configured. Set state['_backend'] or state['_backend_factory']."
        )
    return backend
```

Resolution order:

1. Check `state["_backend"]` — use cached backend if available
2. Call `state["_backend_factory"](state)` — create and cache the backend
3. Raise `RuntimeError` if neither is set

The `_backend_factory` is set in `state` by the `before_agent_callback` during agent initialization.

---

## Filesystem Tools

Defined in `adk_deepagents.tools.filesystem`. All paths must be absolute (start with `/`). Paths are validated against traversal attacks (`..`, `~`, Windows drive letters).

### `ls`

List files and directories at a given path.

```python
def ls(path: str, tool_context: ToolContext) -> dict:
```

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `path` | `str` | Yes | Absolute path to list (must start with `/`) |

**Returns:**

```python
# Success
{"status": "success", "entries": [
    {"path": "/src", "is_dir": True, "size": 0, "modified_at": "2025-01-15T10:30:00+00:00"},
    {"path": "/readme.md", "is_dir": False, "size": 142, "modified_at": "2025-01-15T10:30:00+00:00"},
]}

# Error (invalid path)
{"status": "error", "message": "Path traversal not allowed: ../etc/passwd"}
```

**Example usage by the LLM:**
```
Tool call: ls(path="/src")
```

---

### `read_file`

Read file contents with optional pagination. Image files (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`) are returned as base64-encoded multimodal content.

```python
def read_file(
    file_path: str,
    tool_context: ToolContext,
    offset: int = 0,
    limit: int = 2000,
) -> dict:
```

**Arguments:**

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `file_path` | `str` | Yes | — | Absolute path to the file |
| `offset` | `int` | No | `0` | Line number to start reading from (0-based) |
| `limit` | `int` | No | `2000` | Maximum number of lines to return |

**Returns:**

```python
# Success (text file)
{"status": "success", "content": "     1\timport os\n     2\timport sys\n     3\t..."}

# Success (image file)
{"status": "success", "content": {
    "type": "image",
    "media_type": "image/png",
    "data": "iVBORw0KGgo..."  # base64-encoded
}}

# Pagination indicator (appended to content)
# "... (150 more lines. Use offset=2000 to continue reading)"

# Error
{"status": "error", "message": "Error: file not found: /nonexistent.txt"}
```

Content is formatted with `cat -n` style line numbers. Long lines (> 5000 chars) are chunked with continuation markers (e.g., `5.1`, `5.2`).

**Example usage by the LLM:**
```
Tool call: read_file(file_path="/src/main.py")
Tool call: read_file(file_path="/src/main.py", offset=100, limit=50)
```

---

### `write_file`

Create a new file with the given content. **Cannot overwrite existing files** — use `edit_file` for modifications.

```python
def write_file(file_path: str, content: str, tool_context: ToolContext) -> dict:
```

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `file_path` | `str` | Yes | Absolute path for the new file |
| `content` | `str` | Yes | Content to write to the file |

**Returns:**

```python
# Success
{"status": "success", "path": "/src/new_file.py"}

# Error (file exists)
{"status": "error", "message": "File already exists: /src/main.py. Use edit_file to modify."}
```

For `StateBackend` and `StoreBackend`, the tool automatically merges the `files_update` dict from `WriteResult` into `tool_context.state["files"]`. For `FilesystemBackend`, the file is written directly to disk.

**Example usage by the LLM:**
```
Tool call: write_file(file_path="/src/utils.py", content="def add(a, b):\n    return a + b\n")
```

---

### `edit_file`

Edit a file by replacing a specific string with a new string. The `old_string` must uniquely identify the text to replace, unless `replace_all` is set to `True`.

```python
def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    tool_context: ToolContext,
    replace_all: bool = False,
) -> dict:
```

**Arguments:**

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `file_path` | `str` | Yes | — | Absolute path to the file |
| `old_string` | `str` | Yes | — | Exact text to find and replace |
| `new_string` | `str` | Yes | — | Replacement text |
| `replace_all` | `bool` | No | `False` | If `True`, replace all occurrences |

**Returns:**

```python
# Success
{"status": "success", "path": "/src/main.py", "occurrences": 1}

# Error (not found)
{"status": "error", "message": "old_string not found in file content"}

# Error (ambiguous)
{"status": "error", "message": "old_string appears 3 times. Provide more context to make it unique, or set replace_all=True."}

# Error (identical)
{"status": "error", "message": "old_string and new_string are identical"}
```

**Uniqueness check:** When `replace_all=False` (default), if `old_string` appears more than once in the file, the edit is rejected. The LLM must either provide more surrounding context to make the match unique, or set `replace_all=True`.

**Example usage by the LLM:**
```
Tool call: edit_file(
    file_path="/src/main.py",
    old_string="def hello():\n    print('hello')",
    new_string="def hello():\n    print('Hello, world!')"
)
```

---

### `glob`

Find files matching a glob pattern. Uses [wcmatch](https://facelessuser.github.io/wcmatch/) for glob matching with brace expansion (`{a,b}`) and globstar (`**`).

```python
def glob(pattern: str, tool_context: ToolContext, path: str = "/") -> dict:
```

**Arguments:**

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `pattern` | `str` | Yes | — | Glob pattern (e.g., `**/*.py`, `src/{a,b}/*.ts`) |
| `path` | `str` | No | `"/"` | Base directory to search from |

**Returns:**

```python
# Success
{"status": "success", "entries": [
    {"path": "/src/main.py", "is_dir": False, "size": 1024, "modified_at": "..."},
    {"path": "/src/utils.py", "is_dir": False, "size": 512, "modified_at": "..."},
]}

# No matches
{"status": "success", "entries": []}
```

**Supported patterns:**

| Pattern | Description |
|---------|-------------|
| `**/*.py` | All Python files recursively |
| `*.txt` | Text files in the base directory |
| `src/{components,utils}/*.ts` | TypeScript files in specific subdirectories |
| `**/*.{js,ts,jsx,tsx}` | Multiple extensions |

**Example usage by the LLM:**
```
Tool call: glob(pattern="**/*.py")
Tool call: glob(pattern="*.md", path="/docs")
```

---

### `grep`

Search for a text pattern within files. Performs **literal** (not regex) matching.

```python
def grep(
    pattern: str,
    tool_context: ToolContext,
    path: str | None = None,
    glob: str | None = None,
    output_mode: str = "files_with_matches",
) -> dict:
```

**Arguments:**

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `pattern` | `str` | Yes | — | Text pattern to search for (literal match) |
| `path` | `str \| None` | No | `None` | Directory to limit the search to |
| `glob` | `str \| None` | No | `None` | Glob pattern to filter files (e.g., `"*.py"`) |
| `output_mode` | `str` | No | `"files_with_matches"` | Output format (see below) |

**Output modes:**

| Mode | Description | Example output |
|------|-------------|---------------|
| `"files_with_matches"` | File paths containing matches (default) | `/src/main.py\n/src/utils.py` |
| `"content"` | File path, line number, and matching line | `/src/main.py:42:    def hello():` |
| `"count"` | File path and match count | `/src/main.py: 3\n/src/utils.py: 1` |

**Returns:**

```python
# Success
{"status": "success", "result": "/src/main.py\n/src/utils.py"}

# No matches
{"status": "success", "result": "No matches found."}
```

Results are truncated if they exceed ~20,000 tokens (80,000 characters). For `FilesystemBackend`, ripgrep (`rg`) is used when available for performance; otherwise, a Python fallback is used.

**Example usage by the LLM:**
```
Tool call: grep(pattern="TODO", output_mode="content")
Tool call: grep(pattern="import flask", path="/src", glob="*.py")
```

---

## Todo Tools

Defined in `adk_deepagents.tools.todos`. Todos are stored in `tool_context.state["todos"]` — a simple list of dicts in session state.

### `write_todos`

Create or update the todo list.

```python
def write_todos(todos: list[dict], tool_context: ToolContext) -> dict:
```

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `todos` | `list[dict]` | Yes | List of todo items |

Each todo item is a dict with the following schema:

```python
{
    "content": str,       # Description of the todo item
    "status": str,        # "pending", "in_progress", or "completed"
}
```

**Returns:**

```python
{"status": "success", "count": 3}
```

**Example usage by the LLM:**
```
Tool call: write_todos(todos=[
    {"content": "Implement user authentication", "status": "in_progress"},
    {"content": "Write unit tests", "status": "pending"},
    {"content": "Update README", "status": "completed"}
])
```

---

### `read_todos`

Read the current todo list.

```python
def read_todos(tool_context: ToolContext) -> dict:
```

**Arguments:** None (only `tool_context` is injected by ADK).

**Returns:**

```python
# With todos
{"todos": [
    {"content": "Implement user authentication", "status": "in_progress"},
    {"content": "Write unit tests", "status": "pending"},
]}

# Empty
{"todos": []}
```

**Example usage by the LLM:**
```
Tool call: read_todos()
```

---

## Execution Tools

Execution tools provide shell command capabilities. They are added to the agent when you set `execution="local"` or `execution="heimdall"` in `create_deep_agent()`.

### Local Execute Tool

Created by `create_local_execute_tool()` in `adk_deepagents.execution.local`. Runs commands directly on the host system using `subprocess.run()` with `shell=True`.

> **⚠️ Warning:** Local execution has no sandboxing. Commands run with the same permissions as the Python process. Use Heimdall MCP for production workloads.

```python
from adk_deepagents.execution.local import create_local_execute_tool

execute = create_local_execute_tool()
```

The returned function has the signature:

```python
def execute(command: str) -> dict:
    """Execute a shell command locally and return the output.

    Args:
        command: The shell command to execute.
    """
```

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `command` | `str` | Yes | Shell command to execute |

**Returns:**

```python
# Success
{
    "status": "success",
    "output": "hello world\n",
    "exit_code": 0,
    "truncated": False,
}

# Error
{
    "status": "error",
    "output": "bash: foo: command not found\n",
    "exit_code": 127,
    "truncated": False,
}

# Timeout
{
    "status": "error",
    "output": "Command timed out after 120.0s",
    "exit_code": -1,
    "truncated": False,
}
```

Internal limits:

- **Timeout:** 120 seconds
- **Max output:** 100,000 bytes (output is truncated and `truncated` is set to `True`)

**Usage with `create_deep_agent()`:**

```python
agent = create_deep_agent(
    name="dev_agent",
    execution="local",
)
```

**Example usage by the LLM:**
```
Tool call: execute(command="python -m pytest tests/ -v")
Tool call: execute(command="ls -la /workspace")
```

---

### Heimdall MCP Tools

Sandboxed Python and Bash execution via the [Heimdall MCP](https://github.com/heimdall-ai/heimdall) server. Requires `google-adk[mcp]` and `@heimdall-ai/heimdall`.

Heimdall exposes these tools via MCP (Model Context Protocol):

| Tool | Description |
|------|-------------|
| `execute_python` | Sandboxed Python execution via Pyodide WebAssembly |
| `execute_bash` | Bash command simulation via just-bash |
| `install_packages` | Python package installation via micropip |

**Prerequisites:**

```bash
pip install google-adk[mcp]
npm i -g @heimdall-ai/heimdall
```

**Usage with `create_deep_agent_async()`:**

Since MCP tools require an async connection, you must use the async factory:

```python
import asyncio
from adk_deepagents import create_deep_agent_async

async def main():
    agent, cleanup = await create_deep_agent_async(
        name="sandbox_agent",
        execution="heimdall",
    )

    # Use the agent...

    # Clean up the MCP connection when done
    if cleanup:
        await cleanup()

asyncio.run(main())
```

**Custom MCP server config (stdio):**

```python
agent, cleanup = await create_deep_agent_async(
    execution={
        "command": "npx",
        "args": ["@heimdall-ai/heimdall"],
        "env": {"HEIMDALL_WORKSPACE": "/workspace"},
    },
)
```

**Custom MCP server config (SSE):**

```python
agent, cleanup = await create_deep_agent_async(
    execution={"uri": "http://localhost:8080"},
)
```

**Example usage by the LLM:**
```
Tool call: execute_python(code="print(2 + 2)")
Tool call: execute_bash(command="ls -la")
Tool call: install_packages(packages=["requests", "pandas"])
```

---

## Sub-Agent Tools

When you pass `subagents=[...]` to `create_deep_agent()`, each sub-agent specification becomes an `AgentTool` — an ADK tool that spawns a child `LlmAgent` to handle a task.

A **general-purpose** sub-agent is always included by default (unless you define one with the name `general_purpose`).

```python
from adk_deepagents import create_deep_agent, SubAgentSpec

agent = create_deep_agent(
    subagents=[
        SubAgentSpec(
            name="researcher",
            description="Searches the codebase for relevant files and patterns.",
            system_prompt="You are a code research assistant. Search thoroughly.",
        ),
    ],
)
```

Each sub-agent:

- Gets its own `LlmAgent` with its own instruction and tools
- Inherits the parent's core tools (filesystem, todo) by default
- Can have custom tools, model, skills, and HITL configuration
- Runs autonomously — it does NOT see the parent's conversation history
- Returns its result to the parent agent for synthesis

The `SubAgentSpec` TypedDict:

```python
class SubAgentSpec(TypedDict, total=False):
    name: str            # required
    description: str     # required — shown to the parent LLM for routing
    system_prompt: str   # instruction for the sub-agent
    tools: Sequence[Callable]  # custom tools (default: parent's core tools)
    model: str           # model override (default: parent's model)
    skills: list[str]    # skill directory paths
    interrupt_on: dict[str, bool]  # HITL config for the sub-agent
```

You can also pass pre-built `LlmAgent` instances directly:

```python
from google.adk.agents import LlmAgent

custom_agent = LlmAgent(
    name="custom",
    model="gemini-2.5-pro",
    instruction="You are a specialized agent.",
    tools=[...],
)

agent = create_deep_agent(
    subagents=[custom_agent],
)
```
