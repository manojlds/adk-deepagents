# adk-deepagents

Re-implementation of [deepagents](https://github.com/deep-agents/deepagents) using [Google ADK](https://github.com/google/adk-python) primitives.

Build autonomous, tool-using AI agents with filesystem access, shell execution, sub-agent delegation, conversation summarization, persistent memory, and skills integration — all powered by Google's Agent Development Kit.

## Features

- **Filesystem tools** — `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` backed by pluggable storage
- **Shell execution** — Local subprocess or sandboxed execution via [Heimdall MCP](https://github.com/heimdall-ai/heimdall)
- **Sub-agent delegation** — Spawn child agents for isolated, parallelizable sub-tasks
- **Conversation summarization** — Automatic context window management for long-running sessions
- **Persistent memory** — Load `AGENTS.md` files into the system prompt across sessions
- **Todo tracking** — Built-in `write_todos` / `read_todos` tools for task management
- **Human-in-the-loop** — Interrupt specific tools to require human approval before execution
- **Pluggable backends** — `StateBackend` (in-memory), `FilesystemBackend` (local disk), `CompositeBackend` (path-based routing)
- **Skills integration** — Optional integration with [adk-skills-agent](https://github.com/deep-agents/adk-skills-agent)

## Installation

Requires Python 3.11+.

```bash
pip install adk-deepagents
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add adk-deepagents
```

### Optional dependencies

```bash
# For skills integration
pip install adk-deepagents[skills]

# For Heimdall MCP sandboxed execution
pip install google-adk[mcp]
npm i -g @heimdall-ai/heimdall
```

## Quickstart

```python
from adk_deepagents import create_deep_agent

# Create an agent with default settings:
# - Gemini 2.5 Flash model
# - Filesystem tools (ls, read, write, edit, glob, grep)
# - Todo tools (write_todos, read_todos)
# - StateBackend (in-memory file storage)
agent = create_deep_agent(
    name="my_agent",
    instruction="You are a helpful coding assistant.",
)
```

### Running interactively

```python
import asyncio
from google.adk.runners import InMemoryRunner

async def main():
    runner = InMemoryRunner(agent=agent, app_name="demo")
    session = await runner.session_service.create_session(
        app_name="demo", user_id="user",
    )

    async for event in runner.run_async(
        session_id=session.id,
        user_id="user",
        new_message="List the files in /",
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    print(part.text)

asyncio.run(main())
```

### Using with ADK CLI

Create an `agent.py` in a directory with a top-level `root_agent`:

```python
# my_project/agent.py
from adk_deepagents import create_deep_agent

root_agent = create_deep_agent(
    name="my_agent",
    instruction="You are a helpful assistant.",
)
```

Then run:

```bash
adk run my_project/
```

## API Reference

### `create_deep_agent()`

The main synchronous factory function. Returns a configured `google.adk.agents.LlmAgent`.

```python
from adk_deepagents import create_deep_agent

agent = create_deep_agent(
    model="gemini-2.5-flash",       # LLM model string
    tools=None,                      # Additional tool functions
    instruction=None,                # Custom system instruction
    subagents=None,                  # Sub-agent specifications
    skills=None,                     # Skill directory paths
    skills_config=None,              # SkillsConfig for adk-skills
    memory=None,                     # AGENTS.md file paths to load
    output_schema=None,              # Pydantic BaseModel for structured output
    backend=None,                    # Backend instance or factory
    execution=None,                  # "local", "heimdall", or MCP config dict
    summarization=None,              # SummarizationConfig
    delegation_mode="static",       # "static", "dynamic", or "both"
    dynamic_task_config=None,        # DynamicTaskConfig for task tool behavior
    interrupt_on=None,               # Tool names requiring approval
    name="deep_agent",              # Agent name
)
```

### `create_deep_agent_async()`

Async variant that resolves MCP tools before creating the agent. Required for `execution="heimdall"` or dict-based MCP configs.

```python
from adk_deepagents import create_deep_agent_async

agent, cleanup = await create_deep_agent_async(
    execution="heimdall",
    # ... same parameters as create_deep_agent()
)

# When done:
if cleanup:
    await cleanup()
```

Returns `(LlmAgent, cleanup_fn | None)`.

## Configuration

### Backends

Backends control how the agent reads and writes files.

#### StateBackend (default)

In-memory file storage backed by the ADK session state dict. Files persist for the duration of the session.

```python
agent = create_deep_agent()  # Uses StateBackend by default
```

#### FilesystemBackend

Reads and writes to the local filesystem.

```python
from adk_deepagents.backends.filesystem import FilesystemBackend

agent = create_deep_agent(
    backend=FilesystemBackend(root_dir="/path/to/project"),
)
```

#### CompositeBackend

Routes file operations to different backends based on path prefixes. Longer prefixes take priority.

```python
from adk_deepagents.backends import StateBackend, CompositeBackend
from adk_deepagents.backends.filesystem import FilesystemBackend

def my_backend_factory(state):
    return CompositeBackend(
        default=StateBackend(state),
        routes={
            "/workspace": FilesystemBackend(root_dir="./workspace"),
        },
    )

agent = create_deep_agent(backend=my_backend_factory)
```

You can also pass a `BackendFactory` — a callable `(state: dict) -> Backend` — for deferred construction from session state.

### Execution

#### Local (subprocess)

Runs shell commands directly on the host. Suitable for development and testing.

```python
agent = create_deep_agent(execution="local")
```

#### Heimdall MCP (sandboxed)

Sandboxed Python (Pyodide/WASM) and Bash execution via the Heimdall MCP server.

```python
agent, cleanup = await create_deep_agent_async(execution="heimdall")
```

Or with custom MCP server config:

```python
agent, cleanup = await create_deep_agent_async(
    execution={
        "command": "npx",
        "args": ["@heimdall-ai/heimdall"],
        "env": {"HEIMDALL_WORKSPACE": "/workspace"},
    },
)

# SSE transport
agent, cleanup = await create_deep_agent_async(
    execution={"uri": "http://localhost:8080"},
)
```

### Sub-agents

Delegate isolated sub-tasks to child agents. A general-purpose sub-agent is always included by default.

```python
from adk_deepagents import create_deep_agent, SubAgentSpec

agent = create_deep_agent(
    subagents=[
        SubAgentSpec(
            name="researcher",
            description="Searches the codebase for relevant files and patterns.",
            system_prompt="You are a code research assistant.",
        ),
        SubAgentSpec(
            name="writer",
            description="Writes and edits code files.",
            model="gemini-2.5-pro",
        ),
    ],
)
```

By default (`delegation_mode="static"`), each sub-agent is exposed as its own tool
(`researcher`, `writer`, etc.) and the parent delegates by calling those tools.

### Dynamic Task Delegation

Use `delegation_mode="dynamic"` to expose a single `task` tool that routes work
to sub-agents at runtime (LangChain deepagents-style). The tool supports `task_id`
to continue the same delegated sub-session across turns.

```python
from adk_deepagents import DynamicTaskConfig, SubAgentSpec, create_deep_agent

agent = create_deep_agent(
    subagents=[
        SubAgentSpec(name="explore", description="Searches files and code patterns."),
        SubAgentSpec(name="coder", description="Writes and edits files."),
    ],
    delegation_mode="dynamic",
    dynamic_task_config=DynamicTaskConfig(
        timeout_seconds=90,
        allow_model_override=False,
    ),
)
```

Use `delegation_mode="both"` to keep static sub-agent tools and add the dynamic
`task` tool side by side.

### Summarization

Automatic context window management. When the conversation exceeds a configurable fraction of the context window, older messages are replaced with a condensed summary.

```python
from adk_deepagents import create_deep_agent, SummarizationConfig

agent = create_deep_agent(
    summarization=SummarizationConfig(
        model="gemini-2.5-flash",
        trigger=("fraction", 0.85),     # Trigger at 85% of context window
        keep=("messages", 6),           # Keep 6 most recent messages
        history_path_prefix="/conversation_history",
    ),
)
```

Offloaded conversation history is saved to the backend at the configured path prefix for later reference.

### Memory

Load persistent `AGENTS.md` files into the agent's system prompt. Memory is loaded once on the first agent invocation and cached in session state.

```python
agent = create_deep_agent(
    memory=["/AGENTS.md", "/docs/CONTEXT.md"],
)
```

Memory files should contain project context, role descriptions, coding conventions, or other persistent knowledge the agent should always have access to.

### Human-in-the-loop

Require human approval before specific tools execute. When a tool is interrupted, its invocation is stored in session state under `_pending_approval`.

```python
agent = create_deep_agent(
    interrupt_on={
        "write_file": True,
        "execute": True,
        "read_file": False,  # Explicitly allow (no interruption)
    },
)
```

### Structured output

Use a Pydantic `BaseModel` subclass to constrain the agent's final output format.

```python
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    summary: str
    issues: list[str]
    confidence: float

agent = create_deep_agent(
    output_schema=AnalysisResult,
    instruction="Analyze the given code and return structured results.",
)
```

## Built-in Tools

Every agent created with `create_deep_agent()` includes these tools:

| Tool | Description |
|------|-------------|
| `ls` | List files and directories at a path |
| `read_file` | Read file contents with optional pagination |
| `write_file` | Create a new file (no overwrites) |
| `edit_file` | Edit a file via string replacement |
| `glob` | Find files matching a glob pattern |
| `grep` | Search for text patterns within files |
| `write_todos` | Create or update a todo list |
| `read_todos` | Read the current todo list |

With `execution="local"` or `execution="heimdall"`, an `execute` tool is also available for shell commands.

With `subagents=[...]` and `delegation_mode="static"` (default), one tool is generated per sub-agent specification.

With `delegation_mode="dynamic"` or `"both"`, a single `task` tool is added for runtime delegation.

## Project Structure

```
adk_deepagents/
├── __init__.py          # Public API exports
├── graph.py             # create_deep_agent() and create_deep_agent_async()
├── types.py             # SubAgentSpec, SummarizationConfig, SkillsConfig
├── prompts.py           # System prompt templates
├── memory.py            # AGENTS.md loading and formatting
├── summarization.py     # Context window management
├── backends/
│   ├── protocol.py      # Backend ABC and data types
│   ├── state.py         # StateBackend (in-memory)
│   ├── filesystem.py    # FilesystemBackend (local disk)
│   ├── composite.py     # CompositeBackend (path-based routing)
│   └── utils.py         # Shared utilities
├── callbacks/
│   ├── before_agent.py  # Memory loading on first invocation
│   ├── before_model.py  # Prompt injection and summarization
│   ├── before_tool.py   # Human-in-the-loop interrupts
│   └── after_tool.py    # Post-tool processing and eviction
├── execution/
│   ├── local.py         # Local subprocess execution
│   ├── heimdall.py      # Heimdall MCP integration
│   └── bridge.py        # Skills-to-Heimdall script execution bridge
├── skills/
│   └── integration.py   # adk-skills registry integration
└── tools/
    ├── filesystem.py    # Filesystem tool implementations
    ├── todos.py         # Todo tool implementations
    └── task.py          # Sub-agent delegation tools
```

## Development

### Setup

```bash
git clone https://github.com/anthropics/adk-deepagents.git
cd adk-deepagents
uv sync
```

### Commands

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/unit_tests/test_summarization.py

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run ty check
```

### Testing

Tests use pytest with pytest-asyncio (`asyncio_mode = "auto"`). Unit tests are in `tests/unit_tests/`, integration tests in `tests/integration_tests/`.

```bash
# Run with verbose output
uv run pytest -v

# Run only unit tests
uv run pytest tests/unit_tests/

# Run tests matching a pattern
uv run pytest -k "test_summarization"
```

## License

MIT
