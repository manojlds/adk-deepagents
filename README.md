<!-- This is the README for adk-deepagents -->
# adk-deepagents

Re-implementation of [deepagents](https://github.com/deep-agents/deepagents) using [Google ADK](https://github.com/google/adk-python) primitives.

Build autonomous, tool-using AI agents with filesystem access, shell execution, sub-agent delegation, conversation summarization, persistent memory, and skills integration — all powered by Google's Agent Development Kit.

## Features

- **Filesystem tools** — `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` backed by pluggable storage
- **Shell execution** — Local subprocess or sandboxed execution via [Heimdall MCP](https://github.com/heimdall-ai/heimdall)
- **Browser automation** — Navigate websites, fill forms, extract data via [@playwright/mcp](https://github.com/microsoft/playwright-mcp) or [agent-browser](https://github.com/vercel-labs/agent-browser) CLI skill
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
uv pip install adk-deepagents
```

Or add it to a project with [uv](https://docs.astral.sh/uv/):

```bash
uv add adk-deepagents
```

### Optional dependencies

```bash
# For skills integration
uv pip install "adk-deepagents[skills]"

# For Heimdall MCP sandboxed execution
uv pip install google-adk
npm i -g @heimdall-ai/heimdall

# For browser automation via Playwright MCP
uv pip install adk-deepagents

# For Temporal-backed dynamic task delegation
uv pip install "adk-deepagents[temporal]"

# For A2A server/client integrations
uv pip install "adk-deepagents[a2a]"
```

## CLI Quickstart (`adk-deepagents`)

The package installs an `adk-deepagents` CLI for interactive and non-interactive workflows.

```bash
# Interactive REPL (default mode)
adk-deepagents

# Interactive REPL with an auto-submitted first prompt
adk-deepagents -m "Summarize this repository"

# One-shot non-interactive run
adk-deepagents -n "Run tests and summarize failures"

# Piped stdin non-interactive run (automation-friendly output)
printf 'Summarize README.md\n' | adk-deepagents -q
```

For the full command/flag reference, see [docs/cli.md](docs/cli.md).

## Python Quickstart

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
uv run adk run my_project/
```

## API Reference

### `create_deep_agent()`

The main synchronous factory function. Returns a configured `google.adk.agents.LlmAgent`.

```python
from adk_deepagents import DeepAgentConfig, create_deep_agent

agent = create_deep_agent(
    name="deep_agent",              # Agent name
    model="gemini-2.5-flash",       # LLM model string
    instruction=None,                # Custom system instruction
    tools=None,                      # Additional tool functions
    subagents=None,                  # Sub-agent specifications
    memory=None,                     # AGENTS.md file paths to load
    skills=None,                     # Skill directory paths
    backend=None,                    # Backend instance or factory
    execution=None,                  # "local", "heimdall", or MCP config dict
    browser=None,                    # "playwright" or BrowserConfig
    config=DeepAgentConfig(
        output_schema=None,          # Pydantic BaseModel for structured output
        summarization=None,          # SummarizationConfig
        delegation_mode="static",   # "static", "dynamic", or "both"
        dynamic_task_config=None,    # DynamicTaskConfig (optional Temporal backend)
        skills_config=None,          # SkillsConfig for adk-skills
        interrupt_on=None,           # Tool names requiring approval
        callbacks=None,              # Optional callback hooks
        error_handling=True,
        message_queue=False,
        message_queue_provider=None,
        multimodal=False,
        http_tools=False,
    ),
)
```

### `create_deep_agent_async()`

Async variant that resolves MCP tools before creating the agent. Required for `execution="heimdall"`, dict-based MCP configs, or `browser="playwright"`.

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

```python
# Browser automation
from adk_deepagents import BrowserConfig, create_deep_agent_async

agent, cleanup = await create_deep_agent_async(
    browser="playwright",  # or BrowserConfig(headless=False)
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

`DynamicTaskConfig` also lets you enforce runtime guardrails like `max_depth`
(recursive delegation depth) and `max_parallel` (simultaneous running tasks).

```python
from adk_deepagents import DeepAgentConfig, DynamicTaskConfig, SubAgentSpec, create_deep_agent

agent = create_deep_agent(
    subagents=[
        SubAgentSpec(name="explore", description="Searches files and code patterns."),
        SubAgentSpec(name="coder", description="Writes and edits files."),
    ],
    config=DeepAgentConfig(
        delegation_mode="dynamic",
        dynamic_task_config=DynamicTaskConfig(
            timeout_seconds=90,
            allow_model_override=False,
        ),
    ),
)
```

Use `delegation_mode="both"` to keep static sub-agent tools and add the dynamic
`task` tool side by side.

#### Temporal-backed dynamic tasks

To run delegated `task()` turns on Temporal workers instead of in-process
sessions, set `DynamicTaskConfig.temporal`:

```python
from adk_deepagents import DeepAgentConfig, DynamicTaskConfig, TemporalTaskConfig, create_deep_agent

agent = create_deep_agent(
    config=DeepAgentConfig(
        delegation_mode="dynamic",
        dynamic_task_config=DynamicTaskConfig(
            temporal=TemporalTaskConfig(
                target_host="127.0.0.1:7233",
                namespace="default",
                task_queue="adk-deepagents-tasks",
            )
        ),
    ),
)
```

See [docs/temporal.md](docs/temporal.md) for worker setup, `mise + pitchfork` services,
and integration test instructions.

When the CLI runs in an environment with `ADK_DEEPAGENTS_TEMPORAL_*` variables,
it auto-enables Temporal-backed dynamic tasks using those settings.

#### A2A exposure and dynamic task backend

Expose a deep agent as an A2A server endpoint:

```python
from adk_deepagents import create_deep_agent, to_a2a_app

agent = create_deep_agent(name="deep_a2a")
app = to_a2a_app(agent, host="0.0.0.0", port=8000)
```

Then run the Starlette app (for example with uvicorn):

```bash
uv run uvicorn my_agent_module:app --host 0.0.0.0 --port 8000
```

Use A2A as the backend for dynamic `task()` delegation:

```python
from adk_deepagents import A2ATaskConfig, DeepAgentConfig, DynamicTaskConfig, create_deep_agent

agent = create_deep_agent(
    config=DeepAgentConfig(
        delegation_mode="dynamic",
        dynamic_task_config=DynamicTaskConfig(
            a2a=A2ATaskConfig(agent_url="http://127.0.0.1:8000")
        ),
    ),
)
```

In A2A-backed mode, delegated turns are sent to the target A2A agent and task
continuity is preserved via `task_id`/A2A `context_id`.

### Summarization

Automatic context window management. When the conversation exceeds a configurable fraction of the context window, older messages are replaced with a condensed summary.

```python
from adk_deepagents import DeepAgentConfig, SummarizationConfig, create_deep_agent

agent = create_deep_agent(
    config=DeepAgentConfig(
        summarization=SummarizationConfig(
            model="gemini-2.5-flash",
            trigger=("fraction", 0.85),     # Trigger at 85% of context window
            keep=("messages", 6),           # Keep 6 most recent messages
            history_path_prefix="/conversation_history",
        ),
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
from adk_deepagents import DeepAgentConfig, create_deep_agent

agent = create_deep_agent(
    config=DeepAgentConfig(
        interrupt_on={
            "write_file": True,
            "execute": True,
            "read_file": False,  # Explicitly allow (no interruption)
        },
    ),
)
```

### Structured output

Use a Pydantic `BaseModel` subclass to constrain the agent's final output format.

```python
from adk_deepagents import DeepAgentConfig, create_deep_agent
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    summary: str
    issues: list[str]
    confidence: float

agent = create_deep_agent(
    instruction="Analyze the given code and return structured results.",
    config=DeepAgentConfig(output_schema=AnalysisResult),
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

With `config=DeepAgentConfig(summarization=SummarizationConfig(...))`, a
`compact_conversation` tool is added for manual context compaction on the next model turn.

With `browser="playwright"`, browser tools (`browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, etc.) are added for web page interaction.

With `subagents=[...]` and `delegation_mode="static"` (default), one tool is generated per sub-agent specification.

With `delegation_mode="dynamic"` or `"both"`, a single `task` tool is added for runtime delegation.

## Project Structure

```
adk_deepagents/
├── __init__.py          # Public API exports
├── graph.py             # create_deep_agent() and create_deep_agent_async()
├── types.py             # DeepAgentConfig and shared config/spec dataclasses
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
├── browser/
│   ├── __init__.py      # Browser module exports
│   ├── playwright_mcp.py # Playwright MCP integration
│   └── prompts.py       # Browser agent system prompts
├── skills/
│   └── integration.py   # adk-skills registry integration
├── temporal/
│   ├── client.py        # Temporal workflow dispatch client
│   ├── workflows.py     # Dynamic task workflow definition
│   ├── activities.py    # Dynamic task activity implementation
│   └── worker.py        # Temporal worker factory
└── tools/
    ├── compact.py       # Manual conversation compaction tool
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

### mise + pitchfork (Temporal + OTEL collector)

The repo includes `mise.toml` and `pitchfork.toml` to run local support
services for Temporal workflows and OTEL trace collection.

`mise.toml` also exports the `ADK_DEEPAGENTS_TEMPORAL_*` and OTEL environment
variables used by the CLI and dev worker.

```bash
# Install/update local tools pinned in mise.toml
mise trust
mise install

# Start services (temporal-server + temporal-worker + otel-collector)
pitchfork start temporal-server temporal-worker otel-collector

# Start only Temporal stack processes
pitchfork start temporal-server temporal-worker

# Start only OTEL collector
pitchfork start otel-collector

# Stop all local dev daemons in this repo
pitchfork stop --all

# Reset local OTEL state file
mise run otel-reset

# Reset local Temporal dev state
mise run temporal-reset
```

`temporal-worker` runs `uv run python -m adk_deepagents.temporal.dev_worker`
and uses `ADK_DEEPAGENTS_TEMPORAL_WORKER_MODEL` (or `ADK_DEEPAGENTS_MODEL`) for
its default model. It also exposes a local liveness probe on
`127.0.0.1:17451` for supervisor health checks.

`temporal-server` also starts Temporal Web UI on `http://127.0.0.1:8233`.

Temporal dynamic-task workflows may stay `Running` for a short period so the
same `task_id` can resume; they auto-complete after idle timeout
(`ADK_DEEPAGENTS_TEMPORAL_IDLE_TIMEOUT_SECONDS`, default `600`).

Trace collector output is written to:

```text
.adk/state/otel/traces.json
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

# Run LLM integration tests via A2A transport bridge
ADK_DEEPAGENTS_LLM_TRANSPORT=a2a uv run pytest -m llm
```

## License

MIT
