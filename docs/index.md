# adk-deepagents

**Re-implementation of [deepagents](https://github.com/deep-agents/deepagents) using [Google ADK](https://github.com/google/adk-python) primitives.**

Build autonomous, tool-using AI agents with filesystem access, shell execution, sub-agent delegation, conversation summarization, persistent memory, and skills integration — all powered by Google's Agent Development Kit.

## Key Features

| Feature | Description |
|---------|-------------|
| **Filesystem tools** | `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` — backed by pluggable storage backends |
| **Shell execution** | Local subprocess or sandboxed execution via [Heimdall MCP](https://github.com/heimdall-ai/heimdall) |
| **Browser automation** | Navigate, interact with, and extract data from websites via [@playwright/mcp](https://github.com/microsoft/playwright-mcp) or [agent-browser](https://github.com/vercel-labs/agent-browser) CLI skill |
| **Sub-agent delegation** | Spawn child agents for isolated, parallelizable sub-tasks with `AgentTool` |
| **Temporal backend (optional)** | Dispatch dynamic `task()` delegation to Temporal workflows/workers |
| **Conversation summarization** | Automatic context window management — older messages are replaced with condensed summaries |
| **Persistent memory** | Load `AGENTS.md` files into the system prompt across sessions |
| **Todo tracking** | Built-in `write_todos` / `read_todos` tools for task management |
| **Human-in-the-loop** | Interrupt specific tools to require human approval via ADK's `ToolConfirmation` |
| **Pluggable backends** | `StateBackend` (in-memory), `FilesystemBackend` (local disk), `CompositeBackend` (path routing), `StoreBackend` (cross-session) |
| **Skills integration** | Optional integration with [adk-skills-agent](https://github.com/deep-agents/adk-skills-agent) |
| **Structured output** | Constrain agent output with Pydantic `BaseModel` schemas |

## Architecture Overview

adk-deepagents maps the deepagents middleware architecture onto ADK's callback + tool system:

```
deepagents (middleware)          adk-deepagents (ADK primitives)
─────────────────────────────    ──────────────────────────────────
middleware.before_model()    →   before_model_callback
  • prompt injection                • system prompt injection
  • summarization trigger           • summarization check + execution
  • memory injection                • memory formatting

middleware.before_agent()    →   before_agent_callback
  • patch dangling tool calls       • dangling tool call detection
  • load memory files               • memory loading from backend

middleware.wrap_tool()       →   before_tool_callback
  • human-in-the-loop               • ADK ToolConfirmation flow

middleware.after_tool()      →   after_tool_callback
  • large result eviction           • result eviction to backend

middleware.filesystem        →   tools/filesystem.py (FunctionTool)
middleware.todolist           →   tools/todos.py (FunctionTool)
middleware.task               →   tools/task.py (AgentTool per sub-agent)

BackendProtocol              →   Backend ABC + BackendFactory
SubAgent                     →   SubAgentSpec TypedDict
```

The main entry point, `create_deep_agent()`, wires all of these together into a single configured `google.adk.agents.LlmAgent`.

## Hello World

```python
import asyncio
from adk_deepagents import create_deep_agent
from google.adk.runners import InMemoryRunner

# Create an agent with sensible defaults
agent = create_deep_agent(
    name="hello_agent",
    instruction="You are a helpful assistant.",
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="hello")
    session = await runner.session_service.create_session(
        app_name="hello", user_id="user",
    )

    async for event in runner.run_async(
        session_id=session.id,
        user_id="user",
        new_message="Write a file called /hello.txt with the text 'Hello, world!'",
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    print(part.text)

asyncio.run(main())
```

Out of the box, the agent has filesystem tools (backed by in-memory `StateBackend`), todo tools, and conversation summarization support.

## Documentation

| Page | Description |
|------|-------------|
| [Getting Started](getting-started.md) | Installation, first agent, running interactively, ADK CLI, models |
| [CLI Guide](cli.md) | CLI commands/flags, interactive controls, non-interactive safety, release smoke checklist |
| [Backends](backends.md) | StateBackend, FilesystemBackend, CompositeBackend, StoreBackend, custom backends |
| [Tools](tools.md) | Filesystem tools, todo tools, execution tools — signatures, arguments, examples |
| [Browser Automation](browser.md) | Playwright MCP integration, agent-browser CLI skill, BrowserConfig |
| [Sub-Agent Delegation](subagents.md) | Static `AgentTool` delegation, `SubAgentSpec`, general-purpose sub-agent behavior |
| [Skills Integration](skills.md) | Root-agent and sub-agent skills, `add_skills_tools`, prompt injection |
| [Task System Internals](task-system.md) | Static vs dynamic delegation internals, state keys, runtime registries, guardrails, failure semantics |
| [Temporal Backend](temporal.md) | Configure Temporal-backed dynamic tasks, workers, and devenv services |

## Project Structure

```
adk_deepagents/
├── __init__.py          # Public API: create_deep_agent, create_deep_agent_async
├── graph.py             # Main factory functions
├── types.py             # SubAgentSpec, SummarizationConfig, SkillsConfig, BrowserConfig
├── prompts.py           # System prompt templates
├── memory.py            # AGENTS.md loading and formatting
├── summarization.py     # Context window management
├── backends/
│   ├── protocol.py      # Backend ABC and data types
│   ├── state.py         # StateBackend (in-memory)
│   ├── filesystem.py    # FilesystemBackend (local disk)
│   ├── composite.py     # CompositeBackend (path-based routing)
│   ├── store.py         # StoreBackend (cross-session shared dict)
│   └── utils.py         # Shared utilities
├── callbacks/
│   ├── before_agent.py  # Memory loading, dangling tool call patching
│   ├── before_model.py  # Prompt injection, summarization trigger
│   ├── before_tool.py   # Human-in-the-loop via ToolConfirmation
│   └── after_tool.py    # Large result eviction
├── execution/
│   ├── local.py         # Local subprocess execution
│   ├── heimdall.py      # Heimdall MCP sandboxed execution
│   └── bridge.py        # Skills-to-Heimdall execution bridge
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
    ├── filesystem.py    # ls, read_file, write_file, edit_file, glob, grep
    ├── todos.py         # write_todos, read_todos
    └── task.py          # Sub-agent delegation via AgentTool
```

## License

MIT
