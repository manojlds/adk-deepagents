# Persistent Memory

## Overview

adk-deepagents supports persistent memory by loading `AGENTS.md` files into the agent's system prompt. Memory files contain project context, role descriptions, accumulated knowledge, and guidelines that persist across sessions. The files are loaded from the configured backend and injected into the system instruction before every LLM call.

## How It Works

Memory loading is a two-phase process split across two callbacks:

### Phase 1: `before_agent_callback` — Load Files

When the agent starts, the `before_agent_callback`:

1. Creates a backend instance from the `backend_factory`
2. Calls `backend.download_files(sources)` to fetch each memory file
3. Decodes file contents as UTF-8
4. Stores the result in `state["memory_contents"]` as a `dict[str, str]` mapping path → content
5. Only runs **once per session** (skips if `"memory_contents"` already exists in state)

### Phase 2: `before_model_callback` — Inject into System Prompt

Before every LLM call, the `before_model_callback`:

1. Reads `state["memory_contents"]`
2. Formats the contents using `_format_memory()`
3. Appends the formatted memory to the system instruction via `_append_to_system_instruction()`

## Memory Loading

The `load_memory` function in `adk_deepagents.memory` handles the loading:

```python
from adk_deepagents.memory import load_memory

contents = load_memory(backend, sources=["./AGENTS.md", "./docs/CONTEXT.md"])
# Returns: {"./AGENTS.md": "file content...", "./docs/CONTEXT.md": "..."}
```

It calls `backend.download_files(sources)` and decodes each response as UTF-8. Files that fail to download or decode are silently skipped.

## Memory Formatting

The `format_memory` function produces the prompt text:

```python
from adk_deepagents.memory import format_memory

prompt = format_memory(contents, sources=["./AGENTS.md"])
```

Each memory file becomes a section with a `###` header:

```
<agent_memory>
### ./AGENTS.md
You are a coding assistant for the foo project...

### ./docs/CONTEXT.md
The project uses Python 3.11 with FastAPI...
</agent_memory>
```

If no files were loaded, the memory block contains `(No memory loaded)`.

## Memory System Prompt

The full injected prompt uses the `MEMORY_SYSTEM_PROMPT` template:

```xml
<agent_memory>
{agent_memory}
</agent_memory>

<memory_guidelines>
You have access to persistent memory stored in AGENTS.md files.
These contain information about the project, your role, and accumulated knowledge.

When to update memory:
- When the user explicitly asks you to remember something
- When you discover important role descriptions or project context
- When you receive feedback about your behavior or approach
- When you find tool-specific information that would help future sessions
- When you notice patterns in how the user works

When NOT to update memory:
- Transient information (current task details, temporary data)
- One-time tasks that won't recur
- Simple factual questions
- Small talk or greetings
- Information already in memory

Never store API keys, passwords, or credentials in memory.
</memory_guidelines>
```

## Configuration

Pass a list of file paths to the `memory` parameter on `create_deep_agent`:

```python
from adk_deepagents import create_deep_agent

agent = create_deep_agent(
    memory=["./AGENTS.md"],
)
```

The paths are resolved by the backend. With `StateBackend`, they are looked up in `state["files"]`. With `FilesystemBackend`, they are resolved relative to the backend's root directory.

## Examples

### Basic Memory Loading

```python
from adk_deepagents import create_deep_agent

agent = create_deep_agent(
    instruction="You are a helpful assistant.",
    memory=["./AGENTS.md"],
)
```

The agent will load `./AGENTS.md` from the default `StateBackend` and inject its contents into the system prompt.

### Multiple Memory Files

```python
from adk_deepagents import create_deep_agent

agent = create_deep_agent(
    memory=[
        "./AGENTS.md",
        "./docs/ARCHITECTURE.md",
        "./docs/STYLE_GUIDE.md",
    ],
)
```

Each file becomes a separate `###`-headed section in the `<agent_memory>` block.

### Memory with FilesystemBackend

```python
from adk_deepagents import create_deep_agent
from adk_deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    memory=["./AGENTS.md"],
    backend=FilesystemBackend(root_dir="/path/to/project", virtual_mode=True),
)
```

The `AGENTS.md` file is read from `/path/to/project/AGENTS.md` on disk.

### Pre-Populating Memory in State for StateBackend

When using the default `StateBackend`, you can pre-populate memory by seeding the session state before running the agent:

```python
from google.adk.runners import InMemoryRunner

agent = create_deep_agent(memory=["./AGENTS.md"])
runner = InMemoryRunner(agent=agent, app_name="my_app")
session = await runner.session_service.create_session(
    app_name="my_app",
    user_id="user",
    state={
        "files": {
            "/AGENTS.md": {
                "content": [
                    "# Project Context",
                    "",
                    "This is a Python web application using FastAPI.",
                    "Always use type hints and write tests.",
                ],
                "created_at": "2025-01-01T00:00:00Z",
                "modified_at": "2025-01-01T00:00:00Z",
            }
        }
    },
)
```
