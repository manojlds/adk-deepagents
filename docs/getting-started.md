# Getting Started

This guide walks you through installing adk-deepagents, creating your first agent, and running it interactively, with the ADK CLI, and with the ADK Web UI.

## Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or **pip**
- A **Google API key** (for the default Gemini model), or an **OpenAI / Anthropic API key** if using LiteLLM

## Installation

### With pip

```bash
pip install adk-deepagents
```

### With uv

```bash
uv add adk-deepagents
```

### Optional dependencies

```bash
# For skills integration (adk-skills-agent)
pip install adk-deepagents[skills]

# For Heimdall MCP sandboxed execution
pip install google-adk[mcp]
npm i -g @heimdall-ai/heimdall
```

## Environment Setup

### Google Gemini (default)

Set your Google API key:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Or create a `.env` file in your project root:

```
GOOGLE_API_KEY=your-api-key-here
```

The library uses [python-dotenv](https://pypi.org/project/python-dotenv/) — `.env` files are loaded automatically in the quickstart examples.

### Vertex AI

For Google Cloud Vertex AI, configure your project instead:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

### OpenAI / Anthropic (via LiteLLM)

To use non-Gemini models, install [LiteLLM](https://docs.litellm.ai/) and set the appropriate API key:

```bash
pip install litellm

# For OpenAI
export OPENAI_API_KEY="your-openai-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"
```

Then pass a LiteLLM-prefixed model string:

```python
agent = create_deep_agent(
    model="litellm/openai/gpt-4o",
    # or: model="litellm/anthropic/claude-3.5-sonnet",
)
```

## Your First Agent

```python
from adk_deepagents import create_deep_agent

agent = create_deep_agent(
    name="my_agent",
    instruction="You are a helpful coding assistant.",
)
```

With this single call you get an agent configured with:

- **Model:** `gemini-2.5-flash` (default)
- **Filesystem tools:** `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`
- **Todo tools:** `write_todos`, `read_todos`
- **Backend:** `StateBackend` — files stored in-memory in the ADK session state
- **Callbacks:** Memory loading, prompt injection, dangling tool call patching, large result eviction

The returned object is a standard `google.adk.agents.LlmAgent` — compatible with all ADK runners and tooling.

## Using Different Models

The default model is `gemini-2.5-flash`. Change it by passing the `model` parameter:

```python
# Gemini 2.5 Pro (larger context, slower)
agent = create_deep_agent(model="gemini-2.5-pro")

# Gemini 2.0 Flash
agent = create_deep_agent(model="gemini-2.0-flash")

# OpenAI via LiteLLM
agent = create_deep_agent(model="litellm/openai/gpt-4o")

# Anthropic via LiteLLM
agent = create_deep_agent(model="litellm/anthropic/claude-3.5-sonnet")
```

When using summarization, the library knows context window sizes for common models (Gemini, GPT-4o, Claude 3) and uses them automatically. For unknown models, you can set the context window explicitly:

```python
from adk_deepagents import create_deep_agent, SummarizationConfig

agent = create_deep_agent(
    model="litellm/openai/gpt-4o-mini",
    summarization=SummarizationConfig(
        context_window=128_000,
    ),
)
```

## Running Interactively

Use ADK's `InMemoryRunner` to run the agent in a Python script:

```python
import asyncio
from dotenv import load_dotenv
from google.genai import types
from google.adk.runners import InMemoryRunner
from adk_deepagents import create_deep_agent

load_dotenv()

agent = create_deep_agent(
    name="interactive_agent",
    instruction="You are a helpful coding assistant.",
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="demo")
    session = await runner.session_service.create_session(
        app_name="demo", user_id="user",
    )

    print("Agent ready. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit"):
            break

        async for event in runner.run_async(
            session_id=session.id,
            user_id="user",
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=user_input)],
            ),
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        print(f"Agent: {part.text}")

asyncio.run(main())
```

### Simpler message passing

For one-shot interactions, you can pass a plain string to `new_message`:

```python
async for event in runner.run_async(
    session_id=session.id,
    user_id="user",
    new_message="List the files in /",
):
    ...
```

## Running with ADK CLI

The ADK CLI discovers agents from a directory containing an `agent.py` with a top-level `root_agent`.

### 1. Create the agent module

Create a directory with an `agent.py` and an `__init__.py`:

```
my_project/
├── __init__.py
└── agent.py
```

```python
# my_project/agent.py
from adk_deepagents import create_deep_agent

root_agent = create_deep_agent(
    name="my_agent",
    instruction="You are a helpful coding assistant.",
)
```

```python
# my_project/__init__.py
```

### 2. Run with ADK

```bash
# Interactive CLI
adk run my_project/

# With a specific prompt
adk run my_project/ --prompt "Create a hello world Python script"
```

## Running with ADK Web UI

The ADK Web UI provides a browser-based chat interface. Start it the same way:

```bash
adk web my_project/
```

This opens a local web server where you can interact with your agent in a graphical chat interface, view tool calls, and inspect session state.

## Complete Working Example

Here's a fully-featured agent with filesystem backend, execution, sub-agents, and summarization:

```python
import asyncio
from dotenv import load_dotenv
from google.genai import types
from google.adk.runners import InMemoryRunner

from adk_deepagents import (
    create_deep_agent,
    SubAgentSpec,
    SummarizationConfig,
)
from adk_deepagents.backends.filesystem import FilesystemBackend

load_dotenv()

agent = create_deep_agent(
    name="coding_assistant",
    model="gemini-2.5-flash",
    instruction=(
        "You are an expert coding assistant. You can read, write, and edit files "
        "in the project directory, run shell commands, and delegate research tasks "
        "to a sub-agent."
    ),
    # Use the local filesystem rooted at the current directory
    backend=FilesystemBackend(root_dir=".", virtual_mode=True),
    # Enable local shell execution
    execution="local",
    # Add a research sub-agent
    subagents=[
        SubAgentSpec(
            name="researcher",
            description="Searches the codebase for relevant files, patterns, and context.",
            system_prompt="You are a code research assistant. Search thoroughly.",
        ),
    ],
    # Enable conversation summarization
    summarization=SummarizationConfig(
        model="gemini-2.5-flash",
        trigger=("fraction", 0.85),
        keep=("messages", 6),
    ),
    # Require approval before writing files or executing commands
    interrupt_on={
        "write_file": True,
        "execute": True,
    },
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="coder")
    session = await runner.session_service.create_session(
        app_name="coder", user_id="user",
    )

    print("Coding assistant ready. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit"):
            break

        async for event in runner.run_async(
            session_id=session.id,
            user_id="user",
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=user_input)],
            ),
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        print(f"Agent: {part.text}")


if __name__ == "__main__":
    asyncio.run(main())
```

## What's Next

- **[Backends](backends.md)** — Learn about the storage backends: StateBackend, FilesystemBackend, CompositeBackend, StoreBackend, and how to write your own.
- **[Tools](tools.md)** — Deep reference for all built-in tools: filesystem, todo, and execution tools with signatures and examples.
