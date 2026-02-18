# Sub-Agent Delegation

## Overview

In deepagents, a single `task` tool routes work to sub-agents by name. In adk-deepagents, each sub-agent becomes its own [`AgentTool`](https://google.github.io/adk-python/) instance, so the parent agent calls them like any other tool. The `build_subagent_tools` function in `adk_deepagents.tools.task` converts a list of `SubAgentSpec` dicts (or pre-built `LlmAgent` objects) into `AgentTool` instances that are added to the parent agent's tool list.

## SubAgentSpec

`SubAgentSpec` is a `TypedDict` that describes a sub-agent:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | ✅ | Unique name for the sub-agent |
| `description` | `str` | ✅ | What the sub-agent does (shown to the parent agent) |
| `system_prompt` | `str` | ❌ | Custom system instruction for the sub-agent |
| `tools` | `Sequence[Callable]` | ❌ | Custom tools; defaults to the parent's core tools |
| `model` | `str` | ❌ | Model string; defaults to the parent agent's model |
| `skills` | `list[str]` | ❌ | Directories to discover Agent Skills from |
| `interrupt_on` | `dict[str, bool]` | ❌ | Tool names requiring human approval |

```python
from adk_deepagents import SubAgentSpec

spec = SubAgentSpec(
    name="researcher",
    description="Research agent for gathering information on topics.",
    system_prompt="You are a research assistant. Gather relevant information.",
    model="gemini-2.5-pro",
)
```

## General-Purpose Sub-Agent

A general-purpose sub-agent is **always included by default** when you pass `subagents` to `create_deep_agent`. It has access to all the same tools as the parent agent and is described as:

> General-purpose agent for researching complex questions, searching for files and content, and executing multi-step tasks.

If you provide a sub-agent named `general_purpose` (or `general-purpose`), the default one is **not** added. This lets you override it with a custom implementation.

## Name Sanitization

ADK requires agent names to be valid Python identifiers matching `[a-zA-Z_][a-zA-Z0-9_]*`. The `_sanitize_agent_name` function handles this automatically:

- Hyphens (`-`) → underscores (`_`)
- Spaces → underscores (`_`)
- Invalid characters → underscores
- Leading digit → prefixed with `_`

```python
_sanitize_agent_name("my-agent")      # → "my_agent"
_sanitize_agent_name("research agent") # → "research_agent"
_sanitize_agent_name("123agent")       # → "_123agent"
```

## build_subagent_tools

The `build_subagent_tools` function converts specs into `AgentTool` instances:

```python
from adk_deepagents.tools.task import build_subagent_tools

tools = build_subagent_tools(
    subagents=[spec1, spec2],
    default_model="gemini-2.5-flash",
    default_tools=core_tools,
    include_general_purpose=True,
    skills_config=None,
)
```

**Parameters:**

| Parameter | Description |
|---|---|
| `subagents` | List of `SubAgentSpec` dicts or pre-built `LlmAgent` instances |
| `default_model` | Model to use when a spec doesn't specify one |
| `default_tools` | Default tools given to sub-agents that don't specify their own |
| `include_general_purpose` | If `True` (default), prepend the general-purpose sub-agent |
| `skills_config` | Optional `SkillsConfig` for sub-agents with `skills` set |

Each spec becomes an `LlmAgent` wrapped in an `AgentTool`:

```python
sub_agent = LlmAgent(
    name=_sanitize_agent_name(spec["name"]),
    model=spec.get("model", default_model),
    instruction=spec.get("system_prompt", DEFAULT_SUBAGENT_PROMPT),
    description=spec["description"],
    tools=sub_tools,
    before_tool_callback=before_tool_cb,
)
AgentTool(agent=sub_agent)
```

## Pre-Built LlmAgent Instances

You can pass pre-built `LlmAgent` objects directly in the `subagents` list. They are wrapped in `AgentTool` as-is, without modification:

```python
from google.adk.agents import LlmAgent
from adk_deepagents import create_deep_agent

custom_agent = LlmAgent(
    name="custom_researcher",
    model="gemini-2.5-pro",
    instruction="You are a specialized researcher.",
    description="Custom research agent with special capabilities.",
    tools=[my_custom_tool],
)

agent = create_deep_agent(
    subagents=[custom_agent],
)
```

## Sub-Agent Skills

Each sub-agent can have its own skills directories. When a spec includes `skills`, the corresponding tools (`use_skill`, `run_script`, `read_reference`) are resolved and added to that sub-agent's tool list:

```python
researcher = SubAgentSpec(
    name="researcher",
    description="Research agent with domain-specific skills.",
    skills=["./skills/research/"],
)
```

## Sub-Agent HITL

Each sub-agent can have its own `interrupt_on` configuration for human-in-the-loop approval, independent of the parent agent:

```python
coder = SubAgentSpec(
    name="coder",
    description="Coding agent that requires approval for file writes.",
    interrupt_on={"write_file": True, "execute": True},
)
```

This creates a dedicated `before_tool_callback` for that sub-agent via `make_before_tool_callback`.

## System Prompt Injection

When sub-agents are configured, the `before_model_callback` injects `TASK_SYSTEM_PROMPT` into the parent agent's system instruction. This includes:

- The sub-agent lifecycle documentation
- A list of all available sub-agents with their names and descriptions

```
## Sub-agent Delegation

You can delegate work to specialized sub-agents using the tools below.
Each sub-agent runs independently with its own tools and context.

**Available sub-agents:**

- **general_purpose**: General-purpose agent for researching complex questions...
- **researcher**: Research agent for gathering information on topics.
```

## Lifecycle

Sub-agent delegation follows a four-phase lifecycle:

1. **Spawn** — The parent agent calls the sub-agent tool with a task description
2. **Run** — The sub-agent works autonomously using its own tools
3. **Return** — The sub-agent's result is returned to the parent
4. **Reconcile** — The parent synthesizes the result into its response

## Tips

- **Parallelize independent tasks** by calling multiple sub-agent tools in a single response
- **Give self-contained instructions** — sub-agents do NOT see the parent's conversation history
- **No shared conversation history** — each sub-agent starts fresh with only the task description
- **Use the right granularity** — one sub-agent per focused task, not one per tiny step

## Examples

### Basic: Single Specialist Sub-Agent

```python
from adk_deepagents import SubAgentSpec, create_deep_agent

researcher = SubAgentSpec(
    name="researcher",
    description="Research agent for gathering information on topics.",
    system_prompt="You are a research assistant. Search for information and return structured findings.",
)

agent = create_deep_agent(
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant. Delegate research tasks to the researcher.",
    subagents=[researcher],
)
```

### Multiple Sub-Agents with Different Models

```python
from adk_deepagents import SubAgentSpec, create_deep_agent

researcher = SubAgentSpec(
    name="researcher",
    description="Deep research agent using a powerful model.",
    model="gemini-2.5-pro",
)

writer = SubAgentSpec(
    name="writer",
    description="Fast content writer for drafting text.",
    model="gemini-2.5-flash",
)

agent = create_deep_agent(
    model="gemini-2.5-flash",
    subagents=[researcher, writer],
)
```

### Sub-Agent with Custom Tools

```python
from adk_deepagents import SubAgentSpec, create_deep_agent


def web_search(query: str) -> str:
    """Search the web for information."""
    # Your search implementation
    return f"Results for: {query}"


researcher = SubAgentSpec(
    name="researcher",
    description="Research agent with web search capability.",
    tools=[web_search],  # Only has web_search, not the default filesystem tools
)

agent = create_deep_agent(
    subagents=[researcher],
)
```

### Pre-Built LlmAgent as Sub-Agent

```python
from google.adk.agents import LlmAgent
from adk_deepagents import create_deep_agent


def specialized_tool(query: str) -> str:
    """A specialized tool only this agent has."""
    return f"Specialized result: {query}"


custom_agent = LlmAgent(
    name="specialist",
    model="gemini-2.5-pro",
    instruction="You are a domain specialist. Use your specialized tool.",
    description="Specialist agent for domain-specific queries.",
    tools=[specialized_tool],
)

agent = create_deep_agent(
    subagents=[custom_agent],
)
```

### Sub-Agent with Skills

```python
from adk_deepagents import SubAgentSpec, create_deep_agent

content_writer = SubAgentSpec(
    name="content_writer",
    description="Content writing agent with blog and social media skills.",
    skills=["./skills/writing/"],
)

agent = create_deep_agent(
    subagents=[content_writer],
)
```

### Sub-Agent with interrupt_on

```python
from adk_deepagents import SubAgentSpec, create_deep_agent

coder = SubAgentSpec(
    name="coder",
    description="Coding agent that writes and executes code.",
    interrupt_on={
        "write_file": True,
        "execute": True,
    },
)

agent = create_deep_agent(
    subagents=[coder],
    execution="local",
)
```
