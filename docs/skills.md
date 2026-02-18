# Skills Integration

## Overview

adk-deepagents integrates with the [adk-skills-agent](https://pypi.org/project/adk-skills-agent/) library to provide Agent Skills support. Agent Skills are `SKILL.md` files that offer progressive disclosure of instructions, script execution, and reference documentation to the agent.

## What Agent Skills Provide

Agent Skills enable:

- **Progressive disclosure** — Skills are listed as summaries; the agent calls `use_skill` to load full instructions only when needed
- **Script execution** — Skills can include runnable scripts via `run_script`
- **Reference docs** — Skills can bundle reference materials accessible via `read_reference`

## Installation

Skills support requires the `adk-skills-agent` package. Install it as an extra:

```bash
pip install adk-deepagents[skills]
```

Or install directly:

```bash
pip install adk-skills-agent
```

## add_skills_tools

The `add_skills_tools` function in `adk_deepagents.skills.integration` discovers skills and adds the corresponding tools:

```python
from adk_deepagents.skills.integration import add_skills_tools

tools = add_skills_tools(
    tools=[],                        # Existing tool list to extend
    skills_dirs=["./skills/"],       # Directories to discover skills from
    skills_config=None,              # Optional SkillsConfig
    state=None,                      # Optional session state dict
)
```

**What it does:**

1. Creates a `SkillsRegistry` with optional config kwargs
2. Calls `registry.discover(directory)` for each directory in `skills_dirs`
3. Creates and appends three tools to the tool list:
   - `use_skill` — Load a skill's full instructions
   - `run_script` — Execute a skill's bundled script
   - `read_reference` — Read a skill's reference documentation
4. Optionally stores the registry and metadata in `state`

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `tools` | `list[Callable]` | Existing tool list to extend |
| `skills_dirs` | `list[str]` | Directories to discover skills from |
| `skills_config` | `SkillsConfig \| None` | Optional configuration |
| `state` | `dict \| None` | Session state for storing registry |

## inject_skills_into_prompt

An alternative to tool-based discovery that injects an `<available_skills>` block directly into the system instruction:

```python
from adk_deepagents.skills.integration import inject_skills_into_prompt

instruction = inject_skills_into_prompt(
    instruction="You are a helpful assistant.",
    state=state,        # Must contain "_skills_registry" from add_skills_tools
    format="xml",       # Output format (default "xml")
)
```

This appends the skills listing to the instruction string using the registry's `inject_skills_prompt` method.

## SkillsConfig

The `SkillsConfig` dataclass holds extra kwargs passed to `SkillsRegistry`:

```python
from adk_deepagents import SkillsConfig

config = SkillsConfig(
    extra={"some_option": "value"},
)
```

The `extra` dict is unpacked as kwargs when creating the `SkillsRegistry`:

```python
registry = SkillsRegistry(**config.extra)
```

## Skills in Sub-Agents

Each sub-agent can have its own skills directories, independent of the parent agent:

```python
from adk_deepagents import SubAgentSpec, create_deep_agent

writer = SubAgentSpec(
    name="writer",
    description="Content writer with writing-specific skills.",
    skills=["./skills/writing/"],
)

reviewer = SubAgentSpec(
    name="reviewer",
    description="Code reviewer with review-specific skills.",
    skills=["./skills/review/"],
)

agent = create_deep_agent(
    skills=["./skills/general/"],   # Parent agent skills
    subagents=[writer, reviewer],    # Each has its own skills
)
```

When building sub-agent tools, `_resolve_skills_tools` is called for sub-agents that have `skills` set. This creates a separate `SkillsRegistry` for each sub-agent and appends the skills tools to that sub-agent's tool list.

## Bridge with Heimdall

Skills that include scripts can be executed in Heimdall's sandbox via the `HeimdallScriptExecutor` bridge (see [Execution](./execution.md#heimdallscriptexecutor-bridge)):

```python
from adk_deepagents.execution.bridge import HeimdallScriptExecutor
from adk_deepagents.execution.heimdall import get_heimdall_tools

tools, cleanup = await get_heimdall_tools()
executor = HeimdallScriptExecutor(tools)

# Execute a skill's script in the sandbox
result = await executor.execute("analyze.py", script_content)
```

## Examples

### Basic Skills Discovery

```python
from adk_deepagents import create_deep_agent

agent = create_deep_agent(
    instruction="You are a helpful assistant with skills.",
    skills=["./skills/"],
)
```

The agent will discover all `SKILL.md` files under `./skills/` and gain `use_skill`, `run_script`, and `read_reference` tools.

### Skills with Custom Config

```python
from adk_deepagents import SkillsConfig, create_deep_agent

agent = create_deep_agent(
    skills=["./skills/"],
    skills_config=SkillsConfig(
        extra={"discovery_depth": 3},
    ),
)
```

### Skills in Sub-Agents

```python
from adk_deepagents import SubAgentSpec, create_deep_agent

content_writer = SubAgentSpec(
    name="content_writer",
    description="Content writing agent with blog and social media skills.",
    skills=["./skills/blog/", "./skills/social/"],
)

agent = create_deep_agent(
    subagents=[content_writer],
)
```

### Skills with Prompt Injection

```python
from adk_deepagents.skills.integration import add_skills_tools, inject_skills_into_prompt

# First, set up skills tools and store registry in state
state = {}
tools = add_skills_tools(
    tools=[],
    skills_dirs=["./skills/"],
    state=state,
)

# Then inject skills listing into the prompt
instruction = inject_skills_into_prompt(
    instruction="You are a helpful assistant.",
    state=state,
)
# instruction now contains an <available_skills> XML block
```
