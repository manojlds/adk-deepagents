# Examples

This guide walks through the example applications included in the `examples/` directory. Each example demonstrates different features of adk-deepagents.

## Quickstart

**Location:** `examples/quickstart/`

The minimal working agent. Demonstrates the simplest possible `create_deep_agent` setup.

### Architecture

```
┌──────────────────────────────────┐
│         quickstart_agent         │
│  Model: gemini-2.5-flash         │
│  Backend: StateBackend (default) │
│                                  │
│  Tools:                          │
│  ├─ ls, read_file, write_file   │
│  ├─ edit_file, glob, grep       │
│  └─ write_todos, read_todos     │
└──────────────────────────────────┘
```

### Code Walkthrough

```python
from adk_deepagents import create_deep_agent

root_agent = create_deep_agent(
    name="quickstart_agent",
    instruction=(
        "You are a helpful coding assistant. Use your filesystem and todo tools "
        "to help the user organize and manage their work."
    ),
)
```

**What you get with defaults:**

- **Model:** Gemini 2.5 Flash
- **Backend:** `StateBackend` (in-memory file storage via session state)
- **Tools:** Filesystem tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`) + todo tools (`write_todos`, `read_todos`)
- **No** sub-agents, skills, memory, execution, or summarization

### How to Run

```bash
# With GOOGLE_API_KEY set
python examples/quickstart/agent.py

# Or with ADK CLI
adk run examples/quickstart/
```

The agent starts an interactive loop where you can ask it to read files, write code, manage todos, and more.

---

## Content Builder

**Location:** `examples/content_builder/`

Demonstrates skills, memory, sub-agents, and a filesystem backend working together.

### Architecture

```
┌───────────────────────────────────────────────────────┐
│                  content_builder                       │
│  Model: gemini-2.5-flash                               │
│  Backend: FilesystemBackend (root_dir=".", virtual)    │
│  Memory: ./AGENTS.md                                   │
│  Skills: ./skills/                                     │
│                                                        │
│  Tools:                                                │
│  ├─ Filesystem (ls, read, write, edit, glob, grep)    │
│  ├─ Todos (write_todos, read_todos)                   │
│  └─ Skills (use_skill, run_script, read_reference)    │
│                                                        │
│  Sub-agents:                                           │
│  ├─ general_purpose (auto-included)                   │
│  └─ researcher                                         │
│     └─ "Research agent for gathering information..."   │
└───────────────────────────────────────────────────────┘
```

### Configuration Choices

| Feature | Choice | Why |
|---|---|---|
| Backend | `FilesystemBackend(root_dir=".", virtual_mode=True)` | Writes output files to disk instead of in-memory state |
| Memory | `["./AGENTS.md"]` | Loads project context from AGENTS.md at startup |
| Skills | `["./skills/"]` | Discovers writing skills (blog, social media) |
| Sub-agents | `[researcher]` | Delegates research to a specialized sub-agent |

### Key Code

```python
from adk_deepagents import SubAgentSpec, create_deep_agent
from adk_deepagents.backends import FilesystemBackend

researcher = SubAgentSpec(
    name="researcher",
    description="Research agent for gathering information on topics.",
    system_prompt="You are a research assistant. Gather relevant information.",
)

root_agent = create_deep_agent(
    name="content_builder",
    model="gemini-2.5-flash",
    instruction="You are a content creation assistant...",
    memory=["./AGENTS.md"],
    skills=["./skills/"],
    subagents=[researcher],
    backend=FilesystemBackend(root_dir=".", virtual_mode=True),
)
```

### How to Run

```bash
# Requires adk-skills-agent for skills support
pip install adk-deepagents[skills]

python examples/content_builder/agent.py
```

---

## Deep Research

**Location:** `examples/deep_research/`

A multi-model research agent with parallel sub-agent delegation, web search, strategic thinking, and conversation summarization.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      deep_research                           │
│  Model: configurable (default gemini-2.5-flash)              │
│  Backend: StateBackend (default)                             │
│  Summarization: trigger=0.75, keep=8 messages                │
│                                                              │
│  Tools:                                                      │
│  ├─ Filesystem (ls, read, write, edit, glob, grep)          │
│  ├─ Todos (write_todos, read_todos)                         │
│  ├─ web_search (Tavily → DuckDuckGo fallback)              │
│  └─ think (structured reflection)                           │
│                                                              │
│  Sub-agents:                                                 │
│  ├─ general_purpose (auto-included)                         │
│  └─ research_agent                                           │
│     ├─ Tools: web_search, think                             │
│     └─ Prompt: researcher instructions with date, limits    │
│                                                              │
│  Workflow:                                                   │
│  1. Plan → create todo list                                 │
│  2. Save request → /research_request.md                     │
│  3. Research → delegate to sub-agents (parallel)            │
│  4. Synthesize → consolidate citations                      │
│  5. Write → /final_report.md                                │
│  6. Verify → cross-check against original request           │
└─────────────────────────────────────────────────────────────┘
```

### Configuration Choices

| Feature | Choice | Why |
|---|---|---|
| Model | Configurable via `--model` | Supports Gemini, OpenAI, Anthropic via litellm |
| Sub-agents | `research_agent` with `web_search` + `think` | Focused research with reflection cycle |
| Summarization | `trigger=0.75`, `keep=8` | Long research sessions need context management |
| Custom tools | `web_search`, `think` | Web search with Tavily/DuckDuckGo; think for reflection |

### Custom Tools

**`web_search(query, max_results=3, topic="general")`**

Searches the web using Tavily (if available) with DuckDuckGo as fallback. Fetches full page content for each result. Supports `topic` hints: `"general"`, `"news"`, `"finance"`.

**`think(reflection)`**

A strategic thinking tool that captures the agent's reasoning. Used after each web search to assess findings and plan next steps.

### Multi-Model Support

```bash
# Default (Gemini)
python examples/deep_research/agent.py

# OpenAI (requires OPENAI_API_KEY + litellm)
python examples/deep_research/agent.py --model openai/gpt-4o

# Anthropic (requires ANTHROPIC_API_KEY + litellm)
python examples/deep_research/agent.py --model anthropic/claude-sonnet-4-20250514
```

### Key Code

```python
from adk_deepagents import SubAgentSpec, SummarizationConfig, create_deep_agent

from .tools import think, web_search

research_subagent = SubAgentSpec(
    name="research_agent",
    description="Delegate a research task to this sub-agent...",
    system_prompt=_build_researcher_prompt(),
    tools=[web_search, think],
)

agent = create_deep_agent(
    name="deep_research",
    model=model,
    instruction=_build_orchestrator_prompt(),
    tools=[web_search, think],
    subagents=[research_subagent],
    summarization=SummarizationConfig(
        model=model if model.startswith("gemini") else "gemini-2.5-flash",
        trigger=("fraction", 0.75),
        keep=("messages", 8),
    ),
)
```

### How to Run

```bash
# Basic
python examples/deep_research/agent.py

# With Tavily (better search results)
export TAVILY_API_KEY=your-key
python examples/deep_research/agent.py

# With ADK CLI
adk run examples/deep_research/
```

---

## Sandboxed Coder

**Location:** `examples/sandboxed_coder/`

Demonstrates Heimdall MCP sandboxed execution with skills, supporting both sync (local) and async (Heimdall) factories.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    sandboxed_coder                       │
│  Model: gemini-2.5-flash                                 │
│  Backend: StateBackend (default)                         │
│  Skills: ./skills/ (code review guidelines)              │
│                                                          │
│  Tools (Heimdall mode):                                  │
│  ├─ Filesystem (ls, read, write, edit, glob, grep)      │
│  ├─ Todos (write_todos, read_todos)                     │
│  ├─ Skills (use_skill, run_script, read_reference)      │
│  ├─ execute_python (Pyodide WebAssembly)                │
│  ├─ execute_bash (just-bash)                            │
│  ├─ install_packages (micropip)                         │
│  └─ Workspace tools (write_file, read_file, list_files) │
│                                                          │
│  Tools (Local mode):                                     │
│  ├─ Filesystem (ls, read, write, edit, glob, grep)      │
│  ├─ Todos (write_todos, read_todos)                     │
│  ├─ Skills (use_skill, run_script, read_reference)      │
│  └─ execute (subprocess.run)                            │
│                                                          │
│  Workflow:                                               │
│  1. Understand → read request                           │
│  2. Plan → create todos                                 │
│  3. Implement → write code to /workspace                │
│  4. Test → execute and verify                           │
│  5. Review → activate code-review skill                 │
└─────────────────────────────────────────────────────────┘
```

### Two Factory Functions

The example provides two factory functions:

#### Sync: `build_agent()` — for ADK CLI

Uses local subprocess execution. Simple but unsandboxed.

```python
def build_agent(model="gemini-2.5-flash"):
    return create_deep_agent(
        name="sandboxed_coder",
        model=model,
        instruction=_build_prompt(),
        skills=[SKILLS_DIR],
        execution="local",
    )
```

#### Async: `build_agent_async()` — for Heimdall MCP

Resolves Heimdall MCP tools asynchronously. Returns `(agent, cleanup)`.

```python
async def build_agent_async(model="gemini-2.5-flash"):
    return await create_deep_agent_async(
        name="sandboxed_coder",
        model=model,
        instruction=_build_prompt(),
        skills=[SKILLS_DIR],
        execution="heimdall",
    )
```

### Configuration Choices

| Feature | Choice | Why |
|---|---|---|
| Execution (sync) | `"local"` | Quick setup for ADK CLI, no Heimdall needed |
| Execution (async) | `"heimdall"` | Sandboxed execution for production use |
| Skills | `[SKILLS_DIR]` | Code review guidelines via SKILL.md |
| Prompt | 4 combined sections | Detailed coding workflow, execution, testing, quality |

### Prompt Structure

The system prompt is composed of four template sections:

1. **CODING_WORKFLOW_INSTRUCTIONS** — Implementation guidelines, file operations, iteration strategy
2. **EXECUTION_INSTRUCTIONS** — `execute_python`, `execute_bash`, `install_packages` usage, cross-language workflows
3. **TESTING_INSTRUCTIONS** — Unit testing patterns, TDD, debugging approach
4. **CODE_QUALITY_INSTRUCTIONS** — Code structure, error handling, performance, security

### How to Run

```bash
# Local execution (no Heimdall needed)
adk run examples/sandboxed_coder/

# Heimdall sandboxed execution
npm i -g @heimdall-ai/heimdall
python -m examples.sandboxed_coder.agent
```

The Heimdall runner uses the async factory and properly cleans up the MCP connection:

```python
async def main():
    agent, cleanup = await build_agent_async()
    try:
        # Run interactive session...
        pass
    finally:
        if cleanup:
            await cleanup()
```
