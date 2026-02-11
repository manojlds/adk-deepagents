# Sandboxed Coder

A coding assistant that writes, executes, and tests code in a sandboxed
environment using [Heimdall MCP](https://github.com/manojlds/heimdall) for
secure execution and [Agent Skills](https://agentskills.io) for code review.

## Features

- **Sandboxed Python execution** — Pyodide WebAssembly sandbox (memory-isolated,
  no network access from user code)
- **Sandboxed Bash execution** — just-bash simulation (no real process spawning)
  with 50+ built-in commands
- **Shared workspace** — Files persist across executions and are shared between
  Python and Bash (`/workspace`)
- **Package installation** — Install Python packages via micropip (numpy, pandas,
  scipy, matplotlib, etc.)
- **Cross-language workflows** — Bash prepares data → Python analyzes → Bash
  post-processes
- **Code review skill** — On-demand code review with structured checklist and
  feedback format
- **Planning** — Todo-based task tracking for multi-step coding tasks

## Architecture

```
┌──────────────────────────────────────────────┐
│          Sandboxed Coder Agent               │
│  - Plans tasks via write_todos               │
│  - Writes code to /workspace                 │
│  - Executes and tests in sandbox             │
│  - Activates code-review skill               │
└──────┬───────────┬───────────┬───────────────┘
       │           │           │
  ┌────▼────┐ ┌────▼────┐ ┌───▼──────┐
  │ Python  │ │  Bash   │ │ Package  │
  │ Sandbox │ │ Sandbox │ │ Install  │
  │(Pyodide)│ │(just-sh)│ │(micropip)│
  └────┬────┘ └────┬────┘ └──────────┘
       │           │
       └─────┬─────┘
             │
      ┌──────▼──────┐
      │  /workspace  │
      │ (shared fs)  │
      └─────────────┘
```

## Prerequisites

### 1. Heimdall MCP server

```bash
npm install -g @heimdall-ai/heimdall
```

Or use npx (the agent runs `npx @heimdall-ai/heimdall` by default).

### 2. Python dependencies

```bash
uv sync
```

### 3. API key

```bash
# For Gemini (default)
export GOOGLE_API_KEY=your-key
```

## Quick Start

```bash
# Run the interactive agent
python -m examples.sandboxed_coder.agent

# Or with ADK CLI
adk run examples/sandboxed_coder/
```

## Example Session

```
You: Write a function that finds the longest common subsequence of two strings,
     then test it.

Agent: I'll write the LCS function and tests.
[writes /workspace/lcs.py with the implementation]
[executes Python to run the code]
[writes /workspace/test_lcs.py with test cases]
[executes Python to run tests]
All 5 tests passed!

You: Now review the code quality.

Agent: [activates code-review skill]
## Code Review Summary
The LCS implementation is correct and well-structured.

## Issues Found
### Minor: Missing type hints
- File: /workspace/lcs.py
- Line: 1
- Suggestion: Add type hints to function signature

## Positive Notes
- Clean dynamic programming approach
- Good edge case handling (empty strings)
- Comprehensive test coverage
```

## Cross-Language Workflow Example

```
You: Download this CSV data, clean it with bash, then analyze with pandas.

Agent: [execute_bash: creates sample CSV data in /workspace/raw.csv]
[execute_bash: uses awk/sed to clean and normalize the CSV]
[execute_python: loads cleaned CSV with pandas, computes statistics]
[execute_python: generates summary and saves to /workspace/report.csv]
```

## Skills

The `skills/` directory contains Agent Skills that the coder can activate
on demand:

| Skill | Description |
|-------|-------------|
| `code-review` | Structured code review with checklist for correctness, security, performance, readability, and testing |

Skills are loaded via [adk-skills](https://github.com/manojlds/adk-skills)
(optional dependency). The agent discovers available skills automatically
and activates them with the `use_skill` tool when needed.

## Security Model

| Layer | Protection |
|-------|-----------|
| Python execution | WebAssembly sandbox — memory-isolated, no network access |
| Bash execution | TypeScript simulation — no real process spawning |
| Filesystem | `/workspace` directory only — no host filesystem access |
| Packages | Pyodide's trusted micropip mechanism only |
| Timeouts | Configurable execution timeouts prevent infinite loops |

## Programmatic Usage

```python
import asyncio
from examples.sandboxed_coder.agent import build_agent_async

async def main():
    agent, cleanup = await build_agent_async()
    try:
        # Use agent with ADK Runner...
        pass
    finally:
        if cleanup:
            await cleanup()

asyncio.run(main())
```

## Files

| File | Description |
|------|-------------|
| `agent.py` | Agent factory, CLI runner, configuration |
| `prompts.py` | Coding workflow, execution, testing, and quality prompts |
| `skills/code-review/SKILL.md` | Code review skill with checklist and feedback format |
