# Deep Research Agent

A deep-research ADK app built on `adk_deepagents` with dynamic task delegation,
search-provider routing, and report quality grading.

## Features

- **Dynamic delegation** — Uses the dynamic `task` tool with specialist roles:
  `planner`, `researcher`, `reporter`, `grader`
- **ADK-native app** — Works with `adk run`, `adk web`, and `adk api_server`
- **Provider-routed web search** — `auto` mode prioritizes `serper` first
- **Hard-fail search semantics** — If selected provider fails, search returns
  an explicit error (no silent provider fallback)
- **Citation-oriented workflow** — Encourages inline citations and source list

## Architecture

```
┌─────────────────────────────────────────────────┐
│           Orchestrator (deep_research)           │
│  - Plans via write_todos + planner task          │
│  - Delegates research with dynamic task tool      │
│  - Drafts report via reporter, grades via grader  │
│  - Writes final report to /final_report.md        │
└────────┬────────────┬────────────┬──────────────┘
         │            │            │
 ┌───────▼──────┐ ┌────▼─────┐ ┌────▼─────┐ ┌────▼─────┐
 │ planner task │ │ research │ │ reporter │ │  grader  │
 │              │ │  tasks   │ │   task   │ │   task   │
 └──────────────┘ └──────────┘ └──────────┘ └──────────┘
```

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Set API keys

```bash
# Model setup (same convention as integration tests)
export LITELLM_MODEL=openai/gpt-4o-mini
export OPENAI_API_KEY=your-key

# Search provider setup (Serper first)
export SERPER_API_KEY=your-serper-key

# Optional provider keys (used only if selected/available)
export TAVILY_API_KEY=your-tavily-key
export BRAVE_SEARCH_API_KEY=your-brave-key

# Optional provider selector (default: auto)
export DEEP_RESEARCH_SEARCH_PROVIDER=auto
```

### 3. Run

```bash
# Interactive runner
python -m examples.deep_research.agent

# ADK CLI runtime
adk run examples/deep_research/

# ADK Dev UI
adk web

# ADK FastAPI server
adk api_server
```

## Search Provider Configuration

Provider selection is controlled by `DEEP_RESEARCH_SEARCH_PROVIDER`:

- `auto` (default): `serper` -> `tavily` -> `brave` -> `duckduckgo`
- `serper`
- `tavily`
- `brave`
- `duckduckgo`

Hard-fail behavior:

- If the selected provider fails, the tool returns an explicit error.
- In `auto` mode, if a keyed provider is selected and fails, it does not silently fallback.
- DuckDuckGo is used in `auto` only when no keyed provider is configured.

## Example Session

```
You: Research the current state of quantum computing in 2025

Agent: I'll build a research plan and delegate focused tasks.
[writes todos]
[task subagent_type=researcher ...]
[task subagent_type=reporter ...]
[task subagent_type=grader ...]

Agent: [writes /final_report.md with structured report and numbered sources]

Agent: I've completed the research. The final report is saved to
/final_report.md with citations and a graded quality pass.
```

## Files

| File | Description |
|------|-------------|
| `agent.py` | Main agent, dynamic delegation configuration, CLI runner |
| `prompts.py` | Orchestrator + planner/researcher/reporter/grader prompts |
| `tools.py` | Provider-routed web search (Serper-first auto) and think tool |
