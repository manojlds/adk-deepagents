# Deep Research Agent

A multi-model deep research agent that uses parallel sub-agent delegation to
conduct thorough web research, synthesize findings, and produce structured
reports with proper citations.

Port of langchain deepagents
[`examples/deep_research/`](https://github.com/langchain-ai/deepagents/tree/main/examples/deep_research).

## Features

- **Multi-model support** — Gemini (default), OpenAI, Anthropic, Groq, or any
  ADK-compatible model
- **Parallel sub-agent delegation** — Research tasks are delegated to
  specialized sub-agents that run independently
- **Web search with full page content** — Uses Tavily (preferred) or
  DuckDuckGo fallback, fetches full page content for deep analysis
- **Strategic thinking tool** — Mandatory reflection after each search to
  assess findings and plan next steps
- **Todo-based planning** — Orchestrator creates and tracks research tasks
- **Citation consolidation** — Unique URLs get one citation number across all
  sub-agent findings
- **Conversation summarization** — Manages long sessions by summarizing older
  context

## Architecture

```
┌─────────────────────────────────────────────────┐
│           Orchestrator (deep_research)           │
│  - Plans research via write_todos                │
│  - Delegates to research sub-agents              │
│  - Consolidates citations                        │
│  - Writes final report to /final_report.md       │
└────────┬────────────┬────────────┬──────────────┘
         │            │            │
    ┌────▼────┐  ┌────▼────┐  ┌───▼─────┐
    │Research │  │Research │  │Research │  (up to 3 parallel)
    │Agent 1  │  │Agent 2  │  │Agent 3  │
    │         │  │         │  │         │
    │ search  │  │ search  │  │ search  │
    │ think   │  │ think   │  │ think   │
    │ search  │  │ search  │  │ search  │
    │ think   │  │ think   │  │ think   │
    └─────────┘  └─────────┘  └─────────┘
```

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Set API keys

```bash
# For Gemini (default)
export GOOGLE_API_KEY=your-key

# Optional: For better web search results
export TAVILY_API_KEY=your-key
```

### 3. Run

```bash
# Default (Gemini 2.5 Flash)
python -m examples.deep_research.agent

# Or with ADK CLI
adk run examples/deep_research/
```

## Multi-Model Support

ADK supports any model through its LLM registry. Non-Gemini models require
[`litellm`](https://docs.litellm.ai/):

```bash
pip install litellm
```

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
python -m examples.deep_research.agent --model openai/gpt-4o
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m examples.deep_research.agent --model anthropic/claude-sonnet-4-20250514
```

### Groq

```bash
export GROQ_API_KEY=...
python -m examples.deep_research.agent --model groq/llama3-70b-8192
```

### Programmatic usage with any model

```python
from examples.deep_research.agent import build_agent

# OpenAI
agent = build_agent("openai/gpt-4o")

# Anthropic
agent = build_agent("anthropic/claude-sonnet-4-20250514")

# Gemini Pro
agent = build_agent("gemini-2.5-pro")
```

## Web Search Configuration

The agent uses a tiered search approach:

1. **Tavily** (preferred) — Set `TAVILY_API_KEY` for high-quality results with
   full page content. Install: `pip install tavily-python`
2. **DuckDuckGo** (fallback) — No API key needed, basic HTML scraping

Both backends fetch full page content (not just snippets) for deeper analysis.

## Example Session

```
You: Research the current state of quantum computing in 2025

Agent: I'll create a research plan for quantum computing in 2025.
[writes todos]
[delegates to research_agent: "Research current quantum computing hardware
 advances in 2025, including qubit counts, error rates, and major milestones"]
[delegates to research_agent: "Research quantum computing applications and
 commercialization progress in 2025"]

Agent: [synthesizes findings, consolidates citations]
[writes /final_report.md with structured report and numbered sources]

Agent: I've completed the research. The final report has been saved to
/final_report.md with 12 cited sources covering hardware advances,
software developments, and commercial applications.
```

## Files

| File | Description |
|------|-------------|
| `agent.py` | Main agent, sub-agent definitions, CLI runner |
| `prompts.py` | Research workflow, delegation, and researcher prompts |
| `tools.py` | Web search (Tavily/DuckDuckGo) and think tools |
