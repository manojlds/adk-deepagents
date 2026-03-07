# Browser Research Agent

A hybrid research agent that combines web search APIs with browser automation
for comprehensive data gathering. Uses web search for discovery and simple
pages, and Playwright MCP browser tools for JavaScript-heavy sites,
interactive content, and pages requiring form interaction.

## Features

- **Hybrid approach** — Web search for discovery + browser for complex pages
- **Runtime specialist bootstrap** — Registers a browser specialist via `register_subagent`
- **Dynamic delegation** — Orchestrator routes specialist work with `task`
- **Accessibility-tree based** — Uses ARIA snapshots for reliable extraction
- **Summarization** — Manages context window for long research sessions

## Architecture

```
┌─────────────────────────────────────────────────┐
│         Orchestrator (browser_research)          │
│  - Plans via write_todos                         │
│  - Uses web_search for discovery                 │
│  - Registers + delegates browser specialist      │
│  - Synthesizes findings into report              │
└────────┬──────────────────┬──────────────────────┘
         │                  │
 ┌───────▼──────┐   ┌──────▼──────────────┐
 │  web_search  │   │ browser_researcher  │
│ (Serper/     │   │ (runtime specialist)│
 │  Tavily/     │   │  - browser_navigate │
 │  Brave/DDG)  │   │  - browser_snapshot │
 └──────────────┘   │  - browser_click    │
                    │  - browser_type     │
                    └──────┬──────────────┘
                           │
                    ┌──────▼──────┐
                    │ @playwright  │
                    │   /mcp       │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Chromium    │
                    └─────────────┘
```

## Prerequisites

### 1. Node.js (for @playwright/mcp)

```bash
node --version  # >= 18
```

### 2. Python dependencies

```bash
uv sync
```

### 3. API keys

```bash
# Model
export GOOGLE_API_KEY=your-key

# Search (at least one)
export SERPER_API_KEY=your-serper-key
# or: export TAVILY_API_KEY=your-tavily-key
# or: export BRAVE_SEARCH_API_KEY=your-brave-key
# Falls back to DuckDuckGo if none set
```

## Quick Start

```bash
python -m examples.browser_research.agent
```

## Example Session

```
You: Research the pricing of the top 3 cloud providers and compare their
     free tier offerings. Check the actual pricing pages.

Agent: I'll plan the research and use both search and browser tools.
[register_subagent name=browser_researcher ...]
[writes research plan to todos]
[web_search: "cloud provider pricing comparison 2025"]
[browser_navigate: https://aws.amazon.com/free/]
[browser_snapshot: reads AWS free tier details]
[browser_navigate: https://cloud.google.com/free]
[browser_snapshot: reads GCP free tier details]
[browser_navigate: https://azure.microsoft.com/en-us/pricing/free-services/]
[browser_snapshot: reads Azure free tier details]

Agent: [writes /report.md with comparison table and citations]

## Cloud Provider Free Tier Comparison

| Feature | AWS | GCP | Azure |
|---------|-----|-----|-------|
| Compute | 750 hrs t2.micro/mo | e2-micro always free | 750 hrs B1S/mo |
| Storage | 5 GB S3 | 5 GB Cloud Storage | 5 GB Blob |
| ...     | ...                 | ...                  | ...             |

### Sources
[1] https://aws.amazon.com/free/
[2] https://cloud.google.com/free
[3] https://azure.microsoft.com/en-us/pricing/free-services/
```

## When to Use This vs. Deep Research

| Scenario | Use This | Use Deep Research |
|----------|----------|-------------------|
| Content behind JavaScript rendering | ✅ | ❌ |
| Interactive pages (tabs, filters) | ✅ | ❌ |
| Simple articles and docs | Either | ✅ |
| Broad topic survey | Either | ✅ |
| Form-gated content | ✅ | ❌ |
| Pricing/comparison pages | ✅ | ❌ |

## Files

| File | Description |
|------|-------------|
| `agent.py` | Agent factory, runtime specialist bootstrap, CLI runner |
| `prompts.py` | Research workflow and browser specialist prompts |
