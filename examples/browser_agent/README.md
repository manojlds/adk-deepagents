# Browser Agent

A browser automation agent that navigates websites, fills forms, extracts
data, and performs web-based tasks using
[@playwright/mcp](https://github.com/microsoft/playwright-mcp) for
structured browser control.

## Features

- **Accessibility-tree based** — Uses ARIA snapshots instead of screenshots
  for token-efficient, reliable page understanding
- **Element refs** — Targets elements by stable ref IDs (e.g., `e1`, `e5`)
  from accessibility snapshots
- **Form automation** — Fill multiple form fields in a single call with
  `browser_fill_form`
- **Data extraction** — Parse structured data from accessibility trees
- **Multi-step workflows** — Navigate, interact, verify across pages
- **Todo planning** — Track progress on complex multi-step tasks

## Architecture

```
┌─────────────────────────────────────────┐
│          Browser Agent                  │
│  - Plans tasks via write_todos          │
│  - Navigates with browser_navigate      │
│  - Reads pages via browser_snapshot     │
│  - Interacts via click/type/fill        │
└──────────┬──────────────────────────────┘
           │
    ┌──────▼──────┐
    │ @playwright  │
    │   /mcp       │
    │ (MCP server) │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Chromium /  │
    │  Firefox /   │
    │  WebKit      │
    └─────────────┘
```

## Prerequisites

### 1. Node.js (for @playwright/mcp)

```bash
# Playwright MCP is started automatically via npx
# Ensure Node.js >= 18 is installed
node --version
```

### 2. Python dependencies

```bash
uv sync
```

### 3. API key

```bash
# For Gemini (default)
export GOOGLE_API_KEY=your-key

# Or for OpenAI/Anthropic via litellm
export LITELLM_MODEL=openai/gpt-4o
export OPENAI_API_KEY=your-key
```

## Quick Start

```bash
# Run the interactive agent
python -m examples.browser_agent.agent
```

## Example Session

```
You: Go to https://news.ycombinator.com and get the top 5 stories

Agent: I'll navigate to Hacker News and extract the top stories.
[browser_navigate: https://news.ycombinator.com]
[browser_snapshot: gets accessibility tree with story titles and links]

Here are the top 5 stories on Hacker News:

| # | Title | Points |
|---|-------|--------|
| 1 | Show HN: I built a ... | 342 |
| 2 | Why Rust is great for ... | 289 |
| 3 | The future of AI agents | 256 |
| 4 | New breakthrough in ... | 234 |
| 5 | How we scaled to ... | 198 |
```

## Browser Configuration

You can customize the browser via `BrowserConfig`:

```python
from adk_deepagents import BrowserConfig

# Headed mode (visible browser window)
config = BrowserConfig(headless=False)

# Firefox instead of Chromium
config = BrowserConfig(browser="firefox")

# Custom viewport
config = BrowserConfig(viewport=(1920, 1080))

# Connect to existing browser
config = BrowserConfig(cdp_endpoint="ws://localhost:9222")

# Load saved authentication state
config = BrowserConfig(storage_state="./auth.json")

# Enable vision tools (coordinate-based clicking)
config = BrowserConfig(caps=["vision"])
```

## Programmatic Usage

```python
import asyncio
from examples.browser_agent.agent import build_agent_async

async def main():
    agent, cleanup = await build_agent_async(headless=False)
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
| `prompts.py` | Browser workflow and data extraction prompts |
