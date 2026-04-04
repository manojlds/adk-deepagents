# Browser Automation

## Overview

adk-deepagents supports browser automation through two complementary approaches:

1. **Playwright MCP** — Programmatic browser control via the official [@playwright/mcp](https://github.com/microsoft/playwright-mcp) MCP server, integrated through ADK's `McpToolset`
2. **agent-browser CLI skill** — CLI-based browser automation via [agent-browser](https://github.com/vercel-labs/agent-browser), integrated through adk-skills SKILL.md discovery

Both approaches use **accessibility tree snapshots** (not screenshots) for token-efficient, reliable page interaction.

## Installation

### For Playwright MCP (Approach 1)

```bash
pip install adk-deepagents
```

Node.js >= 18 is required — `@playwright/mcp` is launched automatically via `npx`.

### For agent-browser CLI skill (Approach 2)

```bash
npm install -g agent-browser
pip install adk-deepagents[skills]
```

## Approach 1: Playwright MCP

### How It Works

The Playwright MCP server is launched as a stdio subprocess via `npx @playwright/mcp@latest`. It exposes ~25 browser tools through the MCP protocol, which ADK's `McpToolset` converts into callable tool functions.

The agent interacts with pages through the **snapshot → ref → action** workflow:

1. **Navigate** to a URL with `browser_navigate`
2. **Snapshot** the page with `browser_snapshot` — returns an accessibility tree with element refs (e.g., `e1`, `e5`)
3. **Interact** using refs — `browser_click(ref="e5")`, `browser_type(ref="e1", text="hello")`
4. **Re-snapshot** after page changes — refs are invalidated on navigation

### Quick Start

```python
from adk_deepagents import BrowserConfig, create_deep_agent_async

agent, cleanup = await create_deep_agent_async(
    name="browser_agent",
    instruction="You are a browser automation agent.",
    browser="playwright",  # or BrowserConfig(...)
)

try:
    # Use agent with ADK Runner...
    pass
finally:
    if cleanup:
        await cleanup()
```

### BrowserConfig

The `BrowserConfig` dataclass controls how the Playwright MCP server is launched:

```python
from adk_deepagents import BrowserConfig

config = BrowserConfig(
    provider="playwright",        # Only "playwright" is supported currently
    headless=True,                # Run without a visible window
    browser="chromium",           # "chromium", "firefox", or "webkit"
    viewport=(1280, 720),         # Browser viewport size
    caps=[],                      # Extra capabilities: "vision", "pdf", "testing"
    cdp_endpoint=None,            # Connect to existing browser via CDP
    storage_state=None,           # Path to saved authentication state
)
```

| Parameter | Default | Description |
|---|---|---|
| `headless` | `True` | Run browser without a visible window |
| `browser` | `"chromium"` | Browser engine: `"chromium"`, `"firefox"`, `"webkit"` |
| `viewport` | `(1280, 720)` | Viewport dimensions `(width, height)` |
| `caps` | `[]` | Extra tool groups: `"vision"` (coordinate clicks), `"pdf"` (PDF export), `"testing"` (assertions) |
| `cdp_endpoint` | `None` | Connect to an existing browser via Chrome DevTools Protocol |
| `storage_state` | `None` | Path to a saved authentication state file (cookies, localStorage) |

### Available Tools

#### Core Tools (always enabled)

| Tool | Description |
|---|---|
| `browser_navigate` | Navigate to a URL |
| `browser_navigate_back` | Go back in history |
| `browser_snapshot` | Capture accessibility tree with element refs |
| `browser_click` | Click an element by ref |
| `browser_type` | Type text into an element |
| `browser_fill_form` | Fill multiple form fields at once |
| `browser_select_option` | Select dropdown option(s) |
| `browser_hover` | Hover over an element |
| `browser_drag` | Drag and drop between elements |
| `browser_press_key` | Press a keyboard key |
| `browser_wait_for` | Wait for text, element, or time |
| `browser_handle_dialog` | Accept/dismiss browser dialogs |
| `browser_file_upload` | Upload files |
| `browser_evaluate` | Execute JavaScript |
| `browser_take_screenshot` | Capture a screenshot |
| `browser_network_requests` | List network requests |
| `browser_console_messages` | Get console messages |
| `browser_close` | Close the browser |
| `browser_resize` | Resize the viewport |
| `browser_tabs` | Manage browser tabs |

#### Vision Tools (`caps=["vision"]`)

Coordinate-based interaction for elements without ARIA roles:

| Tool | Description |
|---|---|
| `browser_mouse_click_xy` | Click at `(x, y)` coordinates |
| `browser_mouse_move_xy` | Move mouse to coordinates |
| `browser_mouse_drag_xy` | Drag between coordinates |

#### Testing Tools (`caps=["testing"]`)

Assertion tools for web testing:

| Tool | Description |
|---|---|
| `browser_verify_element_visible` | Assert element is visible |
| `browser_verify_text_visible` | Assert text is visible |
| `browser_verify_value` | Assert element value |
| `browser_generate_locator` | Generate a Playwright locator |

### get_playwright_browser_tools

For manual tool resolution (similar to `get_heimdall_tools`):

```python
from adk_deepagents.browser.playwright_mcp import get_playwright_browser_tools
from adk_deepagents import BrowserConfig, create_deep_agent

config = BrowserConfig(headless=False, browser="firefox")
tools, cleanup = await get_playwright_browser_tools(config=config)

try:
    agent = create_deep_agent(
        tools=tools,
        browser="_resolved",  # Signal that browser tools are already resolved
    )
    # Use agent...
finally:
    await cleanup()
```

### Authentication

To reuse a saved browser session (cookies, localStorage):

```python
config = BrowserConfig(
    storage_state="./auth.json",  # Saved from a previous session
)
```

### Connecting to Existing Browser

To connect to a running browser via Chrome DevTools Protocol:

```python
config = BrowserConfig(
    cdp_endpoint="ws://localhost:9222",
)
```

## Approach 2: agent-browser CLI Skill

### How It Works

The agent-browser CLI is exposed as an [Agent Skill](https://agentskills.io) via a `SKILL.md` file. When the agent needs browser capabilities, it:

1. Activates the skill with `use_skill("agent-browser")`
2. Learns the CLI commands from the skill instructions
3. Runs `agent-browser` commands via the `execute` tool (shell)

This mirrors how AI coding agents (Claude Code, Cursor) use agent-browser.

### Quick Start

```python
from adk_deepagents import create_deep_agent

agent = create_deep_agent(
    name="browser_skill",
    instruction="You are an agent with browser skills.",
    skills=["./examples/skills/"],  # Contains agent-browser/SKILL.md
    execution="local",               # Shell access for CLI commands
)
```

### Prerequisites

```bash
npm install -g agent-browser
pip install adk-deepagents[skills]
```

### Skill File

The skill file at `examples/skills/agent-browser/SKILL.md` teaches the agent the core workflow:

1. `agent-browser open <url>` — Navigate
2. `agent-browser snapshot -i` — Get element refs (`@e1`, `@e2`)
3. `agent-browser fill @e1 "text"` — Interact using refs
4. `agent-browser snapshot -i` — Re-snapshot after changes

### When to Use This vs Playwright MCP

| Scenario | CLI Skill | Playwright MCP |
|---|---|---|
| Agent has shell access | ✅ | Either |
| No async needed | ✅ | ❌ |
| Programmatic, structured tool calls | ❌ | ✅ |
| ADK CLI compatible (`adk run`) | ✅ | ❌ (needs async) |
| Production autonomous agents | Either | ✅ |

## Examples

### Browser Agent (`examples/browser_agent/`)

Standalone browser agent using Playwright MCP. Navigates websites, fills forms, extracts data.

```bash
python -m examples.browser_agent.agent
```

### Browser Research (`examples/browser_research/`)

Hybrid research agent: web search APIs + browser for JS-heavy pages.

```bash
python -m examples.browser_research.agent
```

### Browser Skill (`examples/browser_skill/`)

CLI-based browser via agent-browser skill + shell execution.

```bash
python -m examples.browser_skill.agent
# or: adk run examples/browser_skill/
```
