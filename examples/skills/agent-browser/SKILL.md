---
name: agent-browser
description: >
  Browser automation CLI for AI agents. Use when the user needs to interact
  with websites, including navigating pages, filling forms, clicking buttons,
  taking screenshots, extracting data, testing web apps, or automating any
  browser task. Triggers include requests to "open a website", "fill out a
  form", "click a button", "take a screenshot", "scrape data from a page",
  "test this web app", "login to a site", or any task requiring web interaction.
---

# Browser Automation with agent-browser

`agent-browser` is a CLI tool for browser automation. It runs a persistent
browser daemon and provides commands to navigate, snapshot, and interact
with web pages.

## Prerequisites

```bash
npm install -g agent-browser
# or use npx (auto-installs on first use)
```

## Core Workflow

Every browser automation follows this pattern:

1. **Navigate**: `agent-browser open <url>`
2. **Snapshot**: `agent-browser snapshot -i` (get element refs like `@e1`, `@e2`)
3. **Interact**: Use refs to click, fill, select
4. **Re-snapshot**: After navigation or DOM changes, get fresh refs

```bash
agent-browser open https://example.com/form
agent-browser snapshot -i
# Output shows refs: @e1 [input "email"], @e2 [input "password"], @e3 [button "Submit"]

agent-browser fill @e1 "user@example.com"
agent-browser fill @e2 "password123"
agent-browser click @e3
agent-browser wait --load networkidle
agent-browser snapshot -i  # Check result
```

## Essential Commands

```bash
# Navigation
agent-browser open <url>              # Navigate to URL
agent-browser close                   # Close browser

# Snapshot (get element refs)
agent-browser snapshot -i             # Interactive elements with refs
agent-browser snapshot -i -C          # Include cursor-interactive elements

# Interaction (use @refs from snapshot)
agent-browser click @e1               # Click element
agent-browser fill @e2 "text"         # Clear and type text
agent-browser type @e2 "text"         # Type without clearing
agent-browser select @e1 "option"     # Select dropdown option
agent-browser check @e1               # Check checkbox
agent-browser press Enter             # Press key
agent-browser scroll down 500         # Scroll page

# Get information
agent-browser get text @e1            # Get element text
agent-browser get url                 # Get current URL
agent-browser get title               # Get page title

# Wait
agent-browser wait @e1                # Wait for element
agent-browser wait --load networkidle # Wait for network idle
agent-browser wait --url "**/page"    # Wait for URL pattern
agent-browser wait 2000               # Wait milliseconds

# Capture
agent-browser screenshot              # Screenshot to temp dir
agent-browser screenshot --full       # Full page screenshot
agent-browser pdf output.pdf          # Save as PDF
```

## Common Patterns

### Form Submission

```bash
agent-browser open https://example.com/signup
agent-browser snapshot -i
agent-browser fill @e1 "Jane Doe"
agent-browser fill @e2 "jane@example.com"
agent-browser select @e3 "California"
agent-browser check @e4
agent-browser click @e5
agent-browser wait --load networkidle
```

### Data Extraction

```bash
agent-browser open https://example.com/products
agent-browser snapshot -i
agent-browser get text @e5           # Get specific element text
agent-browser get text body > page.txt  # Get all page text

# JSON output for parsing
agent-browser snapshot -i --json
```

### Authentication with State Persistence

```bash
# Login once and save state
agent-browser open https://app.example.com/login
agent-browser snapshot -i
agent-browser fill @e1 "$USERNAME"
agent-browser fill @e2 "$PASSWORD"
agent-browser click @e3
agent-browser wait --url "**/dashboard"
agent-browser state save auth.json

# Reuse in future sessions
agent-browser state load auth.json
agent-browser open https://app.example.com/dashboard
```

### Command Chaining

Commands can be chained when you don't need intermediate output:

```bash
agent-browser open https://example.com && agent-browser wait --load networkidle && agent-browser snapshot -i
```

Run commands separately when you need to read output first (e.g., snapshot
to discover refs, then interact using those refs).

## Ref Lifecycle (Important)

Refs (`@e1`, `@e2`, etc.) are invalidated when the page changes. Always
re-snapshot after:

- Clicking links or buttons that navigate
- Form submissions
- Dynamic content loading (dropdowns, modals)

```bash
agent-browser click @e5              # Navigates to new page
agent-browser snapshot -i            # MUST re-snapshot for new refs
agent-browser click @e1              # Use new refs
```

## Session Management

```bash
# Named sessions for isolation
agent-browser --session site1 open https://site-a.com
agent-browser --session site2 open https://site-b.com

# Always close when done
agent-browser close
```

## Security

All security features are opt-in:

```bash
# Domain allowlist
export AGENT_BROWSER_ALLOWED_DOMAINS="example.com,*.example.com"

# Content boundaries (prevents prompt injection)
export AGENT_BROWSER_CONTENT_BOUNDARIES=1

# Output limits
export AGENT_BROWSER_MAX_OUTPUT=50000
```
