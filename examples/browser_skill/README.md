# Browser Skill Agent

A browser automation agent that uses the
[agent-browser](https://github.com/vercel-labs/agent-browser) CLI tool
through Agent Skills integration. The agent discovers the browser skill
on demand and learns how to use the CLI for web interaction.

## How It Works

Unlike the `browser_agent` example (which uses Playwright MCP for
programmatic tool calls), this example uses the **skill-based approach**:

1. The agent has shell execution (`execution="local"`) and skills discovery
2. When a browser task is requested, the agent activates the `agent-browser`
   skill via `use_skill("agent-browser")`
3. The skill teaches the agent the `agent-browser` CLI commands and patterns
4. The agent runs `agent-browser` commands through the `execute` tool

This mirrors how AI coding agents (Claude Code, Cursor, etc.) use
agent-browser — through CLI commands guided by a skill file.

## Comparison: Skill vs MCP Approach

| | Browser Skill (this) | Browser Agent (MCP) |
|---|---|---|
| **Integration** | adk-skills + shell | McpToolset (ADK-native) |
| **How agent uses it** | `execute("agent-browser open ...")` | `browser_navigate(url=...)` |
| **Async required** | No | Yes |
| **Best for** | Agents with shell access | Autonomous programmatic agents |
| **Setup** | `npm i -g agent-browser` | Automatic via npx |

## Prerequisites

### 1. agent-browser CLI

```bash
npm install -g agent-browser
```

### 2. Python dependencies

```bash
uv sync
```

### 3. API key

```bash
export GOOGLE_API_KEY=your-key
```

## Quick Start

```bash
# Interactive runner
python -m examples.browser_skill.agent

# Or with ADK CLI
adk run examples/browser_skill/
```

## Example Session

```
You: Open https://news.ycombinator.com and get the top stories

Agent: I'll activate the browser skill first to learn the commands.
[use_skill("agent-browser")]

Agent: Now I'll navigate and extract the stories.
[execute: agent-browser open https://news.ycombinator.com]
[execute: agent-browser wait --load networkidle]
[execute: agent-browser snapshot -i]

Agent: Here are the top stories from Hacker News:
1. Show HN: I built a ...
2. Why Rust is great for ...
...
[execute: agent-browser close]
```

## Files

| File | Description |
|------|-------------|
| `agent.py` | Agent factory with skills + local execution |
| `prompts.py` | Browser skill activation instructions |
| `../skills/agent-browser/SKILL.md` | The agent-browser skill definition |
