"""Browser automation integration for adk-deepagents.

Provides two integration paths:

1. **Playwright MCP** — Programmatic browser control via ``@playwright/mcp``
   MCP server, integrated through ADK's ``McpToolset``.

2. **agent-browser CLI skill** — CLI-based browser automation via
   ``agent-browser``, integrated through adk-skills ``SKILL.md`` discovery.
"""

from adk_deepagents.browser.playwright_mcp import get_playwright_browser_tools

__all__ = [
    "get_playwright_browser_tools",
]
