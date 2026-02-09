"""Tools for the deep research example agent.

Provides web search (with optional Tavily integration) and a strategic
thinking tool that ADK can expose as ``FunctionTool`` instances.
"""

from __future__ import annotations

import html
import logging
import os
import re
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

_MAX_RESULT_CHARS = 15_000
_USER_AGENT = (
    "Mozilla/5.0 (compatible; DeepResearchAgent/1.0; +https://github.com/manojlds/adk-deepagents)"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_html_tags(raw_html: str) -> str:
    """Convert HTML to plain text using a simple regex-based approach."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", raw_html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch_page(url: str, timeout: int = 10) -> str | None:
    """Fetch a URL and return plain-text content, or *None* on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return _strip_html_tags(raw)
    except Exception:
        logger.debug("Failed to fetch %s", url, exc_info=True)
        return None


def _truncate(text: str, max_chars: int = _MAX_RESULT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def _format_result(index: int, title: str, url: str, content: str) -> str:
    content = _truncate(content)
    return f"[{index}] {title}\n    URL: {url}\n{content}"


# ---------------------------------------------------------------------------
# Tavily-backed search
# ---------------------------------------------------------------------------


def _search_tavily(query: str, max_results: int, topic: str) -> str | None:
    """Attempt a Tavily search.  Returns formatted results or *None*."""
    try:
        from tavily import TavilyClient  # type: ignore[import-untyped]
    except ImportError:
        return None

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return None

    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, max_results=max_results, topic=topic)
    results: list[dict] = response.get("results", [])

    if not results:
        return "No results found."

    parts: list[str] = []
    for i, result in enumerate(results, start=1):
        title = result.get("title", "")
        url = result.get("url", "")
        snippet = result.get("content", "")

        full_text = _fetch_page(url)
        content = full_text if full_text else snippet

        parts.append(_format_result(i, title, url, content))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# DuckDuckGo fallback
# ---------------------------------------------------------------------------


def _search_duckduckgo(query: str, max_results: int) -> str:
    """Basic DuckDuckGo HTML scraping fallback."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            page = resp.read().decode("utf-8", errors="replace")
    except Exception:
        logger.debug("DuckDuckGo fetch failed", exc_info=True)
        return "Web search unavailable."

    link_pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL
    )
    snippet_pattern = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)

    links = link_pattern.findall(page)
    snippets = snippet_pattern.findall(page)

    if not links:
        return "No results found."

    parts: list[str] = []
    for i, (href, raw_title) in enumerate(links[:max_results], start=1):
        title = _strip_html_tags(raw_title)
        snippet = _strip_html_tags(snippets[i - 1]) if i - 1 < len(snippets) else ""

        full_text = _fetch_page(href)
        content = full_text if full_text else snippet

        parts.append(_format_result(i, title, href, content))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def web_search(query: str, max_results: int = 3, topic: str = "general") -> str:
    """Search the web and return relevant results with page content.

    Uses Tavily when available (requires ``tavily-python`` and the
    ``TAVILY_API_KEY`` environment variable).  Falls back to a basic
    DuckDuckGo HTML scrape otherwise.

    Supported Tavily topics: ``"general"``, ``"news"``, ``"finance"``.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 3).
        topic: Search topic hint — ``"general"``, ``"news"``, or
            ``"finance"`` (Tavily only).

    Returns:
        Formatted search results with title, URL, and content.
    """
    tavily_result = _search_tavily(query, max_results, topic)
    if tavily_result is not None:
        return tavily_result

    logger.debug("Tavily unavailable, falling back to DuckDuckGo")
    return _search_duckduckgo(query, max_results)


def think(reflection: str) -> str:
    """Pause for structured strategic thinking about research progress.

    Use this tool after each web search to reflect on what has been
    learned so far, identify gaps in the research, evaluate source
    quality, and plan the next steps.

    Args:
        reflection: Your strategic thinking — assess findings, note
            what is still missing, and outline next actions.

    Returns:
        Confirmation that the reflection was captured.
    """
    logger.debug("Think: %s", reflection)
    return f"Reflection captured: {reflection}"
