"""Tools for the deep research example agent.

Provides a provider-routed web search tool (Serper-first in auto mode) and a
strategic thinking helper.
"""

from __future__ import annotations

import html
import json
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
_SEARCH_PROVIDER_ENV = "DEEP_RESEARCH_SEARCH_PROVIDER"
_DEFAULT_SEARCH_PROVIDER = "auto"


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


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: int = 15) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request_headers = {
        "Content-Type": "application/json",
        "User-Agent": _USER_AGENT,
        **headers,
    }
    request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _truncate(text: str, max_chars: int = _MAX_RESULT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def _format_result(index: int, title: str, url: str, content: str, provider: str) -> str:
    content = _truncate(content)
    return f"[{index}] {title}\n    URL: {url}\n    Provider: {provider}\n{content}"


def _format_results(results: list[dict[str, str]]) -> str:
    if not results:
        return "No results found."
    return "\n\n".join(
        _format_result(
            i,
            item.get("title", ""),
            item.get("url", ""),
            item.get("content", ""),
            item.get("provider", "unknown"),
        )
        for i, item in enumerate(results, start=1)
    )


# ---------------------------------------------------------------------------
# Tavily-backed search
# ---------------------------------------------------------------------------


def _search_tavily(query: str, max_results: int, topic: str) -> list[dict[str, str]]:
    """Search Tavily and return normalized results."""
    try:
        from tavily import TavilyClient  # type: ignore[import-untyped]
    except ImportError as err:
        raise RuntimeError("tavily-python is not installed") from err

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")

    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, max_results=max_results, topic=topic)
    results: list[dict] = response.get("results", [])

    formatted: list[dict[str, str]] = []
    for result in results:
        title = result.get("title", "")
        url = result.get("url", "")
        snippet = result.get("content", "")

        full_text = _fetch_page(url)
        content = full_text if full_text else snippet
        formatted.append(
            {
                "title": title,
                "url": url,
                "content": content,
                "provider": "tavily",
            }
        )
    return formatted


def _search_serper(query: str, max_results: int) -> list[dict[str, str]]:
    """Search Serper and return normalized results."""
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY is not set")

    payload = {
        "q": query,
        "num": max_results,
    }
    response = _post_json(
        "https://google.serper.dev/search",
        payload,
        headers={"X-API-KEY": api_key},
    )

    organic = response.get("organic", [])
    if not isinstance(organic, list):
        raise RuntimeError("Unexpected Serper response: organic is not a list")

    formatted: list[dict[str, str]] = []
    for item in organic[:max_results]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", ""))
        url = str(item.get("link", ""))
        snippet = str(item.get("snippet", ""))
        full_text = _fetch_page(url)
        content = full_text if full_text else snippet
        formatted.append(
            {
                "title": title,
                "url": url,
                "content": content,
                "provider": "serper",
            }
        )
    return formatted


def _search_brave(query: str, max_results: int) -> list[dict[str, str]]:
    """Search Brave Search API and return normalized results."""
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("BRAVE_SEARCH_API_KEY is not set")

    encoded = urllib.parse.urlencode({"q": query, "count": max_results})
    url = f"https://api.search.brave.com/res/v1/web/search?{encoded}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
            "User-Agent": _USER_AGENT,
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))

    web_data = payload.get("web", {})
    if not isinstance(web_data, dict):
        raise RuntimeError("Unexpected Brave response: web is not an object")
    results = web_data.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError("Unexpected Brave response: web.results is not a list")

    formatted: list[dict[str, str]] = []
    for item in results[:max_results]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", ""))
        result_url = str(item.get("url", ""))
        snippet = str(item.get("description", ""))
        full_text = _fetch_page(result_url)
        content = full_text if full_text else snippet
        formatted.append(
            {
                "title": title,
                "url": result_url,
                "content": content,
                "provider": "brave",
            }
        )
    return formatted


# ---------------------------------------------------------------------------
# DuckDuckGo fallback
# ---------------------------------------------------------------------------


def _search_duckduckgo(query: str, max_results: int) -> list[dict[str, str]]:
    """Basic DuckDuckGo HTML scraping fallback."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            page = resp.read().decode("utf-8", errors="replace")
    except Exception as err:
        logger.debug("DuckDuckGo fetch failed", exc_info=True)
        raise RuntimeError("DuckDuckGo search unavailable") from err

    link_pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL
    )
    snippet_pattern = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)

    links = link_pattern.findall(page)
    snippets = snippet_pattern.findall(page)

    formatted: list[dict[str, str]] = []
    for i, (href, raw_title) in enumerate(links[:max_results], start=1):
        title = _strip_html_tags(raw_title)
        snippet = _strip_html_tags(snippets[i - 1]) if i - 1 < len(snippets) else ""

        full_text = _fetch_page(href)
        content = full_text if full_text else snippet
        formatted.append(
            {
                "title": title,
                "url": href,
                "content": content,
                "provider": "duckduckgo",
            }
        )
    return formatted


def _resolve_provider() -> str:
    provider = os.environ.get(_SEARCH_PROVIDER_ENV, _DEFAULT_SEARCH_PROVIDER).strip().lower()
    valid = {"auto", "serper", "tavily", "brave", "duckduckgo"}
    if provider not in valid:
        raise RuntimeError(
            f"Invalid {_SEARCH_PROVIDER_ENV}={provider!r}. Expected one of: "
            "auto, serper, tavily, brave, duckduckgo"
        )
    return provider


def _resolve_auto_provider() -> str:
    if os.environ.get("SERPER_API_KEY"):
        return "serper"
    if os.environ.get("TAVILY_API_KEY"):
        return "tavily"
    if os.environ.get("BRAVE_SEARCH_API_KEY"):
        return "brave"
    return "duckduckgo"


def _dispatch_provider(
    provider: str, query: str, max_results: int, topic: str
) -> list[dict[str, str]]:
    if provider == "serper":
        return _search_serper(query, max_results)
    if provider == "tavily":
        return _search_tavily(query, max_results, topic)
    if provider == "brave":
        return _search_brave(query, max_results)
    return _search_duckduckgo(query, max_results)


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def web_search(query: str, max_results: int = 3, topic: str = "general") -> str:
    """Search the web and return relevant results with page content.

    Uses provider routing configured by ``DEEP_RESEARCH_SEARCH_PROVIDER``:

    - ``auto`` (default): ``serper`` -> ``tavily`` -> ``brave`` -> ``duckduckgo``
    - ``serper``
    - ``tavily``
    - ``brave``
    - ``duckduckgo``

    In ``auto`` mode, the first provider with configured credentials is used.
    If the chosen provider fails, this tool hard-fails with an error (no
    silent fallback).

    Supported Tavily topics: ``"general"``, ``"news"``, ``"finance"``.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 3).
        topic: Search topic hint — ``"general"``, ``"news"``, or
            ``"finance"`` (Tavily only).

    Returns:
        Formatted search results with title, URL, and content.
    """
    provider = _resolve_provider()
    if provider == "auto":
        provider = _resolve_auto_provider()

    try:
        results = _dispatch_provider(provider, query, max_results, topic)
    except Exception as exc:
        logger.exception("Web search provider '%s' failed", provider)
        return f"Web search error via provider '{provider}': {exc}"

    return _format_results(results)


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
