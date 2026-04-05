"""HTTP tools — fetch_url and http_request with SSRF protection.

Provides web fetching capabilities with safety checks to prevent
Server-Side Request Forgery attacks.

Ported from OpenSWE's fetch_url and http_request tools.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from contextlib import suppress
from typing import Any

from google.adk.tools.tool_context import ToolContext

from adk_deepagents.backends.utils import truncate_if_too_long
from adk_deepagents.tools.ssrf import is_url_safe

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; DeepAgents/1.0)"
DEFAULT_TIMEOUT = 30


def _html_to_markdown(html: str) -> str:
    """Convert HTML to Markdown, falling back to raw HTML if markdownify is unavailable."""
    try:
        from markdownify import markdownify  # ty: ignore[unresolved-import]

        return markdownify(html, heading_style="ATX", strip=["img", "script", "style"])
    except ImportError:
        logger.debug("markdownify not installed, returning raw HTML")
        return html


def fetch_url(
    url: str,
    tool_context: ToolContext,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """Fetch a URL and return its content as Markdown.

    Performs an HTTP GET request, converts HTML responses to Markdown for
    readability, and applies SSRF protection to block requests to private
    networks.

    Args:
        url: The URL to fetch (must be http:// or https://).
        timeout: Request timeout in seconds. Defaults to 30.
    """
    safe, reason = is_url_safe(url)
    if not safe:
        return {"status": "error", "message": f"URL blocked: {reason}"}

    headers = {"User-Agent": DEFAULT_USER_AGENT}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            raw_bytes = response.read()

            # Detect encoding
            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()

            try:
                text = raw_bytes.decode(charset)
            except (UnicodeDecodeError, LookupError):
                text = raw_bytes.decode("utf-8", errors="replace")

            # Convert HTML to Markdown
            if "html" in content_type.lower():
                text = _html_to_markdown(text)

            text = truncate_if_too_long(text)

            return {
                "status": "success",
                "url": url,
                "content_type": content_type,
                "content": text,
            }

    except urllib.error.HTTPError as e:
        return {
            "status": "error",
            "message": f"HTTP {e.code}: {e.reason}",
            "url": url,
        }
    except urllib.error.URLError as e:
        return {
            "status": "error",
            "message": f"URL error: {e.reason}",
            "url": url,
        }
    except TimeoutError:
        return {
            "status": "error",
            "message": f"Request timed out after {timeout}s",
            "url": url,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Request failed: {type(e).__name__}: {e}",
            "url": url,
        }


def http_request(
    url: str,
    tool_context: ToolContext,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """Make an HTTP request and return the response.

    A general-purpose HTTP client supporting GET, POST, PUT, and DELETE.
    Applies SSRF protection to block requests to private networks.

    Args:
        url: The URL to request (must be http:// or https://).
        method: HTTP method (GET, POST, PUT, DELETE). Defaults to GET.
        headers: Optional request headers as a dict.
        body: Optional request body (string).
        timeout: Request timeout in seconds. Defaults to 30.
    """
    safe, reason = is_url_safe(url)
    if not safe:
        return {"status": "error", "message": f"URL blocked: {reason}"}

    method = method.upper()
    if method not in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"):
        return {"status": "error", "message": f"Unsupported HTTP method: {method}"}

    req_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        req_headers.update(headers)

    data = body.encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            raw_bytes = response.read()

            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()

            try:
                response_text = raw_bytes.decode(charset)
            except (UnicodeDecodeError, LookupError):
                response_text = raw_bytes.decode("utf-8", errors="replace")

            response_text = truncate_if_too_long(response_text)

            result: dict[str, Any] = {
                "status": "success",
                "url": url,
                "status_code": response.status,
                "content_type": content_type,
                "body": response_text,
            }

            # Try to parse JSON responses
            if "json" in content_type.lower():
                with suppress(json.JSONDecodeError):
                    result["json"] = json.loads(response_text)

            return result

    except urllib.error.HTTPError as e:
        body_text = ""
        with suppress(Exception):
            body_text = e.read().decode("utf-8", errors="replace")[:2000]
        return {
            "status": "error",
            "message": f"HTTP {e.code}: {e.reason}",
            "url": url,
            "status_code": e.code,
            "body": body_text,
        }
    except urllib.error.URLError as e:
        return {
            "status": "error",
            "message": f"URL error: {e.reason}",
            "url": url,
        }
    except TimeoutError:
        return {
            "status": "error",
            "message": f"Request timed out after {timeout}s",
            "url": url,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Request failed: {type(e).__name__}: {e}",
            "url": url,
        }
