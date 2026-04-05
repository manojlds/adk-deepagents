"""Multimodal utilities — image URL extraction and fetching.

Detects image URLs in conversation text and fetches them as base64
inline data parts for multimodal model support.

Ported from OpenSWE's multimodal message handling.
"""

from __future__ import annotations

import base64
import logging
import re
import urllib.error
import urllib.request

from google.genai import types

from adk_deepagents.tools.ssrf import is_url_safe

logger = logging.getLogger(__name__)

# Maximum image size in bytes (10 MB)
MAX_IMAGE_SIZE = 10 * 1024 * 1024

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; DeepAgents/1.0)"

# Supported image extensions
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"})

# Media type mapping
MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}

# Regex for Markdown image syntax: ![alt](url)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

# Regex for bare image URLs
_BARE_IMAGE_URL_RE = re.compile(
    r"(?<!\()(?<!\])\bhttps?://[^\s<>\"']+\.(?:png|jpg|jpeg|gif|webp)(?:\?[^\s<>\"']*)?\b",
    re.IGNORECASE,
)


def extract_image_urls(text: str) -> list[str]:
    """Extract image URLs from text.

    Finds URLs in Markdown image syntax ``![alt](url)`` and bare image
    URLs with recognized extensions.

    Returns a deduplicated list preserving order.
    """
    urls: list[str] = []
    seen: set[str] = set()

    # Markdown images
    for match in _MARKDOWN_IMAGE_RE.finditer(text):
        url = match.group(1).strip()
        if url not in seen:
            seen.add(url)
            urls.append(url)

    # Bare image URLs
    for match in _BARE_IMAGE_URL_RE.finditer(text):
        url = match.group(0)
        if url not in seen:
            seen.add(url)
            urls.append(url)

    return urls


def _detect_media_type(url: str, content_type: str | None = None) -> str:
    """Detect the media type from URL extension or Content-Type header."""
    if content_type and content_type.startswith("image/"):
        return content_type.split(";")[0].strip()

    # Fall back to extension
    url_lower = url.lower().split("?")[0]  # Strip query params
    for ext, media in MEDIA_TYPES.items():
        if url_lower.endswith(ext):
            return media

    return "image/png"  # Default


def fetch_image_as_part(url: str) -> types.Part | None:
    """Fetch an image URL and return it as an inline data Part.

    Returns ``None`` if the URL is blocked by SSRF protection, the
    download fails, or the image exceeds the size limit.

    Parameters
    ----------
    url:
        The image URL to fetch.
    """
    safe, reason = is_url_safe(url)
    if not safe:
        logger.debug("Image URL blocked by SSRF: %s (%s)", url, reason)
        return None

    headers = {"User-Agent": DEFAULT_USER_AGENT}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            content_type = response.headers.get("Content-Type", "")

            # Check size via Content-Length if available
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_IMAGE_SIZE:
                logger.debug("Image too large (%s bytes): %s", content_length, url)
                return None

            data = response.read(MAX_IMAGE_SIZE + 1)
            if len(data) > MAX_IMAGE_SIZE:
                logger.debug("Image exceeded size limit: %s", url)
                return None

            media_type = _detect_media_type(url, content_type)

            return types.Part(
                inline_data=types.Blob(
                    mime_type=media_type,
                    data=base64.b64encode(data),
                ),
            )

    except Exception:
        logger.debug("Failed to fetch image: %s", url, exc_info=True)
        return None


def process_multimodal_content(
    contents: list[types.Content],
    *,
    fetched_cache: dict[str, types.Part | None] | None = None,
) -> bool:
    """Scan the last user message for image URLs and attach inline parts.

    Modifies the contents list in place. Only processes the most recent
    user message to avoid re-fetching on every turn.

    Parameters
    ----------
    contents:
        The LLM request contents list.
    fetched_cache:
        Optional cache mapping URL -> Part to avoid re-downloading.

    Returns
    -------
    bool
        Whether any images were attached.
    """
    if not contents:
        return False

    cache: dict[str, types.Part | None] = fetched_cache if fetched_cache is not None else {}

    # Find the last user message
    last_user_idx = -1
    for i in range(len(contents) - 1, -1, -1):
        if contents[i].role == "user":
            last_user_idx = i
            break

    if last_user_idx < 0:
        return False

    user_content = contents[last_user_idx]
    if not user_content.parts:
        return False

    # Collect image URLs from text parts
    all_urls: list[str] = []
    for part in user_content.parts:
        if part.text:
            all_urls.extend(extract_image_urls(part.text))

    if not all_urls:
        return False

    # Fetch and attach images
    new_parts: list[types.Part] = []
    for url in all_urls:
        if url in cache:
            cached = cache[url]
            if cached is not None:
                new_parts.append(cached)
            continue

        image_part = fetch_image_as_part(url)
        cache[url] = image_part
        if image_part is not None:
            new_parts.append(image_part)

    if not new_parts:
        return False

    # Append image parts to the user message
    user_content.parts.extend(new_parts)
    return True
