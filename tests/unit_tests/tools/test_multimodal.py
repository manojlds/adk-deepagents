"""Tests for multimodal utilities."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

from google.genai import types

from adk_deepagents.tools.multimodal import (
    extract_image_urls,
    fetch_image_as_part,
    process_multimodal_content,
)


class TestExtractImageUrls:
    def test_markdown_image(self):
        text = "Look at this: ![screenshot](https://example.com/img.png)"
        urls = extract_image_urls(text)
        assert urls == ["https://example.com/img.png"]

    def test_bare_image_url(self):
        text = "Check https://example.com/photo.jpg for details"
        urls = extract_image_urls(text)
        assert urls == ["https://example.com/photo.jpg"]

    def test_multiple_images(self):
        text = "![a](https://a.com/1.png) and ![b](https://b.com/2.jpg) also https://c.com/3.webp"
        urls = extract_image_urls(text)
        assert len(urls) == 3

    def test_no_images(self):
        text = "Just some text with https://example.com/page and nothing else"
        urls = extract_image_urls(text)
        assert urls == []

    def test_deduplication(self):
        text = "![a](https://example.com/img.png) and https://example.com/img.png again"
        urls = extract_image_urls(text)
        assert len(urls) == 1

    def test_url_with_query_params(self):
        text = "See https://example.com/image.png?size=large"
        urls = extract_image_urls(text)
        assert len(urls) == 1
        assert "size=large" in urls[0]

    def test_gif_extension(self):
        text = "Animation: https://example.com/anim.gif"
        urls = extract_image_urls(text)
        assert len(urls) == 1

    def test_webp_extension(self):
        text = "Image: https://example.com/photo.webp"
        urls = extract_image_urls(text)
        assert len(urls) == 1


class TestFetchImageAsPart:
    @patch("adk_deepagents.tools.multimodal.is_url_safe", return_value=(False, "private"))
    def test_ssrf_blocked(self, mock_safe):
        result = fetch_image_as_part("http://10.0.0.1/img.png")
        assert result is None

    @patch("adk_deepagents.tools.multimodal.urllib.request.urlopen")
    @patch("adk_deepagents.tools.multimodal.is_url_safe", return_value=(True, ""))
    def test_successful_fetch(self, mock_safe, mock_urlopen):
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.headers = {"Content-Type": "image/png", "Content-Length": "100"}
        mock_response.read.return_value = b"\x89PNG" + b"\x00" * 50
        mock_urlopen.return_value = mock_response

        result = fetch_image_as_part("https://example.com/test.png")
        assert result is not None
        assert result.inline_data is not None
        assert result.inline_data.mime_type == "image/png"

    @patch("adk_deepagents.tools.multimodal.urllib.request.urlopen")
    @patch("adk_deepagents.tools.multimodal.is_url_safe", return_value=(True, ""))
    def test_too_large_by_header(self, mock_safe, mock_urlopen):
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.headers = {
            "Content-Type": "image/png",
            "Content-Length": str(20 * 1024 * 1024),
        }
        mock_urlopen.return_value = mock_response

        result = fetch_image_as_part("https://example.com/huge.png")
        assert result is None

    @patch(
        "adk_deepagents.tools.multimodal.urllib.request.urlopen",
        side_effect=Exception("connection error"),
    )
    @patch("adk_deepagents.tools.multimodal.is_url_safe", return_value=(True, ""))
    def test_fetch_failure(self, mock_safe, mock_urlopen):
        result = fetch_image_as_part("https://example.com/broken.png")
        assert result is None


class TestProcessMultimodalContent:
    def test_no_contents(self):
        assert process_multimodal_content([]) is False

    def test_no_user_message(self):
        contents = [types.Content(role="model", parts=[types.Part(text="Hello")])]
        assert process_multimodal_content(contents) is False

    def test_no_images_in_text(self):
        contents = [types.Content(role="user", parts=[types.Part(text="Just text")])]
        assert process_multimodal_content(contents) is False

    @patch("adk_deepagents.tools.multimodal.fetch_image_as_part")
    def test_image_attached(self, mock_fetch):
        mock_part = types.Part(
            inline_data=types.Blob(mime_type="image/png", data=base64.b64decode("aW1hZ2U="))
        )
        mock_fetch.return_value = mock_part

        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text="Look at https://example.com/img.png")],
            )
        ]
        result = process_multimodal_content(contents)
        assert result is True
        assert contents[0].parts is not None
        assert len(contents[0].parts) == 2

    @patch("adk_deepagents.tools.multimodal.fetch_image_as_part")
    def test_cache_used(self, mock_fetch):
        cached_part = types.Part(
            inline_data=types.Blob(mime_type="image/png", data=base64.b64decode("Y2FjaGVk"))
        )
        cache = {"https://example.com/img.png": cached_part}

        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text="Look at https://example.com/img.png")],
            )
        ]
        process_multimodal_content(contents, fetched_cache=cache)
        mock_fetch.assert_not_called()
        assert contents[0].parts is not None
        assert len(contents[0].parts) == 2

    @patch("adk_deepagents.tools.multimodal.fetch_image_as_part", return_value=None)
    def test_failed_fetch_cached_as_none(self, mock_fetch):
        cache: dict = {}
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text="See https://example.com/broken.png")],
            )
        ]
        process_multimodal_content(contents, fetched_cache=cache)
        assert cache["https://example.com/broken.png"] is None
        assert contents[0].parts is not None
        assert len(contents[0].parts) == 1
