"""Tests for HTTP tools."""

from __future__ import annotations

import urllib.error
from email.message import Message
from unittest.mock import MagicMock, patch

from adk_deepagents.tools.http import fetch_url, http_request


def _make_tool_context() -> MagicMock:
    ctx = MagicMock()
    ctx.state = {}
    return ctx


class TestFetchUrl:
    @patch("adk_deepagents.tools.http.is_url_safe", return_value=(False, "private IP"))
    def test_ssrf_blocked(self, mock_safe):
        result = fetch_url("http://10.0.0.1", _make_tool_context())
        assert result["status"] == "error"
        assert "blocked" in result["message"].lower()

    @patch("adk_deepagents.tools.http.urllib.request.urlopen")
    @patch("adk_deepagents.tools.http.is_url_safe", return_value=(True, ""))
    def test_successful_fetch(self, mock_safe, mock_urlopen):
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.read.return_value = b"Hello World"
        mock_urlopen.return_value = mock_response

        result = fetch_url("https://example.com", _make_tool_context())
        assert result["status"] == "success"
        assert result["content"] == "Hello World"

    @patch("adk_deepagents.tools.http.urllib.request.urlopen")
    @patch("adk_deepagents.tools.http.is_url_safe", return_value=(True, ""))
    def test_html_conversion(self, mock_safe, mock_urlopen):
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_response.read.return_value = b"<h1>Title</h1><p>Content</p>"
        mock_urlopen.return_value = mock_response

        result = fetch_url("https://example.com", _make_tool_context())
        assert result["status"] == "success"
        # Should either be markdown (if markdownify installed) or raw HTML
        assert "Title" in result["content"]

    @patch(
        "adk_deepagents.tools.http.urllib.request.urlopen",
        side_effect=urllib.error.HTTPError("url", 404, "Not Found", Message(), None),
    )
    @patch("adk_deepagents.tools.http.is_url_safe", return_value=(True, ""))
    def test_http_error(self, mock_safe, mock_urlopen):
        result = fetch_url("https://example.com/missing", _make_tool_context())
        assert result["status"] == "error"
        assert "404" in result["message"]

    @patch(
        "adk_deepagents.tools.http.urllib.request.urlopen",
        side_effect=TimeoutError(),
    )
    @patch("adk_deepagents.tools.http.is_url_safe", return_value=(True, ""))
    def test_timeout(self, mock_safe, mock_urlopen):
        result = fetch_url("https://slow.example.com", _make_tool_context(), timeout=5)
        assert result["status"] == "error"
        assert "timed out" in result["message"].lower()


class TestHttpRequest:
    @patch("adk_deepagents.tools.http.is_url_safe", return_value=(False, "reserved"))
    def test_ssrf_blocked(self, mock_safe):
        result = http_request("http://169.254.169.254", _make_tool_context())
        assert result["status"] == "error"

    def test_unsupported_method(self):
        with patch("adk_deepagents.tools.http.is_url_safe", return_value=(True, "")):
            result = http_request("https://example.com", _make_tool_context(), method="TRACE")
            assert result["status"] == "error"
            assert "unsupported" in result["message"].lower()

    @patch("adk_deepagents.tools.http.urllib.request.urlopen")
    @patch("adk_deepagents.tools.http.is_url_safe", return_value=(True, ""))
    def test_post_request(self, mock_safe, mock_urlopen):
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.status = 201
        mock_response.read.return_value = b'{"id": 1}'
        mock_urlopen.return_value = mock_response

        result = http_request(
            "https://api.example.com/items",
            _make_tool_context(),
            method="POST",
            headers={"Content-Type": "application/json"},
            body='{"name": "test"}',
        )
        assert result["status"] == "success"
        assert result["status_code"] == 201

    @patch("adk_deepagents.tools.http.urllib.request.urlopen")
    @patch("adk_deepagents.tools.http.is_url_safe", return_value=(True, ""))
    def test_get_request(self, mock_safe, mock_urlopen):
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.status = 200
        mock_response.read.return_value = b"OK"
        mock_urlopen.return_value = mock_response

        result = http_request("https://example.com", _make_tool_context())
        assert result["status"] == "success"
        assert result["body"] == "OK"
