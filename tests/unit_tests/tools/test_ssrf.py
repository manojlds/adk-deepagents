"""Tests for SSRF protection."""

from __future__ import annotations

import socket
from unittest.mock import patch

from adk_deepagents.tools.ssrf import _is_private_or_reserved, is_url_safe


class TestIsPrivateOrReserved:
    def test_loopback_ipv4(self):
        assert _is_private_or_reserved("127.0.0.1") is True

    def test_loopback_ipv6(self):
        assert _is_private_or_reserved("::1") is True

    def test_private_10(self):
        assert _is_private_or_reserved("10.0.0.1") is True

    def test_private_172(self):
        assert _is_private_or_reserved("172.16.0.1") is True

    def test_private_192(self):
        assert _is_private_or_reserved("192.168.1.1") is True

    def test_link_local(self):
        assert _is_private_or_reserved("169.254.1.1") is True

    def test_link_local_ipv6(self):
        assert _is_private_or_reserved("fe80::1") is True

    def test_public_ip(self):
        assert _is_private_or_reserved("8.8.8.8") is False

    def test_public_ipv6(self):
        assert _is_private_or_reserved("2001:4860:4860::8888") is False

    def test_multicast(self):
        assert _is_private_or_reserved("224.0.0.1") is True

    def test_invalid_address(self):
        assert _is_private_or_reserved("not-an-ip") is True

    def test_ipv4_mapped_ipv6_private(self):
        assert _is_private_or_reserved("::ffff:192.168.1.1") is True

    def test_ipv4_mapped_ipv6_public(self):
        assert _is_private_or_reserved("::ffff:8.8.8.8") is False


class TestIsUrlSafe:
    def test_no_scheme(self):
        safe, reason = is_url_safe("example.com/path")
        assert safe is False
        assert "no scheme" in reason.lower() or "scheme" in reason.lower()

    def test_non_http_scheme(self):
        safe, reason = is_url_safe("ftp://example.com")
        assert safe is False
        assert "ftp" in reason

    def test_no_hostname(self):
        safe, reason = is_url_safe("http://")
        assert safe is False
        assert "hostname" in reason.lower()

    @patch("adk_deepagents.tools.ssrf.socket.getaddrinfo")
    def test_public_url_safe(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 80)),
        ]
        safe, reason = is_url_safe("https://example.com")
        assert safe is True
        assert reason == ""

    @patch("adk_deepagents.tools.ssrf.socket.getaddrinfo")
    def test_private_ip_blocked(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("192.168.1.1", 80)),
        ]
        safe, reason = is_url_safe("http://internal.corp")
        assert safe is False
        assert "blocked" in reason.lower() or "private" in reason.lower()

    @patch("adk_deepagents.tools.ssrf.socket.getaddrinfo")
    def test_loopback_blocked(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("127.0.0.1", 80)),
        ]
        safe, reason = is_url_safe("http://localhost")
        assert safe is False

    @patch(
        "adk_deepagents.tools.ssrf.socket.getaddrinfo",
        side_effect=socket.gaierror("Name resolution failed"),
    )
    def test_dns_failure(self, mock_getaddrinfo):
        safe, reason = is_url_safe("http://nonexistent.invalid")
        assert safe is False
        assert "dns" in reason.lower() or "resolution" in reason.lower()

    @patch("adk_deepagents.tools.ssrf.socket.getaddrinfo")
    def test_multiple_ips_one_private(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 80)),
            (2, 1, 6, "", ("10.0.0.1", 80)),
        ]
        safe, reason = is_url_safe("https://example.com")
        assert safe is False
