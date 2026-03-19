"""SSRF protection — blocks requests to private and reserved IP ranges.

Resolves hostnames to IP addresses and checks against blocked ranges
to prevent Server-Side Request Forgery attacks.

Ported from OpenSWE's ``_is_url_safe()`` pattern.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _is_private_or_reserved(addr: str) -> bool:
    """Check if an IP address string is private, loopback, or reserved."""
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        # If we can't parse it, block it to be safe
        return True

    if ip.is_private:
        return True
    if ip.is_loopback:
        return True
    if ip.is_link_local:
        return True
    if ip.is_reserved:
        return True
    if ip.is_multicast:
        return True

    # Additional checks for IPv4-mapped IPv6 addresses
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        return _is_private_or_reserved(str(ip.ipv4_mapped))

    return False


def is_url_safe(url: str) -> tuple[bool, str]:
    """Check if a URL is safe to request (not targeting private infrastructure).

    Parameters
    ----------
    url:
        The URL to check.

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` if safe, ``(False, reason)`` if blocked.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False, f"Failed to parse URL: {url}"

    if not parsed.scheme:
        return False, "URL has no scheme (expected http:// or https://)"

    if parsed.scheme not in ("http", "https"):
        return False, f"URL scheme '{parsed.scheme}' not allowed (only http/https)"

    hostname = parsed.hostname
    if not hostname:
        return False, "URL has no hostname"

    # Resolve hostname to IP addresses
    try:
        addr_infos = socket.getaddrinfo(hostname, parsed.port or 80, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        return False, f"DNS resolution failed for '{hostname}': {e}"

    if not addr_infos:
        return False, f"No DNS results for '{hostname}'"

    for addr_info in addr_infos:
        ip_str = str(addr_info[4][0])
        if _is_private_or_reserved(ip_str):
            return False, f"URL resolves to blocked address {ip_str} (private/reserved range)"

    return True, ""
