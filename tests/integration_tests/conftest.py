"""Shared fixtures for integration tests."""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip integration tests if OPENCODE_API_KEY is not set."""
    if os.environ.get("OPENCODE_API_KEY"):
        return
    skip_marker = pytest.mark.skip(reason="OPENCODE_API_KEY not set — skipping integration tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)
