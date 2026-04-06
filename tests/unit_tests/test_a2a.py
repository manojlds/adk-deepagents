"""Tests for A2A integration helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adk_deepagents.a2a import to_a2a_app


def test_to_a2a_app_forwards_arguments():
    fake_to_a2a = MagicMock(return_value="app")
    fake_module = type("_Module", (), {"to_a2a": fake_to_a2a})()

    with patch("importlib.import_module", return_value=fake_module):
        app = to_a2a_app(
            agent="agent",
            host="127.0.0.1",
            port=9999,
            protocol="https",
            agent_card={"name": "card"},
            runner="runner",
        )

    assert app == "app"
    fake_to_a2a.assert_called_once_with(
        "agent",
        host="127.0.0.1",
        port=9999,
        protocol="https",
        agent_card={"name": "card"},
        runner="runner",
    )


def test_to_a2a_app_raises_helpful_error_when_sdk_missing():
    with (
        patch("importlib.import_module", side_effect=ImportError),
        pytest.raises(
            ImportError,
            match="a2a-sdk",
        ),
    ):
        to_a2a_app(agent="agent")
