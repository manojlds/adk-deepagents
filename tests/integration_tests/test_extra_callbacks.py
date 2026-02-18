"""Integration tests — extra_callbacks composition.

Verifies that _compose_callbacks correctly chains built-in and user-provided
callbacks, including short-circuit behavior.  No API key required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.genai import types

from adk_deepagents.graph import _compose_callbacks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_callback_context(state=None):
    ctx = MagicMock(spec=CallbackContext)
    ctx.state = state if state is not None else {}
    return ctx


def _make_llm_request():
    req = MagicMock(spec=LlmRequest)
    req.config = types.GenerateContentConfig()
    req.contents = []
    return req


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComposeCallbacksBeforeAgent:
    def test_extra_before_agent_called(self):
        calls = []

        def builtin_cb(ctx):
            calls.append("builtin")
            return None

        def extra_cb(ctx):
            calls.append("extra")
            return None

        composed = _compose_callbacks(builtin_cb, extra_cb)
        ctx = _make_callback_context()
        result = composed(ctx)
        assert result is None
        assert calls == ["builtin", "extra"]

    def test_builtin_shortcircuit_skips_extra(self):
        calls = []
        sentinel = types.Content(role="model", parts=[types.Part(text="stop")])

        def builtin_cb(ctx):
            calls.append("builtin")
            return sentinel

        def extra_cb(ctx):
            calls.append("extra")
            return None

        composed = _compose_callbacks(builtin_cb, extra_cb)
        ctx = _make_callback_context()
        result = composed(ctx)
        assert result is sentinel
        assert calls == ["builtin"]


class TestComposeCallbacksBeforeModel:
    def test_extra_before_model_called(self):
        calls = []

        def builtin_cb(ctx, req):
            calls.append("builtin")
            return None

        def extra_cb(ctx, req):
            calls.append("extra")
            return None

        composed = _compose_callbacks(builtin_cb, extra_cb)
        ctx = _make_callback_context()
        req = _make_llm_request()
        result = composed(ctx, req)
        assert result is None
        assert calls == ["builtin", "extra"]


class TestComposeCallbacksEdgeCases:
    def test_none_builtin(self):
        def extra_cb(ctx):
            return "extra_result"

        composed = _compose_callbacks(None, extra_cb)
        assert composed is extra_cb

    def test_none_extra(self):
        def builtin_cb(ctx):
            return "builtin_result"

        composed = _compose_callbacks(builtin_cb, None)
        assert composed is builtin_cb

    def test_both_none(self):
        composed = _compose_callbacks(None, None)
        assert composed is None

    def test_multiple_extras(self):
        calls = []

        def builtin_cb(ctx):
            calls.append("builtin")
            return None

        def extra1(ctx):
            calls.append("extra1")
            return None

        def extra2(ctx):
            calls.append("extra2")
            return None

        # Compose builtin + extra1, then compose result + extra2
        composed = _compose_callbacks(builtin_cb, extra1)
        composed = _compose_callbacks(composed, extra2)
        ctx = _make_callback_context()
        result = composed(ctx)
        assert result is None
        assert calls == ["builtin", "extra1", "extra2"]
