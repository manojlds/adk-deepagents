"""Tests for after_model callback (empty message guard)."""

from __future__ import annotations

from unittest.mock import MagicMock

from google.adk.models import LlmResponse
from google.genai import types

from adk_deepagents.callbacks.after_model import (
    EMPTY_RESPONSE_NUDGE,
    _is_empty_response,
    make_after_model_callback,
)


def _make_callback_context() -> MagicMock:
    ctx = MagicMock()
    ctx.state = {}
    return ctx


class TestIsEmptyResponse:
    def test_none_content(self):
        resp = LlmResponse(content=None)
        assert _is_empty_response(resp) is True

    def test_empty_parts(self):
        resp = LlmResponse(content=types.Content(role="model", parts=[]))
        assert _is_empty_response(resp) is True

    def test_whitespace_only_text(self):
        resp = LlmResponse(content=types.Content(role="model", parts=[types.Part(text="   ")]))
        assert _is_empty_response(resp) is True

    def test_text_content(self):
        resp = LlmResponse(content=types.Content(role="model", parts=[types.Part(text="Hello")]))
        assert _is_empty_response(resp) is False

    def test_function_call(self):
        resp = LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(name="read_file", args={"path": "/test"})
                    )
                ],
            )
        )
        assert _is_empty_response(resp) is False

    def test_text_and_function_call(self):
        resp = LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(text="Let me read that file."),
                    types.Part(
                        function_call=types.FunctionCall(name="read_file", args={"path": "/test"})
                    ),
                ],
            )
        )
        assert _is_empty_response(resp) is False


class TestAfterModelCallback:
    def test_non_empty_response_returns_none(self):
        cb = make_after_model_callback()
        resp = LlmResponse(content=types.Content(role="model", parts=[types.Part(text="Hello")]))
        result = cb(_make_callback_context(), resp)
        assert result is None

    def test_empty_response_returns_nudge(self):
        cb = make_after_model_callback()
        resp = LlmResponse(content=None)
        result = cb(_make_callback_context(), resp)
        assert result is not None
        assert isinstance(result, LlmResponse)
        assert result.content.parts[0].text == EMPTY_RESPONSE_NUDGE

    def test_empty_parts_returns_nudge(self):
        cb = make_after_model_callback()
        resp = LlmResponse(content=types.Content(role="model", parts=[]))
        result = cb(_make_callback_context(), resp)
        assert result is not None
        assert EMPTY_RESPONSE_NUDGE in result.content.parts[0].text

    def test_whitespace_only_returns_nudge(self):
        cb = make_after_model_callback()
        resp = LlmResponse(content=types.Content(role="model", parts=[types.Part(text="  \n  ")]))
        result = cb(_make_callback_context(), resp)
        assert result is not None

    def test_function_call_not_nudged(self):
        cb = make_after_model_callback()
        resp = LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(function_call=types.FunctionCall(name="ls", args={"path": "/"}))],
            )
        )
        result = cb(_make_callback_context(), resp)
        assert result is None
