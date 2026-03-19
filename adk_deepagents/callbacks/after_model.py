"""After-model callback — empty message guard.

Detects empty model responses (no text content and no tool calls) and
injects a synthetic nudge to prevent no-op turns. This enables the
model to self-correct rather than silently producing nothing.

Ported from OpenSWE's ``ensure_no_empty_msg`` pattern.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.genai import types

logger = logging.getLogger(__name__)

EMPTY_RESPONSE_NUDGE = (
    "I notice I produced an empty response. Let me re-examine the task "
    "and provide a substantive response with concrete actions."
)


def _is_empty_response(llm_response: LlmResponse) -> bool:
    """Check if an LLM response has no meaningful content."""
    content = llm_response.content
    if content is None:
        return True

    parts = content.parts
    if not parts:
        return True

    for part in parts:
        # Has text content
        if part.text and part.text.strip():
            return False
        # Has a function call
        if getattr(part, "function_call", None) is not None:
            return False

    return True


def make_after_model_callback() -> Callable:
    """Create an ``after_model_callback`` that guards against empty responses.

    When the model produces a response with no text and no tool calls,
    the callback replaces it with a nudge message that prompts the model
    to try again.
    """

    def after_model_callback(
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        if not _is_empty_response(llm_response):
            return None  # Response is fine, proceed normally

        logger.warning("Empty model response detected, injecting nudge")

        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=EMPTY_RESPONSE_NUDGE)],
            ),
        )

    return after_model_callback
