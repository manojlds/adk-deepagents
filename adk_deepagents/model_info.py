"""Model metadata resolution — context window sizes.

Resolves model context window sizes dynamically using available metadata
sources instead of relying on a hardcoded dict.

Resolution order:
1. ``litellm.get_model_info()`` — offline lookup from litellm's model
   database (covers thousands of models across all providers).
2. ``google.genai.Client.models.get()`` — live API call for Gemini models
   (requires ``GOOGLE_API_KEY`` or Vertex AI credentials).
3. Static fallback table — small dict of known defaults for when neither
   source is available.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_WINDOW = 200_000
"""Conservative default when no model metadata is available."""

# ---------------------------------------------------------------------------
# Static fallback — last-resort values for models that cannot be resolved
# dynamically (e.g., litellm not installed, API key unavailable).
# ---------------------------------------------------------------------------

_FALLBACK_CONTEXT_WINDOWS: dict[str, int] = {
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    "gemini-1.5-flash": 1_048_576,
    "gemini-1.5-pro": 2_097_152,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-3.5-sonnet": 200_000,
    "claude-sonnet-4": 1_000_000,
}


# ---------------------------------------------------------------------------
# LiteLLM lookup
# ---------------------------------------------------------------------------


def _lookup_via_litellm(model: str) -> int | None:
    """Try ``litellm.get_model_info()`` for context window size.

    Returns ``None`` if litellm is not installed or the model is unknown.
    """
    try:
        import litellm

        info = litellm.get_model_info(model)
        max_input = info.get("max_input_tokens")
        if isinstance(max_input, int) and max_input > 0:
            return max_input
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# google.genai lookup (Gemini models only)
# ---------------------------------------------------------------------------


def _is_gemini_model(model: str) -> bool:
    """Return ``True`` if *model* looks like a Gemini model name."""
    normalized = model.lower().replace("gemini/", "")
    return normalized.startswith("gemini-") or normalized.startswith("models/gemini-")


def _lookup_via_genai(model: str) -> int | None:
    """Try ``google.genai.Client.models.get()`` for Gemini models.

    This makes a live API call and requires ``GOOGLE_API_KEY`` or
    Vertex AI credentials to be configured.

    Returns ``None`` on any failure (missing credentials, network error, etc.).
    """
    if not _is_gemini_model(model):
        return None

    try:
        from google.genai import Client

        client = Client()
        model_name = model.replace("gemini/", "")
        model_info = client.models.get(model=model_name)
        token_limit = getattr(model_info, "input_token_limit", None)
        if isinstance(token_limit, int) and token_limit > 0:
            return token_limit
    except Exception:
        logger.debug("google.genai model lookup failed for %s", model, exc_info=True)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def resolve_context_window(model: str) -> int:
    """Resolve the context window size for *model*.

    Tries multiple sources in order and caches the result:

    1. ``litellm.get_model_info()`` — fast offline lookup
    2. ``google.genai`` API — live lookup for Gemini models
    3. Static fallback table
    4. ``DEFAULT_CONTEXT_WINDOW`` (200k)

    Parameters
    ----------
    model:
        Model name string (e.g. ``"gemini-2.5-flash"``, ``"gpt-4o"``).

    Returns
    -------
    int
        Context window size in tokens.
    """
    # 1. litellm (offline metadata database)
    result = _lookup_via_litellm(model)
    if result is not None:
        logger.debug("Context window for %s from litellm: %d", model, result)
        return result

    # 2. google.genai API (live, Gemini only)
    result = _lookup_via_genai(model)
    if result is not None:
        logger.debug("Context window for %s from google.genai: %d", model, result)
        return result

    # 3. Static fallback
    fallback = _FALLBACK_CONTEXT_WINDOWS.get(model)
    if fallback is not None:
        logger.debug("Context window for %s from fallback table: %d", model, fallback)
        return fallback

    logger.debug(
        "Context window for %s not found in any source, using default %d",
        model,
        DEFAULT_CONTEXT_WINDOW,
    )
    return DEFAULT_CONTEXT_WINDOW
