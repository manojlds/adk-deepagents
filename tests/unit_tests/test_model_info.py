"""Tests for model_info — dynamic context window resolution."""

from unittest.mock import patch

from adk_deepagents.model_info import (
    _FALLBACK_CONTEXT_WINDOWS,
    DEFAULT_CONTEXT_WINDOW,
    _is_gemini_model,
    _lookup_via_litellm,
    resolve_context_window,
)

# ---------------------------------------------------------------------------
# _is_gemini_model
# ---------------------------------------------------------------------------


def test_is_gemini_model_plain():
    assert _is_gemini_model("gemini-2.5-flash") is True
    assert _is_gemini_model("gemini-1.5-pro") is True


def test_is_gemini_model_with_prefix():
    assert _is_gemini_model("gemini/gemini-2.5-flash") is True
    assert _is_gemini_model("models/gemini-2.5-flash") is True


def test_is_gemini_model_non_gemini():
    assert _is_gemini_model("gpt-4o") is False
    assert _is_gemini_model("claude-3-opus") is False


# ---------------------------------------------------------------------------
# _lookup_via_litellm
# ---------------------------------------------------------------------------


def test_lookup_via_litellm_known_model():
    """litellm should resolve well-known models."""
    result = _lookup_via_litellm("gpt-4o")
    assert result is not None
    assert result > 0


def test_lookup_via_litellm_gemini():
    """litellm should resolve Gemini models with gemini/ prefix."""
    result = _lookup_via_litellm("gemini-2.5-flash")
    assert result is not None
    assert result > 0


def test_lookup_via_litellm_unknown_model():
    """litellm returns None for unknown models."""
    result = _lookup_via_litellm("totally-fake-model-12345")
    assert result is None


def test_lookup_via_litellm_import_error():
    """Returns None if litellm is not available."""
    with patch.dict("sys.modules", {"litellm": None}):
        result = _lookup_via_litellm("gpt-4o")
        assert result is None


# ---------------------------------------------------------------------------
# resolve_context_window
# ---------------------------------------------------------------------------


def test_resolve_known_model():
    """Known models should return a positive context window."""
    # Clear lru_cache to avoid cross-test pollution
    resolve_context_window.cache_clear()
    result = resolve_context_window("gemini-2.5-flash")
    assert result > 0
    assert result == 1_048_576  # known value


def test_resolve_unknown_model_falls_back():
    """Unknown models without litellm data use DEFAULT_CONTEXT_WINDOW."""
    resolve_context_window.cache_clear()
    result = resolve_context_window("completely-unknown-model-xyz")
    assert result == DEFAULT_CONTEXT_WINDOW


def test_resolve_caches_result():
    """Second call should hit the cache."""
    resolve_context_window.cache_clear()
    r1 = resolve_context_window("gpt-4o")
    r2 = resolve_context_window("gpt-4o")
    assert r1 == r2
    # Check cache was hit (cache_info shows hits > 0)
    info = resolve_context_window.cache_info()
    assert info.hits >= 1


def test_resolve_fallback_table_used_when_litellm_fails():
    """When litellm can't resolve, falls back to static table."""
    resolve_context_window.cache_clear()
    model = "gemini-1.5-pro"

    with (
        patch("adk_deepagents.model_info._lookup_via_litellm", return_value=None),
        patch("adk_deepagents.model_info._lookup_via_genai", return_value=None),
    ):
        result = resolve_context_window(model)

    assert result == _FALLBACK_CONTEXT_WINDOWS[model]


def test_resolve_litellm_takes_priority_over_fallback():
    """litellm result takes priority over the static fallback table."""
    resolve_context_window.cache_clear()
    model = "gemini-2.5-flash"

    with patch("adk_deepagents.model_info._lookup_via_litellm", return_value=999_999):
        result = resolve_context_window(model)

    assert result == 999_999


def test_resolve_genai_used_when_litellm_fails_for_gemini():
    """google.genai is tried when litellm can't resolve a Gemini model."""
    resolve_context_window.cache_clear()
    model = "gemini-2.5-flash"

    with (
        patch("adk_deepagents.model_info._lookup_via_litellm", return_value=None),
        patch("adk_deepagents.model_info._lookup_via_genai", return_value=888_888),
    ):
        result = resolve_context_window(model)

    assert result == 888_888
