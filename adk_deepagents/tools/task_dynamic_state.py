"""Backend and state coercion helpers for dynamic tasks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

from google.adk.tools.tool_context import ToolContext

from adk_deepagents.backends.protocol import BackendFactory
from adk_deepagents.backends.runtime import get_or_create_backend_for_session

logger = logging.getLogger(__name__)


def _coerce_backend_factory(value: Any) -> BackendFactory | None:
    if callable(value):
        return cast(BackendFactory, value)
    return None


def _backend_context_from_backend(backend: Any) -> dict[str, Any] | None:
    root_value = getattr(backend, "_root", None)
    if isinstance(root_value, Path):
        root_dir = str(root_value)
    elif isinstance(root_value, str) and root_value:
        root_dir = root_value
    else:
        return None

    context: dict[str, Any] = {
        "kind": "filesystem",
        "root_dir": root_dir,
        "virtual_mode": bool(getattr(backend, "_virtual_mode", True)),
    }

    raw_mapped_sources = getattr(backend, "_memory_source_paths", None)
    if isinstance(raw_mapped_sources, dict):
        mapped_sources: dict[str, str] = {}
        for raw_key, raw_path in raw_mapped_sources.items():
            if not isinstance(raw_key, str) or not raw_key:
                continue
            if isinstance(raw_path, Path):
                mapped_sources[raw_key] = str(raw_path)
            elif isinstance(raw_path, str) and raw_path:
                mapped_sources[raw_key] = raw_path

        if mapped_sources:
            context["memory_source_paths"] = mapped_sources

    raw_respect_gitignore = getattr(backend, "_respect_gitignore", None)
    if isinstance(raw_respect_gitignore, bool):
        context["respect_gitignore"] = raw_respect_gitignore

    raw_exclude_patterns = getattr(backend, "_exclude_patterns", None)
    if isinstance(raw_exclude_patterns, (tuple, list)):
        normalized_patterns = [
            pattern
            for pattern in raw_exclude_patterns
            if isinstance(pattern, str) and pattern.strip()
        ]
        if normalized_patterns:
            context["exclude_patterns"] = normalized_patterns

    return context


def _extract_temporal_backend_context(
    *,
    tool_context: ToolContext,
    adk_parent_session_id: str | None,
    runtime_backend_factory: BackendFactory | None,
) -> dict[str, Any] | None:
    state_dict = cast(dict[str, Any], tool_context.state)

    backend: Any = None
    if isinstance(adk_parent_session_id, str) and adk_parent_session_id:
        backend = get_or_create_backend_for_session(adk_parent_session_id, state_dict)

    if backend is None and runtime_backend_factory is not None:
        try:
            backend = runtime_backend_factory(state_dict)
        except Exception:  # pragma: no cover - defensive
            logger.debug(
                "Failed to reconstruct parent backend for Temporal snapshot", exc_info=True
            )

    if backend is None:
        backend = tool_context.state.get("_backend")

    if backend is None:
        state_backend_factory = _coerce_backend_factory(tool_context.state.get("_backend_factory"))
        if state_backend_factory is not None:
            try:
                backend = state_backend_factory(state_dict)
            except Exception:  # pragma: no cover - defensive
                logger.debug(
                    "Failed to build backend from tool_context state factory", exc_info=True
                )

    context = _backend_context_from_backend(backend)
    if context is not None:
        return context

    # Final fallback: preserve the caller's current workspace root.
    return {
        "kind": "filesystem",
        "root_dir": str(Path.cwd().resolve()),
        "virtual_mode": True,
    }


def _coerce_files_state(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _coerce_todos_state(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _coerce_positive_int(value: Any, fallback: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    return fallback
