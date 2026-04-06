"""Helpers for exposing deep agents over A2A."""

from __future__ import annotations

import importlib
from typing import Any


def to_a2a_app(
    agent: Any,
    *,
    host: str = "localhost",
    port: int = 8000,
    protocol: str = "http",
    agent_card: Any = None,
    runner: Any = None,
) -> Any:
    """Convert an ADK/deep agent into an A2A Starlette app.

    This is a thin wrapper around ``google.adk.a2a.utils.agent_to_a2a.to_a2a``
    with an optional-dependency guard and a stable import surface.
    """
    try:
        module = importlib.import_module("google.adk.a2a.utils.agent_to_a2a")
    except ImportError:
        raise ImportError(
            "A2A support requires the 'a2a-sdk' package. "
            "Install it with: pip install adk-deepagents[a2a]"
        ) from None

    to_a2a = getattr(module, "to_a2a", None)
    if not callable(to_a2a):  # pragma: no cover - defensive path
        raise RuntimeError("google.adk.a2a.utils.agent_to_a2a.to_a2a is not available")

    return to_a2a(
        agent,
        host=host,
        port=port,
        protocol=protocol,
        agent_card=agent_card,
        runner=runner,
    )
