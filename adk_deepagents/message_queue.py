"""Thread-safe message queue helpers."""

from __future__ import annotations

import threading
from typing import Any


class SharedMessageQueue:
    """Thread-safe in-process message buffer.

    Producers call :meth:`push` to enqueue plain text messages.
    Consumers call :meth:`drain` to atomically fetch and clear all queued
    messages in the ``[{"text": ...}, ...]`` shape expected by
    ``DeepAgentConfig.message_queue_provider``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._messages: list[dict[str, Any]] = []

    def push(self, text: str) -> None:
        with self._lock:
            self._messages.append({"text": text})

    def drain(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self._messages:
                return []
            messages = list(self._messages)
            self._messages.clear()
            return messages
