"""Temporal activity implementation for dynamic sub-agent turns."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ACTIVITY_APP_NAME = "temporal_dynamic_task"
_ACTIVITY_USER_ID = "temporal_task_user"


@dataclass
class TaskSnapshot:
    """Serializable snapshot of one dynamic task turn."""

    subagent_type: str
    prompt: str
    depth: int = 1
    files: dict[str, Any] = field(default_factory=dict)
    todos: list[Any] = field(default_factory=list)
    history: list[dict[str, str]] = field(default_factory=list)
    model_override: str | None = None
    subagent_spec: dict[str, Any] | None = None
    subagent_spec_hash: str | None = None
    timeout_seconds: float = 120.0
    backend_context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSnapshot:
        return cls(
            subagent_type=data.get("subagent_type", "general_purpose"),
            prompt=data.get("prompt", ""),
            depth=data.get("depth", 1),
            files=data.get("files", {}),
            todos=data.get("todos", []),
            history=data.get("history", []),
            model_override=data.get("model_override"),
            subagent_spec=data.get("subagent_spec")
            if isinstance(data.get("subagent_spec"), dict)
            else None,
            subagent_spec_hash=data.get("subagent_spec_hash")
            if isinstance(data.get("subagent_spec_hash"), str)
            else None,
            timeout_seconds=data.get("timeout_seconds", 120.0),
            backend_context=data.get("backend_context")
            if isinstance(data.get("backend_context"), dict)
            else None,
        )


@dataclass
class TaskResult:
    """Serializable activity result for dynamic task execution."""

    result: str = ""
    function_calls: list[str] = field(default_factory=list)
    files: dict[str, Any] = field(default_factory=dict)
    todos: list[Any] = field(default_factory=list)
    timed_out: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskResult:
        return cls(
            result=data.get("result", ""),
            function_calls=data.get("function_calls", []),
            files=data.get("files", {}),
            todos=data.get("todos", []),
            timed_out=data.get("timed_out", False),
            error=data.get("error"),
        )


def _build_backend_factory(backend_context: dict[str, Any] | None) -> Any | None:
    if not isinstance(backend_context, dict):
        return None

    if backend_context.get("kind") != "filesystem":
        return None

    raw_root_dir = backend_context.get("root_dir")
    if not isinstance(raw_root_dir, str) or not raw_root_dir.strip():
        return None

    root_dir = raw_root_dir.strip()
    virtual_mode = bool(backend_context.get("virtual_mode", True))

    raw_mapped_sources = backend_context.get("memory_source_paths")
    if isinstance(raw_mapped_sources, dict):
        try:
            from adk_deepagents.backends.memory_mapped_filesystem import (
                MemoryMappedFilesystemBackend,
            )

            mapped_sources: dict[str, Path] = {}
            for raw_key, raw_value in raw_mapped_sources.items():
                if not isinstance(raw_key, str) or not raw_key.strip():
                    continue
                if not isinstance(raw_value, str) or not raw_value.strip():
                    continue
                mapped_sources[raw_key] = Path(raw_value)

            raw_excludes = backend_context.get("exclude_patterns")
            exclude_patterns = (
                [
                    pattern
                    for pattern in raw_excludes
                    if isinstance(pattern, str) and pattern.strip()
                ]
                if isinstance(raw_excludes, list)
                else []
            )

            raw_respect_gitignore = backend_context.get("respect_gitignore")
            respect_gitignore = (
                raw_respect_gitignore if isinstance(raw_respect_gitignore, bool) else True
            )

            def _memory_mapped_factory(_state: dict[str, Any]) -> Any:
                return MemoryMappedFilesystemBackend(
                    root_dir=Path(root_dir),
                    memory_source_paths=mapped_sources,
                    respect_gitignore=respect_gitignore,
                    exclude_patterns=exclude_patterns,
                )

            return _memory_mapped_factory
        except Exception:  # pragma: no cover - defensive fallback
            logger.debug("Unable to restore memory-mapped backend context", exc_info=True)

    from adk_deepagents.backends.filesystem import FilesystemBackend

    def _filesystem_factory(_state: dict[str, Any]) -> Any:
        return FilesystemBackend(root_dir=root_dir, virtual_mode=virtual_mode)

    return _filesystem_factory


def create_run_task_activity(*, agent_builder: Any) -> Any:
    """Create the Temporal activity that executes one dynamic task turn."""
    from temporalio import activity

    @activity.defn(name="run_dynamic_task")
    async def run_dynamic_task(snapshot_dict: dict[str, Any]) -> dict[str, Any]:
        snapshot = TaskSnapshot.from_dict(snapshot_dict)

        from google.adk.runners import InMemoryRunner
        from google.genai import types

        try:
            child_agent = agent_builder(
                snapshot.subagent_type,
                snapshot.model_override,
                snapshot.subagent_spec,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to build sub-agent for Temporal task")
            return TaskResult(error=f"Agent build failed: {exc}").to_dict()

        from adk_deepagents.backends.runtime import clear_session_backend, register_backend_factory

        runner = InMemoryRunner(agent=child_agent, app_name=_ACTIVITY_APP_NAME)
        session = await runner.session_service.create_session(
            app_name=_ACTIVITY_APP_NAME,
            user_id=_ACTIVITY_USER_ID,
            state={
                "files": snapshot.files,
                "todos": snapshot.todos,
                "_dynamic_delegation_depth": snapshot.depth,
            },
        )

        cleanup_backend_registration = False
        backend_factory = _build_backend_factory(snapshot.backend_context)
        if backend_factory is None:
            logger.warning(
                "Temporal activity missing backend context; falling back to default state backend"
            )
        if backend_factory is not None:
            register_backend_factory(session.id, backend_factory)
            cleanup_backend_registration = True

        prompt = snapshot.prompt
        if snapshot.history:
            from adk_deepagents.tools.task_dynamic import _build_resume_prompt

            prompt = _build_resume_prompt(history=snapshot.history, prompt=snapshot.prompt)

        content = types.Content(role="user", parts=[types.Part(text=prompt)])
        texts: list[str] = []
        function_calls: list[str] = []

        async def _collect() -> None:
            async for event in runner.run_async(
                session_id=session.id,
                user_id=_ACTIVITY_USER_ID,
                new_message=content,
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            texts.append(part.text)
                        if hasattr(part, "function_call") and part.function_call:
                            name = part.function_call.name
                            if isinstance(name, str) and name:
                                function_calls.append(name)

        timed_out = False
        error: str | None = None

        try:
            await asyncio.wait_for(_collect(), timeout=snapshot.timeout_seconds)
        except TimeoutError:
            timed_out = True
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Temporal activity execution failed")
            error = f"{type(exc).__name__}: {exc}"

        session_state: dict[str, Any] = {}
        try:
            final_session = await runner.session_service.get_session(
                app_name=_ACTIVITY_APP_NAME,
                user_id=_ACTIVITY_USER_ID,
                session_id=session.id,
            )
            if final_session is not None and isinstance(final_session.state, dict):
                session_state = final_session.state
        except Exception:  # pragma: no cover - defensive
            logger.debug("Unable to read Temporal activity session state", exc_info=True)
        finally:
            if cleanup_backend_registration:
                clear_session_backend(session.id)

        return TaskResult(
            result="\n".join(texts).strip(),
            function_calls=function_calls,
            files=session_state.get("files", {}),
            todos=session_state.get("todos", []),
            timed_out=timed_out,
            error=error,
        ).to_dict()

    return run_dynamic_task
