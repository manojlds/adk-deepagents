"""Temporal activity implementation for dynamic sub-agent turns."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass, field
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

        return TaskResult(
            result="\n".join(texts).strip(),
            function_calls=function_calls,
            files=session_state.get("files", {}),
            todos=session_state.get("todos", []),
            timed_out=timed_out,
            error=error,
        ).to_dict()

    return run_dynamic_task
