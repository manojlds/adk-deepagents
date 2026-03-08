"""Background agent service that bridges the ADK runner with TUI events."""

from __future__ import annotations

import asyncio
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from google.genai import types

from adk_deepagents.cli.interactive import (
    _build_confirmation_response_message,
    _build_runner,
    _extract_confirmation_requests,
    _format_approval_args_preview,
    _format_model_name,
    _InteractiveApprovalContext,
    _ModelCommandContext,
    _ThreadCommandContext,
    _ToolConfirmationRequest,
    handle_slash_command,
)

REQUEST_CONFIRMATION_FUNCTION_CALL_NAME = "adk_request_confirmation"

_DETAIL_VALUE_PREVIEW_LIMIT = 80
_DETAIL_TEXT_PREVIEW_LIMIT = 180


def _truncate_preview(text: str, *, limit: int = _DETAIL_VALUE_PREVIEW_LIMIT) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _as_preview(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return _truncate_preview(stripped)

    if isinstance(value, (int, float, bool)):
        return str(value)

    return None


def _coerce_payload_dict(raw_payload: Any) -> dict[str, Any]:
    if isinstance(raw_payload, dict):
        return raw_payload

    if isinstance(raw_payload, str):
        try:
            parsed = json.loads(raw_payload)
        except json.JSONDecodeError:
            return {}

        if isinstance(parsed, dict):
            return parsed

    return {}


def _format_key_values(
    payload: dict[str, Any],
    *,
    keys: tuple[str, ...],
    key_aliases: dict[str, str] | None = None,
) -> str | None:
    pairs: list[str] = []
    aliases = key_aliases or {}
    for key in keys:
        preview = _as_preview(payload.get(key))
        if preview is None:
            continue
        label = aliases.get(key, key)
        pairs.append(f"{label}={preview}")

    if not pairs:
        return None

    return ", ".join(pairs)


def _format_tool_call_detail(tool_name: str, tool_args: dict[str, Any]) -> str | None:
    if not tool_args:
        return None

    if tool_name == "glob":
        return _format_key_values(tool_args, keys=("pattern", "path"))

    if tool_name == "grep":
        return _format_key_values(tool_args, keys=("pattern", "path", "glob", "output_mode"))

    if tool_name == "task":
        detail = _format_key_values(
            tool_args,
            keys=("subagent_type", "task_id", "description", "prompt"),
            key_aliases={"subagent_type": "subagent"},
        )
        if detail is not None:
            return detail

    if tool_name == "register_subagent":
        detail = _format_key_values(tool_args, keys=("name", "model", "description"))
        tool_names = tool_args.get("tool_names")
        if isinstance(tool_names, list):
            if detail is None:
                detail = ""
            detail = (detail + ", " if detail else "") + f"tool_names={len(tool_names)}"
        return detail or None

    if tool_name == "execute":
        return _format_key_values(tool_args, keys=("command",))

    if tool_name in {"ls", "read_file", "write_file", "edit_file"}:
        return _format_key_values(
            tool_args,
            keys=("path", "file_path", "offset", "limit"),
            key_aliases={"file_path": "path"},
        )

    if tool_name in {"write_todos", "read_todos"}:
        todos = tool_args.get("todos")
        if isinstance(todos, list):
            return f"todos={len(todos)}"

    return _format_key_values(tool_args, keys=("path", "name", "description", "command"))


def _format_tool_response_detail(tool_name: str, response: dict[str, Any]) -> str | None:
    status = _as_preview(response.get("status"))

    if tool_name in {"glob", "ls"}:
        entries = response.get("entries")
        if isinstance(entries, list):
            prefix = f"status={status}, " if status else ""
            return f"{prefix}entries={len(entries)}"

    if tool_name == "grep":
        result = response.get("result")
        if isinstance(result, str):
            lines = len(result.splitlines()) if result else 0
            prefix = f"status={status}, " if status else ""
            return f"{prefix}result_lines={lines}"

    if tool_name == "task":
        detail = _format_key_values(
            response,
            keys=("status", "subagent_type", "task_id", "created_subagent"),
            key_aliases={"subagent_type": "subagent"},
        )
        error_value = _as_preview(response.get("error"))
        if error_value:
            if detail is None:
                return f"error={error_value}"
            return f"{detail}, error={error_value}"
        return detail

    if tool_name == "register_subagent":
        return _format_key_values(
            response,
            keys=("status", "subagent_type", "model"),
            key_aliases={"subagent_type": "subagent"},
        )

    if tool_name == "execute":
        detail = _format_key_values(response, keys=("exit_code", "truncated", "status"))
        output = response.get("output")
        if isinstance(output, str) and output.strip():
            output_preview = _truncate_preview(output.strip(), limit=_DETAIL_TEXT_PREVIEW_LIMIT)
            if detail is None:
                return f"output={output_preview}"
            return f"{detail}, output={output_preview}"
        return detail

    if tool_name in {"read_file", "write_file", "edit_file"}:
        return _format_key_values(
            response,
            keys=("status", "path", "occurrences"),
        )

    error_value = _as_preview(response.get("error"))
    if error_value is not None:
        if status is None:
            return f"error={error_value}"
        return f"status={status}, error={error_value}"

    if status is not None:
        return f"status={status}"

    return None


@dataclass
class UiUpdate:
    """Event pushed from the agent service to the TUI."""

    kind: Literal[
        "user_message",
        "assistant_delta",
        "thought_delta",
        "tool_call",
        "tool_result",
        "system",
        "error",
        "approval_request",
        "turn_started",
        "turn_finished",
        "clear_transcript",
        "exit",
    ]
    text: str | None = None
    tool_name: str | None = None
    tool_detail: str | None = None
    request_id: str | None = None
    approval_tool_name: str | None = None
    approval_hint: str | None = None
    approval_args_preview: str | None = None


@dataclass
class AgentService:
    """Manages the ADK runner lifecycle and emits UI-friendly updates."""

    agent_name: str
    user_id: str
    model: str | None
    db_path: Path
    auto_approve: bool
    session_id: str
    memory_sources: list[str] = field(default_factory=list)
    memory_source_paths: dict[str, Path] = field(default_factory=dict)
    skills_dirs: list[str] = field(default_factory=list)

    updates: asyncio.Queue[UiUpdate] = field(default_factory=asyncio.Queue)

    _runner: Any = field(default=None, init=False, repr=False)
    _thread_context: _ThreadCommandContext | None = field(default=None, init=False, repr=False)
    _model_context: _ModelCommandContext | None = field(default=None, init=False, repr=False)
    _approval_context: _InteractiveApprovalContext | None = field(
        default=None, init=False, repr=False
    )
    _busy: bool = field(default=False, init=False, repr=False)
    _pending_approval: asyncio.Future[tuple[bool, bool]] | None = field(
        default=None, init=False, repr=False
    )

    def initialize(self) -> None:
        """Build the runner and internal contexts. Must be called once."""
        self._runner = _build_runner(
            agent_name=self.agent_name,
            model=self.model,
            db_path=self.db_path,
            memory_sources=self.memory_sources,
            memory_source_paths=self.memory_source_paths or {},
            skills_dirs=self.skills_dirs,
        )
        self._thread_context = _ThreadCommandContext(
            db_path=self.db_path,
            user_id=self.user_id,
            agent_name=self.agent_name,
            model=self.model,
            active_session_id=self.session_id,
        )
        self._approval_context = _InteractiveApprovalContext(auto_approve=self.auto_approve)

        def _switch_model(new_model: str | None) -> None:
            self._runner = _build_runner(
                agent_name=self.agent_name,
                model=new_model,
                db_path=self.db_path,
                memory_sources=self.memory_sources,
                memory_source_paths=self.memory_source_paths or {},
                skills_dirs=self.skills_dirs,
            )

        self._model_context = _ModelCommandContext(model=self.model, switch_model=_switch_model)

    async def handle_input(self, text: str) -> None:
        """Process user input — slash command or normal prompt."""
        text = text.strip()
        if not text:
            return

        if text.startswith("/"):
            await self._handle_slash_command(text)
            return

        if self._busy:
            await self.updates.put(UiUpdate(kind="system", text="A turn is already running."))
            return

        await self.updates.put(UiUpdate(kind="user_message", text=text))
        self._busy = True
        asyncio.create_task(self._run_turn(text))

    def resolve_approval(self, approved: bool, always: bool = False) -> None:
        """Resolve a pending tool approval from the TUI."""
        if self._pending_approval and not self._pending_approval.done():
            self._pending_approval.set_result((approved, always))

    async def _handle_slash_command(self, command: str) -> None:
        assert self._thread_context is not None
        assert self._model_context is not None

        prev_session = self._thread_context.active_session_id
        prev_model = self._model_context.model

        out = io.StringIO()
        err = io.StringIO()

        # Run in a thread because slash commands call session_store helpers
        # that use asyncio.run(), which cannot nest inside the running loop.
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: handle_slash_command(
                command,
                stdout=out,
                stderr=err,
                thread_context=self._thread_context,
                model_context=self._model_context,
            ),
        )

        for line in out.getvalue().splitlines():
            await self.updates.put(UiUpdate(kind="system", text=line))
        for line in err.getvalue().splitlines():
            if "[error]" in line.lower():
                await self.updates.put(UiUpdate(kind="error", text=line))
            else:
                await self.updates.put(UiUpdate(kind="system", text=line))

        if self._thread_context.active_session_id != prev_session:
            await self.updates.put(UiUpdate(kind="clear_transcript"))
            await self.updates.put(
                UiUpdate(
                    kind="system",
                    text=f"Active thread: {self._thread_context.active_session_id}",
                )
            )

        if self._model_context.model != prev_model:
            await self.updates.put(
                UiUpdate(
                    kind="system",
                    text=f"Model: {_format_model_name(self._model_context.model)}",
                )
            )

        if result == "exit":
            await self.updates.put(UiUpdate(kind="exit"))

    async def _run_turn(self, prompt: str) -> None:
        assert self._thread_context is not None
        assert self._approval_context is not None

        pending_messages: list[types.Content] = [
            types.Content(role="user", parts=[types.Part(text=prompt)])
        ]
        await self.updates.put(UiUpdate(kind="turn_started"))

        try:
            while pending_messages:
                next_message = pending_messages.pop(0)
                pending_confirmations: list[_ToolConfirmationRequest] = []
                seen_confirmation_ids: set[str] = set()

                async for event in self._runner.run_async(
                    user_id=self._thread_context.user_id,
                    session_id=self._thread_context.active_session_id,
                    new_message=next_message,
                ):
                    if getattr(event, "author", None) == "user":
                        continue

                    self._emit_event_updates(event)

                    for request in _extract_confirmation_requests(event):
                        if request.request_id not in seen_confirmation_ids:
                            seen_confirmation_ids.add(request.request_id)
                            pending_confirmations.append(request)

                for request in pending_confirmations:
                    approved = await self._await_approval(request)
                    pending_messages.append(
                        _build_confirmation_response_message(
                            request_id=request.request_id,
                            approved=approved,
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            await self.updates.put(UiUpdate(kind="error", text=str(exc)))
        finally:
            self._busy = False
            await self.updates.put(UiUpdate(kind="turn_finished"))

    def _emit_event_updates(self, event: Any) -> None:
        """Parse an ADK event and queue UI updates (non-async, fire-and-forget)."""
        error_message = getattr(event, "error_message", None)
        if isinstance(error_message, str) and error_message.strip():
            self.updates.put_nowait(UiUpdate(kind="error", text=error_message.strip()))

        content = getattr(event, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if not parts:
            return

        for part in parts:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                is_thought = getattr(part, "thought", False)
                kind = "thought_delta" if is_thought else "assistant_delta"
                self.updates.put_nowait(UiUpdate(kind=kind, text=text))

            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                tool_name = getattr(function_call, "name", None) or "unknown_tool"
                if tool_name != REQUEST_CONFIRMATION_FUNCTION_CALL_NAME:
                    args_dict = _coerce_payload_dict(getattr(function_call, "args", None))
                    detail = _format_tool_call_detail(tool_name, args_dict)
                    self.updates.put_nowait(
                        UiUpdate(kind="tool_call", tool_name=tool_name, tool_detail=detail)
                    )

            function_response = getattr(part, "function_response", None)
            if function_response is not None:
                tool_name = getattr(function_response, "name", None) or "unknown_tool"
                if tool_name == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME:
                    continue

                response = getattr(function_response, "response", None)
                if isinstance(response, dict):
                    detail = _format_tool_response_detail(tool_name, response)
                    if detail is not None:
                        self.updates.put_nowait(
                            UiUpdate(kind="tool_result", tool_name=tool_name, tool_detail=detail)
                        )

                    for key in ("error", "stderr"):
                        value = response.get(key)
                        if isinstance(value, str) and value.strip():
                            self.updates.put_nowait(
                                UiUpdate(kind="error", text=f"{tool_name}: {value.strip()}")
                            )

    async def _await_approval(self, request: _ToolConfirmationRequest) -> bool:
        assert self._approval_context is not None

        if self._approval_context.auto_approve:
            await self.updates.put(
                UiUpdate(kind="system", text=f"Auto-approved '{request.tool_name}'.")
            )
            return True

        loop = asyncio.get_running_loop()
        self._pending_approval = loop.create_future()

        args_preview = _format_approval_args_preview(request.tool_args)
        await self.updates.put(
            UiUpdate(
                kind="approval_request",
                request_id=request.request_id,
                approval_tool_name=request.tool_name,
                approval_hint=request.hint,
                approval_args_preview=args_preview,
            )
        )

        approved, always = await self._pending_approval
        self._pending_approval = None

        if always:
            self._approval_context.auto_approve = True
            await self.updates.put(
                UiUpdate(kind="system", text="Auto-approve enabled for remaining tool calls.")
            )

        return approved
