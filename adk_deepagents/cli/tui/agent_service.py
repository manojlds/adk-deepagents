"""Background agent service that bridges the ADK runner with TUI events."""

from __future__ import annotations

import asyncio
import difflib
import io
import json
import logging
import os
import re
import subprocess
import threading
from contextlib import suppress
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
    _merge_ingested_trajectory,
    _ModelCommandContext,
    _ThreadCommandContext,
    _ToolConfirmationRequest,
    handle_slash_command,
)
from adk_deepagents.cli.tui.models import (
    AgentProfile,
    AgentRegistry,
    ConversationLog,
    MessageRecord,
)
from adk_deepagents.types import DynamicTaskConfig, SummarizationConfig

log = logging.getLogger("adk_deepagents.tui.service")

REQUEST_CONFIRMATION_FUNCTION_CALL_NAME = "adk_request_confirmation"

_DETAIL_VALUE_PREVIEW_LIMIT = 80
_DETAIL_TEXT_PREVIEW_LIMIT = 180
_STREAM_CHUNK_SIZE = 28
_ACTIVITY_FRAMES: tuple[str, ...] = ("|", "/", "-", "\\")
_FILE_REF_MAX_SIZE = 100_000  # Skip files larger than 100KB.
_FILE_REF_PATTERN = re.compile(r"(?<!\w)@([\w./\-~][\w./\-~]*)")

ActivityPhase = Literal["working", "thinking", "tool", "responding", "approval"]


class _SharedMessageQueue:
    """Thread-safe in-process message buffer for TUI ↔ callback communication.

    The TUI's ``queue_message()`` appends messages here.  The
    ``before_model_callback`` calls ``drain()`` on every LLM invocation
    to retrieve and clear queued messages — no session state needed.
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


def _expand_file_references(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Expand ``@path/to/file`` references in user input.

    Returns the original text (unchanged) and a list of ``(path, content)``
    pairs for every successfully read file.  Non-existent or unreadable
    paths are silently skipped.
    """
    refs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in _FILE_REF_PATTERN.finditer(text):
        raw_path = match.group(1)
        resolved = Path(raw_path).expanduser()
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        resolved = resolved.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if not resolved.is_file():
            continue
        try:
            if resolved.stat().st_size > _FILE_REF_MAX_SIZE:
                continue
            content = resolved.read_text(errors="replace")
            refs.append((raw_path, content))
        except OSError:
            continue
    return text, refs


def _activity_label_for_phase(phase: ActivityPhase) -> str:
    if phase == "thinking":
        return "Thinking"
    if phase == "tool":
        return "Running tools"
    if phase == "responding":
        return "Responding"
    if phase == "approval":
        return "Awaiting approval"
    return "Working"


def _chunk_stream_text(text: str, *, chunk_size: int = _STREAM_CHUNK_SIZE) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            newline_cut = text.rfind("\n", start + 1, end)
            space_cut = text.rfind(" ", start + 1, end)
            cut = max(newline_cut, space_cut)
            if cut > start:
                end = cut + 1

        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        start = end

    return chunks


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
            keys=(
                "status",
                "subagent_type",
                "task_id",
                "created_subagent",
                "queued",
            ),
            key_aliases={"subagent_type": "subagent"},
        )

        queue_wait = response.get("queue_wait_seconds")
        if isinstance(queue_wait, (int, float)):
            queue_wait_detail = f"queue_wait={queue_wait:.3f}s"
            detail = queue_wait_detail if detail is None else f"{detail}, {queue_wait_detail}"

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


# Minimum line count for tool output to be worth formatting.
_TOOL_OUTPUT_MIN_LINES = 3
# File extensions to language hints for code-fence rendering.
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".rb": "ruby",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
    ".md": "markdown",
    ".dockerfile": "dockerfile",
    ".tf": "hcl",
}


def _guess_language_from_path(path: str | None) -> str:
    """Guess a code-fence language hint from a file path."""
    if not path:
        return ""
    from pathlib import PurePosixPath

    suffix = PurePosixPath(path).suffix.lower()
    return _EXT_TO_LANG.get(suffix, "")


def _extract_tool_output(tool_name: str, response: dict[str, Any]) -> str | None:
    """Extract displayable output text from a tool response.

    Returns a Markdown-formatted string with code fences for content that
    benefits from syntax highlighting, or ``None`` when there is nothing
    interesting to show.
    """
    if tool_name == "read_file":
        content = response.get("content")
        if isinstance(content, str) and content.strip():
            lines = content.strip().splitlines()
            if len(lines) >= _TOOL_OUTPUT_MIN_LINES:
                path = response.get("path", "")
                lang = _guess_language_from_path(path)
                return f"```{lang}\n{content.strip()}\n```"

    if tool_name == "execute":
        output = response.get("output")
        if isinstance(output, str) and output.strip():
            lines = output.strip().splitlines()
            if len(lines) >= _TOOL_OUTPUT_MIN_LINES:
                return f"```\n{output.strip()}\n```"

    if tool_name == "grep":
        result = response.get("result")
        if isinstance(result, str) and result.strip():
            lines = result.strip().splitlines()
            if len(lines) >= _TOOL_OUTPUT_MIN_LINES:
                return f"```\n{result.strip()}\n```"

    if tool_name == "glob":
        entries = response.get("entries")
        if isinstance(entries, list) and len(entries) >= _TOOL_OUTPUT_MIN_LINES:
            text = "\n".join(str(e) for e in entries)
            return f"```\n{text}\n```"

    return None


def _extract_diff_content(
    tool_name: str,
    response: dict[str, Any],
    call_args: dict[str, Any] | None = None,
) -> str | None:
    """Extract or generate unified diff text from a tool response.

    For ``edit_file`` responses with ``status: "success"``, a diff is
    synthesised from the ``old_string`` / ``new_string`` call arguments
    using :mod:`difflib`.  For ``execute`` responses whose output looks
    like ``git diff`` output, the raw text is returned.  A literal
    ``"diff"`` key in the response is also honoured.

    Returns ``None`` when no diff can be produced.
    """
    # edit_file: generate a diff from old_string / new_string when the edit succeeded.
    if tool_name == "edit_file" and call_args and response.get("status") == "success":
        old_string = call_args.get("old_string")
        new_string = call_args.get("new_string")
        file_path = call_args.get("file_path", response.get("path", "file"))
        if isinstance(old_string, str) and isinstance(new_string, str) and old_string != new_string:
            return _generate_unified_diff(old_string, new_string, file_path=str(file_path))

    # Literal "diff" key in the response (defensive / future-proof).
    diff_value = response.get("diff")
    if isinstance(diff_value, str) and diff_value.strip():
        stripped = diff_value.strip()
        # Basic validation: must contain typical diff markers.
        if any(stripped.startswith(p) for p in ("---", "@@", "diff --")):
            return stripped
        if "\n@@" in stripped or "\n---" in stripped:
            return stripped

    # execute responses may contain diff output in the "output" key.
    if tool_name == "execute":
        output = response.get("output")
        if isinstance(output, str) and output.strip():
            stripped = output.strip()
            if stripped.startswith("diff --git") or stripped.startswith("--- a/"):
                return stripped

    return None


def _generate_unified_diff(
    old_text: str,
    new_text: str,
    *,
    file_path: str = "file",
    context_lines: int = 3,
) -> str | None:
    """Produce a unified diff string from two text fragments.

    Returns ``None`` if the texts are identical.
    """
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            n=context_lines,
        )
    )
    if not diff_lines:
        return None
    return "".join(diff_lines).rstrip()


@dataclass(frozen=True)
class TrajectorySummary:
    """Lightweight trajectory metadata for TUI review overlays."""

    trace_id: str
    status: str
    agent_name: str
    steps: int
    score: float | None
    is_golden: bool
    start_time_ns: int


def _load_trajectory_summaries(
    *,
    trajectories_dir: Path,
    otel_traces_path: Path | None,
    sync_from_otel: bool,
) -> list[TrajectorySummary]:
    """Load trajectory summaries, optionally syncing from OTEL traces first."""
    from adk_deepagents.optimization.store import TrajectoryStore

    store = TrajectoryStore(trajectories_dir)

    if sync_from_otel and otel_traces_path is not None and otel_traces_path.exists():
        from adk_deepagents.telemetry.trace_reader import read_traces_file

        new_trajectories = read_traces_file(otel_traces_path)
        for traj in new_trajectories:
            existing = store.load(traj.trace_id)
            if existing is None:
                store.save(traj)
            else:
                merged, changed = _merge_ingested_trajectory(existing, traj)
                if changed:
                    store.save(merged)

    summaries: list[TrajectorySummary] = []
    for trace_id in store.list_ids():
        trajectory = store.load(trace_id)
        if trajectory is None:
            continue
        summaries.append(
            TrajectorySummary(
                trace_id=trajectory.trace_id,
                status=trajectory.status,
                agent_name=trajectory.agent_name or "-",
                steps=len(trajectory.steps),
                score=trajectory.score,
                is_golden=trajectory.is_golden,
                start_time_ns=trajectory.start_time_ns,
            )
        )

    summaries.sort(key=lambda item: (item.start_time_ns, item.trace_id), reverse=True)
    return summaries


@dataclass
class UiUpdate:
    """Event pushed from the agent service to the TUI."""

    kind: Literal[
        "user_message",
        "assistant_delta",
        "thought_delta",
        "tool_call",
        "tool_result",
        "diff_content",
        "system",
        "error",
        "approval_request",
        "activity",
        "turn_started",
        "turn_finished",
        "clear_transcript",
        "queued_message",
        "exit",
    ]
    text: str | None = None
    tool_name: str | None = None
    tool_detail: str | None = None
    tool_output: str | None = None
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
    dynamic_task_config: DynamicTaskConfig | None = None
    memory_sources: list[str] = field(default_factory=list)
    memory_source_paths: dict[str, Path] = field(default_factory=dict)
    skills_dirs: list[str] = field(default_factory=list)
    trajectories_dir: Path | None = None
    otel_traces_path: Path | None = None

    updates: asyncio.Queue[UiUpdate] = field(default_factory=asyncio.Queue)

    _runner: Any = field(default=None, init=False, repr=False)
    _thread_context: _ThreadCommandContext | None = field(default=None, init=False, repr=False)
    _model_context: _ModelCommandContext | None = field(default=None, init=False, repr=False)
    _approval_context: _InteractiveApprovalContext | None = field(
        default=None, init=False, repr=False
    )
    _busy: bool = field(default=False, init=False, repr=False)
    _conversation_log: ConversationLog = field(
        default_factory=ConversationLog, init=False, repr=False
    )
    _agent_registry: AgentRegistry = field(default_factory=AgentRegistry, init=False, repr=False)
    _active_agent_name: str = field(default="", init=False, repr=False)
    # Accumulated assistant text for the current streaming message.
    _current_assistant_text: str = field(default="", init=False, repr=False)
    _activity_phase: ActivityPhase = field(default="working", init=False, repr=False)
    _activity_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _turn_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _pending_approval: asyncio.Future[tuple[bool, bool]] | None = field(
        default=None, init=False, repr=False
    )
    _pending_edit_args: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _queued_messages: list[str] = field(default_factory=list, init=False, repr=False)
    _shared_queue: _SharedMessageQueue = field(
        default_factory=_SharedMessageQueue, init=False, repr=False
    )

    def __post_init__(self) -> None:
        # Eagerly set active agent name so the property is usable before initialize().
        self._active_agent_name = self.agent_name
        # Ensure the startup agent is registered as a profile so Tab cycling
        # works even when the CLI --agent value isn't one of the builtins.
        if self._agent_registry.get(self.agent_name) is None:
            self._agent_registry.add(
                AgentProfile(name=self.agent_name, description="Default agent", mode="primary")
            )

    def initialize(self) -> None:
        """Build the runner and internal contexts. Must be called once."""
        self._active_agent_name = self.agent_name
        self._runner = _build_runner(
            agent_name=self.agent_name,
            model=self.model,
            db_path=self.db_path,
            dynamic_task_config=self.dynamic_task_config,
            memory_sources=self.memory_sources,
            memory_source_paths=self.memory_source_paths or {},
            skills_dirs=self.skills_dirs,
            message_queue_provider=self._shared_queue.drain,
            summarization=SummarizationConfig(),
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
                agent_name=self._active_agent_name,
                model=new_model,
                db_path=self.db_path,
                dynamic_task_config=self.dynamic_task_config,
                memory_sources=self.memory_sources,
                memory_source_paths=self.memory_source_paths or {},
                skills_dirs=self.skills_dirs,
                message_queue_provider=self._shared_queue.drain,
                summarization=SummarizationConfig(),
            )

        self._model_context = _ModelCommandContext(model=self.model, switch_model=_switch_model)

    async def queue_message(self, text: str) -> None:
        """Buffer a message for injection into the next LLM call.

        The message is pushed into a shared in-process buffer that the
        ``before_model_callback`` drains on every LLM invocation.  This
        means mid-turn steering works immediately — no session state
        round-trip is needed.  A ``queued_message`` UI update is emitted
        so the TUI can display the message right away.
        """
        log.debug("[queue_message] text=%r", text)
        self._queued_messages.append(text)
        self._shared_queue.push(text)
        await self.updates.put(UiUpdate(kind="queued_message", text=text))
        self._log_record(MessageRecord(role="queued", text=text))
        log.debug("[queue_message] UI update enqueued")

    async def handle_input(self, text: str) -> None:
        """Process user input — slash command, bash shortcut, or normal prompt."""
        text = text.strip()
        if not text:
            return

        log.debug("[handle_input] text=%r busy=%s", text, self._busy)

        if text.startswith("/"):
            await self._handle_slash_command(text)
            return

        if text.startswith("!"):
            await self._handle_bash_shortcut(text)
            return

        if self._busy:
            log.debug("[handle_input] agent is busy — queuing message")
            await self.queue_message(text)
            return

        # Expand @file references — append file contents as context.
        prompt, file_refs = _expand_file_references(text)
        if file_refs:
            parts = [prompt]
            for ref_path, ref_content in file_refs:
                parts.append(f"\n\n--- @{ref_path} ---\n{ref_content}")
            prompt = "".join(parts)
            # Show the user which files were attached.
            ref_names = ", ".join(f"@{p}" for p, _ in file_refs)
            await self.updates.put(UiUpdate(kind="system", text=f"Attached: {ref_names}"))

        await self.updates.put(UiUpdate(kind="user_message", text=text))
        self._log_record(MessageRecord(role="user", text=text))
        self._busy = True
        self._turn_task = asyncio.create_task(self._run_turn(prompt))

    def resolve_approval(self, approved: bool, always: bool = False) -> None:
        """Resolve a pending tool approval from the TUI."""
        if self._pending_approval and not self._pending_approval.done():
            self._pending_approval.set_result((approved, always))

    async def open_editor(self) -> str | None:
        """Open the user's ``$EDITOR`` and return the composed text.

        Returns ``None`` if the editor is not configured, the user saves
        an empty file, or the process fails.
        """
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
        if not editor:
            await self.updates.put(
                UiUpdate(
                    kind="error",
                    text="No $EDITOR set. Export EDITOR=vim (or your preferred editor).",
                )
            )
            return None

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as tmp:
            tmp_path = tmp.name

        try:
            # Run the editor in a thread so the event loop isn't blocked.
            loop = asyncio.get_running_loop()
            returncode = await loop.run_in_executor(
                None,
                lambda: subprocess.call(editor.split() + [tmp_path]),  # noqa: S603
            )
            if returncode != 0:
                await self.updates.put(
                    UiUpdate(kind="error", text=f"Editor exited with code {returncode}.")
                )
                return None

            content = Path(tmp_path).read_text().strip()
            return content if content else None
        except Exception as exc:  # noqa: BLE001
            await self.updates.put(UiUpdate(kind="error", text=f"Editor error: {exc}"))
            return None
        finally:
            with suppress(OSError):
                os.unlink(tmp_path)

    # -----------------------------------------------------------------
    # Agent switching
    # -----------------------------------------------------------------

    @property
    def agent_registry(self) -> AgentRegistry:
        return self._agent_registry

    @property
    def active_agent_name(self) -> str:
        return self._active_agent_name

    @property
    def active_agent_profile(self) -> AgentProfile | None:
        return self._agent_registry.get(self._active_agent_name)

    async def switch_agent(self, profile: AgentProfile) -> None:
        """Switch to a different agent profile, rebuilding the runner.

        The model is taken from the profile if set, otherwise falls back
        to the current model.  The session (thread) is preserved — only
        the agent personality/tools change.
        """
        if profile.name == self._active_agent_name:
            await self.updates.put(
                UiUpdate(kind="system", text=f"Already using agent '{profile.name}'.")
            )
            return

        if self._busy:
            await self.updates.put(
                UiUpdate(
                    kind="error",
                    text="Cannot switch agents while a turn is running. Interrupt first.",
                )
            )
            return

        model = profile.model or (self._model_context.model if self._model_context else self.model)
        self._active_agent_name = profile.name

        self._runner = _build_runner(
            agent_name=profile.name,
            model=model,
            db_path=self.db_path,
            dynamic_task_config=self.dynamic_task_config,
            memory_sources=self.memory_sources,
            memory_source_paths=self.memory_source_paths or {},
            skills_dirs=self.skills_dirs,
            message_queue_provider=self._shared_queue.drain,
            instruction=profile.prompt,
            summarization=SummarizationConfig(),
        )

        if self._thread_context is not None:
            self._thread_context.agent_name = profile.name

        label = profile.description or profile.name
        await self.updates.put(
            UiUpdate(kind="system", text=f"Switched to agent: {profile.name} — {label}")
        )

    # -----------------------------------------------------------------
    # Conversation log & export
    # -----------------------------------------------------------------

    @property
    def conversation_log(self) -> ConversationLog:
        return self._conversation_log

    def _log_record(self, record: MessageRecord) -> None:
        """Append a record to the conversation log."""
        self._conversation_log.append(record)

    async def export_conversation(self) -> str | None:
        """Export the conversation to Markdown and open in ``$EDITOR``.

        Returns the exported Markdown text, or ``None`` if the
        conversation is empty or the editor fails.
        """
        md = self._conversation_log.to_markdown()
        if not md.strip():
            await self.updates.put(UiUpdate(kind="system", text="Nothing to export."))
            return None

        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
        if not editor:
            # No editor — just emit the markdown as a system message.
            await self.updates.put(
                UiUpdate(kind="system", text="Exported conversation (set $EDITOR to open):")
            )
            await self.updates.put(UiUpdate(kind="system", text=md))
            return md

        import tempfile

        with tempfile.NamedTemporaryFile(
            suffix=".md", prefix="adk-export-", delete=False, mode="w"
        ) as tmp:
            tmp.write(md)
            tmp_path = tmp.name

        try:
            loop = asyncio.get_running_loop()
            returncode = await loop.run_in_executor(
                None,
                lambda: subprocess.call(editor.split() + [tmp_path]),  # noqa: S603
            )
            if returncode != 0:
                await self.updates.put(
                    UiUpdate(kind="error", text=f"Editor exited with code {returncode}.")
                )
            else:
                await self.updates.put(
                    UiUpdate(kind="system", text=f"Conversation exported to {tmp_path}")
                )
            return md
        except Exception as exc:  # noqa: BLE001
            await self.updates.put(UiUpdate(kind="error", text=f"Export error: {exc}"))
            return md

    async def list_trajectory_summaries(
        self,
        *,
        sync_from_otel: bool = True,
    ) -> list[TrajectorySummary]:
        """Return trajectory summaries for TUI trajectory review widgets."""
        trajectories_dir = self.trajectories_dir
        if trajectories_dir is None:
            return []

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: _load_trajectory_summaries(
                trajectories_dir=trajectories_dir,
                otel_traces_path=self.otel_traces_path,
                sync_from_otel=sync_from_otel,
            ),
        )

    async def handle_evaluate_command(self, args: str) -> None:
        """Evaluate a trajectory with the LLM judge."""
        trajectories_dir = self.trajectories_dir
        if trajectories_dir is None:
            await self.updates.put(UiUpdate(kind="error", text="Trajectory store is unavailable."))
            return

        parts = args.split()
        if not parts:
            await self.updates.put(
                UiUpdate(
                    kind="error",
                    text="Usage: /trajectories evaluate <trace_id_prefix>",
                )
            )
            return

        trace_prefix = parts[0]

        from adk_deepagents.optimization.store import TrajectoryStore

        store = TrajectoryStore(trajectories_dir)
        all_ids = store.list_ids()
        matches = [tid for tid in all_ids if tid.startswith(trace_prefix)]
        if not matches:
            await self.updates.put(
                UiUpdate(
                    kind="error",
                    text=f"No trajectory matching '{trace_prefix}'.",
                )
            )
            return
        if len(matches) > 1:
            await self.updates.put(
                UiUpdate(
                    kind="error",
                    text=(
                        f"Ambiguous prefix '{trace_prefix}', "
                        f"matches: {len(matches)}. Be more specific."
                    ),
                )
            )
            return

        trace_id = matches[0]
        trajectory = store.load(trace_id)
        if trajectory is None:
            await self.updates.put(
                UiUpdate(
                    kind="error",
                    text=f"Failed to load trajectory '{trace_id[:12]}'.",
                )
            )
            return

        await self.updates.put(UiUpdate(kind="system", text=f"Evaluating {trace_id[:12]}..."))

        try:
            from adk_deepagents.optimization.evaluator import (
                evaluate_trajectory,
            )

            model = self.model or "gemini-2.5-flash"
            feedback = await evaluate_trajectory(trajectory, model=model)
        except Exception as exc:  # noqa: BLE001
            await self.updates.put(UiUpdate(kind="error", text=f"Evaluation failed: {exc}"))
            return

        # Save feedback and score.
        store.add_feedback(trace_id, feedback)
        if feedback.rating is not None:
            store.set_score(trace_id, feedback.rating)

        # Display results.
        score_str = f"{feedback.rating:.3f}" if feedback.rating is not None else "N/A"
        await self.updates.put(UiUpdate(kind="system", text=f"Score: {score_str}"))
        await self.updates.put(UiUpdate(kind="system", text=f"Summary: {feedback.comment}"))

        criteria = feedback.metadata.get("criteria", [])
        for c in criteria:
            await self.updates.put(
                UiUpdate(
                    kind="system",
                    text=(f"  {c['name']}: {c['score']:.2f} — {c['reasoning'][:100]}"),
                )
            )

        strengths = feedback.metadata.get("strengths", [])
        if strengths:
            await self.updates.put(UiUpdate(kind="system", text="Strengths:"))
            for s in strengths:
                await self.updates.put(UiUpdate(kind="system", text=f"  + {s}"))

        issues = feedback.metadata.get("issues", [])
        if issues:
            await self.updates.put(UiUpdate(kind="system", text="Issues:"))
            for issue in issues:
                await self.updates.put(UiUpdate(kind="system", text=f"  - {issue}"))

    async def handle_replay_command(self, args: str) -> None:
        """Replay a trajectory with the current agent configuration."""
        trajectories_dir = self.trajectories_dir
        if trajectories_dir is None:
            await self.updates.put(UiUpdate(kind="error", text="Trajectory store is unavailable."))
            return

        parts = args.split()
        if not parts:
            await self.updates.put(
                UiUpdate(
                    kind="error",
                    text="Usage: /trajectories replay <trace_id_prefix>",
                )
            )
            return

        trace_prefix = parts[0]

        from adk_deepagents.optimization.store import TrajectoryStore

        store = TrajectoryStore(trajectories_dir)
        all_ids = store.list_ids()
        matches = [tid for tid in all_ids if tid.startswith(trace_prefix)]
        if not matches:
            await self.updates.put(
                UiUpdate(
                    kind="error",
                    text=f"No trajectory matching '{trace_prefix}'.",
                )
            )
            return
        if len(matches) > 1:
            await self.updates.put(
                UiUpdate(
                    kind="error",
                    text=(f"Ambiguous prefix '{trace_prefix}', matches: {len(matches)}."),
                )
            )
            return

        trace_id = matches[0]
        trajectory = store.load(trace_id)
        if trajectory is None:
            await self.updates.put(
                UiUpdate(
                    kind="error",
                    text=f"Failed to load trajectory '{trace_id[:12]}'.",
                )
            )
            return

        await self.updates.put(UiUpdate(kind="system", text=f"Replaying {trace_id[:12]}..."))

        try:
            from adk_deepagents import create_deep_agent
            from adk_deepagents.optimization.replay import (
                BuiltAgent,
                ReplayConfig,
                replay_trajectory,
            )

            model = self.model or "gemini-2.5-flash"

            def builder() -> BuiltAgent:
                agent = create_deep_agent(
                    model=model,
                    name=self.agent_name,
                    instruction=None,
                    execution="local",
                    memory=(list(self.memory_sources) if self.memory_sources else None),
                    skills=(list(self.skills_dirs) if self.skills_dirs else None),
                )
                return BuiltAgent(agent=agent)

            config = ReplayConfig(tool_approval="auto_approve")
            result = await replay_trajectory(
                trajectory,
                agent_builder=builder,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            await self.updates.put(UiUpdate(kind="error", text=f"Replay failed: {exc}"))
            return

        # Save replay trajectory.
        if result.replay_trajectory is not None:
            store.save(result.replay_trajectory)
            store.set_tag(
                result.replay_trajectory.trace_id,
                "optimization_role",
                "replay",
            )
            store.set_tag(
                result.replay_trajectory.trace_id,
                "optimization_parent_trace_id",
                trace_id,
            )
            await self.updates.put(
                UiUpdate(
                    kind="system",
                    text=(f"Replay saved: {result.replay_trajectory.trace_id[:12]}"),
                )
            )

        # Display results.
        await self.updates.put(
            UiUpdate(
                kind="system",
                text=f"Turns: {len(result.per_turn_outputs)}",
            )
        )
        await self.updates.put(UiUpdate(kind="system", text=f"Events: {len(result.events)}"))
        for i, turn_output in enumerate(result.per_turn_outputs):
            preview = turn_output[:200].replace("\n", " ")
            await self.updates.put(UiUpdate(kind="system", text=f"  Turn {i + 1}: {preview}"))

    async def handle_optimize_loop_command(self, args: str) -> None:
        """Run the optimization loop on stored trajectories."""
        trajectories_dir = self.trajectories_dir
        if trajectories_dir is None:
            await self.updates.put(UiUpdate(kind="error", text="Trajectory store is unavailable."))
            return

        # Parse arguments.
        parts = args.split()
        golden_only = "--golden-only" in parts
        max_iter = 2
        for i, p in enumerate(parts):
            if p == "--max-iter" and i + 1 < len(parts):
                try:
                    max_iter = int(parts[i + 1])
                except ValueError:
                    await self.updates.put(
                        UiUpdate(
                            kind="error",
                            text="--max-iter must be an integer.",
                        )
                    )
                    return

        from adk_deepagents.optimization.store import TrajectoryStore

        store = TrajectoryStore(trajectories_dir)

        # Get trajectories.
        if golden_only:
            trajs = store.list_trajectories(is_golden=True)
        else:
            trajs = store.list_trajectories()

        if not trajs:
            label = "golden trajectories" if golden_only else "trajectories"
            await self.updates.put(
                UiUpdate(
                    kind="system",
                    text=(
                        f"No {label} found. Use /trajectories mark <id>"
                        " to mark golden trajectories first."
                    ),
                )
            )
            return

        await self.updates.put(
            UiUpdate(
                kind="system",
                text=(
                    f"Starting optimization loop: {len(trajs)}"
                    f" trajectories, max {max_iter} iterations"
                ),
            )
        )

        try:
            from adk_deepagents import create_deep_agent
            from adk_deepagents.optimization import (
                BuiltAgent,
                OptimizationCandidate,
                ReplayConfig,
                run_optimization_loop,
            )

            model = self.model or "gemini-2.5-flash"

            base_kwargs: dict[str, Any] = {
                "name": self.agent_name,
                "execution": "local",
            }
            if self.memory_sources:
                base_kwargs["memory"] = list(self.memory_sources)
            if self.skills_dirs:
                base_kwargs["skills"] = list(self.skills_dirs)

            base_candidate = OptimizationCandidate(
                agent_kwargs=base_kwargs,
            )

            def agent_builder_factory(
                candidate: OptimizationCandidate,
            ) -> BuiltAgent:
                kwargs = {**candidate.agent_kwargs, "model": model}
                agent = create_deep_agent(**kwargs)
                return BuiltAgent(agent=agent)

            updates_queue = self.updates

            def on_iteration(iteration_result: Any) -> None:
                it = iteration_result
                msgs = [f"--- Iteration {it.iteration} ---"]
                if it.average_score is not None:
                    msgs.append(f"  Avg score: {it.average_score:.3f}")
                if it.average_delta is not None:
                    msgs.append(f"  Avg delta: {it.average_delta:+.3f}")
                msgs.append(f"  Regressions: {it.regressions}")
                for s in it.suggestions:
                    tag = " [auto]" if s.auto_applicable else " [manual]"
                    msgs.append(f"  • {s.kind}{tag}: {s.rationale[:80]}")
                for msg in msgs:
                    with suppress(Exception):
                        updates_queue.put_nowait(UiUpdate(kind="system", text=msg))

            replay_config = ReplayConfig(tool_approval="auto_approve")

            result = await run_optimization_loop(
                trajectories=trajs,
                base_candidate=base_candidate,
                agent_builder_factory=agent_builder_factory,
                evaluator_model=model,
                replay_config=replay_config,
                store=store,
                max_iterations=max_iter,
                apply_mode="prompt_and_skills",
                on_iteration=on_iteration,
            )

        except Exception as exc:  # noqa: BLE001
            await self.updates.put(UiUpdate(kind="error", text=f"Optimization failed: {exc}"))
            return

        # Report results.
        await self.updates.put(
            UiUpdate(
                kind="system",
                text=f"Optimization complete: {result.stopped_reason}",
            )
        )
        await self.updates.put(
            UiUpdate(
                kind="system",
                text=f"Iterations: {len(result.iterations)}",
            )
        )

        for it in result.iterations:
            score = f"{it.average_score:.3f}" if it.average_score is not None else "N/A"
            delta = f"{it.average_delta:+.3f}" if it.average_delta is not None else "N/A"
            await self.updates.put(
                UiUpdate(
                    kind="system",
                    text=(f"  Iter {it.iteration}: score={score} delta={delta}"),
                )
            )

        optimized_instruction = result.best_candidate.agent_kwargs.get("instruction", "")
        if optimized_instruction:
            await self.updates.put(UiUpdate(kind="system", text="Optimized instruction:"))
            for line in optimized_instruction.split("\n"):
                await self.updates.put(UiUpdate(kind="system", text=f"  {line}"))

        all_suggestions = [s for it in result.iterations for s in it.suggestions]
        manual = [s for s in all_suggestions if not s.auto_applicable]
        if manual:
            await self.updates.put(
                UiUpdate(
                    kind="system",
                    text=f"Manual suggestions ({len(manual)}):",
                )
            )
            for s in manual:
                await self.updates.put(
                    UiUpdate(
                        kind="system",
                        text=(f"  [{s.kind}] {s.target}: {s.proposal[:100]}"),
                    )
                )

    async def handle_trajectory_command(self, args: str) -> None:
        """Handle /trajectories subcommands, emitting UiUpdate messages."""
        parts = args.strip().split(None, 1)
        subcommand = parts[0] if parts else ""

        if subcommand == "evaluate":
            rest = parts[1] if len(parts) > 1 else ""
            await self.handle_evaluate_command(rest)
            return

        if subcommand == "replay":
            rest = parts[1] if len(parts) > 1 else ""
            await self.handle_replay_command(rest)
            return

        trajectories_dir = self.trajectories_dir
        if trajectories_dir is None:
            await self.updates.put(UiUpdate(kind="error", text="Trajectory store is unavailable."))
            return

        out = io.StringIO()
        err = io.StringIO()

        from adk_deepagents.cli.interactive import _handle_trajectory_slash_command

        raw_command = f"/trajectories {args}".strip() if args else "/trajectories"

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: _handle_trajectory_slash_command(
                raw_command,
                trajectories_dir=trajectories_dir,
                otel_traces_path=self.otel_traces_path,
                stdout=out,
                stderr=err,
            ),
        )

        for line in out.getvalue().splitlines():
            await self.updates.put(UiUpdate(kind="system", text=line))
        for line in err.getvalue().splitlines():
            if "[error]" in line.lower():
                await self.updates.put(UiUpdate(kind="error", text=line))
            else:
                await self.updates.put(UiUpdate(kind="system", text=line))

    async def _handle_bash_shortcut(self, text: str) -> None:
        """Execute a shell command (``!cmd``) and display the output."""
        command = text[1:].strip()
        if not command:
            await self.updates.put(UiUpdate(kind="error", text="Usage: !<command>"))
            return

        await self.updates.put(UiUpdate(kind="user_message", text=text))
        await self.updates.put(UiUpdate(kind="system", text=f"$ {command}"))

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(  # noqa: S603
                    command,
                    shell=True,  # noqa: S602
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=os.getcwd(),
                ),
            )
            if result.stdout.strip():
                await self.updates.put(UiUpdate(kind="system", text=result.stdout.rstrip()))
            if result.stderr.strip():
                await self.updates.put(UiUpdate(kind="error", text=result.stderr.rstrip()))
            if result.returncode != 0:
                await self.updates.put(
                    UiUpdate(kind="system", text=f"Exit code: {result.returncode}")
                )
        except subprocess.TimeoutExpired:
            await self.updates.put(UiUpdate(kind="error", text="Command timed out (30s limit)."))
        except Exception as exc:  # noqa: BLE001
            await self.updates.put(UiUpdate(kind="error", text=f"Command failed: {exc}"))

    def cancel_turn(self) -> bool:
        """Cancel the currently running agent turn.

        Returns ``True`` if a turn was actually cancelled, ``False`` if
        nothing was running.
        """
        if not self._busy or self._turn_task is None:
            return False
        self._turn_task.cancel()
        return True

    def _set_activity_phase(self, phase: ActivityPhase) -> None:
        self._activity_phase = phase

    async def _run_activity_indicator(self) -> None:
        frame_index = 0
        while self._busy:
            frame = _ACTIVITY_FRAMES[frame_index % len(_ACTIVITY_FRAMES)]
            frame_index += 1
            label = _activity_label_for_phase(self._activity_phase)
            await self.updates.put(UiUpdate(kind="activity", text=f"{label} {frame}"))
            await asyncio.sleep(0.12)

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
                trajectories_dir=self.trajectories_dir,
                otel_traces_path=self.otel_traces_path,
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
            self._conversation_log.clear()
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

        if result == "compact":
            # Trigger the compact tool by sending a prompt to the agent.
            compact_prompt = "Please compact and summarize the conversation so far."
            await self.updates.put(UiUpdate(kind="user_message", text=compact_prompt))
            self._log_record(MessageRecord(role="user", text=compact_prompt))
            self._busy = True
            self._turn_task = asyncio.create_task(self._run_turn(compact_prompt))

    async def _run_turn(self, prompt: str) -> None:
        assert self._thread_context is not None
        assert self._approval_context is not None

        pending_messages: list[types.Content] = [
            types.Content(role="user", parts=[types.Part(text=prompt)])
        ]
        self._set_activity_phase("working")
        await self.updates.put(UiUpdate(kind="turn_started"))
        self._activity_task = asyncio.create_task(self._run_activity_indicator())

        try:
            while pending_messages:
                next_message = pending_messages.pop(0)
                pending_confirmations: list[_ToolConfirmationRequest] = []
                seen_confirmation_ids: set[str] = set()

                # Clear the local UI-tracking list; the shared queue is
                # drained directly by the before_model_callback.
                self._queued_messages.clear()

                async for event in self._runner.run_async(
                    user_id=self._thread_context.user_id,
                    session_id=self._thread_context.active_session_id,
                    new_message=next_message,
                ):
                    if getattr(event, "author", None) == "user":
                        continue

                    await self._emit_event_updates(event)

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
        except asyncio.CancelledError:
            await self.updates.put(UiUpdate(kind="system", text="Generation interrupted."))
        except Exception as exc:  # noqa: BLE001
            await self.updates.put(UiUpdate(kind="error", text=str(exc)))
        finally:
            # Flush any accumulated assistant text to the conversation log.
            if self._current_assistant_text:
                self._log_record(MessageRecord(role="assistant", text=self._current_assistant_text))
                self._current_assistant_text = ""
            self._busy = False
            self._turn_task = None
            self._pending_edit_args.clear()
            activity_task = self._activity_task
            self._activity_task = None
            if activity_task is not None:
                activity_task.cancel()
                with suppress(asyncio.CancelledError):
                    await activity_task
            await self.updates.put(UiUpdate(kind="activity", text=None))
            await self.updates.put(UiUpdate(kind="turn_finished"))

        # After the turn completes, check whether any messages were
        # queued while the agent was busy.  If so, combine them into a
        # single follow-up prompt and start a new turn automatically.
        # This handles the common case where the user sends a message
        # during a simple single-LLM-call turn that finishes before the
        # before_model_callback gets a chance to drain the queue.
        remaining = self._shared_queue.drain()
        if remaining:
            combined = "\n\n---\n\n".join(
                m["text"] for m in remaining if isinstance(m, dict) and m.get("text")
            )
            if combined.strip():
                await self.updates.put(UiUpdate(kind="user_message", text=combined.strip()))
                self._log_record(MessageRecord(role="user", text=combined.strip()))
                self._busy = True
                self._turn_task = asyncio.create_task(self._run_turn(combined.strip()))

    async def _emit_event_updates(self, event: Any) -> None:
        """Parse an ADK event and enqueue ordered UI updates."""

        error_message = getattr(event, "error_message", None)
        if isinstance(error_message, str) and error_message.strip():
            await self.updates.put(UiUpdate(kind="error", text=error_message.strip()))
            self._log_record(MessageRecord(role="error", text=error_message.strip()))

        content = getattr(event, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if not parts:
            return

        for part in parts:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                is_thought = getattr(part, "thought", False)
                kind = "thought_delta" if is_thought else "assistant_delta"
                self._set_activity_phase("thinking" if is_thought else "responding")
                if not is_thought:
                    self._current_assistant_text += text
                for chunk in _chunk_stream_text(text):
                    await self.updates.put(UiUpdate(kind=kind, text=chunk))
                    await asyncio.sleep(0)

            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                self._set_activity_phase("tool")
                tool_name = getattr(function_call, "name", None) or "unknown_tool"
                if tool_name != REQUEST_CONFIRMATION_FUNCTION_CALL_NAME:
                    args_dict = _coerce_payload_dict(getattr(function_call, "args", None))
                    detail = _format_tool_call_detail(tool_name, args_dict)
                    await self.updates.put(
                        UiUpdate(kind="tool_call", tool_name=tool_name, tool_detail=detail)
                    )
                    # Flush accumulated assistant text before tool call.
                    if self._current_assistant_text:
                        self._log_record(
                            MessageRecord(role="assistant", text=self._current_assistant_text)
                        )
                        self._current_assistant_text = ""
                    self._log_record(
                        MessageRecord(
                            role="tool_call",
                            text=detail or "",
                            tool_name=tool_name,
                        )
                    )
                    # Stash args for edit_file keyed by call ID so we can
                    # generate a diff when the response arrives — which may
                    # come in a later event (e.g. after HITL approval).
                    if tool_name == "edit_file":
                        call_id = getattr(function_call, "id", None) or ""
                        self._pending_edit_args[call_id] = args_dict

            function_response = getattr(part, "function_response", None)
            if function_response is not None:
                self._set_activity_phase("tool")
                tool_name = getattr(function_response, "name", None) or "unknown_tool"
                if tool_name == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME:
                    continue

                response = getattr(function_response, "response", None)
                if isinstance(response, dict):
                    detail = _format_tool_response_detail(tool_name, response)
                    # Extract formatted tool output for syntax highlighting.
                    tool_output = _extract_tool_output(tool_name, response)
                    if detail is not None:
                        await self.updates.put(
                            UiUpdate(
                                kind="tool_result",
                                tool_name=tool_name,
                                tool_detail=detail,
                                tool_output=tool_output,
                            )
                        )
                        self._log_record(
                            MessageRecord(
                                role="tool_result",
                                text=detail,
                                tool_name=tool_name,
                            )
                        )

                    # Emit diff content for syntax-highlighted rendering.
                    # Look up stashed call args by the response's call ID,
                    # falling back to any single pending entry.
                    call_args: dict[str, Any] | None = None
                    if tool_name == "edit_file":
                        resp_id = getattr(function_response, "id", None) or ""
                        resp_status = response.get("status")
                        # Only consume pending args on terminal statuses;
                        # intermediate statuses like "awaiting_approval"
                        # should leave them for the final response.
                        is_terminal = resp_status in {"success", "error", None}
                        if is_terminal:
                            call_args = self._pending_edit_args.pop(resp_id, None)
                            if call_args is None and len(self._pending_edit_args) == 1:
                                # Fallback: if there's exactly one pending
                                # entry (common case), use it regardless of
                                # ID mismatch.
                                call_args = self._pending_edit_args.pop(
                                    next(iter(self._pending_edit_args))
                                )
                        else:
                            call_args = self._pending_edit_args.get(resp_id)
                            if call_args is None and len(self._pending_edit_args) == 1:
                                call_args = next(iter(self._pending_edit_args.values()))

                    diff_text = _extract_diff_content(tool_name, response, call_args=call_args)
                    if diff_text is not None:
                        await self.updates.put(UiUpdate(kind="diff_content", text=diff_text))
                        self._log_record(MessageRecord(role="diff", text=diff_text))

                    for key in ("error", "stderr"):
                        value = response.get(key)
                        if isinstance(value, str) and value.strip():
                            await self.updates.put(
                                UiUpdate(kind="error", text=f"{tool_name}: {value.strip()}")
                            )

    async def _await_approval(self, request: _ToolConfirmationRequest) -> bool:
        assert self._approval_context is not None

        if self._approval_context.auto_approve:
            await self.updates.put(
                UiUpdate(kind="system", text=f"Auto-approved '{request.tool_name}'.")
            )
            return True

        self._set_activity_phase("approval")
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
