"""Custom Textual widgets for the adk-deepagents TUI."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, Markdown, OptionList, Static, TextArea
from textual.widgets.option_list import Option

log = logging.getLogger("adk_deepagents.tui.widgets")

# ---------------------------------------------------------------------------
# File scanning for @ file picker
# ---------------------------------------------------------------------------

_IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        ".env",
        "node_modules",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
        "dist",
        "build",
        ".eggs",
        "*.egg-info",
    }
)

_FILE_PICKER_MAX_FILES = 5000
_FILE_PICKER_MAX_DEPTH = 8
_FILE_PICKER_MAX_SHOWN = 30


def _scan_project_files(root: str | Path | None = None) -> list[str]:
    """Scan project files returning relative paths, respecting ignore patterns.

    Returns up to ``_FILE_PICKER_MAX_FILES`` relative paths sorted
    alphabetically.  Directories in ``_IGNORE_DIRS`` are pruned.
    """
    root = Path.cwd() if root is None else Path(root)

    results: list[str] = []

    def _walk(directory: Path, depth: int) -> None:
        if depth > _FILE_PICKER_MAX_DEPTH:
            return
        if len(results) >= _FILE_PICKER_MAX_FILES:
            return
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError:
            return
        for entry in entries:
            if len(results) >= _FILE_PICKER_MAX_FILES:
                return
            name = entry.name
            if entry.is_dir(follow_symlinks=False):
                if name in _IGNORE_DIRS or name.endswith(".egg-info"):
                    continue
                _walk(Path(entry.path), depth + 1)
            elif entry.is_file(follow_symlinks=False):
                try:
                    rel = Path(entry.path).relative_to(root)
                    results.append(str(rel))
                except ValueError:
                    pass

    _walk(root, 0)
    results.sort()
    return results


# Cached file list (invalidated per-session — good enough for interactive use).
_cached_project_files: list[str] | None = None
_cached_project_root: str | None = None


def _get_project_files(root: str | Path | None = None) -> list[str]:
    """Return cached project file list, rescanning if root changes."""
    global _cached_project_files, _cached_project_root
    resolved = str(Path(root) if root else Path.cwd())
    if _cached_project_files is None or _cached_project_root != resolved:
        _cached_project_files = _scan_project_files(resolved)
        _cached_project_root = resolved
    return _cached_project_files


def invalidate_file_cache() -> None:
    """Force rescan on next ``_get_project_files()`` call."""
    global _cached_project_files, _cached_project_root
    _cached_project_files = None
    _cached_project_root = None


def _extract_at_query(text: str, cursor_col: int, cursor_row: int) -> str | None:
    """Extract the ``@query`` token at the cursor position, or *None*.

    Scans backward from the cursor to find an ``@`` character that is not
    preceded by a word character (to avoid matching email addresses).
    Returns the substring after ``@`` up to the cursor position.
    """
    lines = text.split("\n")
    if cursor_row < 0 or cursor_row >= len(lines):
        return None
    line = lines[cursor_row]
    col = min(cursor_col, len(line))

    # Walk backward from cursor to find '@'.
    i = col - 1
    while i >= 0 and line[i] not in (" ", "\t", "@"):
        i -= 1
    if i < 0 or line[i] != "@":
        return None
    # '@' must not be preceded by a word character (skip emails).
    if i > 0 and line[i - 1].isalnum():
        return None
    return line[i + 1 : col]


# ---------------------------------------------------------------------------
# MessageDisplay – scrollable area for interleaved user/assistant messages
# ---------------------------------------------------------------------------


class MessageDisplay(VerticalScroll):
    """Scrollable area showing interleaved user/assistant messages.

    Assistant text is rendered as Markdown for rich formatting.  Tool calls,
    system messages, and thinking blocks remain as styled ``Static`` widgets.
    """

    DEFAULT_CSS = """
    MessageDisplay {
        height: 1fr;
        padding: 0 1;
    }

    MessageDisplay .user-msg {
        color: $accent;
        margin-bottom: 1;
    }

    MessageDisplay .assistant-md {
        margin-bottom: 1;
    }

    MessageDisplay .thought-msg {
        color: $text-muted;
        text-opacity: 70%;
        text-style: italic;
        margin-bottom: 1;
    }

    MessageDisplay .thought-msg.hidden {
        display: none;
    }

    MessageDisplay .tool-msg {
        color: $success;
        text-opacity: 80%;
        margin-bottom: 0;
    }

    MessageDisplay .tool-detail-msg {
        color: $text-muted;
        margin-bottom: 0;
        text-opacity: 75%;
    }

    MessageDisplay .tool-detail-msg.hidden {
        display: none;
    }

    MessageDisplay .tool-result-msg {
        color: $accent;
        margin-bottom: 1;
        text-opacity: 85%;
    }

    MessageDisplay .tool-result-msg.hidden {
        display: none;
    }

    MessageDisplay .system-msg {
        color: $text-muted;
        text-style: italic;
        margin-bottom: 0;
    }

    MessageDisplay .error-msg {
        color: $error;
        margin-bottom: 0;
    }

    MessageDisplay .approval-box {
        margin-top: 1;
        margin-bottom: 1;
        padding: 1;
        border: solid $warning;
        height: auto;
    }

    MessageDisplay .approval-box .approval-text {
        margin-bottom: 1;
    }

    MessageDisplay .approval-box .approval-buttons {
        height: 3;
    }

    MessageDisplay .approval-box .approval-buttons Button {
        margin-right: 1;
    }

    MessageDisplay .diff-line-added {
        color: #a6e3a1;
        margin: 0;
        padding: 0;
    }

    MessageDisplay .diff-line-removed {
        color: #f38ba8;
        margin: 0;
        padding: 0;
    }

    MessageDisplay .diff-line-hunk {
        color: #89b4fa;
        text-style: bold;
        margin: 0;
        padding: 0;
    }

    MessageDisplay .diff-line-context {
        color: $text-muted;
        margin: 0;
        padding: 0;
    }

    MessageDisplay .queued-msg {
        color: $warning;
        text-opacity: 80%;
        text-style: italic;
        margin-bottom: 1;
    }
    """

    _current_assistant_md: Markdown | None = None
    _current_assistant_text: str = ""
    _current_thought: Static | None = None
    _current_thought_text: str = ""
    _in_thought: bool = False

    # Whether tool call/result details are shown.
    _show_tool_details: bool = True
    # Whether thinking/reasoning blocks are shown.
    _show_thinking: bool = True

    def add_user_message(self, text: str) -> None:
        """Append a user message bubble."""
        self._end_assistant()
        widget = Static(f"> {text}", classes="user-msg")
        self.mount(widget)
        self.scroll_end(animate=False)

    def add_queued_message(self, text: str) -> None:
        """Append a queued message indicator (sent while agent was busy)."""
        self._end_assistant()
        widget = Static(f"> [queued] {text}", classes="queued-msg")
        self.mount(widget)
        self.scroll_end(animate=False)

    def start_assistant_message(self) -> None:
        """Begin a new assistant message for streaming deltas (markdown)."""
        self._end_thought()
        if self._current_assistant_md is not None:
            return
        self._current_assistant_text = ""
        widget = Markdown("", classes="assistant-md")
        self._current_assistant_md = widget
        self.mount(widget)

    def append_assistant_delta(self, text: str) -> None:
        """Append streaming text to the current assistant markdown block."""
        if self._in_thought:
            self._end_thought()
        if self._current_assistant_md is None:
            self.start_assistant_message()
        assert self._current_assistant_md is not None
        self._current_assistant_text += text
        self._current_assistant_md.update(self._current_assistant_text)
        self.scroll_end(animate=False)

    def start_thought_block(self) -> None:
        """Begin a new thought/reasoning block for streaming deltas."""
        self._end_assistant()
        if self._current_thought is not None:
            return
        self._in_thought = True
        self._current_thought_text = ""
        classes = "thought-msg" if self._show_thinking else "thought-msg hidden"
        widget = Static("thinking... ", classes=classes)
        self._current_thought = widget
        self.mount(widget)

    def append_thought_delta(self, text: str) -> None:
        """Append streaming text to the current thought block."""
        if self._current_thought is None:
            self.start_thought_block()
        assert self._current_thought is not None
        self._current_thought_text += text
        self._current_thought.update("thinking... " + self._current_thought_text)
        self.scroll_end(animate=False)

    def end_assistant_message(self) -> None:
        """Close the current assistant message so the next one starts fresh."""
        self._end_thought()
        self._end_assistant()

    def add_tool_call(self, tool_name: str, *, detail: str | None = None) -> None:
        """Show a tool call indicator with optional details."""
        self._end_assistant()
        widget = Static(f"$ {tool_name}", classes="tool-msg")
        self.mount(widget)
        if detail:
            classes = "tool-detail-msg" if self._show_tool_details else "tool-detail-msg hidden"
            self.mount(Static(f"  {detail}", classes=classes))
        self.scroll_end(animate=False)

    def add_tool_result(self, tool_name: str, *, detail: str | None = None) -> None:
        """Show a concise tool result summary."""
        self._end_assistant()
        rendered = f"  -> {detail}" if detail else f"  -> {tool_name} completed"
        classes = "tool-result-msg" if self._show_tool_details else "tool-result-msg hidden"
        self.mount(Static(rendered, classes=classes))
        self.scroll_end(animate=False)

    def add_system_message(self, text: str, *, error: bool = False) -> None:
        """Show a system or error message."""
        self._end_assistant()
        cls = "error-msg" if error else "system-msg"
        widget = Static(text, classes=cls)
        self.mount(widget)
        self.scroll_end(animate=False)

    def add_diff_block(self, diff_text: str) -> None:
        """Render a unified diff with per-line coloring."""
        self._end_assistant()
        for line in diff_text.splitlines():
            if line.startswith("@@"):
                self.mount(Static(line, classes="diff-line-hunk"))
            elif line.startswith("+"):
                self.mount(Static(line, classes="diff-line-added"))
            elif line.startswith("-"):
                self.mount(Static(line, classes="diff-line-removed"))
            else:
                self.mount(Static(line, classes="diff-line-context"))
        self.scroll_end(animate=False)

    def add_approval_prompt(
        self, *, tool_name: str, hint: str | None, args_preview: str, request_id: str
    ) -> None:
        """Show an inline approval prompt with action buttons."""
        self._end_assistant()
        label = f"Tool '{tool_name}' requests approval"
        if hint:
            label += f"\n   Hint: {hint}"
        label += f"\n   Args: {args_preview}"

        box = ApprovalBox(request_id=request_id, label_text=label)
        self.mount(box)
        self.scroll_end(animate=False)

    def clear_transcript(self) -> None:
        """Remove all messages."""
        self._current_assistant_md = None
        self._current_thought = None
        self._in_thought = False
        self.remove_children()

    def toggle_tool_details(self) -> bool:
        """Toggle tool detail visibility retroactively. Returns new state."""
        self._show_tool_details = not self._show_tool_details
        # Retroactively show/hide existing detail and result widgets.
        for widget in self.query(".tool-detail-msg"):
            if self._show_tool_details:
                widget.remove_class("hidden")
            else:
                widget.add_class("hidden")
        for widget in self.query(".tool-result-msg"):
            if self._show_tool_details:
                widget.remove_class("hidden")
            else:
                widget.add_class("hidden")
        return self._show_tool_details

    def toggle_thinking(self) -> bool:
        """Toggle thinking block visibility retroactively. Returns new state."""
        self._show_thinking = not self._show_thinking
        for widget in self.query(".thought-msg"):
            if self._show_thinking:
                widget.remove_class("hidden")
            else:
                widget.add_class("hidden")
        return self._show_thinking

    def _end_assistant(self) -> None:
        self._current_assistant_md = None

    def _end_thought(self) -> None:
        self._current_thought = None
        self._in_thought = False


# ---------------------------------------------------------------------------
# ApprovalBox – inline approve / reject / always
# ---------------------------------------------------------------------------


class ApprovalBox(Static):
    """Inline approval widget with approve/reject/always buttons."""

    class Resolved(Message):
        """Fired when the user clicks an approval button."""

        def __init__(self, request_id: str, approved: bool, always: bool) -> None:
            super().__init__()
            self.request_id = request_id
            self.approved = approved
            self.always = always

    def __init__(self, *, request_id: str, label_text: str) -> None:
        super().__init__(classes="approval-box")
        self._request_id = request_id
        self._label_text = label_text

    def compose(self) -> ComposeResult:
        yield Static(self._label_text, classes="approval-text")
        with Horizontal(classes="approval-buttons"):
            yield Button("Approve", id="btn-approve", variant="success")
            yield Button("Reject", id="btn-reject", variant="error")
            yield Button("Always", id="btn-always", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "btn-approve":
            self.post_message(self.Resolved(self._request_id, approved=True, always=False))
        elif event.button.id == "btn-reject":
            self.post_message(self.Resolved(self._request_id, approved=False, always=False))
        elif event.button.id == "btn-always":
            self.post_message(self.Resolved(self._request_id, approved=True, always=True))
        self.add_class("resolved")


# ---------------------------------------------------------------------------
# Slash command registry
# ---------------------------------------------------------------------------

SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show available commands"),
    ("/threads", "List recent threads"),
    ("/threads <sel>", "Switch thread by index, id, or 'latest'"),
    ("/clear", "Start a new thread"),
    ("/model", "Show the active model"),
    ("/model <name>", "Switch model (or 'default' to reset)"),
    ("/details", "Toggle tool detail visibility"),
    ("/thinking", "Toggle thinking block visibility"),
    ("/theme", "Switch TUI color theme"),
    ("/editor", "Open external editor to compose a message"),
    ("/compact", "Compact/summarize the session context"),
    ("/quit", "Exit the TUI"),
]


# ---------------------------------------------------------------------------
# SubmittableTextArea – TextArea that submits on Enter, newline on Shift+Enter
# ---------------------------------------------------------------------------


class SubmittableTextArea(TextArea):
    """A ``TextArea`` where **Enter** submits and **Shift+Enter** inserts a newline.

    Textual's stock ``TextArea._on_key`` intercepts the ``enter`` key and
    inserts ``"\\n"`` *before* the event can bubble to the parent widget.
    This subclass overrides that behaviour so that plain Enter fires a
    ``Submitted`` message while Shift+Enter (or Ctrl+Enter) still inserts
    a newline.

    Supports **input history**: Up/Down arrows recall previous submissions
    when the cursor is at the first/last line of the input.
    """

    _history: list[str]
    _history_index: int
    _draft: str

    class Submitted(Message):
        """Fired when the user presses Enter (without Shift/Ctrl)."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._history = []
        self._history_index = -1
        self._draft = ""

    def _push_history(self, text: str) -> None:
        """Record a submitted message in history (deduplicating consecutive)."""
        if text and (not self._history or self._history[-1] != text):
            self._history.append(text)
        self._history_index = -1
        self._draft = ""

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter" and not self.read_only:
            # Plain Enter → submit the text instead of inserting a newline.
            event.stop()
            event.prevent_default()
            value = self.text.strip()
            log.debug("[_on_key] enter pressed, read_only=%s, value=%r", self.read_only, value)
            if value:
                self._push_history(value)
                self.post_message(self.Submitted(value))
                self.clear()
            return

        if event.key in {"shift+enter", "ctrl+enter"} and not self.read_only:
            # Shift/Ctrl+Enter → insert a newline (mimic stock behaviour).
            event.stop()
            event.prevent_default()
            start, end = self.selection
            self._replace_via_keyboard("\n", start, end)
            return

        if event.key == "up" and not self.read_only and self._history:
            # Navigate backward through history when cursor is on the first line.
            cursor_row = self.selection.end[0]
            if cursor_row == 0:
                event.stop()
                event.prevent_default()
                if self._history_index == -1:
                    # Save current text as draft before navigating.
                    self._draft = self.text
                    self._history_index = len(self._history) - 1
                elif self._history_index > 0:
                    self._history_index -= 1
                else:
                    return  # Already at oldest entry.
                self.clear()
                self.insert(self._history[self._history_index])
                return

        if event.key == "down" and not self.read_only and self._history:
            # Navigate forward through history when cursor is on the last line.
            cursor_row = self.selection.end[0]
            last_row = self.document.line_count - 1
            if cursor_row == last_row and self._history_index != -1:
                event.stop()
                event.prevent_default()
                if self._history_index < len(self._history) - 1:
                    self._history_index += 1
                    self.clear()
                    self.insert(self._history[self._history_index])
                else:
                    # Past the newest entry → restore draft.
                    self._history_index = -1
                    self.clear()
                    if self._draft:
                        self.insert(self._draft)
                    self._draft = ""
                return

        # Everything else → delegate to the stock handler.
        await super()._on_key(event)


# ---------------------------------------------------------------------------
# PromptInput – multi-line TextArea with slash-command picker
# ---------------------------------------------------------------------------


class PromptInput(Static):
    """Multi-line input box with slash-command picker and submit handling.

    * **Enter** submits the message.
    * **Shift+Enter** (or **Ctrl+Enter**) inserts a newline.
    * Typing ``/`` at the start of input opens the slash-command picker.
    """

    DEFAULT_CSS = """
    PromptInput {
        height: auto;
        min-height: 3;
        max-height: 12;
        padding: 0 1;
    }

    PromptInput SubmittableTextArea {
        width: 1fr;
        min-height: 3;
        max-height: 10;
        border: tall $accent;
    }

    PromptInput SubmittableTextArea:focus {
        border: tall $accent;
    }

    PromptInput .prompt-placeholder {
        color: $text-muted;
        text-opacity: 60%;
        height: 1;
        display: none;
    }

    PromptInput .prompt-placeholder.visible {
        display: block;
    }

    PromptInput #command-picker {
        display: none;
        max-height: 12;
        width: 1fr;
        margin-bottom: 1;
        border: round $accent;
        background: $surface;
    }

    PromptInput #command-picker.visible {
        display: block;
    }

    PromptInput #command-picker > .option-list--option {
        padding: 0 1;
    }

    PromptInput #file-picker {
        display: none;
        max-height: 12;
        width: 1fr;
        margin-bottom: 1;
        border: round $accent;
        background: $surface;
    }

    PromptInput #file-picker.visible {
        display: block;
    }

    PromptInput #file-picker > .option-list--option {
        padding: 0 1;
    }
    """

    class Submitted(Message):
        """Fired when the user presses Enter (without modifier)."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def compose(self) -> ComposeResult:
        yield OptionList(id="command-picker")
        yield OptionList(id="file-picker")
        yield Static("Send a message or /help", classes="prompt-placeholder visible")
        yield SubmittableTextArea(id="prompt-input", language=None)

    def on_mount(self) -> None:
        ta = self.query_one("#prompt-input", SubmittableTextArea)
        ta.show_line_numbers = False

    def on_submittable_text_area_submitted(self, event: SubmittableTextArea.Submitted) -> None:
        """Relay the submission from the TextArea as a PromptInput.Submitted."""
        log.debug("[on_submittable_text_area_submitted] relaying value=%r", event.value)
        event.stop()
        self.query_one("#command-picker", OptionList).remove_class("visible")
        self.query_one("#file-picker", OptionList).remove_class("visible")
        self.post_message(self.Submitted(event.value))

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Show/hide the slash-command picker, file picker, and manage placeholder."""
        ta = self.query_one("#prompt-input", SubmittableTextArea)
        text = ta.text
        placeholder = self.query_one(".prompt-placeholder", Static)

        # Placeholder visibility
        if text:
            placeholder.remove_class("visible")
        else:
            placeholder.add_class("visible")

        cmd_picker = self.query_one("#command-picker", OptionList)
        file_picker = self.query_one("#file-picker", OptionList)

        # --- Slash-command picker (only when text starts with '/') ---
        if text.startswith("/"):
            file_picker.remove_class("visible")
            prefix = text.lower().strip()
            cmd_picker.clear_options()
            for cmd, desc in SLASH_COMMANDS:
                if cmd.startswith(prefix) or prefix == "/":
                    cmd_picker.add_option(Option(f"{cmd}  — {desc}", id=cmd))

            if cmd_picker.option_count > 0:
                cmd_picker.add_class("visible")
                cmd_picker.highlighted = 0
            else:
                cmd_picker.remove_class("visible")
            return

        cmd_picker.remove_class("visible")

        # --- File picker (when '@' is typed) ---
        cursor_row, cursor_col = ta.selection.end
        at_query = _extract_at_query(text, cursor_col, cursor_row)
        if at_query is None:
            file_picker.remove_class("visible")
            return

        query = at_query.lower()
        project_files = _get_project_files()
        file_picker.clear_options()
        count = 0
        for fpath in project_files:
            if count >= _FILE_PICKER_MAX_SHOWN:
                break
            if query and query not in fpath.lower():
                continue
            file_picker.add_option(Option(fpath, id=fpath))
            count += 1

        if file_picker.option_count > 0:
            file_picker.add_class("visible")
            file_picker.highlighted = 0
        else:
            file_picker.remove_class("visible")

    def on_key(self, event: events.Key) -> None:
        """Forward arrow keys to the active picker and handle Escape."""
        cmd_picker = self.query_one("#command-picker", OptionList)
        file_picker = self.query_one("#file-picker", OptionList)

        # Determine which picker is currently visible (at most one).
        active_picker: OptionList | None = None
        if cmd_picker.has_class("visible"):
            active_picker = cmd_picker
        elif file_picker.has_class("visible"):
            active_picker = file_picker

        # Close picker on Escape
        if event.key == "escape" and active_picker is not None:
            active_picker.remove_class("visible")
            event.prevent_default()
            event.stop()
            return

        # Forward arrow keys to picker when visible
        if active_picker is not None and event.key in {"up", "down"}:
            active_picker.focus()
            return

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle selection from either the command picker or the file picker."""
        event.stop()
        cmd_picker = self.query_one("#command-picker", OptionList)
        file_picker = self.query_one("#file-picker", OptionList)

        # --- File picker selection ---
        if file_picker.has_class("visible"):
            file_picker.remove_class("visible")
            selected_path = event.option_id or ""
            if not selected_path:
                return
            ta = self.query_one("#prompt-input", SubmittableTextArea)
            text = ta.text
            cursor_row, cursor_col = ta.selection.end
            lines = text.split("\n")
            if 0 <= cursor_row < len(lines):
                line = lines[cursor_row]
                col = min(cursor_col, len(line))
                # Find the '@' that started this query.
                i = col - 1
                while i >= 0 and line[i] not in (" ", "\t", "@"):
                    i -= 1
                if i >= 0 and line[i] == "@":
                    # Replace @query with @selected_path
                    new_line = line[:i] + "@" + selected_path + line[col:]
                    lines[cursor_row] = new_line
                    new_text = "\n".join(lines)
                    ta.clear()
                    ta.insert(new_text)
            ta.focus()
            return

        # --- Command picker selection ---
        cmd_picker.remove_class("visible")

        cmd_id = event.option_id or ""
        ta = self.query_one("#prompt-input", SubmittableTextArea)

        # Commands with no arguments — submit immediately
        if cmd_id in {
            "/help",
            "/clear",
            "/quit",
            "/details",
            "/compact",
            "/thinking",
            "/theme",
            "/editor",
        }:
            ta.clear()
            self.post_message(self.Submitted(cmd_id))
        else:
            # Place command in the input and strip the placeholder args
            if "<" in cmd_id:
                text = cmd_id.split("<")[0].rstrip() + " "
            else:
                text = cmd_id + " " if " " not in cmd_id else cmd_id
            ta.clear()
            ta.insert(text)
            ta.focus()

    def disable_input(self) -> None:
        ta = self.query_one("#prompt-input", SubmittableTextArea)
        ta.read_only = True
        self.query_one("#command-picker", OptionList).remove_class("visible")
        self.query_one("#file-picker", OptionList).remove_class("visible")

    def enable_input(self) -> None:
        ta = self.query_one("#prompt-input", SubmittableTextArea)
        ta.read_only = False
        ta.focus()
        placeholder = self.query_one(".prompt-placeholder", Static)
        if not ta.text:
            placeholder.add_class("visible")
            placeholder.update("Send a message or /help")

    def set_activity_status(self, status: str | None) -> None:
        placeholder = self.query_one(".prompt-placeholder", Static)
        if status:
            placeholder.update(status)
            placeholder.add_class("visible")
        else:
            placeholder.update("Send a message or /help")
            ta = self.query_one("#prompt-input", SubmittableTextArea)
            if not ta.text:
                placeholder.add_class("visible")
            else:
                placeholder.remove_class("visible")


# ---------------------------------------------------------------------------
# CommandPalette – searchable overlay listing available actions/commands
# ---------------------------------------------------------------------------


class CommandPaletteItem:
    """An action that can appear in the command palette."""

    def __init__(
        self,
        *,
        action: str,
        label: str,
        description: str = "",
        keybind: str = "",
    ) -> None:
        self.action = action
        self.label = label
        self.description = description
        self.keybind = keybind


# Default palette items (will be enriched by the app with resolved keybinds).
DEFAULT_PALETTE_ITEMS: list[CommandPaletteItem] = [
    CommandPaletteItem(
        action="help", label="Help", description="Show help and commands", keybind=""
    ),
    CommandPaletteItem(
        action="session_new",
        label="New Session",
        description="Start a new thread",
        keybind="",
    ),
    CommandPaletteItem(
        action="session_list",
        label="Sessions",
        description="List and switch sessions",
        keybind="",
    ),
    CommandPaletteItem(
        action="model_list",
        label="Models",
        description="Show or switch model",
        keybind="",
    ),
    CommandPaletteItem(
        action="tool_details_toggle",
        label="Toggle Details",
        description="Show/hide tool call details",
        keybind="",
    ),
    CommandPaletteItem(
        action="thinking_toggle",
        label="Toggle Thinking",
        description="Show/hide thinking blocks",
        keybind="",
    ),
    CommandPaletteItem(
        action="theme_picker",
        label="Theme",
        description="Switch TUI color theme",
        keybind="",
    ),
    CommandPaletteItem(
        action="session_compact",
        label="Compact",
        description="Compact/summarize the session",
        keybind="",
    ),
    CommandPaletteItem(
        action="editor_open",
        label="Editor",
        description="Open external editor to compose a message",
        keybind="",
    ),
    CommandPaletteItem(
        action="session_interrupt",
        label="Interrupt",
        description="Cancel the running turn",
        keybind="",
    ),
    CommandPaletteItem(
        action="app_quit",
        label="Quit",
        description="Exit the TUI",
        keybind="",
    ),
]


class CommandPalette(Vertical):
    """Modal command palette overlay with fuzzy search.

    Mount this widget, call :meth:`show`, and listen for
    :class:`CommandPalette.ActionSelected` messages.
    """

    DEFAULT_CSS = """
    CommandPalette {
        display: none;
        width: 60;
        max-width: 80%;
        height: auto;
        max-height: 20;
        background: $surface;
        border: solid $accent;
        padding: 0 1;
        layer: overlay;
        align-horizontal: center;
        offset-y: 2;
    }

    CommandPalette.visible {
        display: block;
    }

    CommandPalette Input {
        width: 1fr;
        margin-bottom: 1;
    }

    CommandPalette OptionList {
        width: 1fr;
        max-height: 14;
    }
    """

    class ActionSelected(Message):
        """Fired when the user selects an action from the palette."""

        def __init__(self, action: str) -> None:
            super().__init__()
            self.action = action

    def __init__(
        self,
        items: list[CommandPaletteItem] | None = None,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._items = items or list(DEFAULT_PALETTE_ITEMS)

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Type to search actions...", id="palette-search")
        yield OptionList(id="palette-list")

    def show(self, items: list[CommandPaletteItem] | None = None) -> None:
        """Open the palette with the given (or default) items."""
        if items is not None:
            self._items = items
        self.add_class("visible")
        self._populate_list("")
        search_input = self.query_one("#palette-search", Input)
        search_input.value = ""
        search_input.focus()

    def hide(self) -> None:
        """Close the palette."""
        self.remove_class("visible")

    def on_input_changed(self, event: Input.Changed) -> None:
        self._populate_list(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Select the highlighted item on Enter."""
        event.stop()
        option_list = self.query_one("#palette-list", OptionList)
        idx = option_list.highlighted
        if idx is not None and 0 <= idx < option_list.option_count:
            option = option_list.get_option_at_index(idx)
            if option.id:
                self.post_message(self.ActionSelected(option.id))
        self.hide()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if event.option_id:
            self.post_message(self.ActionSelected(event.option_id))
        self.hide()

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.hide()
            event.prevent_default()
            event.stop()
        elif event.key in {"up", "down"}:
            self.query_one("#palette-list", OptionList).focus()

    def _populate_list(self, query: str) -> None:
        option_list = self.query_one("#palette-list", OptionList)
        option_list.clear_options()
        q = query.lower().strip()
        for item in self._items:
            text = f"{item.label}  — {item.description}"
            if item.keybind:
                text += f"  [{item.keybind}]"
            if not q or q in item.label.lower() or q in item.description.lower():
                option_list.add_option(Option(text, id=item.action))
        if option_list.option_count > 0:
            option_list.highlighted = 0


# ---------------------------------------------------------------------------
# ThemePicker – overlay for switching color themes
# ---------------------------------------------------------------------------


class ThemePicker(Vertical):
    """Modal overlay for choosing a TUI color theme.

    Shows all available themes in an ``OptionList``.  Fires
    :class:`ThemePicker.ThemeSelected` when the user picks one.
    """

    DEFAULT_CSS = """
    ThemePicker {
        display: none;
        width: 50;
        max-width: 70%;
        height: auto;
        max-height: 20;
        background: $surface;
        border: solid $accent;
        padding: 0 1;
        layer: overlay;
        align-horizontal: center;
        offset-y: 2;
    }

    ThemePicker.visible {
        display: block;
    }

    ThemePicker Input {
        width: 1fr;
        margin-bottom: 1;
    }

    ThemePicker OptionList {
        width: 1fr;
        max-height: 14;
    }
    """

    class ThemeSelected(Message):
        """Fired when the user selects a theme."""

        def __init__(self, theme_name: str) -> None:
            super().__init__()
            self.theme_name = theme_name

    def __init__(
        self,
        theme_names: list[tuple[str, str]] | None = None,
        *,
        current_theme: str = "",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._theme_entries: list[tuple[str, str]] = theme_names or []
        self._current_theme = current_theme

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Type to filter themes...", id="theme-search")
        yield OptionList(id="theme-list")

    def show(
        self,
        theme_entries: list[tuple[str, str]] | None = None,
        current_theme: str = "",
    ) -> None:
        """Open the picker. *theme_entries* are ``(name, label)`` pairs."""
        if theme_entries is not None:
            self._theme_entries = theme_entries
        if current_theme:
            self._current_theme = current_theme
        self.add_class("visible")
        self._populate_list("")
        search_input = self.query_one("#theme-search", Input)
        search_input.value = ""
        search_input.focus()

    def hide(self) -> None:
        self.remove_class("visible")

    def on_input_changed(self, event: Input.Changed) -> None:
        self._populate_list(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        option_list = self.query_one("#theme-list", OptionList)
        idx = option_list.highlighted
        if idx is not None and 0 <= idx < option_list.option_count:
            option = option_list.get_option_at_index(idx)
            if option.id:
                self.post_message(self.ThemeSelected(option.id))
        self.hide()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if event.option_id:
            self.post_message(self.ThemeSelected(event.option_id))
        self.hide()

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.hide()
            event.prevent_default()
            event.stop()
        elif event.key in {"up", "down"}:
            self.query_one("#theme-list", OptionList).focus()

    def _populate_list(self, query: str) -> None:
        option_list = self.query_one("#theme-list", OptionList)
        option_list.clear_options()
        q = query.lower().strip()
        for theme_name, theme_label in self._theme_entries:
            marker = " *" if theme_name == self._current_theme else ""
            text = f"{theme_label}{marker}"
            if not q or q in theme_name.lower() or q in theme_label.lower():
                option_list.add_option(Option(text, id=theme_name))
        if option_list.option_count > 0:
            option_list.highlighted = 0
