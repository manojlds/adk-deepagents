"""Custom Textual widgets for the adk-deepagents TUI."""

from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option


class MessageDisplay(VerticalScroll):
    """Scrollable area showing interleaved user/assistant messages."""

    DEFAULT_CSS = """
    MessageDisplay {
        height: 1fr;
        padding: 0 1;
    }

    MessageDisplay .user-msg {
        color: $accent;
        margin-bottom: 1;
    }

    MessageDisplay .assistant-msg {
        color: $text;
        margin-bottom: 1;
    }

    MessageDisplay .thought-msg {
        color: $text-muted;
        text-opacity: 70%;
        text-style: italic;
        margin-bottom: 1;
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

    MessageDisplay .tool-result-msg {
        color: $accent;
        margin-bottom: 1;
        text-opacity: 85%;
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
    """

    _current_assistant: Static | None = None
    _current_assistant_text: str = ""
    _current_thought: Static | None = None
    _current_thought_text: str = ""
    _in_thought: bool = False

    def add_user_message(self, text: str) -> None:
        """Append a user message bubble."""
        self._end_assistant()
        widget = Static(f"> {text}", classes="user-msg")
        self.mount(widget)
        self.scroll_end(animate=False)

    def start_assistant_message(self) -> None:
        """Begin a new assistant message for streaming deltas."""
        self._end_thought()
        if self._current_assistant is not None:
            return
        self._current_assistant_text = ""
        widget = Static("", classes="assistant-msg")
        self._current_assistant = widget
        self.mount(widget)

    def append_assistant_delta(self, text: str) -> None:
        """Append streaming text to the current assistant message."""
        if self._in_thought:
            self._end_thought()
        if self._current_assistant is None:
            self.start_assistant_message()
        assert self._current_assistant is not None
        self._current_assistant_text += text
        self._current_assistant.update(self._current_assistant_text)
        self.scroll_end(animate=False)

    def start_thought_block(self) -> None:
        """Begin a new thought/reasoning block for streaming deltas."""
        self._end_assistant()
        if self._current_thought is not None:
            return
        self._in_thought = True
        self._current_thought_text = ""
        widget = Static("thinking... ", classes="thought-msg")
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
            self.mount(Static(f"  {detail}", classes="tool-detail-msg"))
        self.scroll_end(animate=False)

    def add_tool_result(self, tool_name: str, *, detail: str | None = None) -> None:
        """Show a concise tool result summary."""
        self._end_assistant()
        rendered = f"  -> {detail}" if detail else f"  -> {tool_name} completed"
        self.mount(Static(rendered, classes="tool-result-msg"))
        self.scroll_end(animate=False)

    def add_system_message(self, text: str, *, error: bool = False) -> None:
        """Show a system or error message."""
        self._end_assistant()
        cls = "error-msg" if error else "system-msg"
        widget = Static(text, classes=cls)
        self.mount(widget)
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
        self._current_assistant = None
        self._current_thought = None
        self._in_thought = False
        self.remove_children()

    def _end_assistant(self) -> None:
        self._current_assistant = None

    def _end_thought(self) -> None:
        self._current_thought = None
        self._in_thought = False


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


SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show available commands"),
    ("/threads", "List recent threads"),
    ("/threads <sel>", "Switch thread by index, id, or 'latest'"),
    ("/clear", "Start a new thread"),
    ("/model", "Show the active model"),
    ("/model <name>", "Switch model (or 'default' to reset)"),
    ("/quit", "Exit the TUI"),
]


class PromptInput(Static):
    """Input box with slash-command picker and submit handling."""

    DEFAULT_CSS = """
    PromptInput {
        height: auto;
        min-height: 3;
        padding: 0 1;
    }

    PromptInput Input {
        width: 1fr;
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
    """

    class Submitted(Message):
        """Fired when the user presses Enter."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def compose(self) -> ComposeResult:
        yield OptionList(id="command-picker")
        yield Input(placeholder="Send a message or /help", id="prompt-input")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Show/hide and filter the command picker as the user types."""
        value = event.value
        picker = self.query_one("#command-picker", OptionList)

        if not value.startswith("/"):
            picker.remove_class("visible")
            return

        prefix = value.lower()
        picker.clear_options()
        for cmd, desc in SLASH_COMMANDS:
            if cmd.startswith(prefix) or prefix == "/":
                picker.add_option(Option(f"{cmd}  — {desc}", id=cmd))

        if picker.option_count > 0:
            picker.add_class("visible")
            picker.highlighted = 0
        else:
            picker.remove_class("visible")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        picker = self.query_one("#command-picker", OptionList)
        picker.remove_class("visible")
        value = event.value.strip()
        if value:
            self.post_message(self.Submitted(value))
            event.input.clear()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Fill the input with the selected slash command."""
        event.stop()
        picker = self.query_one("#command-picker", OptionList)
        picker.remove_class("visible")

        cmd_id = event.option_id or ""
        inp = self.query_one("#prompt-input", Input)

        if cmd_id in {"/help", "/clear", "/quit"}:
            inp.clear()
            self.post_message(self.Submitted(cmd_id))
        else:
            inp.value = cmd_id + " " if "<" in cmd_id else cmd_id
            # Strip the angle-bracket placeholder for commands that take args
            if "<" in inp.value:
                inp.value = cmd_id.split("<")[0].rstrip()
                inp.value += " "
            inp.focus()

    def on_key(self, event: events.Key) -> None:
        """Forward arrow keys from input to the picker when visible."""
        picker = self.query_one("#command-picker", OptionList)
        if not picker.has_class("visible"):
            return

        if event.key == "escape":
            picker.remove_class("visible")
            event.prevent_default()
            event.stop()
        elif event.key in {"up", "down"}:
            picker.focus()

    def disable_input(self) -> None:
        self.query_one("#prompt-input", Input).disabled = True
        self.query_one("#command-picker", OptionList).remove_class("visible")

    def enable_input(self) -> None:
        inp = self.query_one("#prompt-input", Input)
        inp.disabled = False
        inp.focus()
