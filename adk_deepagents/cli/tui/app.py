"""Main Textual application for the adk-deepagents TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

from adk_deepagents.cli.tui.agent_service import AgentService, UiUpdate
from adk_deepagents.cli.tui.keybindings import KeybindConfig, load_keybind_config
from adk_deepagents.cli.tui.widgets import (
    DEFAULT_PALETTE_ITEMS,
    ApprovalBox,
    CommandPalette,
    CommandPaletteItem,
    MessageDisplay,
    PromptInput,
)
from adk_deepagents.types import DynamicTaskConfig


@dataclass
class TuiConfig:
    """All configuration needed to start the TUI."""

    agent_name: str
    user_id: str
    session_id: str
    db_path: Path
    model: str | None = None
    first_prompt: str | None = None
    auto_approve: bool = False
    dynamic_task_config: DynamicTaskConfig | None = None
    memory_sources: list[str] = field(default_factory=list)
    memory_source_paths: dict[str, Path] = field(default_factory=dict)
    skills_dirs: list[str] = field(default_factory=list)
    keybinds_raw: dict[str, Any] | None = None


def _build_bindings(kb: KeybindConfig) -> list[Binding]:
    """Translate the keybind config into Textual ``Binding`` objects.

    Only a subset of actions map to Textual-level bindings (the ones that
    make sense as global shortcuts).  The rest are dispatched via the
    command palette or leader-key sequences.
    """
    # Map action names to (description, show_in_footer) tuples.
    _ACTION_META: dict[str, tuple[str, bool]] = {
        "app_quit": ("Quit", True),
        "command_palette": ("Palette", True),
        "session_interrupt": ("Interrupt", False),
        "session_new": ("New Session", False),
        "session_list": ("Sessions", False),
        "model_list": ("Models", False),
        "session_compact": ("Compact", False),
        "help": ("Help", False),
        "tool_details_toggle": ("Toggle Details", False),
        "editor_open": ("Editor", False),
        "messages_half_page_up": ("Half Page Up", False),
        "messages_half_page_down": ("Half Page Down", False),
        "messages_page_up": ("Page Up", False),
        "messages_page_down": ("Page Down", False),
        "messages_first": ("Scroll Top", False),
        "messages_last": ("Scroll Bottom", False),
    }

    bindings: list[Binding] = []
    for action, (desc, show) in _ACTION_META.items():
        for combo in kb.keys_for(action):
            # Skip combos that contain spaces (leader sequences) — Textual
            # doesn't support multi-key combos natively, so we handle those
            # separately via on_key.
            if " " in combo:
                continue
            bindings.append(Binding(combo, f"app_action('{action}')", desc, show=show))
    return bindings


def _enrich_palette_items(
    items: list[CommandPaletteItem], kb: KeybindConfig
) -> list[CommandPaletteItem]:
    """Attach resolved keybind display strings to palette items."""
    enriched: list[CommandPaletteItem] = []
    for item in items:
        enriched.append(
            CommandPaletteItem(
                action=item.action,
                label=item.label,
                description=item.description,
                keybind=kb.display_for(item.action),
            )
        )
    return enriched


class DeepAgentTui(App[None]):
    """Full-screen TUI for interacting with adk-deepagents."""

    TITLE = "adk-deepagents"
    SUB_TITLE = "agent"

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
        layers: default overlay;
    }

    #messages {
        height: 1fr;
    }

    #composer {
        height: auto;
        min-height: 3;
    }

    #command-palette {
        dock: top;
        margin: 2 4;
    }
    """

    # BINDINGS is set dynamically in __init__ based on keybind config.
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, config: TuiConfig) -> None:
        self._keybind_config = load_keybind_config(config.keybinds_raw)
        # Build palette items with resolved keybinds before super().__init__
        # so they're ready when compose() runs.
        self._palette_items = _enrich_palette_items(DEFAULT_PALETTE_ITEMS, self._keybind_config)
        super().__init__()
        self._config = config
        self._service = AgentService(
            agent_name=config.agent_name,
            user_id=config.user_id,
            model=config.model,
            db_path=config.db_path,
            auto_approve=config.auto_approve,
            session_id=config.session_id,
            dynamic_task_config=config.dynamic_task_config,
            memory_sources=config.memory_sources,
            memory_source_paths=config.memory_source_paths,
            skills_dirs=config.skills_dirs,
        )
        self._waiting_for_approval = False
        # Leader-key state: when the leader key is pressed, we set this flag
        # and wait for the next key to form the full combo.
        self._leader_pressed = False
        # Dynamic bindings — must be set as instance attribute *after*
        # super().__init__() so Textual picks them up.
        for b in _build_bindings(self._keybind_config):
            self._bindings.bind(b.key, b.action, b.description, show=b.show)

    def compose(self) -> ComposeResult:
        yield Header()
        yield MessageDisplay(id="messages")
        yield PromptInput(id="composer")
        yield CommandPalette(items=self._palette_items, id="command-palette")
        yield Footer()

    async def on_mount(self) -> None:
        self._service.initialize()
        self.run_worker(self._pump_updates(), exclusive=False)

        messages = self.query_one(MessageDisplay)
        messages.add_system_message(f"Thread {self._config.session_id} — type /help for commands")

        if self._config.first_prompt:
            await self._service.handle_input(self._config.first_prompt)

    # -----------------------------------------------------------------
    # Key handling for leader-key sequences
    # -----------------------------------------------------------------

    def on_key(self, event: Any) -> None:
        """Handle leader-key sequences (e.g. ``ctrl+x n``).

        Textual doesn't natively support multi-key combos, so we track
        whether the leader key was pressed and compose the full action on
        the follow-up key.
        """
        key = getattr(event, "key", None)
        if key is None:
            return

        leader = self._keybind_config.leader

        if self._leader_pressed:
            self._leader_pressed = False
            full_combo = f"{leader} {key}"
            action = self._resolve_leader_combo(full_combo)
            if action:
                event.prevent_default()
                event.stop()
                self._handle_action(action)
            return

        if key == leader:
            self._leader_pressed = True
            event.prevent_default()
            event.stop()

    def _resolve_leader_combo(self, combo: str) -> str | None:
        """Find the action bound to *combo*, if any."""
        for action, combos in self._keybind_config.bindings.items():
            if combo in combos:
                return action
        return None

    # -----------------------------------------------------------------
    # Action dispatch (from keybindings or command palette)
    # -----------------------------------------------------------------

    def action_app_action(self, action: str) -> None:
        """Textual action handler — dispatches to ``_handle_action``."""
        self._handle_action(action)

    def _handle_action(self, action: str) -> None:
        """Central dispatcher for all TUI actions."""
        if action == "app_quit":
            self.exit()
        elif action == "command_palette":
            self.query_one(CommandPalette).show()
        elif action == "session_interrupt":
            self._do_interrupt()
        elif action == "session_new":
            self.run_worker(self._service.handle_input("/clear"))
        elif action == "session_list":
            self.run_worker(self._service.handle_input("/threads"))
        elif action == "model_list":
            self.run_worker(self._service.handle_input("/model"))
        elif action == "session_compact":
            self.run_worker(self._service.handle_input("/compact"))
        elif action == "help":
            self.run_worker(self._service.handle_input("/help"))
        elif action == "tool_details_toggle":
            self._do_toggle_details()
        elif action == "messages_half_page_up":
            self.query_one(MessageDisplay).scroll_up(animate=False)
        elif action == "messages_half_page_down":
            self.query_one(MessageDisplay).scroll_down(animate=False)
        elif action == "messages_page_up":
            self.query_one(MessageDisplay).scroll_page_up(animate=False)
        elif action == "messages_page_down":
            self.query_one(MessageDisplay).scroll_page_down(animate=False)
        elif action == "messages_first":
            self.query_one(MessageDisplay).scroll_home(animate=False)
        elif action == "messages_last":
            self.query_one(MessageDisplay).scroll_end(animate=False)

    # -----------------------------------------------------------------
    # Action implementations
    # -----------------------------------------------------------------

    def _do_interrupt(self) -> None:
        """Cancel the running agent turn, if any."""
        cancelled = self._service.cancel_turn()
        if not cancelled:
            messages = self.query_one(MessageDisplay)
            messages.add_system_message("Nothing to interrupt.")

    def _do_toggle_details(self) -> None:
        """Toggle tool detail visibility."""
        messages = self.query_one(MessageDisplay)
        new_state = messages.toggle_tool_details()
        label = "shown" if new_state else "hidden"
        messages.add_system_message(f"Tool details: {label}")

    # -----------------------------------------------------------------
    # UI update pump
    # -----------------------------------------------------------------

    async def _pump_updates(self) -> None:
        """Consume UI updates from the agent service forever."""
        while True:
            update = await self._service.updates.get()
            self._apply_update(update)

    def _apply_update(self, update: UiUpdate) -> None:
        messages = self.query_one(MessageDisplay)
        composer = self.query_one(PromptInput)

        if update.kind == "user_message":
            messages.end_assistant_message()
            messages.add_user_message(update.text or "")

        elif update.kind == "assistant_delta":
            messages.append_assistant_delta(update.text or "")

        elif update.kind == "thought_delta":
            messages.append_thought_delta(update.text or "")

        elif update.kind == "tool_call":
            messages.end_assistant_message()
            messages.add_tool_call(
                update.tool_name or "unknown_tool",
                detail=update.tool_detail,
            )

        elif update.kind == "tool_result":
            messages.end_assistant_message()
            messages.add_tool_result(
                update.tool_name or "unknown_tool",
                detail=update.tool_detail,
            )

        elif update.kind == "system":
            messages.end_assistant_message()
            messages.add_system_message(update.text or "")

        elif update.kind == "error":
            messages.end_assistant_message()
            messages.add_system_message(update.text or "", error=True)

        elif update.kind == "approval_request":
            messages.end_assistant_message()
            messages.add_approval_prompt(
                tool_name=update.approval_tool_name or "unknown_tool",
                hint=update.approval_hint,
                args_preview=update.approval_args_preview or "{}",
                request_id=update.request_id or "",
            )
            composer.disable_input()
            self._waiting_for_approval = True

        elif update.kind == "activity":
            composer.set_activity_status(update.text)

        elif update.kind == "turn_started":
            composer.disable_input()
            composer.set_activity_status("Working...")

        elif update.kind == "turn_finished":
            messages.end_assistant_message()
            if not self._waiting_for_approval:
                composer.enable_input()
                composer.set_activity_status(None)

        elif update.kind == "clear_transcript":
            messages.clear_transcript()

        elif update.kind == "exit":
            self.exit()

    # -----------------------------------------------------------------
    # Widget event handlers
    # -----------------------------------------------------------------

    async def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        """Handle user input submission."""
        value = event.value.strip()
        if value == "/details":
            self._do_toggle_details()
            return
        await self._service.handle_input(value)

    def on_approval_box_resolved(self, event: ApprovalBox.Resolved) -> None:
        self._service.resolve_approval(event.approved, event.always)
        self._waiting_for_approval = False
        self.query_one(PromptInput).enable_input()

    def on_command_palette_action_selected(self, event: CommandPalette.ActionSelected) -> None:
        """Handle command palette selection."""
        self._handle_action(event.action)
