"""Main Textual application for the adk-deepagents TUI."""

from __future__ import annotations

import logging
import re
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer, Header

from adk_deepagents.cli.tui.agent_service import AgentService, UiUpdate
from adk_deepagents.cli.tui.keybindings import KeybindConfig, load_keybind_config
from adk_deepagents.cli.tui.themes import (
    BUILTIN_THEMES,
    DEFAULT_THEME_NAME,
    Theme,
    get_theme,
)
from adk_deepagents.cli.tui.widgets import (
    DEFAULT_PALETTE_ITEMS,
    AgentPicker,
    ApprovalBox,
    CommandPalette,
    CommandPaletteItem,
    MessageDisplay,
    PromptInput,
    Sidebar,
    ThemePicker,
)
from adk_deepagents.types import DynamicTaskConfig

log = logging.getLogger("adk_deepagents.tui.app")


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
    theme_name: str | None = None


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
        "agent_cycle": ("Next Agent", False),
        "agent_cycle_reverse": ("Prev Agent", False),
        "agent_list": ("Agents", False),
        "session_compact": ("Compact", False),
        "help": ("Help", False),
        "tool_details_toggle": ("Toggle Details", False),
        "thinking_toggle": ("Toggle Thinking", False),
        "theme_picker": ("Theme", False),
        "editor_open": ("Editor", False),
        "sidebar_toggle": ("Sidebar", False),
        "session_export": ("Export", False),
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

    #main-area {
        height: 1fr;
        layout: horizontal;
    }

    #messages {
        height: 1fr;
        width: 1fr;
    }

    #composer {
        height: auto;
        min-height: 3;
    }

    #command-palette {
        dock: top;
        margin: 2 4;
    }

    #theme-picker {
        dock: top;
        margin: 2 4;
    }

    #agent-picker {
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

        # Resolve theme.
        theme_name = config.theme_name or DEFAULT_THEME_NAME
        self._active_theme: Theme = (
            get_theme(theme_name)
            or get_theme(DEFAULT_THEME_NAME)
            or next(iter(BUILTIN_THEMES.values()))
        )

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
        with Horizontal(id="main-area"):
            yield Sidebar(id="sidebar")
            yield MessageDisplay(id="messages")
        yield PromptInput(id="composer")
        yield CommandPalette(items=self._palette_items, id="command-palette")
        yield ThemePicker(id="theme-picker")
        yield AgentPicker(id="agent-picker")
        yield Footer()

    async def on_mount(self) -> None:
        # Apply initial theme.
        self._apply_theme(self._active_theme)

        self._service.initialize()
        self.run_worker(self._pump_updates(), exclusive=False, exit_on_error=False)

        messages = self.query_one(MessageDisplay)
        messages.add_system_message(f"Thread {self._config.session_id} — type /help for commands")

        # Initialize sidebar with agent/model/session info.
        self._update_sidebar_info()

        # Wire agent names to the prompt input for @agent autocomplete.
        self._sync_agent_names()

        if self._config.first_prompt:
            await self._service.handle_input(self._config.first_prompt)

    # -----------------------------------------------------------------
    # Theme management
    # -----------------------------------------------------------------

    def _apply_theme(self, theme: Theme) -> None:
        """Apply the given theme's CSS to the app."""
        self._active_theme = theme
        with suppress(Exception):
            self.screen.styles.background = theme.background

        # Store theme colors as an app-level attribute for widgets to access.
        self.theme_colors: dict[str, str] = {
            "background": theme.background,
            "surface": theme.surface,
            "panel": theme.panel,
            "text": theme.text,
            "text_muted": theme.text_muted,
            "primary": theme.primary,
            "secondary": theme.secondary,
            "accent": theme.accent,
            "success": theme.success,
            "warning": theme.warning,
            "error": theme.error,
            "info": theme.info,
            "border": theme.border,
            "border_active": theme.border_active,
            "diff_added": theme.diff_added,
            "diff_removed": theme.diff_removed,
            "thought": theme.thought,
        }

        # Update diff line colors in existing message display.
        try:
            md = self.query_one(MessageDisplay)
            for w in md.query(".diff-line-added"):
                w.styles.color = theme.diff_added
            for w in md.query(".diff-line-removed"):
                w.styles.color = theme.diff_removed
            for w in md.query(".diff-line-hunk"):
                w.styles.color = theme.diff_hunk_header
        except Exception:  # noqa: BLE001
            pass

    def _open_theme_picker(self) -> None:
        """Show the theme picker overlay."""
        entries = [
            (name, t.label)
            for name, t in sorted(BUILTIN_THEMES.items(), key=lambda kv: kv[1].label)
        ]
        picker = self.query_one(ThemePicker)
        picker.show(theme_entries=entries, current_theme=self._active_theme.name)

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
        elif action == "thinking_toggle":
            self._do_toggle_thinking()
        elif action == "theme_picker":
            self._open_theme_picker()
        elif action == "editor_open":
            self.run_worker(self._do_open_editor())
        elif action == "agent_cycle":
            self.run_worker(self._do_agent_cycle(reverse=False))
        elif action == "agent_cycle_reverse":
            self.run_worker(self._do_agent_cycle(reverse=True))
        elif action == "agent_list":
            self._open_agent_picker()
        elif action == "sidebar_toggle":
            self._do_toggle_sidebar()
        elif action == "session_export":
            self.run_worker(self._do_export())
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

    def _do_toggle_thinking(self) -> None:
        """Toggle thinking block visibility."""
        messages = self.query_one(MessageDisplay)
        new_state = messages.toggle_thinking()
        label = "shown" if new_state else "hidden"
        messages.add_system_message(f"Thinking blocks: {label}")

    async def _do_open_editor(self) -> None:
        """Open $EDITOR to compose a message, then submit it."""
        text = await self._service.open_editor()
        if text:
            await self._service.handle_input(text)

    async def _do_agent_cycle(self, *, reverse: bool = False) -> None:
        """Cycle to the next (or previous) primary agent."""
        registry = self._service.agent_registry
        current = self._service.active_agent_name
        nxt = registry.cycle_prev(current) if reverse else registry.cycle_next(current)
        if nxt is not None:
            await self._service.switch_agent(nxt)
            self._update_sidebar_info()

    def _open_agent_picker(self) -> None:
        """Show the agent picker overlay."""
        registry = self._service.agent_registry
        entries = [(p.name, p.description, p.mode) for p in registry.all_visible()]
        picker = self.query_one(AgentPicker)
        picker.show(agent_entries=entries, current_agent=self._service.active_agent_name)

    def _do_toggle_sidebar(self) -> None:
        """Toggle the sidebar panel."""
        sidebar = self.query_one(Sidebar)
        visible = sidebar.toggle()
        if visible:
            self._update_sidebar_info()

    async def _do_export(self) -> None:
        """Export the conversation to Markdown."""
        await self._service.export_conversation()

    def _update_sidebar_info(self) -> None:
        """Refresh sidebar labels with current agent/model/session info."""
        sidebar = self.query_one(Sidebar)
        profile = self._service.active_agent_profile
        if profile:
            sidebar.update_agent_info(profile.name, profile.description)
        else:
            sidebar.update_agent_info(self._service.active_agent_name)
        model_name = self._config.model or "default"
        sidebar.update_model_info(model_name)
        if self._service._thread_context is not None:
            sidebar.update_session_info(self._service._thread_context.active_session_id)
        else:
            sidebar.update_session_info(self._config.session_id)

    def _sync_agent_names(self) -> None:
        """Push the current agent names to the prompt input for @mention autocomplete."""
        registry = self._service.agent_registry
        names = [(p.name, p.description) for p in registry.all_visible()]
        composer = self.query_one(PromptInput)
        composer.set_agent_names(names)

    # -----------------------------------------------------------------
    # UI update pump
    # -----------------------------------------------------------------

    async def _pump_updates(self) -> None:
        """Consume UI updates from the agent service forever."""
        while True:
            update = await self._service.updates.get()
            log.debug("[_pump_updates] received update kind=%s", update.kind)
            try:
                self._apply_update(update)
            except Exception:
                log.exception("[_pump_updates] error applying update kind=%s", update.kind)

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

        elif update.kind == "diff_content":
            messages.end_assistant_message()
            messages.add_diff_block(update.text or "")

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
            composer.set_activity_status("Working...")

        elif update.kind == "turn_finished":
            messages.end_assistant_message()
            if not self._waiting_for_approval:
                composer.enable_input()
                composer.set_activity_status(None)

        elif update.kind == "clear_transcript":
            messages.clear_transcript()

        elif update.kind == "queued_message":
            log.debug("[_apply_update] rendering queued_message: %r", update.text)
            messages.end_assistant_message()
            messages.add_queued_message(update.text or "")

        elif update.kind == "exit":
            self.exit()

    # -----------------------------------------------------------------
    # Widget event handlers
    # -----------------------------------------------------------------

    async def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        """Handle user input submission."""
        value = event.value.strip()
        log.debug("[on_prompt_input_submitted] value=%r", value)
        if value == "/details":
            self._do_toggle_details()
            return
        if value == "/thinking":
            self._do_toggle_thinking()
            return
        if value == "/theme":
            self._open_theme_picker()
            return
        if value == "/editor":
            self.run_worker(self._do_open_editor())
            return
        if value == "/export":
            self.run_worker(self._do_export())
            return
        if value == "/agent":
            # Show current agent info.
            profile = self._service.active_agent_profile
            name = self._service.active_agent_name
            desc = profile.description if profile else ""
            msg = f"Agent: {name}"
            if desc:
                msg += f" — {desc}"
            messages = self.query_one(MessageDisplay)
            messages.add_system_message(msg)
            return
        if value.startswith("/agent "):
            # Switch to named agent.
            agent_name = value[len("/agent ") :].strip()
            await self._do_switch_agent_by_name(agent_name)
            return
        # Check for @agent_name mention at the start of the message.
        routed = await self._maybe_route_agent_mention(value)
        if not routed:
            await self._service.handle_input(value)

    def on_approval_box_resolved(self, event: ApprovalBox.Resolved) -> None:
        self._service.resolve_approval(event.approved, event.always)
        self._waiting_for_approval = False
        self.query_one(PromptInput).enable_input()

    def on_command_palette_action_selected(self, event: CommandPalette.ActionSelected) -> None:
        """Handle command palette selection."""
        self._handle_action(event.action)

    def on_theme_picker_theme_selected(self, event: ThemePicker.ThemeSelected) -> None:
        """Handle theme selection from the picker."""
        theme = get_theme(event.theme_name)
        if theme is not None:
            self._apply_theme(theme)
            messages = self.query_one(MessageDisplay)
            messages.add_system_message(f"Theme: {theme.label}")

    async def on_agent_picker_agent_selected(self, event: AgentPicker.AgentSelected) -> None:
        """Handle agent selection from the agent picker overlay."""
        await self._do_switch_agent_by_name(event.agent_name)

    async def on_sidebar_session_selected(self, event: Sidebar.SessionSelected) -> None:
        """Handle session selection from the sidebar."""
        await self._service.handle_input(f"/threads {event.session_id}")
        self._update_sidebar_info()

    async def _do_switch_agent_by_name(self, agent_name: str) -> None:
        """Switch to the agent with the given name."""
        profile = self._service.agent_registry.get(agent_name)
        if profile is None:
            messages = self.query_one(MessageDisplay)
            available = ", ".join(p.name for p in self._service.agent_registry.all_visible())
            messages.add_system_message(f"Unknown agent '{agent_name}'. Available: {available}")
            return
        await self._service.switch_agent(profile)
        self._update_sidebar_info()

    # Pattern: @agent_name at the start of a message, followed by whitespace and text.
    _AGENT_MENTION_RE = re.compile(r"^@(\w[\w-]*)\s+(.*)", re.DOTALL)

    async def _maybe_route_agent_mention(self, value: str) -> bool:
        """If *value* starts with ``@agent_name ...``, route to that agent.

        Temporarily switches to the mentioned agent, sends the remaining
        text, then switches back to the original agent.  Returns ``True``
        if the message was routed, ``False`` otherwise (caller should
        handle normally).
        """
        m = self._AGENT_MENTION_RE.match(value)
        if m is None:
            return False

        target_name = m.group(1)
        body = m.group(2).strip()
        if not body:
            return False

        registry = self._service.agent_registry
        target_profile = registry.get(target_name)
        if target_profile is None:
            # Not a known agent name — treat as normal message.
            return False

        original_name = self._service.active_agent_name
        if target_name == original_name:
            # Already on this agent — just send the body without the prefix.
            await self._service.handle_input(body)
            return True

        # Switch to target agent, send, then switch back.
        await self._service.switch_agent(target_profile)
        self._update_sidebar_info()
        await self._service.handle_input(body)

        # Switch back to the original agent after the turn completes.
        original_profile = registry.get(original_name)
        if original_profile is not None and original_name != self._service.active_agent_name:
            await self._service.switch_agent(original_profile)
            self._update_sidebar_info()

        return True
