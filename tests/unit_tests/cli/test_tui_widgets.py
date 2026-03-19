"""Unit tests for TUI widgets (cli/tui/widgets.py).

These tests cover the non-Textual-app aspects of the widgets:
data structures, message classes, constants, and SLASH_COMMANDS.
Full integration tests requiring a Textual App context are left
for integration test suites.
"""

from __future__ import annotations

from adk_deepagents.cli.tui.widgets import (
    DEFAULT_PALETTE_ITEMS,
    SLASH_COMMANDS,
    ApprovalBox,
    CommandPalette,
    CommandPaletteItem,
    MessageDisplay,
    PromptInput,
)


class TestSlashCommands:
    def test_is_list_of_tuples(self):
        assert isinstance(SLASH_COMMANDS, list)
        for item in SLASH_COMMANDS:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_all_commands_start_with_slash(self):
        for cmd, _desc in SLASH_COMMANDS:
            assert cmd.startswith("/"), f"Command {cmd!r} missing leading slash"

    def test_known_commands_present(self):
        cmd_names = {cmd.split()[0] for cmd, _ in SLASH_COMMANDS}
        assert "/help" in cmd_names
        assert "/quit" in cmd_names
        assert "/threads" in cmd_names
        assert "/clear" in cmd_names
        assert "/model" in cmd_names

    def test_new_commands_present(self):
        cmd_names = {cmd.split()[0] for cmd, _ in SLASH_COMMANDS}
        assert "/details" in cmd_names
        assert "/compact" in cmd_names

    def test_descriptions_are_nonempty(self):
        for cmd, desc in SLASH_COMMANDS:
            assert desc.strip(), f"Empty description for {cmd}"


class TestApprovalBoxResolved:
    def test_resolved_approved(self):
        msg = ApprovalBox.Resolved(request_id="req-1", approved=True, always=False)
        assert msg.request_id == "req-1"
        assert msg.approved is True
        assert msg.always is False

    def test_resolved_rejected(self):
        msg = ApprovalBox.Resolved(request_id="req-2", approved=False, always=False)
        assert msg.approved is False

    def test_resolved_always(self):
        msg = ApprovalBox.Resolved(request_id="req-3", approved=True, always=True)
        assert msg.always is True


class TestPromptInputSubmitted:
    def test_submitted_message(self):
        msg = PromptInput.Submitted(value="hello world")
        assert msg.value == "hello world"

    def test_submitted_empty(self):
        msg = PromptInput.Submitted(value="")
        assert msg.value == ""


class TestMessageDisplayClassAttributes:
    """Test class-level defaults on MessageDisplay."""

    def test_default_css_exists(self):
        assert MessageDisplay.DEFAULT_CSS is not None
        assert isinstance(MessageDisplay.DEFAULT_CSS, str)
        assert "MessageDisplay" in MessageDisplay.DEFAULT_CSS

    def test_default_state(self):
        assert MessageDisplay._current_assistant_md is None
        assert MessageDisplay._current_assistant_text == ""
        assert MessageDisplay._current_thought is None
        assert MessageDisplay._current_thought_text == ""
        assert MessageDisplay._in_thought is False

    def test_tool_details_default_visible(self):
        assert MessageDisplay._show_tool_details is True


class TestApprovalBoxClass:
    """Test ApprovalBox class structure."""

    def test_has_resolved_message(self):
        assert hasattr(ApprovalBox, "Resolved")
        assert issubclass(ApprovalBox.Resolved, object)


class TestPromptInputClass:
    """Test PromptInput class structure."""

    def test_has_submitted_message(self):
        assert hasattr(PromptInput, "Submitted")

    def test_default_css_exists(self):
        assert PromptInput.DEFAULT_CSS is not None
        assert isinstance(PromptInput.DEFAULT_CSS, str)


# ---------------------------------------------------------------------------
# CommandPaletteItem tests
# ---------------------------------------------------------------------------


class TestCommandPaletteItem:
    """Test CommandPaletteItem data structure."""

    def test_basic_construction(self):
        item = CommandPaletteItem(action="app_quit", label="Quit", description="Exit the TUI")
        assert item.action == "app_quit"
        assert item.label == "Quit"
        assert item.description == "Exit the TUI"
        assert item.keybind == ""

    def test_with_keybind(self):
        item = CommandPaletteItem(
            action="help",
            label="Help",
            description="Show help",
            keybind="ctrl+x h",
        )
        assert item.keybind == "ctrl+x h"


class TestDefaultPaletteItems:
    """Test the DEFAULT_PALETTE_ITEMS list."""

    def test_is_nonempty_list(self):
        assert isinstance(DEFAULT_PALETTE_ITEMS, list)
        assert len(DEFAULT_PALETTE_ITEMS) > 0

    def test_all_items_have_action(self):
        for item in DEFAULT_PALETTE_ITEMS:
            assert isinstance(item, CommandPaletteItem)
            assert item.action

    def test_expected_actions_present(self):
        actions = {item.action for item in DEFAULT_PALETTE_ITEMS}
        assert "app_quit" in actions
        assert "help" in actions
        assert "session_interrupt" in actions
        assert "tool_details_toggle" in actions

    def test_all_have_label_and_description(self):
        for item in DEFAULT_PALETTE_ITEMS:
            assert item.label.strip(), f"Empty label for {item.action}"
            assert item.description.strip(), f"Empty description for {item.action}"


class TestCommandPaletteClass:
    """Test CommandPalette class structure (no Textual app required)."""

    def test_has_action_selected_message(self):
        assert hasattr(CommandPalette, "ActionSelected")

    def test_action_selected_message(self):
        msg = CommandPalette.ActionSelected(action="app_quit")
        assert msg.action == "app_quit"

    def test_default_css_exists(self):
        assert CommandPalette.DEFAULT_CSS is not None
        assert isinstance(CommandPalette.DEFAULT_CSS, str)
