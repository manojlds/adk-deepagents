"""Unit tests for TUI widgets (cli/tui/widgets.py).

These tests cover the non-Textual-app aspects of the widgets:
data structures, message classes, constants, and SLASH_COMMANDS.
Full integration tests requiring a Textual App context are left
for integration test suites.
"""

from __future__ import annotations

import os
import tempfile

import pytest
from textual.app import App, ComposeResult

from adk_deepagents.cli.tui.widgets import (
    _FILE_PICKER_MAX_SHOWN,
    _IGNORE_DIRS,
    DEFAULT_PALETTE_ITEMS,
    SLASH_COMMANDS,
    AgentPicker,
    ApprovalBox,
    CommandPalette,
    CommandPaletteItem,
    MessageDisplay,
    PromptInput,
    Sidebar,
    SubmittableTextArea,
    ThemePicker,
    TrajectoryPicker,
    TrajectoryPickerEntry,
    _extract_at_query,
    _scan_project_files,
    invalidate_file_cache,
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

    def test_phase2_commands_present(self):
        cmd_names = {cmd.split()[0] for cmd, _ in SLASH_COMMANDS}
        assert "/thinking" in cmd_names
        assert "/theme" in cmd_names

    def test_phase3_commands_present(self):
        cmd_names = {cmd.split()[0] for cmd, _ in SLASH_COMMANDS}
        assert "/editor" in cmd_names

    def test_trajectory_commands_present(self):
        cmd_names = {cmd.split()[0] for cmd, _ in SLASH_COMMANDS}
        assert "/trajectories" in cmd_names
        assert any(cmd.startswith("/trajectories review") for cmd, _ in SLASH_COMMANDS)
        assert any(cmd.startswith("/trajectories show <") for cmd, _ in SLASH_COMMANDS)
        assert any(cmd.startswith("/trajectories show <id> --detail") for cmd, _ in SLASH_COMMANDS)
        assert any(cmd.startswith("/trajectories golden") for cmd, _ in SLASH_COMMANDS)
        assert any(cmd.startswith("/trajectories mark <") for cmd, _ in SLASH_COMMANDS)
        assert any(cmd.startswith("/trajectories unmark <") for cmd, _ in SLASH_COMMANDS)
        assert any(cmd.startswith("/trajectories rate <") for cmd, _ in SLASH_COMMANDS)
        assert any(cmd.startswith("/trajectories feedback <") for cmd, _ in SLASH_COMMANDS)
        assert any(cmd.startswith("/trajectories tag <") for cmd, _ in SLASH_COMMANDS)
        assert any(cmd.startswith("/trajectories untag <") for cmd, _ in SLASH_COMMANDS)
        assert any(cmd.startswith("/trajectories export") for cmd, _ in SLASH_COMMANDS)
        assert any(cmd.startswith("/trajectories export <path>") for cmd, _ in SLASH_COMMANDS)

    def test_optimize_commands_present(self):
        cmd_names = {cmd.split()[0] for cmd, _ in SLASH_COMMANDS}
        assert "/optimize" in cmd_names
        assert any(cmd.startswith("/optimize gepa") for cmd, _ in SLASH_COMMANDS)

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

    def test_show_thinking_default_visible(self):
        assert MessageDisplay._show_thinking is True


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

    def test_phase2_actions_present(self):
        actions = {item.action for item in DEFAULT_PALETTE_ITEMS}
        assert "thinking_toggle" in actions
        assert "theme_picker" in actions

    def test_phase3_actions_present(self):
        actions = {item.action for item in DEFAULT_PALETTE_ITEMS}
        assert "editor_open" in actions

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


class TestSubmittableTextArea:
    """Test SubmittableTextArea class structure."""

    def test_has_submitted_message(self):
        assert hasattr(SubmittableTextArea, "Submitted")

    def test_submitted_message_value(self):
        msg = SubmittableTextArea.Submitted(value="hello")
        assert msg.value == "hello"

    def test_is_textarea_subclass(self):
        from textual.widgets import TextArea

        assert issubclass(SubmittableTextArea, TextArea)

    def test_history_attributes_exist(self):
        """SubmittableTextArea should have input history attributes."""
        # Can't instantiate outside a Textual app, so check class annotations.
        annotations = SubmittableTextArea.__annotations__
        assert "_history" in annotations
        assert "_history_index" in annotations
        assert "_draft" in annotations

    def test_push_history_method_exists(self):
        assert hasattr(SubmittableTextArea, "_push_history")


# ---------------------------------------------------------------------------
# ThemePicker tests
# ---------------------------------------------------------------------------


class TestThemePickerClass:
    """Test ThemePicker class structure (no Textual app required)."""

    def test_has_theme_selected_message(self):
        assert hasattr(ThemePicker, "ThemeSelected")

    def test_theme_selected_message(self):
        msg = ThemePicker.ThemeSelected(theme_name="catppuccin")
        assert msg.theme_name == "catppuccin"

    def test_default_css_exists(self):
        assert ThemePicker.DEFAULT_CSS is not None
        assert isinstance(ThemePicker.DEFAULT_CSS, str)

    def test_is_vertical_subclass(self):
        from textual.containers import Vertical

        assert issubclass(ThemePicker, Vertical)


# ---------------------------------------------------------------------------
# File picker utility tests
# ---------------------------------------------------------------------------


class TestExtractAtQuery:
    """Test _extract_at_query() helper for detecting @-tokens at cursor."""

    def test_simple_at_query(self):
        assert _extract_at_query("@foo", cursor_col=4, cursor_row=0) == "foo"

    def test_at_with_empty_query(self):
        assert _extract_at_query("@", cursor_col=1, cursor_row=0) == ""

    def test_at_mid_line(self):
        text = "check @src/main"
        assert _extract_at_query(text, cursor_col=15, cursor_row=0) == "src/main"

    def test_at_after_space(self):
        text = "look at @widgets.py please"
        assert _extract_at_query(text, cursor_col=19, cursor_row=0) == "widgets.py"

    def test_no_at_returns_none(self):
        assert _extract_at_query("hello world", cursor_col=5, cursor_row=0) is None

    def test_email_ignored(self):
        """An @ preceded by a word character (email) should return None."""
        assert _extract_at_query("user@example.com", cursor_col=16, cursor_row=0) is None

    def test_multiline_second_row(self):
        text = "line one\n@myfile"
        assert _extract_at_query(text, cursor_col=7, cursor_row=1) == "myfile"

    def test_cursor_at_start_no_at(self):
        assert _extract_at_query("@foo", cursor_col=0, cursor_row=0) is None

    def test_cursor_right_after_at(self):
        assert _extract_at_query("@", cursor_col=1, cursor_row=0) == ""

    def test_at_with_path_separators(self):
        text = "@src/cli/tui/wid"
        assert _extract_at_query(text, cursor_col=16, cursor_row=0) == "src/cli/tui/wid"

    def test_invalid_row(self):
        assert _extract_at_query("@foo", cursor_col=4, cursor_row=5) is None

    def test_tab_separated(self):
        text = "hello\t@bar"
        assert _extract_at_query(text, cursor_col=10, cursor_row=0) == "bar"


class TestScanProjectFiles:
    """Test _scan_project_files() with a temporary directory tree."""

    def test_basic_scan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files.
            open(os.path.join(tmpdir, "main.py"), "w").close()
            open(os.path.join(tmpdir, "README.md"), "w").close()
            os.makedirs(os.path.join(tmpdir, "src"))
            open(os.path.join(tmpdir, "src", "app.py"), "w").close()

            result = _scan_project_files(tmpdir)
            assert "main.py" in result
            assert "README.md" in result
            assert os.path.join("src", "app.py") in result

    def test_ignores_git_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, ".git", "objects"))
            open(os.path.join(tmpdir, ".git", "objects", "pack"), "w").close()
            open(os.path.join(tmpdir, "main.py"), "w").close()

            result = _scan_project_files(tmpdir)
            assert "main.py" in result
            assert not any(".git" in p for p in result)

    def test_ignores_pycache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "__pycache__"))
            open(os.path.join(tmpdir, "__pycache__", "foo.pyc"), "w").close()
            open(os.path.join(tmpdir, "main.py"), "w").close()

            result = _scan_project_files(tmpdir)
            assert "main.py" in result
            assert not any("__pycache__" in p for p in result)

    def test_ignores_node_modules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "node_modules", "express"))
            open(os.path.join(tmpdir, "node_modules", "express", "index.js"), "w").close()
            open(os.path.join(tmpdir, "app.js"), "w").close()

            result = _scan_project_files(tmpdir)
            assert "app.js" in result
            assert not any("node_modules" in p for p in result)

    def test_ignores_venv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, ".venv", "lib"))
            open(os.path.join(tmpdir, ".venv", "lib", "site.py"), "w").close()
            open(os.path.join(tmpdir, "main.py"), "w").close()

            result = _scan_project_files(tmpdir)
            assert "main.py" in result
            assert not any(".venv" in p for p in result)

    def test_results_sorted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "z.py"), "w").close()
            open(os.path.join(tmpdir, "a.py"), "w").close()
            open(os.path.join(tmpdir, "m.py"), "w").close()

            result = _scan_project_files(tmpdir)
            assert result == sorted(result)

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _scan_project_files(tmpdir)
            assert result == []

    def test_nonexistent_root_returns_empty(self):
        result = _scan_project_files("/nonexistent/path/that/does/not/exist")
        assert result == []


class TestIgnoreDirs:
    """Test that _IGNORE_DIRS contains expected entries."""

    def test_git_in_ignore(self):
        assert ".git" in _IGNORE_DIRS

    def test_pycache_in_ignore(self):
        assert "__pycache__" in _IGNORE_DIRS

    def test_venv_in_ignore(self):
        assert ".venv" in _IGNORE_DIRS

    def test_node_modules_in_ignore(self):
        assert "node_modules" in _IGNORE_DIRS


class TestFilePickerConstants:
    """Test file picker configuration constants."""

    def test_max_shown_is_reasonable(self):
        assert _FILE_PICKER_MAX_SHOWN > 0
        assert _FILE_PICKER_MAX_SHOWN <= 100

    def test_invalidate_cache(self):
        """invalidate_file_cache should reset the cache globals."""
        invalidate_file_cache()
        # After invalidation, the module-level cache should be None.
        from adk_deepagents.cli.tui import widgets

        assert widgets._cached_project_files is None
        assert widgets._cached_project_root is None


class TestPromptInputFilePicker:
    """Test PromptInput class structure includes file picker CSS."""

    def test_file_picker_css_present(self):
        assert "#file-picker" in PromptInput.DEFAULT_CSS

    def test_file_picker_visible_css_present(self):
        assert "#file-picker.visible" in PromptInput.DEFAULT_CSS


# ---------------------------------------------------------------------------
# Textual pilot test: PromptInput submission event chain
# ---------------------------------------------------------------------------


class _SubmitTestApp(App):
    """Minimal app to test PromptInput submission chain."""

    submitted_values: list[str]

    def compose(self) -> ComposeResult:
        yield PromptInput()

    def on_mount(self) -> None:
        self.submitted_values = []

    def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        self.submitted_values.append(event.value)


class TestPromptInputSubmissionPilot:
    """Use Textual's pilot to test that Enter fires PromptInput.Submitted."""

    @pytest.mark.asyncio
    async def test_enter_fires_submitted(self) -> None:
        """Type text and press Enter; verify Submitted event reaches the app."""
        app = _SubmitTestApp()
        async with app.run_test() as pilot:
            ta = app.query_one("#prompt-input", SubmittableTextArea)
            ta.focus()
            # Type some text
            await pilot.press("h", "e", "l", "l", "o")
            await pilot.pause()
            assert ta.text == "hello"
            # Press Enter to submit
            await pilot.press("enter")
            await pilot.pause()
            assert app.submitted_values == ["hello"]
            # TextArea should be cleared after submission
            assert ta.text == ""

    @pytest.mark.asyncio
    async def test_multiple_submits(self) -> None:
        """Two rapid submissions should both fire Submitted events."""
        app = _SubmitTestApp()
        async with app.run_test() as pilot:
            ta = app.query_one("#prompt-input", SubmittableTextArea)
            ta.focus()
            # First message
            await pilot.press("f", "i", "r", "s", "t")
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.submitted_values) == 1
            assert app.submitted_values[0] == "first"

            # Second message
            await pilot.press("s", "e", "c", "o", "n", "d")
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.submitted_values) == 2
            assert app.submitted_values[1] == "second"


# ---------------------------------------------------------------------------
# Phase 4: AgentPicker tests
# ---------------------------------------------------------------------------


class TestAgentPickerClass:
    """Test AgentPicker class structure (no Textual app required)."""

    def test_has_agent_selected_message(self):
        assert hasattr(AgentPicker, "AgentSelected")

    def test_agent_selected_message(self):
        msg = AgentPicker.AgentSelected(agent_name="build")
        assert msg.agent_name == "build"

    def test_default_css_exists(self):
        assert AgentPicker.DEFAULT_CSS is not None
        assert isinstance(AgentPicker.DEFAULT_CSS, str)

    def test_is_vertical_subclass(self):
        from textual.containers import Vertical

        assert issubclass(AgentPicker, Vertical)


class TestTrajectoryPickerClass:
    """Test TrajectoryPicker class structure (no Textual app required)."""

    def test_has_action_selected_message(self):
        assert hasattr(TrajectoryPicker, "ActionSelected")

    def test_action_selected_message(self):
        msg = TrajectoryPicker.ActionSelected("set_score", trace_id="abc", score=0.8)
        assert msg.action == "set_score"
        assert msg.trace_id == "abc"
        assert msg.score == 0.8

    def test_default_css_exists(self):
        assert TrajectoryPicker.DEFAULT_CSS is not None
        assert isinstance(TrajectoryPicker.DEFAULT_CSS, str)

    def test_entry_dataclass(self):
        entry = TrajectoryPickerEntry(
            trace_id="abc",
            status="ok",
            agent_name="demo",
            steps=2,
            score=0.7,
            is_golden=True,
        )
        assert entry.trace_id == "abc"
        assert entry.status == "ok"
        assert entry.agent_name == "demo"
        assert entry.steps == 2
        assert entry.score == 0.7
        assert entry.is_golden is True


# ---------------------------------------------------------------------------
# Phase 4: Sidebar tests
# ---------------------------------------------------------------------------


class TestSidebarClass:
    """Test Sidebar class structure (no Textual app required)."""

    def test_has_session_selected_message(self):
        assert hasattr(Sidebar, "SessionSelected")

    def test_session_selected_message(self):
        msg = Sidebar.SessionSelected(session_id="s123")
        assert msg.session_id == "s123"

    def test_default_css_exists(self):
        assert Sidebar.DEFAULT_CSS is not None
        assert isinstance(Sidebar.DEFAULT_CSS, str)

    def test_is_vertical_subclass(self):
        from textual.containers import Vertical

        assert issubclass(Sidebar, Vertical)


# ---------------------------------------------------------------------------
# Phase 4: Slash commands for agent/export
# ---------------------------------------------------------------------------


class TestPhase4SlashCommands:
    def test_agent_command_present(self):
        cmd_names = {cmd.split()[0] for cmd, _ in SLASH_COMMANDS}
        assert "/agent" in cmd_names

    def test_export_command_present(self):
        cmd_names = {cmd.split()[0] for cmd, _ in SLASH_COMMANDS}
        assert "/export" in cmd_names

    def test_agent_with_name_command_present(self):
        """The agent command should have a variant with <name> arg."""
        assert any(cmd.startswith("/agent <") for cmd, _ in SLASH_COMMANDS)


# ---------------------------------------------------------------------------
# Phase 4: Palette items for agent/export/sidebar
# ---------------------------------------------------------------------------


class TestPhase4PaletteItems:
    def test_agent_list_action_present(self):
        actions = {item.action for item in DEFAULT_PALETTE_ITEMS}
        assert "agent_list" in actions

    def test_agent_cycle_action_present(self):
        actions = {item.action for item in DEFAULT_PALETTE_ITEMS}
        assert "agent_cycle" in actions

    def test_session_export_action_present(self):
        actions = {item.action for item in DEFAULT_PALETTE_ITEMS}
        assert "session_export" in actions

    def test_sidebar_toggle_action_present(self):
        actions = {item.action for item in DEFAULT_PALETTE_ITEMS}
        assert "sidebar_toggle" in actions

    def test_trajectory_picker_action_present(self):
        actions = {item.action for item in DEFAULT_PALETTE_ITEMS}
        assert "trajectory_picker" in actions


# ---------------------------------------------------------------------------
# Phase 4: PromptInput agent mention picker CSS
# ---------------------------------------------------------------------------


class TestPromptInputAgentMentionPicker:
    """Test PromptInput class structure includes agent-mention-picker CSS."""

    def test_agent_mention_picker_css_present(self):
        assert "#agent-mention-picker" in PromptInput.DEFAULT_CSS

    def test_agent_mention_picker_visible_css_present(self):
        assert "#agent-mention-picker.visible" in PromptInput.DEFAULT_CSS


# ---------------------------------------------------------------------------
# Phase 5: Auto-scroll and half-page scroll tests
# ---------------------------------------------------------------------------


class TestMessageDisplayAutoScroll:
    """Test auto-scroll constants and method existence on MessageDisplay."""

    def test_auto_scroll_threshold_is_positive(self):
        assert MessageDisplay._AUTO_SCROLL_THRESHOLD > 0

    def test_auto_scroll_threshold_value(self):
        assert MessageDisplay._AUTO_SCROLL_THRESHOLD == 5

    def test_has_is_near_bottom_method(self):
        assert hasattr(MessageDisplay, "_is_near_bottom")
        assert callable(MessageDisplay._is_near_bottom)

    def test_has_auto_scroll_method(self):
        assert hasattr(MessageDisplay, "_auto_scroll")
        assert callable(MessageDisplay._auto_scroll)


# ---------------------------------------------------------------------------
# Phase 5: Tool output CSS tests
# ---------------------------------------------------------------------------


class TestToolOutputCSS:
    """Test that MessageDisplay DEFAULT_CSS includes tool-output-md styles."""

    def test_tool_output_md_in_css(self):
        assert ".tool-output-md" in MessageDisplay.DEFAULT_CSS

    def test_tool_output_md_hidden_in_css(self):
        assert ".tool-output-md.hidden" in MessageDisplay.DEFAULT_CSS


# ---------------------------------------------------------------------------
# Phase 5: Compact command in slash commands list
# ---------------------------------------------------------------------------


class TestPhase5SlashCommands:
    def test_compact_command_present(self):
        cmd_names = {cmd.split()[0] for cmd, _ in SLASH_COMMANDS}
        assert "/compact" in cmd_names

    def test_compact_description_nonempty(self):
        compact_items = [(c, d) for c, d in SLASH_COMMANDS if c.startswith("/compact")]
        assert len(compact_items) >= 1
        for _cmd, desc in compact_items:
            assert desc.strip()


# ---------------------------------------------------------------------------
# Phase 5: Palette item for compact
# ---------------------------------------------------------------------------


class TestPhase5PaletteItems:
    def test_compact_action_present(self):
        actions = {item.action for item in DEFAULT_PALETTE_ITEMS}
        assert "session_compact" in actions
