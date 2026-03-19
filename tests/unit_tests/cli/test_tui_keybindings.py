"""Unit tests for TUI keybinding configuration (cli/tui/keybindings.py)."""

from __future__ import annotations

from adk_deepagents.cli.tui.keybindings import (
    DEFAULT_KEYBINDS,
    DEFAULT_LEADER,
    KeybindConfig,
    load_keybind_config,
)


class TestDefaults:
    """Test default keybindings and leader key."""

    def test_default_leader(self):
        assert DEFAULT_LEADER == "ctrl+x"

    def test_default_keybinds_is_dict(self):
        assert isinstance(DEFAULT_KEYBINDS, dict)
        assert len(DEFAULT_KEYBINDS) > 0

    def test_known_actions_present(self):
        assert "app_quit" in DEFAULT_KEYBINDS
        assert "command_palette" in DEFAULT_KEYBINDS
        assert "session_interrupt" in DEFAULT_KEYBINDS
        assert "session_new" in DEFAULT_KEYBINDS
        assert "help" in DEFAULT_KEYBINDS

    def test_leader_placeholder_in_defaults(self):
        """Some defaults should use <leader> placeholder."""
        assert "<leader>" in DEFAULT_KEYBINDS["session_new"]
        assert "<leader>" in DEFAULT_KEYBINDS["help"]


class TestKeybindConfig:
    """Test the KeybindConfig dataclass."""

    def test_keys_for_missing_action(self):
        config = KeybindConfig(leader="ctrl+x", bindings={})
        assert config.keys_for("nonexistent") == []

    def test_keys_for_existing_action(self):
        config = KeybindConfig(
            leader="ctrl+x",
            bindings={"help": ["ctrl+x h"]},
        )
        assert config.keys_for("help") == ["ctrl+x h"]

    def test_first_key_for(self):
        config = KeybindConfig(
            leader="ctrl+x",
            bindings={"help": ["ctrl+x h", "f1"]},
        )
        assert config.first_key_for("help") == "ctrl+x h"

    def test_first_key_for_missing(self):
        config = KeybindConfig(leader="ctrl+x", bindings={})
        assert config.first_key_for("missing") is None

    def test_display_for(self):
        config = KeybindConfig(
            leader="ctrl+x",
            bindings={"help": ["ctrl+x h"]},
        )
        assert config.display_for("help") == "ctrl+x h"

    def test_display_for_missing(self):
        config = KeybindConfig(leader="ctrl+x", bindings={})
        assert config.display_for("missing") == ""


class TestLoadKeybindConfig:
    """Test load_keybind_config with various inputs."""

    def test_defaults_loaded_when_none(self):
        config = load_keybind_config(None)
        assert config.leader == DEFAULT_LEADER
        assert len(config.bindings) > 0

    def test_defaults_loaded_when_empty(self):
        config = load_keybind_config({})
        assert config.leader == DEFAULT_LEADER
        assert len(config.bindings) > 0

    def test_leader_expanded_in_combos(self):
        config = load_keybind_config(None)
        # "session_new" is bound to "<leader> n" by default → "ctrl+x n"
        combos = config.keys_for("session_new")
        assert len(combos) > 0
        assert combos[0] == "ctrl+x n"

    def test_custom_leader(self):
        config = load_keybind_config({"leader": "ctrl+a"})
        assert config.leader == "ctrl+a"
        combos = config.keys_for("session_new")
        assert len(combos) > 0
        assert combos[0] == "ctrl+a n"

    def test_override_binding(self):
        config = load_keybind_config({"app_quit": "ctrl+q"})
        combos = config.keys_for("app_quit")
        assert combos == ["ctrl+q"]

    def test_disable_binding_with_none(self):
        config = load_keybind_config({"app_quit": "none"})
        combos = config.keys_for("app_quit")
        assert combos == []

    def test_multiple_combos(self):
        config = load_keybind_config({"help": "f1, <leader> h"})
        combos = config.keys_for("help")
        assert len(combos) == 2
        assert "f1" in combos
        assert "ctrl+x h" in combos

    def test_command_palette_binding(self):
        config = load_keybind_config(None)
        combos = config.keys_for("command_palette")
        assert "ctrl+p" in combos

    def test_simple_bindings_not_expanded(self):
        """Bindings without <leader> should pass through unchanged."""
        config = load_keybind_config(None)
        combos = config.keys_for("app_quit")
        assert combos == ["ctrl+c"]
