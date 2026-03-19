"""Configurable keybinding system with leader-key support.

Keybindings are stored in ``config.toml`` under a ``[tui.keybinds]`` table and
can be overridden per-project via environment variables.  The leader key
defaults to ``ctrl+x`` and is expanded in binding strings as ``<leader>``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

# Default leader key (same convention as OpenCode).
DEFAULT_LEADER = "ctrl+x"

# Default keybindings map.  Keys are action names; values are key-combo
# strings.  Multiple combos for the same action are separated by commas.
# ``<leader>`` is expanded at load time to the configured leader key.
DEFAULT_KEYBINDS: dict[str, str] = {
    # Application lifecycle
    "app_quit": "ctrl+c",
    # Session / thread management
    "session_new": "<leader> n",
    "session_list": "<leader> l",
    # Model management
    "model_list": "<leader> m",
    # Conversation management
    "session_compact": "<leader> c",
    "session_interrupt": "escape",
    # Navigation helpers
    "command_palette": "ctrl+p",
    "help": "<leader> h",
    # Tool details toggle
    "tool_details_toggle": "<leader> d",
    # Editor
    "editor_open": "<leader> e",
    # Scroll (vim-style)
    "messages_half_page_up": "ctrl+u",
    "messages_half_page_down": "ctrl+d",
    "messages_page_up": "pageup",
    "messages_page_down": "pagedown",
    "messages_first": "home",
    "messages_last": "end",
    # Input
    "input_submit": "enter",
    "input_newline": "shift+enter",
}


@dataclass
class KeybindConfig:
    """Resolved keybinding configuration."""

    leader: str = DEFAULT_LEADER
    bindings: dict[str, list[str]] = field(default_factory=dict)

    def keys_for(self, action: str) -> list[str]:
        """Return the list of key combos for *action*, or an empty list."""
        return self.bindings.get(action, [])

    def first_key_for(self, action: str) -> str | None:
        """Return the first key combo for *action*, or ``None``."""
        combos = self.bindings.get(action)
        return combos[0] if combos else None

    def display_for(self, action: str) -> str:
        """Human-friendly string for display (e.g. in footer)."""
        combo = self.first_key_for(action)
        if combo is None:
            return ""
        return combo


def _expand_leader(combo: str, leader: str) -> str:
    """Replace ``<leader>`` with the actual leader key."""
    return combo.replace("<leader>", leader)


def _parse_combos(raw: str, leader: str) -> list[str]:
    """Split a comma-separated combo string and expand leader."""
    combos: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if part and part != "none":
            combos.append(_expand_leader(part, leader))
    return combos


def load_keybind_config(raw_config: dict[str, Any] | None = None) -> KeybindConfig:
    """Build a :class:`KeybindConfig` from raw TOML/dict data.

    Parameters
    ----------
    raw_config:
        Optional dict (from ``config.toml [tui.keybinds]``).  Keys are
        action names, values are comma-separated key-combo strings.
        A special ``"leader"`` key overrides the leader.
    """
    merged = copy.deepcopy(DEFAULT_KEYBINDS)

    leader = DEFAULT_LEADER
    if raw_config:
        leader = raw_config.get("leader", DEFAULT_LEADER)
        for key, value in raw_config.items():
            if key == "leader":
                continue
            if isinstance(value, str):
                merged[key] = value

    bindings: dict[str, list[str]] = {}
    for action, raw_combo in merged.items():
        combos = _parse_combos(raw_combo, leader)
        if combos:
            bindings[action] = combos

    return KeybindConfig(leader=leader, bindings=bindings)
