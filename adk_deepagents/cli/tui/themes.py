"""Theme system for the adk-deepagents TUI.

Themes define semantic color tokens that map to Textual CSS variables.
Built-in themes are provided; users can set the active theme via
``config.toml [tui] theme = "<name>"`` or the ``/theme`` slash command.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """A set of semantic color tokens for the TUI.

    All colors are CSS-compatible strings (hex ``#rrggbb``, named colors,
    or ``""``) used to populate Textual design tokens at runtime.
    """

    name: str
    label: str
    description: str = ""

    # -- Core surfaces --
    background: str = "#1e1e2e"
    surface: str = "#181825"
    panel: str = "#313244"

    # -- Text --
    text: str = "#cdd6f4"
    text_muted: str = "#a6adc8"

    # -- Accents --
    primary: str = "#89b4fa"
    secondary: str = "#f5c2e7"
    accent: str = "#89b4fa"

    # -- Semantic --
    success: str = "#a6e3a1"
    warning: str = "#f9e2af"
    error: str = "#f38ba8"
    info: str = "#89dceb"

    # -- Borders --
    border: str = "#45475a"
    border_active: str = "#89b4fa"

    # -- Diff --
    diff_added: str = "#a6e3a1"
    diff_removed: str = "#f38ba8"
    diff_context: str = "#a6adc8"
    diff_hunk_header: str = "#89b4fa"
    diff_added_bg: str = "#1a3a2a"
    diff_removed_bg: str = "#3a1a2a"

    # -- Thought / reasoning blocks --
    thought: str = "#a6adc8"


# ---------------------------------------------------------------------------
# Built-in themes
# ---------------------------------------------------------------------------

BUILTIN_THEMES: dict[str, Theme] = {}


def _register(theme: Theme) -> Theme:
    """Register a theme in the built-in registry."""
    BUILTIN_THEMES[theme.name] = theme
    return theme


DEFAULT_THEME_NAME = "catppuccin"

# --- Catppuccin Mocha (default) ---
_register(
    Theme(
        name="catppuccin",
        label="Catppuccin Mocha",
        description="Soothing pastel theme for the high-spirited",
        background="#1e1e2e",
        surface="#181825",
        panel="#313244",
        text="#cdd6f4",
        text_muted="#a6adc8",
        primary="#89b4fa",
        secondary="#f5c2e7",
        accent="#89b4fa",
        success="#a6e3a1",
        warning="#f9e2af",
        error="#f38ba8",
        info="#89dceb",
        border="#45475a",
        border_active="#89b4fa",
        diff_added="#a6e3a1",
        diff_removed="#f38ba8",
        diff_context="#a6adc8",
        diff_hunk_header="#89b4fa",
        diff_added_bg="#1a3a2a",
        diff_removed_bg="#3a1a2a",
        thought="#a6adc8",
    )
)

# --- Tokyo Night ---
_register(
    Theme(
        name="tokyonight",
        label="Tokyo Night",
        description="A clean dark theme inspired by Tokyo city lights",
        background="#1a1b26",
        surface="#16161e",
        panel="#292e42",
        text="#c0caf5",
        text_muted="#565f89",
        primary="#7aa2f7",
        secondary="#bb9af7",
        accent="#7aa2f7",
        success="#9ece6a",
        warning="#e0af68",
        error="#f7768e",
        info="#7dcfff",
        border="#3b4261",
        border_active="#7aa2f7",
        diff_added="#9ece6a",
        diff_removed="#f7768e",
        diff_context="#565f89",
        diff_hunk_header="#7aa2f7",
        diff_added_bg="#1a2f1a",
        diff_removed_bg="#2f1a1a",
        thought="#565f89",
    )
)

# --- Gruvbox Dark ---
_register(
    Theme(
        name="gruvbox",
        label="Gruvbox Dark",
        description="Retro groove color scheme",
        background="#282828",
        surface="#1d2021",
        panel="#3c3836",
        text="#ebdbb2",
        text_muted="#a89984",
        primary="#83a598",
        secondary="#d3869b",
        accent="#83a598",
        success="#b8bb26",
        warning="#fabd2f",
        error="#fb4934",
        info="#8ec07c",
        border="#504945",
        border_active="#83a598",
        diff_added="#b8bb26",
        diff_removed="#fb4934",
        diff_context="#a89984",
        diff_hunk_header="#83a598",
        diff_added_bg="#2a3a1a",
        diff_removed_bg="#3a2020",
        thought="#a89984",
    )
)

# --- Nord ---
_register(
    Theme(
        name="nord",
        label="Nord",
        description="Arctic, north-bluish color palette",
        background="#2e3440",
        surface="#272c36",
        panel="#3b4252",
        text="#eceff4",
        text_muted="#7b88a1",
        primary="#88c0d0",
        secondary="#b48ead",
        accent="#88c0d0",
        success="#a3be8c",
        warning="#ebcb8b",
        error="#bf616a",
        info="#81a1c1",
        border="#434c5e",
        border_active="#88c0d0",
        diff_added="#a3be8c",
        diff_removed="#bf616a",
        diff_context="#7b88a1",
        diff_hunk_header="#88c0d0",
        diff_added_bg="#2a3a2a",
        diff_removed_bg="#3a2a2a",
        thought="#7b88a1",
    )
)

# --- One Dark ---
_register(
    Theme(
        name="onedark",
        label="One Dark",
        description="Atom One Dark theme adaptation",
        background="#282c34",
        surface="#21252b",
        panel="#2c313a",
        text="#abb2bf",
        text_muted="#5c6370",
        primary="#61afef",
        secondary="#c678dd",
        accent="#61afef",
        success="#98c379",
        warning="#e5c07b",
        error="#e06c75",
        info="#56b6c2",
        border="#3e4451",
        border_active="#61afef",
        diff_added="#98c379",
        diff_removed="#e06c75",
        diff_context="#5c6370",
        diff_hunk_header="#61afef",
        diff_added_bg="#1a2f1a",
        diff_removed_bg="#2f1a1a",
        thought="#5c6370",
    )
)

# --- Matrix ---
_register(
    Theme(
        name="matrix",
        label="Matrix",
        description="Green-on-black hacker aesthetic",
        background="#0a0a0a",
        surface="#050505",
        panel="#1a1a1a",
        text="#00ff41",
        text_muted="#007a1f",
        primary="#00ff41",
        secondary="#00cc33",
        accent="#00ff41",
        success="#00ff41",
        warning="#ccff00",
        error="#ff0033",
        info="#00ccff",
        border="#003300",
        border_active="#00ff41",
        diff_added="#00ff41",
        diff_removed="#ff0033",
        diff_context="#007a1f",
        diff_hunk_header="#00ff41",
        diff_added_bg="#001a00",
        diff_removed_bg="#1a0000",
        thought="#007a1f",
    )
)

# --- Ayu Dark ---
_register(
    Theme(
        name="ayu",
        label="Ayu Dark",
        description="Simple, bright colors on a dark background",
        background="#0b0e14",
        surface="#0d1017",
        panel="#1c1f27",
        text="#bfbdb6",
        text_muted="#636a76",
        primary="#39bae6",
        secondary="#d2a6ff",
        accent="#39bae6",
        success="#7fd962",
        warning="#ffb454",
        error="#d95757",
        info="#59c2ff",
        border="#2a2d37",
        border_active="#39bae6",
        diff_added="#7fd962",
        diff_removed="#d95757",
        diff_context="#636a76",
        diff_hunk_header="#39bae6",
        diff_added_bg="#152a15",
        diff_removed_bg="#2a1515",
        thought="#636a76",
    )
)

# --- Kanagawa ---
_register(
    Theme(
        name="kanagawa",
        label="Kanagawa",
        description="Dark theme inspired by Katsushika Hokusai",
        background="#1f1f28",
        surface="#16161d",
        panel="#2a2a37",
        text="#dcd7ba",
        text_muted="#727169",
        primary="#7e9cd8",
        secondary="#957fb8",
        accent="#7e9cd8",
        success="#98bb6c",
        warning="#e6c384",
        error="#e82424",
        info="#7fb4ca",
        border="#363646",
        border_active="#7e9cd8",
        diff_added="#98bb6c",
        diff_removed="#e82424",
        diff_context="#727169",
        diff_hunk_header="#7e9cd8",
        diff_added_bg="#1a2f1a",
        diff_removed_bg="#2f1010",
        thought="#727169",
    )
)

# --- Everforest Dark ---
_register(
    Theme(
        name="everforest",
        label="Everforest",
        description="Comfortable and pleasant green forest theme",
        background="#2d353b",
        surface="#272e33",
        panel="#343f44",
        text="#d3c6aa",
        text_muted="#859289",
        primary="#7fbbb3",
        secondary="#d699b6",
        accent="#7fbbb3",
        success="#a7c080",
        warning="#dbbc7f",
        error="#e67e80",
        info="#83c092",
        border="#475258",
        border_active="#7fbbb3",
        diff_added="#a7c080",
        diff_removed="#e67e80",
        diff_context="#859289",
        diff_hunk_header="#7fbbb3",
        diff_added_bg="#2a3a25",
        diff_removed_bg="#3a2525",
        thought="#859289",
    )
)


def get_theme(name: str) -> Theme | None:
    """Look up a built-in theme by name (case-insensitive)."""
    return BUILTIN_THEMES.get(name.lower().strip())


def list_theme_names() -> list[str]:
    """Return sorted list of available theme names."""
    return sorted(BUILTIN_THEMES.keys())


def theme_to_css(theme: Theme) -> str:
    """Generate Textual CSS variable overrides for *theme*.

    These CSS custom properties are applied at the ``Screen`` level to
    override Textual's built-in design tokens.
    """
    return f"""\
Screen {{
    background: {theme.background};
}}

/* -- Textual design-token overrides via nested CSS -- */
* {{
    /* Surface / panel */
    scrollbar-background: {theme.surface};
    scrollbar-color: {theme.border};
    scrollbar-color-hover: {theme.primary};
    scrollbar-color-active: {theme.primary};
}}
"""


def theme_to_widget_colors(theme: Theme) -> dict[str, str]:
    """Return a dict of color token names → values for widget-level styling.

    Widgets read these via ``self.app.theme_colors[token_name]`` to
    dynamically style themselves.  This avoids needing to re-parse CSS.
    """
    return {
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
        "diff_context": theme.diff_context,
        "diff_hunk_header": theme.diff_hunk_header,
        "diff_added_bg": theme.diff_added_bg,
        "diff_removed_bg": theme.diff_removed_bg,
        "thought": theme.thought,
    }
