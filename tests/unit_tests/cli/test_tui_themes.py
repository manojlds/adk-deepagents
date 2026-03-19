"""Unit tests for TUI theme system (cli/tui/themes.py)."""

from __future__ import annotations

from adk_deepagents.cli.tui.themes import (
    BUILTIN_THEMES,
    DEFAULT_THEME_NAME,
    Theme,
    get_theme,
    list_theme_names,
    theme_to_css,
    theme_to_widget_colors,
)


class TestThemeDataclass:
    """Test the Theme frozen dataclass."""

    def test_basic_construction(self):
        theme = Theme(name="test", label="Test Theme")
        assert theme.name == "test"
        assert theme.label == "Test Theme"
        assert theme.background  # has default

    def test_frozen(self):
        theme = Theme(name="test", label="Test")
        try:
            theme.name = "changed"  # type: ignore[misc]
            raise AssertionError("Should have raised AttributeError")
        except AttributeError:
            pass

    def test_all_color_fields_are_strings(self):
        theme = Theme(name="test", label="Test")
        for attr_name in (
            "background",
            "surface",
            "panel",
            "text",
            "text_muted",
            "primary",
            "secondary",
            "accent",
            "success",
            "warning",
            "error",
            "info",
            "border",
            "border_active",
            "diff_added",
            "diff_removed",
            "diff_context",
            "diff_hunk_header",
            "diff_added_bg",
            "diff_removed_bg",
            "thought",
        ):
            value = getattr(theme, attr_name)
            assert isinstance(value, str), f"{attr_name} should be str, got {type(value)}"

    def test_custom_colors(self):
        theme = Theme(name="custom", label="Custom", background="#000000", text="#ffffff")
        assert theme.background == "#000000"
        assert theme.text == "#ffffff"


class TestBuiltinThemes:
    """Test the built-in theme registry."""

    def test_registry_is_nonempty(self):
        assert len(BUILTIN_THEMES) > 0

    def test_default_theme_exists(self):
        assert DEFAULT_THEME_NAME in BUILTIN_THEMES

    def test_all_themes_have_name_and_label(self):
        for name, theme in BUILTIN_THEMES.items():
            assert theme.name == name
            assert theme.label.strip()

    def test_expected_themes_present(self):
        names = set(BUILTIN_THEMES.keys())
        assert "catppuccin" in names
        assert "tokyonight" in names
        assert "gruvbox" in names
        assert "nord" in names
        assert "onedark" in names
        assert "matrix" in names
        assert "ayu" in names
        assert "kanagawa" in names
        assert "everforest" in names

    def test_all_themes_have_hex_backgrounds(self):
        for name, theme in BUILTIN_THEMES.items():
            assert theme.background.startswith("#"), f"{name} background missing #"

    def test_theme_count(self):
        assert len(BUILTIN_THEMES) >= 9


class TestGetTheme:
    """Test the get_theme lookup function."""

    def test_existing_theme(self):
        theme = get_theme("catppuccin")
        assert theme is not None
        assert theme.name == "catppuccin"

    def test_case_insensitive(self):
        theme = get_theme("CATPPUCCIN")
        assert theme is not None
        assert theme.name == "catppuccin"

    def test_whitespace_stripped(self):
        theme = get_theme("  nord  ")
        assert theme is not None
        assert theme.name == "nord"

    def test_missing_theme_returns_none(self):
        assert get_theme("nonexistent") is None

    def test_empty_string(self):
        assert get_theme("") is None


class TestListThemeNames:
    """Test the list_theme_names function."""

    def test_returns_sorted_list(self):
        names = list_theme_names()
        assert isinstance(names, list)
        assert names == sorted(names)

    def test_all_registered_themes_listed(self):
        names = set(list_theme_names())
        for name in BUILTIN_THEMES:
            assert name in names


class TestThemeToCss:
    """Test CSS generation from a theme."""

    def test_returns_string(self):
        theme = get_theme("catppuccin")
        assert theme is not None
        css = theme_to_css(theme)
        assert isinstance(css, str)

    def test_contains_screen_block(self):
        theme = get_theme("catppuccin")
        assert theme is not None
        css = theme_to_css(theme)
        assert "Screen" in css

    def test_contains_background_color(self):
        theme = Theme(name="test", label="Test", background="#123456")
        css = theme_to_css(theme)
        assert "#123456" in css


class TestThemeToWidgetColors:
    """Test widget color dict generation."""

    def test_returns_dict(self):
        theme = get_theme("catppuccin")
        assert theme is not None
        colors = theme_to_widget_colors(theme)
        assert isinstance(colors, dict)

    def test_has_expected_keys(self):
        theme = get_theme("catppuccin")
        assert theme is not None
        colors = theme_to_widget_colors(theme)
        expected_keys = {
            "background",
            "surface",
            "panel",
            "text",
            "text_muted",
            "primary",
            "secondary",
            "accent",
            "success",
            "warning",
            "error",
            "info",
            "border",
            "border_active",
            "diff_added",
            "diff_removed",
            "diff_context",
            "diff_hunk_header",
            "diff_added_bg",
            "diff_removed_bg",
            "thought",
        }
        assert expected_keys.issubset(set(colors.keys()))

    def test_values_match_theme(self):
        theme = Theme(name="t", label="T", background="#aabbcc", text="#112233")
        colors = theme_to_widget_colors(theme)
        assert colors["background"] == "#aabbcc"
        assert colors["text"] == "#112233"
