"""Unit tests for CLI configuration loading/saving."""

from __future__ import annotations

from pathlib import Path

from adk_deepagents.cli.config import load_config, save_config
from adk_deepagents.cli.models import AppConfig, CliConfig, ModelsConfig, WarningsConfig


def test_load_config_returns_defaults_when_missing(tmp_path: Path) -> None:
    config = load_config(tmp_path / "missing.toml")

    assert config.models.default == "gemini-2.5-flash"
    assert config.models.recent is None
    assert config.warnings.suppress == []
    assert config.cli.auto_approve is False


def test_load_config_parses_known_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[models]
default = "openai/gpt-4o"
recent = "gemini-2.5-pro"

[warnings]
suppress = ["ripgrep"]

[cli]
auto_approve = true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.models.default == "openai/gpt-4o"
    assert config.models.recent == "gemini-2.5-pro"
    assert config.warnings.suppress == ["ripgrep"]
    assert config.cli.auto_approve is True


def test_save_config_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    expected = AppConfig(
        models=ModelsConfig(default="anthropic/claude-sonnet-4", recent="gemini-2.5-flash"),
        warnings=WarningsConfig(suppress=["ripgrep", "sandbox"]),
        cli=CliConfig(auto_approve=True),
    )

    save_config(expected, config_path)
    actual = load_config(config_path)

    assert actual == expected
