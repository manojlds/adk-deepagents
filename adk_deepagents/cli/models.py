"""Dataclasses and defaults for CLI configuration and profiles."""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_AGENT_NAME = "agent"
DEFAULT_MEMORY_TEMPLATE = """# AGENTS.md

This file stores persistent preferences and conventions for your CLI agent profile.

- Add coding style preferences.
- Add testing expectations.
- Add project-specific reminders you want loaded every session.
"""


@dataclass
class ModelsConfig:
    """Model selection defaults for the CLI."""

    default: str = DEFAULT_MODEL
    recent: str | None = None


@dataclass
class WarningsConfig:
    """Config for optional warning suppression."""

    suppress: list[str] = field(default_factory=list)


@dataclass
class CliConfig:
    """General CLI runtime preferences."""

    auto_approve: bool = False


@dataclass
class AppConfig:
    """Top-level CLI config file model."""

    models: ModelsConfig = field(default_factory=ModelsConfig)
    warnings: WarningsConfig = field(default_factory=WarningsConfig)
    cli: CliConfig = field(default_factory=CliConfig)
