"""Configuration loading and saving for the CLI."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from adk_deepagents.cli.models import AppConfig, CliConfig, ModelsConfig, WarningsConfig
from adk_deepagents.cli.paths import ensure_cli_home

CONFIG_FILENAME = "config.toml"


def _as_section(raw: object) -> dict[str, Any]:
    """Return a dict section when *raw* is a TOML table."""
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items()}


def _quote(value: str) -> str:
    """Quote a TOML string value."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def get_config_path() -> Path:
    """Return default config path under CLI home."""
    return ensure_cli_home() / CONFIG_FILENAME


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load config from disk or return defaults when missing."""
    path = config_path or get_config_path()
    if not path.exists():
        return AppConfig()

    data = tomllib.loads(path.read_text(encoding="utf-8"))

    models_data = _as_section(data.get("models"))
    warnings_data = _as_section(data.get("warnings"))
    cli_data = _as_section(data.get("cli"))

    model_default = models_data.get("default")
    model_recent = models_data.get("recent")
    warnings_suppress = warnings_data.get("suppress")
    auto_approve = cli_data.get("auto_approve")

    models = ModelsConfig(
        default=model_default
        if isinstance(model_default, str) and model_default
        else ModelsConfig().default,
        recent=model_recent if isinstance(model_recent, str) and model_recent else None,
    )
    warnings = WarningsConfig(
        suppress=[entry for entry in warnings_suppress if isinstance(entry, str)]
        if isinstance(warnings_suppress, list)
        else []
    )
    cli = CliConfig(auto_approve=bool(auto_approve) if isinstance(auto_approve, bool) else False)

    return AppConfig(models=models, warnings=warnings, cli=cli)


def save_config(config: AppConfig, config_path: Path | None = None) -> Path:
    """Persist config to disk and return path."""
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "[models]",
        f"default = {_quote(config.models.default)}",
        f"recent = {_quote(config.models.recent)}" if config.models.recent else 'recent = ""',
        "",
        "[warnings]",
        "suppress = [" + ", ".join(_quote(entry) for entry in config.warnings.suppress) + "]",
        "",
        "[cli]",
        f"auto_approve = {'true' if config.cli.auto_approve else 'false'}",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def ensure_config(config_path: Path | None = None) -> tuple[Path, AppConfig]:
    """Load config and create default file if missing."""
    path = config_path or get_config_path()
    config = load_config(path)
    if not path.exists():
        save_config(config, path)
    return path, config
