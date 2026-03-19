"""CLI config/bootstrap helpers for adk-deepagents."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

CLI_HOME_ENV_VAR = "ADK_DEEPAGENTS_HOME"
CLI_HOME_DIRNAME = ".adk-deepagents"
CONFIG_FILENAME = "config.toml"
PROFILES_DIRNAME = "profiles"
PROFILE_MEMORY_FILENAME = "AGENTS.md"
SESSIONS_DB_FILENAME = "sessions.db"

DEFAULT_AGENT_NAME = "agent"

_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class CliPaths:
    """Filesystem locations used by the CLI."""

    root_dir: Path
    config_path: Path
    profiles_dir: Path
    sessions_db_path: Path


@dataclass
class CliDefaults:
    """Persisted CLI defaults loaded from config.toml."""

    default_agent: str = DEFAULT_AGENT_NAME
    default_model: str | None = None
    dynamic_task_max_parallel: int | None = None
    dynamic_task_concurrency_policy: Literal["error", "wait"] | None = None
    dynamic_task_queue_timeout_seconds: float | None = None
    tui_keybinds: dict[str, Any] | None = field(default=None)
    tui_theme: str | None = None


def resolve_cli_paths(home_dir: Path | None = None) -> CliPaths:
    """Resolve root/config/profile paths for the CLI home directory."""
    if home_dir is None:
        override = os.environ.get(CLI_HOME_ENV_VAR)
        root_dir = Path(override).expanduser() if override else Path.home() / CLI_HOME_DIRNAME
    else:
        root_dir = home_dir.expanduser()

    return CliPaths(
        root_dir=root_dir,
        config_path=root_dir / CONFIG_FILENAME,
        profiles_dir=root_dir / PROFILES_DIRNAME,
        sessions_db_path=root_dir / SESSIONS_DB_FILENAME,
    )


def bootstrap_cli_home(paths: CliPaths) -> None:
    """Create CLI directories/files for first-run bootstrap."""
    paths.root_dir.mkdir(parents=True, exist_ok=True)
    paths.profiles_dir.mkdir(parents=True, exist_ok=True)

    if not paths.config_path.exists():
        save_cli_defaults(paths, CliDefaults())

    ensure_profile_memory(paths, DEFAULT_AGENT_NAME)


def load_cli_defaults(paths: CliPaths) -> CliDefaults:
    """Load persisted defaults from config.toml."""
    if not paths.config_path.exists():
        return CliDefaults()

    config_data = tomllib.loads(paths.config_path.read_text(encoding="utf-8"))

    default_agent_raw = config_data.get("default_agent", DEFAULT_AGENT_NAME)
    if not isinstance(default_agent_raw, str):
        raise ValueError("config.toml default_agent must be a string.")

    default_agent = _normalize_profile_name(default_agent_raw)

    default_model_raw = config_data.get("default_model")
    default_model: str | None
    if default_model_raw is None:
        default_model = None
    elif isinstance(default_model_raw, str):
        stripped = default_model_raw.strip()
        default_model = stripped or None
    else:
        raise ValueError("config.toml default_model must be a string when set.")

    dynamic_task_max_parallel: int | None = None
    dynamic_task_concurrency_policy: Literal["error", "wait"] | None = None
    dynamic_task_queue_timeout_seconds: float | None = None

    dynamic_task_raw = config_data.get("dynamic_task")
    if dynamic_task_raw is not None:
        if not isinstance(dynamic_task_raw, dict):
            raise ValueError("config.toml dynamic_task must be a table.")

        max_parallel_raw = dynamic_task_raw.get("max_parallel")
        if max_parallel_raw is not None:
            if isinstance(max_parallel_raw, bool) or not isinstance(max_parallel_raw, int):
                raise ValueError("config.toml dynamic_task.max_parallel must be an integer.")
            if max_parallel_raw < 1:
                raise ValueError("config.toml dynamic_task.max_parallel must be >= 1.")
            dynamic_task_max_parallel = max_parallel_raw

        concurrency_policy_raw = dynamic_task_raw.get("concurrency_policy")
        if concurrency_policy_raw is not None:
            if not isinstance(concurrency_policy_raw, str):
                raise ValueError(
                    "config.toml dynamic_task.concurrency_policy must be 'error' or 'wait'."
                )
            normalized_policy = concurrency_policy_raw.strip().lower()
            if normalized_policy == "error":
                dynamic_task_concurrency_policy = "error"
            elif normalized_policy == "wait":
                dynamic_task_concurrency_policy = "wait"
            else:
                raise ValueError(
                    "config.toml dynamic_task.concurrency_policy must be 'error' or 'wait'."
                )

        queue_timeout_raw = dynamic_task_raw.get("queue_timeout_seconds")
        if queue_timeout_raw is not None:
            if isinstance(queue_timeout_raw, bool) or not isinstance(
                queue_timeout_raw, (int, float)
            ):
                raise ValueError("config.toml dynamic_task.queue_timeout_seconds must be a number.")
            queue_timeout = float(queue_timeout_raw)
            if queue_timeout < 0:
                raise ValueError("config.toml dynamic_task.queue_timeout_seconds must be >= 0.")
            dynamic_task_queue_timeout_seconds = queue_timeout

    tui_keybinds: dict[str, Any] | None = None
    tui_theme: str | None = None
    tui_raw = config_data.get("tui")
    if tui_raw is not None:
        if not isinstance(tui_raw, dict):
            raise ValueError("config.toml [tui] must be a table.")
        keybinds_raw = tui_raw.get("keybinds")
        if keybinds_raw is not None:
            if not isinstance(keybinds_raw, dict):
                raise ValueError("config.toml [tui.keybinds] must be a table.")
            tui_keybinds = keybinds_raw
        theme_raw = tui_raw.get("theme")
        if theme_raw is not None:
            if not isinstance(theme_raw, str):
                raise ValueError("config.toml [tui] theme must be a string.")
            tui_theme = theme_raw.strip() or None

    return CliDefaults(
        default_agent=default_agent,
        default_model=default_model,
        dynamic_task_max_parallel=dynamic_task_max_parallel,
        dynamic_task_concurrency_policy=dynamic_task_concurrency_policy,
        dynamic_task_queue_timeout_seconds=dynamic_task_queue_timeout_seconds,
        tui_keybinds=tui_keybinds,
        tui_theme=tui_theme,
    )


def save_cli_defaults(paths: CliPaths, defaults: CliDefaults) -> None:
    """Persist defaults to config.toml."""
    default_agent = _normalize_profile_name(defaults.default_agent)

    lines = [f'default_agent = "{_toml_escape(default_agent)}"']
    if defaults.default_model:
        lines.append(f'default_model = "{_toml_escape(defaults.default_model)}"')

    has_dynamic_task_settings = any(
        value is not None
        for value in (
            defaults.dynamic_task_max_parallel,
            defaults.dynamic_task_concurrency_policy,
            defaults.dynamic_task_queue_timeout_seconds,
        )
    )
    if has_dynamic_task_settings:
        lines.append("")
        lines.append("[dynamic_task]")
        if defaults.dynamic_task_max_parallel is not None:
            lines.append(f"max_parallel = {defaults.dynamic_task_max_parallel}")
        if defaults.dynamic_task_concurrency_policy is not None:
            lines.append(
                f'concurrency_policy = "{_toml_escape(defaults.dynamic_task_concurrency_policy)}"'
            )
        if defaults.dynamic_task_queue_timeout_seconds is not None:
            lines.append(f"queue_timeout_seconds = {defaults.dynamic_task_queue_timeout_seconds}")

    if defaults.tui_keybinds or defaults.tui_theme:
        lines.append("")
        if defaults.tui_theme:
            lines.append("[tui]")
            lines.append(f'theme = "{_toml_escape(defaults.tui_theme)}"')
            lines.append("")
        if defaults.tui_keybinds:
            lines.append("[tui.keybinds]")
            for action, combo in sorted(defaults.tui_keybinds.items()):
                lines.append(f'{action} = "{_toml_escape(str(combo))}"')

    _atomic_write_text(paths.config_path, "\n".join(lines) + "\n")


def list_profiles(paths: CliPaths) -> list[str]:
    """Return available profile directory names sorted alphabetically."""
    if not paths.profiles_dir.exists():
        return []

    names = []
    for child in paths.profiles_dir.iterdir():
        if not child.is_dir():
            continue
        if not _PROFILE_NAME_RE.fullmatch(child.name):
            continue
        names.append(child.name)

    return sorted(names)


def ensure_profile_memory(paths: CliPaths, profile_name: str) -> Path:
    """Ensure a profile directory and memory template file exist."""
    profile_dir = _resolve_profile_dir(paths, profile_name)
    profile_dir.mkdir(parents=True, exist_ok=True)

    memory_path = profile_dir / PROFILE_MEMORY_FILENAME
    if not memory_path.exists():
        _atomic_write_text(memory_path, default_profile_template(profile_dir.name))

    return memory_path


def reset_profile_memory(paths: CliPaths, profile_name: str) -> Path:
    """Reset a profile's AGENTS.md to the default template."""
    memory_path = ensure_profile_memory(paths, profile_name)

    if memory_path.is_symlink():
        raise ValueError("Refusing to reset a symlinked profile memory file.")

    _atomic_write_text(memory_path, default_profile_template(memory_path.parent.name))
    return memory_path


def default_profile_template(profile_name: str) -> str:
    """Default AGENTS.md contents for a CLI profile."""
    return (
        f"# {profile_name} profile memory\n"
        "\n"
        "Use this file for persistent profile-specific instructions and context.\n"
    )


def _resolve_profile_dir(paths: CliPaths, profile_name: str) -> Path:
    normalized = _normalize_profile_name(profile_name)

    profiles_root = paths.profiles_dir.resolve()
    candidate = (paths.profiles_dir / normalized).resolve()

    if candidate != profiles_root and profiles_root not in candidate.parents:
        raise ValueError("Profile path resolves outside the profiles directory.")

    return candidate


def _normalize_profile_name(profile_name: str) -> str:
    normalized = profile_name.strip()
    if not normalized:
        raise ValueError("Profile name cannot be empty.")
    if not _PROFILE_NAME_RE.fullmatch(normalized):
        raise ValueError(
            "Profile name may only contain letters, numbers, '.', '-', and '_' and must "
            "start with a letter or number."
        )
    return normalized


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
