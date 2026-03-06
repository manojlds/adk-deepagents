"""Path resolution and profile bootstrap helpers for the CLI."""

from __future__ import annotations

import os
from pathlib import Path

from adk_deepagents.cli.models import DEFAULT_AGENT_NAME, DEFAULT_MEMORY_TEMPLATE

CLI_HOME_ENV = "ADK_DEEPAGENTS_HOME"
CLI_HOME_DIRNAME = ".adk-deepagents"
AGENT_MEMORY_FILENAME = "AGENTS.md"
SESSIONS_DB_FILENAME = "sessions.db"


def get_cli_home() -> Path:
    """Return CLI app-data directory.

    Uses ``ADK_DEEPAGENTS_HOME`` when set, otherwise defaults to
    ``~/.adk-deepagents``.
    """
    env_home = os.environ.get(CLI_HOME_ENV)
    if env_home:
        return Path(env_home).expanduser().resolve()
    return (Path.home() / CLI_HOME_DIRNAME).resolve()


def ensure_cli_home() -> Path:
    """Create and return the CLI app-data directory."""
    home = get_cli_home()
    home.mkdir(parents=True, exist_ok=True)
    return home


def get_agent_dir(agent_name: str) -> Path:
    """Return the profile directory for ``agent_name``."""
    return ensure_cli_home() / agent_name


def get_agent_memory_path(agent_name: str) -> Path:
    """Return the AGENTS.md path for ``agent_name``."""
    return get_agent_dir(agent_name) / AGENT_MEMORY_FILENAME


def get_sessions_db_path() -> Path:
    """Return SQLite sessions database path under CLI home."""
    return ensure_cli_home() / SESSIONS_DB_FILENAME


def ensure_agent_profile(agent_name: str = DEFAULT_AGENT_NAME) -> Path:
    """Create profile directory if missing and return it."""
    profile_dir = get_agent_dir(agent_name)
    profile_dir.mkdir(parents=True, exist_ok=True)
    return profile_dir


def reset_agent_memory(
    agent_name: str,
    *,
    template: str = DEFAULT_MEMORY_TEMPLATE,
) -> Path:
    """Overwrite profile ``AGENTS.md`` with the default template."""
    memory_path = get_agent_memory_path(agent_name)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(template.rstrip() + "\n", encoding="utf-8")
    return memory_path


def bootstrap_agent_profile(agent_name: str = DEFAULT_AGENT_NAME) -> Path:
    """Ensure the profile directory and memory file exist."""
    profile_dir = ensure_agent_profile(agent_name)
    memory_path = profile_dir / AGENT_MEMORY_FILENAME
    if not memory_path.exists():
        reset_agent_memory(agent_name)
    return profile_dir


def list_agent_profiles() -> list[str]:
    """Return sorted profile directory names under CLI home."""
    home = ensure_cli_home()
    profiles = [entry.name for entry in home.iterdir() if entry.is_dir()]
    return sorted(profiles)
