"""TUI (Terminal User Interface) for adk-deepagents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from adk_deepagents.types import DynamicTaskConfig


def run_tui(
    *,
    first_prompt: str | None,
    model: str | None,
    agent_name: str,
    user_id: str,
    session_id: str,
    db_path: Path,
    auto_approve: bool,
    dynamic_task_config: DynamicTaskConfig | None = None,
    memory_sources: Sequence[str] = (),
    memory_source_paths: Mapping[str, Path] | None = None,
    skills_dirs: Sequence[str] = (),
    keybinds_raw: dict[str, Any] | None = None,
) -> int:
    """Launch the Textual TUI and block until exit."""
    from adk_deepagents.cli.tui.app import DeepAgentTui, TuiConfig

    config = TuiConfig(
        first_prompt=first_prompt,
        model=model,
        agent_name=agent_name,
        user_id=user_id,
        session_id=session_id,
        db_path=db_path,
        auto_approve=auto_approve,
        dynamic_task_config=dynamic_task_config,
        memory_sources=list(memory_sources),
        memory_source_paths=dict(memory_source_paths) if memory_source_paths else {},
        skills_dirs=list(skills_dirs),
        keybinds_raw=keybinds_raw,
    )

    app = DeepAgentTui(config)
    app.run()
    return 0
