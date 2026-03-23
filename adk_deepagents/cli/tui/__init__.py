"""TUI (Terminal User Interface) for adk-deepagents."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from adk_deepagents.types import DynamicTaskConfig

_TUI_LOG_FILE = Path.home() / ".adk-deepagents-tui-debug.log"


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
    theme_name: str | None = None,
    trajectories_dir: Path | None = None,
    otel_traces_path: Path | None = None,
) -> int:
    """Launch the Textual TUI and block until exit."""
    # Enable debug logging to a file for TUI diagnostics.
    handler = logging.FileHandler(_TUI_LOG_FILE, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    tui_logger = logging.getLogger("adk_deepagents.tui")
    tui_logger.setLevel(logging.DEBUG)
    tui_logger.addHandler(handler)

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
        theme_name=theme_name,
        trajectories_dir=trajectories_dir,
        otel_traces_path=otel_traces_path,
    )

    app = DeepAgentTui(config)
    app.run()
    return 0
