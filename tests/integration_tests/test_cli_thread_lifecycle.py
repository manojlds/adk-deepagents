"""Integration tests for CLI thread persistence across process restarts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _run_cli(home_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["ADK_DEEPAGENTS_HOME"] = str(home_dir)

    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "adk_deepagents.cli", *args],  # noqa: S607
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )


def _extract_first_thread_id(list_output: str) -> str | None:
    lines = list_output.strip().splitlines()
    if len(lines) < 3:
        return None

    return lines[2].split("\t", maxsplit=1)[0]


def test_session_continuity_across_process_restarts(tmp_path: Path) -> None:
    home_dir = tmp_path / ".adk-deepagents"

    created = _run_cli(home_dir, "--agent", "demo")
    assert created.returncode == 0

    listed = _run_cli(home_dir, "threads", "list", "--agent", "demo")
    assert listed.returncode == 0
    thread_id = _extract_first_thread_id(listed.stdout)
    assert thread_id is not None

    resume_latest = _run_cli(home_dir, "--agent", "demo", "--resume")
    assert resume_latest.returncode == 0

    resume_explicit = _run_cli(home_dir, "--agent", "demo", "--resume", thread_id)
    assert resume_explicit.returncode == 0

    deleted = _run_cli(home_dir, "threads", "delete", thread_id, "--agent", "demo")
    assert deleted.returncode == 0

    missing = _run_cli(home_dir, "--agent", "demo", "--resume", thread_id)
    assert missing.returncode == 1
    assert "failed to resolve thread" in missing.stderr
