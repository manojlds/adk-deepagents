"""Data extraction benchmark runner.

Loads tasks from the tasks/ directory, runs them through the agent,
and verifies output using the deterministic JSON verifier.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from adk_deepagents.optimization.benchmark import (
    BenchmarkResult,
    BenchmarkRunner,
    LocalBenchmarkRunner,
    TaskSpec,
)
from examples.data_extraction.verify import verify_json_output

logger = logging.getLogger(__name__)


def load_tasks(
    tasks_dir: str | Path,
    *,
    split: str = "train",
) -> list[TaskSpec]:
    """Load task specs from the tasks directory for a given split.

    Reads splits.yaml to determine which tasks belong to the split,
    then loads each task's input file and expected output.
    """
    tasks_path = Path(tasks_dir)
    splits_file = tasks_path / "splits.yaml"

    if splits_file.exists():
        splits = yaml.safe_load(splits_file.read_text(encoding="utf-8"))
        task_ids = splits.get(split, [])
    else:
        # No splits file — use all task directories
        task_ids = [
            d.name
            for d in sorted(tasks_path.iterdir())
            if d.is_dir() and (d / "task.yaml").exists()
        ]

    specs: list[TaskSpec] = []
    for task_id in task_ids:
        task_dir = tasks_path / task_id
        task_yaml = task_dir / "task.yaml"

        if not task_yaml.exists():
            logger.warning("Task %s has no task.yaml, skipping", task_id)
            continue

        task_config = yaml.safe_load(task_yaml.read_text(encoding="utf-8"))

        # Load input file
        input_file = task_config.get("input_file", "input.txt")
        input_path = task_dir / input_file
        workspace_files: dict[str, str] = {}
        if input_path.exists():
            workspace_files[f"/workspace/{input_file}"] = input_path.read_text(encoding="utf-8")

        # Load expected output
        expected_file = task_dir / "expected.json"
        expected_output: dict[str, Any] | None = None
        if expected_file.exists():
            expected_output = json.loads(expected_file.read_text(encoding="utf-8"))

        specs.append(
            TaskSpec(
                task_id=task_id,
                instruction=task_config["instruction"],
                workspace_files=workspace_files,
                expected_output=expected_output,
                timeout_seconds=task_config.get("timeout_seconds", 120.0),
                metadata={"split": split, "task_dir": str(task_dir)},
            )
        )

    return specs


def _verify_extraction(
    task: TaskSpec,
    agent_output: str,
    final_state: dict[str, Any],
) -> float:
    """Verify data extraction output against expected JSON.

    Looks for output.json in the agent's workspace state.
    """
    if task.expected_output is None:
        return 0.0

    # Try to find output in state files
    files = final_state.get("files", {})
    output_data: dict[str, Any] | None = None

    for candidate_path in ["/workspace/output.json", "output.json", "/output.json"]:
        file_entry = files.get(candidate_path)
        if file_entry is not None:
            content = file_entry.get("content", [])
            raw = "\n".join(content) if isinstance(content, list) else str(content)
            try:
                output_data = json.loads(raw)
                break
            except (json.JSONDecodeError, ValueError):
                continue

    if output_data is None:
        # Try parsing from agent's text output
        try:
            output_data = json.loads(agent_output)
        except (json.JSONDecodeError, ValueError):
            return 0.0

    score, _details = verify_json_output(task.expected_output, output_data)
    return score


class DataExtractionBenchmark(BenchmarkRunner):
    """Benchmark runner for data extraction tasks.

    Loads tasks from the tasks/ directory, runs them through a deep agent
    with StateBackend, and verifies output with the JSON field verifier.
    """

    def __init__(
        self,
        tasks_dir: str | Path,
        *,
        split: str = "train",
        pass_threshold: float = 0.5,
        max_concurrency: int = 4,
    ) -> None:
        self._tasks_dir = Path(tasks_dir)
        self._split = split
        self._pass_threshold = pass_threshold
        self._max_concurrency = max_concurrency
        self._tasks = load_tasks(tasks_dir, split=split)
        self._runner = LocalBenchmarkRunner(
            self._tasks,
            verify_fn=_verify_extraction,
            pass_threshold=pass_threshold,
            max_concurrency=max_concurrency,
        )

    def list_task_ids(self) -> list[str]:
        return self._runner.list_task_ids()

    async def run(
        self,
        agent_factory: Callable[..., Any],
        *,
        task_ids: list[str] | None = None,
    ) -> BenchmarkResult:
        return await self._runner.run(agent_factory, task_ids=task_ids)
