"""Persistent trajectory store backed by JSON files."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from adk_deepagents.optimization.trajectory import (
    AgentStep,
    FeedbackEntry,
    ModelCall,
    ToolCall,
    Trajectory,
)

logger = logging.getLogger(__name__)

_INDEX_FILE = "_index.json"


def _trajectory_to_dict(t: Trajectory) -> dict[str, Any]:
    """Serialize a Trajectory to a JSON-compatible dict."""
    return asdict(t)


def _trajectory_from_dict(data: dict[str, Any]) -> Trajectory:
    """Deserialize a Trajectory from a dict."""
    steps: list[AgentStep] = []
    for step_data in data.get("steps", []):
        mc_data = step_data.get("model_call")
        model_call = ModelCall(**mc_data) if mc_data is not None else None

        tool_calls = [ToolCall(**tc) for tc in step_data.get("tool_calls", [])]

        steps.append(
            AgentStep(
                agent_name=step_data.get("agent_name", ""),
                model_call=model_call,
                tool_calls=tool_calls,
            )
        )

    feedback = [FeedbackEntry(**fb) for fb in data.get("feedback", [])]

    return Trajectory(
        trace_id=data["trace_id"],
        session_id=data.get("session_id"),
        agent_name=data.get("agent_name"),
        steps=steps,
        start_time_ns=data.get("start_time_ns", 0),
        end_time_ns=data.get("end_time_ns", 0),
        status=data.get("status", "unset"),
        score=data.get("score"),
        is_golden=data.get("is_golden", False),
        feedback=feedback,
        tags=data.get("tags", {}),
    )


class TrajectoryStore:
    """File-backed store for agent execution trajectories.

    Each trajectory is stored as a separate JSON file named ``{trace_id}.json``
    in the store directory.  A lightweight ``_index.json`` caches metadata for
    fast queries without reading every file.
    """

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict[str, Any]] = self._load_index()

    def _index_path(self) -> Path:
        return self._dir / _INDEX_FILE

    def _trajectory_path(self, trace_id: str) -> Path:
        return self._dir / f"{trace_id}.json"

    def _load_index(self) -> dict[str, dict[str, Any]]:
        idx_path = self._index_path()
        if not idx_path.exists():
            return {}
        try:
            return json.loads(idx_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt trajectory index at %s, rebuilding", idx_path)
            return self._rebuild_index()

    def _save_index(self) -> None:
        self._index_path().write_text(
            json.dumps(self._index, separators=(",", ":")),
            encoding="utf-8",
        )

    def _rebuild_index(self) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for json_file in self._dir.glob("*.json"):
            if json_file.name == _INDEX_FILE:
                continue
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                trace_id = data.get("trace_id", json_file.stem)
                index[trace_id] = _make_index_entry(data)
            except (json.JSONDecodeError, OSError):
                logger.warning("Skipping corrupt trajectory file: %s", json_file)
        self._index = index
        self._save_index()
        return index

    def save(self, trajectory: Trajectory) -> None:
        """Persist a trajectory to disk."""
        data = _trajectory_to_dict(trajectory)
        self._trajectory_path(trajectory.trace_id).write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
        self._index[trajectory.trace_id] = _make_index_entry(data)
        self._save_index()

    def load(self, trace_id: str) -> Trajectory | None:
        """Load a trajectory by trace_id.  Returns None if not found."""
        path = self._trajectory_path(trace_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return _trajectory_from_dict(data)
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning("Failed to load trajectory %s: %s", trace_id, exc)
            return None

    def delete(self, trace_id: str) -> bool:
        """Delete a trajectory.  Returns True if it existed."""
        path = self._trajectory_path(trace_id)
        existed = path.exists()
        if existed:
            path.unlink()
        self._index.pop(trace_id, None)
        self._save_index()
        return existed

    def list_ids(self) -> list[str]:
        """Return all stored trace_ids."""
        return list(self._index.keys())

    def list_trajectories(
        self,
        *,
        agent_name: str | None = None,
        status: str | None = None,
        is_golden: bool | None = None,
        min_score: float | None = None,
        tag: tuple[str, str] | None = None,
    ) -> list[Trajectory]:
        """Query trajectories with optional filters.

        All filters are ANDed together.
        """
        matching: list[Trajectory] = []
        for trace_id, entry in self._index.items():
            if agent_name is not None and entry.get("agent_name") != agent_name:
                continue
            if status is not None and entry.get("status") != status:
                continue
            if is_golden is not None and entry.get("is_golden") != is_golden:
                continue
            if min_score is not None:
                entry_score = entry.get("score")
                if entry_score is None or entry_score < min_score:
                    continue
            if tag is not None:
                entry_tags = entry.get("tags", {})
                if entry_tags.get(tag[0]) != tag[1]:
                    continue

            traj = self.load(trace_id)
            if traj is not None:
                matching.append(traj)
        return matching

    def mark_golden(self, trace_id: str, *, golden: bool = True) -> bool:
        """Mark or unmark a trajectory as golden.  Returns False if not found."""
        traj = self.load(trace_id)
        if traj is None:
            return False
        traj.is_golden = golden
        self.save(traj)
        return True

    def set_score(self, trace_id: str, score: float) -> bool:
        """Set the optimization score for a trajectory.  Returns False if not found."""
        traj = self.load(trace_id)
        if traj is None:
            return False
        traj.score = score
        self.save(traj)
        return True

    def set_tag(self, trace_id: str, key: str, value: str) -> bool:
        """Set a tag on a trajectory.  Returns False if not found."""
        traj = self.load(trace_id)
        if traj is None:
            return False
        traj.tags[key] = value
        self.save(traj)
        return True

    def remove_tag(self, trace_id: str, key: str) -> bool:
        """Remove a tag from a trajectory. Returns False if trajectory or key is missing."""
        traj = self.load(trace_id)
        if traj is None or key not in traj.tags:
            return False
        del traj.tags[key]
        self.save(traj)
        return True

    def add_feedback(self, trace_id: str, feedback: FeedbackEntry) -> bool:
        """Append feedback to a trajectory.  Returns False if not found."""
        traj = self.load(trace_id)
        if traj is None:
            return False
        if feedback.timestamp_ns == 0:
            feedback.timestamp_ns = time.time_ns()
        traj.feedback.append(feedback)
        self.save(traj)
        return True

    def export_dataset(
        self,
        *,
        is_golden: bool | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Export trajectories in a format suitable for GEPA's reflective dataset.

        Each entry contains the trajectory's inputs (prompts/tool calls),
        outputs (model responses), score, and feedback as structured data.
        """
        trajectories = self.list_trajectories(is_golden=is_golden, min_score=min_score)
        dataset: list[dict[str, Any]] = []

        for traj in trajectories:
            entry: dict[str, Any] = {
                "trace_id": traj.trace_id,
                "agent_name": traj.agent_name,
                "status": traj.status,
                "score": traj.score,
                "is_golden": traj.is_golden,
                "duration_ms": traj.duration_ms,
                "total_input_tokens": traj.total_input_tokens,
                "total_output_tokens": traj.total_output_tokens,
                "steps": [],
                "feedback": [asdict(fb) for fb in traj.feedback],
            }

            for step in traj.steps:
                step_data: dict[str, Any] = {"agent_name": step.agent_name}
                if step.model_call:
                    step_data["model"] = step.model_call.model
                    step_data["request"] = step.model_call.request
                    step_data["response"] = step.model_call.response
                step_data["tool_calls"] = [
                    {"name": tc.name, "args": tc.args, "response": tc.response, "error": tc.error}
                    for tc in step.tool_calls
                ]
                entry["steps"].append(step_data)

            dataset.append(entry)

        return dataset

    def export_dataset_jsonl(
        self,
        output_path: str | Path,
        *,
        is_golden: bool | None = None,
        min_score: float | None = None,
        dataset: list[dict[str, Any]] | None = None,
    ) -> int:
        """Write exported dataset entries to JSONL and return the number written."""
        rows = dataset
        if rows is None:
            rows = self.export_dataset(is_golden=is_golden, min_score=min_score)

        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as handle:
            for entry in rows:
                handle.write(json.dumps(entry, ensure_ascii=False))
                handle.write("\n")

        return len(rows)


def _make_index_entry(data: dict[str, Any]) -> dict[str, Any]:
    """Extract lightweight metadata for the index from a full trajectory dict."""
    return {
        "agent_name": data.get("agent_name"),
        "status": data.get("status", "unset"),
        "is_golden": data.get("is_golden", False),
        "score": data.get("score"),
        "start_time_ns": data.get("start_time_ns", 0),
        "tags": data.get("tags", {}),
    }
