"""Tests for TrajectoryStore."""

from __future__ import annotations

import json

import pytest

from adk_deepagents.optimization.store import (
    TrajectoryStore,
    _make_index_entry,
    _trajectory_from_dict,
    _trajectory_to_dict,
)
from adk_deepagents.optimization.trajectory import (
    AgentStep,
    FeedbackEntry,
    ModelCall,
    ToolCall,
    Trajectory,
)


def _sample_trajectory(
    trace_id: str = "abc123",
    *,
    agent_name: str = "test_agent",
    status: str = "ok",
    score: float | None = None,
    is_golden: bool = False,
    tags: dict[str, str] | None = None,
) -> Trajectory:
    return Trajectory(
        trace_id=trace_id,
        session_id="session_1",
        agent_name=agent_name,
        steps=[
            AgentStep(
                agent_name=agent_name,
                model_call=ModelCall(
                    model="gemini-2.5-flash",
                    input_tokens=100,
                    output_tokens=50,
                    duration_ms=500.0,
                    request={"system": "you are helpful"},
                    response={"text": "hello"},
                    finish_reason="stop",
                ),
                tool_calls=[
                    ToolCall(
                        name="read_file",
                        args={"path": "/test.txt"},
                        response={"content": "hello"},
                        duration_ms=10.0,
                    ),
                ],
            ),
        ],
        start_time_ns=1_000_000_000,
        end_time_ns=2_000_000_000,
        status=status,
        score=score,
        is_golden=is_golden,
        tags=tags or {},
    )


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_round_trip(self):
        traj = _sample_trajectory(score=0.9, is_golden=True, tags={"env": "test"})
        traj.feedback.append(FeedbackEntry(source="user", rating=1.0, comment="great"))
        data = _trajectory_to_dict(traj)
        restored = _trajectory_from_dict(data)

        assert restored.trace_id == traj.trace_id
        assert restored.session_id == traj.session_id
        assert restored.agent_name == traj.agent_name
        assert restored.status == traj.status
        assert restored.score == traj.score
        assert restored.is_golden == traj.is_golden
        assert restored.tags == {"env": "test"}
        assert restored.start_time_ns == traj.start_time_ns
        assert restored.end_time_ns == traj.end_time_ns
        assert len(restored.steps) == 1
        assert restored.steps[0].model_call is not None
        assert restored.steps[0].model_call.model == "gemini-2.5-flash"
        assert restored.steps[0].model_call.input_tokens == 100
        assert restored.steps[0].model_call.request == {"system": "you are helpful"}
        assert len(restored.steps[0].tool_calls) == 1
        assert restored.steps[0].tool_calls[0].name == "read_file"
        assert len(restored.feedback) == 1
        assert restored.feedback[0].source == "user"
        assert restored.feedback[0].rating == 1.0

    def test_round_trip_no_model_call(self):
        traj = Trajectory(
            trace_id="no_mc",
            steps=[AgentStep(agent_name="a", tool_calls=[])],
        )
        data = _trajectory_to_dict(traj)
        restored = _trajectory_from_dict(data)
        assert restored.steps[0].model_call is None

    def test_round_trip_minimal(self):
        traj = Trajectory(trace_id="minimal")
        data = _trajectory_to_dict(traj)
        restored = _trajectory_from_dict(data)
        assert restored.trace_id == "minimal"
        assert restored.steps == []
        assert restored.feedback == []


# ---------------------------------------------------------------------------
# Index entry
# ---------------------------------------------------------------------------


class TestIndexEntry:
    def test_make_index_entry(self):
        data = _trajectory_to_dict(_sample_trajectory(score=0.8, is_golden=True))
        entry = _make_index_entry(data)
        assert entry["agent_name"] == "test_agent"
        assert entry["status"] == "ok"
        assert entry["is_golden"] is True
        assert entry["score"] == 0.8
        assert entry["start_time_ns"] == 1_000_000_000

    def test_make_index_entry_defaults(self):
        entry = _make_index_entry({"trace_id": "x"})
        assert entry["agent_name"] is None
        assert entry["status"] == "unset"
        assert entry["is_golden"] is False
        assert entry["score"] is None


# ---------------------------------------------------------------------------
# Store: save / load / delete
# ---------------------------------------------------------------------------


class TestStorePathTraversal:
    def test_rejects_path_traversal(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        traj = _sample_trajectory("../../etc/passwd")
        with pytest.raises(ValueError, match="Invalid trace_id"):
            store.save(traj)

    def test_rejects_slash_in_trace_id(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        with pytest.raises(ValueError, match="Invalid trace_id"):
            store.load("sub/dir")


class TestStoreBasicOps:
    def test_save_and_load(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        traj = _sample_trajectory()
        store.save(traj)

        loaded = store.load("abc123")
        assert loaded is not None
        assert loaded.trace_id == "abc123"
        assert loaded.agent_name == "test_agent"
        assert len(loaded.steps) == 1

    def test_load_missing(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        assert store.load("nonexistent") is None

    def test_delete_existing(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        store.save(_sample_trajectory())
        assert store.delete("abc123") is True
        assert store.load("abc123") is None
        assert "abc123" not in store.list_ids()

    def test_delete_missing(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        assert store.delete("nonexistent") is False

    def test_list_ids(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        store.save(_sample_trajectory("t1"))
        store.save(_sample_trajectory("t2"))
        store.save(_sample_trajectory("t3"))
        ids = store.list_ids()
        assert sorted(ids) == ["t1", "t2", "t3"]

    def test_save_overwrites(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        store.save(_sample_trajectory(score=0.5))
        store.save(_sample_trajectory(score=0.9))
        loaded = store.load("abc123")
        assert loaded is not None
        assert loaded.score == 0.9

    def test_creates_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        store = TrajectoryStore(nested)
        store.save(_sample_trajectory())
        assert nested.exists()
        assert store.load("abc123") is not None


# ---------------------------------------------------------------------------
# Store: index persistence and rebuild
# ---------------------------------------------------------------------------


class TestStoreIndex:
    def test_index_persists_across_instances(self, tmp_path):
        store_dir = tmp_path / "store"
        store1 = TrajectoryStore(store_dir)
        store1.save(_sample_trajectory("t1"))
        store1.save(_sample_trajectory("t2"))

        store2 = TrajectoryStore(store_dir)
        assert sorted(store2.list_ids()) == ["t1", "t2"]

    def test_rebuild_index_on_corrupt(self, tmp_path):
        store_dir = tmp_path / "store"
        store = TrajectoryStore(store_dir)
        store.save(_sample_trajectory("t1"))

        # Corrupt the index
        (store_dir / "_index.json").write_text("{{bad json", encoding="utf-8")

        store2 = TrajectoryStore(store_dir)
        assert "t1" in store2.list_ids()

    def test_rebuild_skips_corrupt_trajectory_files(self, tmp_path):
        store_dir = tmp_path / "store"
        store_dir.mkdir()
        (store_dir / "good.json").write_text(
            json.dumps(_trajectory_to_dict(_sample_trajectory("good"))),
            encoding="utf-8",
        )
        (store_dir / "bad.json").write_text("not json", encoding="utf-8")
        # Corrupt the index to force rebuild
        (store_dir / "_index.json").write_text("{{", encoding="utf-8")

        store = TrajectoryStore(store_dir)
        assert "good" in store.list_ids()
        assert "bad" not in store.list_ids()


# ---------------------------------------------------------------------------
# Store: annotation methods
# ---------------------------------------------------------------------------


class TestStoreAnnotation:
    def test_mark_golden(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        store.save(_sample_trajectory())
        assert store.mark_golden("abc123") is True

        loaded = store.load("abc123")
        assert loaded is not None
        assert loaded.is_golden is True

    def test_mark_golden_unset(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        store.save(_sample_trajectory(is_golden=True))
        store.mark_golden("abc123", golden=False)

        loaded = store.load("abc123")
        assert loaded is not None
        assert loaded.is_golden is False

    def test_mark_golden_missing(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        assert store.mark_golden("missing") is False

    def test_set_score(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        store.save(_sample_trajectory())
        assert store.set_score("abc123", 0.95) is True

        loaded = store.load("abc123")
        assert loaded is not None
        assert loaded.score == 0.95

    def test_set_score_missing(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        assert store.set_score("missing", 0.5) is False

    def test_set_tag(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        store.save(_sample_trajectory())
        assert store.set_tag("abc123", "env", "production") is True

        loaded = store.load("abc123")
        assert loaded is not None
        assert loaded.tags["env"] == "production"

    def test_set_tag_missing(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        assert store.set_tag("missing", "k", "v") is False

    def test_remove_tag(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        store.save(_sample_trajectory(tags={"env": "production"}))

        assert store.remove_tag("abc123", "env") is True
        loaded = store.load("abc123")
        assert loaded is not None
        assert "env" not in loaded.tags

    def test_remove_tag_missing_key(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        store.save(_sample_trajectory(tags={"env": "production"}))
        assert store.remove_tag("abc123", "missing") is False

    def test_remove_tag_missing_trajectory(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        assert store.remove_tag("missing", "env") is False

    def test_add_feedback(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        store.save(_sample_trajectory())
        fb = FeedbackEntry(source="user", rating=0.8, comment="good job")
        assert store.add_feedback("abc123", fb) is True

        loaded = store.load("abc123")
        assert loaded is not None
        assert len(loaded.feedback) == 1
        assert loaded.feedback[0].source == "user"
        assert loaded.feedback[0].rating == 0.8
        assert loaded.feedback[0].comment == "good job"
        assert loaded.feedback[0].timestamp_ns > 0  # auto-set

    def test_add_feedback_preserves_explicit_timestamp(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        store.save(_sample_trajectory())
        fb = FeedbackEntry(source="auto", timestamp_ns=999)
        store.add_feedback("abc123", fb)

        loaded = store.load("abc123")
        assert loaded is not None
        assert loaded.feedback[0].timestamp_ns == 999

    def test_add_feedback_missing(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        fb = FeedbackEntry(source="user")
        assert store.add_feedback("missing", fb) is False

    def test_add_multiple_feedback(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        store.save(_sample_trajectory())
        store.add_feedback("abc123", FeedbackEntry(source="user", rating=0.5))
        store.add_feedback("abc123", FeedbackEntry(source="auto", rating=0.9))

        loaded = store.load("abc123")
        assert loaded is not None
        assert len(loaded.feedback) == 2


# ---------------------------------------------------------------------------
# Store: list_trajectories (filtered queries)
# ---------------------------------------------------------------------------


class TestStoreQuery:
    def _populate(self, store: TrajectoryStore) -> None:
        store.save(_sample_trajectory("t1", agent_name="alpha", status="ok", score=0.9))
        store.save(
            _sample_trajectory("t2", agent_name="alpha", status="error", score=0.3, is_golden=True)
        )
        store.save(_sample_trajectory("t3", agent_name="beta", status="ok", score=0.7))
        store.save(_sample_trajectory("t4", agent_name="beta", status="ok", tags={"env": "prod"}))

    def test_no_filter(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        self._populate(store)
        results = store.list_trajectories()
        assert len(results) == 4

    def test_filter_agent_name(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        self._populate(store)
        results = store.list_trajectories(agent_name="alpha")
        assert len(results) == 2
        assert all(t.agent_name == "alpha" for t in results)

    def test_filter_status(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        self._populate(store)
        results = store.list_trajectories(status="error")
        assert len(results) == 1
        assert results[0].trace_id == "t2"

    def test_filter_is_golden(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        self._populate(store)
        results = store.list_trajectories(is_golden=True)
        assert len(results) == 1
        assert results[0].trace_id == "t2"

    def test_filter_min_score(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        self._populate(store)
        results = store.list_trajectories(min_score=0.7)
        assert len(results) == 2
        assert {t.trace_id for t in results} == {"t1", "t3"}

    def test_filter_tag(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        self._populate(store)
        results = store.list_trajectories(tag=("env", "prod"))
        assert len(results) == 1
        assert results[0].trace_id == "t4"

    def test_combined_filters(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        self._populate(store)
        results = store.list_trajectories(agent_name="beta", status="ok", min_score=0.7)
        assert len(results) == 1
        assert results[0].trace_id == "t3"

    def test_filter_no_match(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        self._populate(store)
        results = store.list_trajectories(agent_name="nonexistent")
        assert results == []


# ---------------------------------------------------------------------------
# Store: export_dataset
# ---------------------------------------------------------------------------


class TestStoreExport:
    def test_export_all(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        traj = _sample_trajectory(score=0.9)
        traj.feedback.append(FeedbackEntry(source="user", rating=1.0, comment="perfect"))
        store.save(traj)

        dataset = store.export_dataset()
        assert len(dataset) == 1
        entry = dataset[0]
        assert entry["trace_id"] == "abc123"
        assert entry["agent_name"] == "test_agent"
        assert entry["score"] == 0.9
        assert entry["total_input_tokens"] == 100
        assert entry["total_output_tokens"] == 50
        assert len(entry["steps"]) == 1
        assert entry["steps"][0]["model"] == "gemini-2.5-flash"
        assert entry["steps"][0]["tool_calls"][0]["name"] == "read_file"
        assert len(entry["feedback"]) == 1

    def test_export_golden_only(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        store.save(_sample_trajectory("t1", is_golden=True, score=1.0))
        store.save(_sample_trajectory("t2", is_golden=False, score=0.5))

        dataset = store.export_dataset(is_golden=True)
        assert len(dataset) == 1
        assert dataset[0]["trace_id"] == "t1"

    def test_export_min_score(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        store.save(_sample_trajectory("t1", score=0.9))
        store.save(_sample_trajectory("t2", score=0.3))

        dataset = store.export_dataset(min_score=0.5)
        assert len(dataset) == 1
        assert dataset[0]["trace_id"] == "t1"

    def test_export_empty_store(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        assert store.export_dataset() == []

    def test_export_jsonl_writes_filtered_dataset(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        store.save(_sample_trajectory("t1", score=0.9, is_golden=True))
        store.save(_sample_trajectory("t2", score=0.2, is_golden=False))

        out_path = tmp_path / "exports" / "dataset.jsonl"
        written = store.export_dataset_jsonl(out_path, min_score=0.5)

        assert written == 1
        assert out_path.exists()
        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["trace_id"] == "t1"
        assert row["score"] == 0.9

    def test_export_jsonl_empty_store_creates_empty_file(self, tmp_path):
        store = TrajectoryStore(tmp_path / "store")
        out_path = tmp_path / "exports" / "dataset.jsonl"

        written = store.export_dataset_jsonl(out_path)

        assert written == 0
        assert out_path.exists()
        assert out_path.read_text(encoding="utf-8") == ""
