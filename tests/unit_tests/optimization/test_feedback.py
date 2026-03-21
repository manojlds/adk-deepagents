"""Unit tests for optimization feedback persistence helpers."""

from __future__ import annotations

from adk_deepagents.optimization.feedback import append_feedback_jsonl, load_feedback_jsonl
from adk_deepagents.optimization.models import FeedbackRecord


def test_append_and_load_feedback_jsonl(tmp_path) -> None:
    path = tmp_path / "feedback.jsonl"
    record = FeedbackRecord(
        feedback_id="fb1",
        source="human",
        score=1.0,
        label="good",
        rationale="helpful answer",
        trace_id="trace-1",
        span_id="span-1",
        session_id="session-1",
    )

    append_feedback_jsonl(record, path)

    loaded = load_feedback_jsonl(path)
    assert len(loaded) == 1
    assert loaded[0].feedback_id == "fb1"
    assert loaded[0].score == 1.0
    assert loaded[0].trace_id == "trace-1"


def test_load_feedback_ignores_invalid_lines(tmp_path) -> None:
    path = tmp_path / "feedback.jsonl"
    path.write_text(
        '{"feedback_id":"ok","source":"human"}\n'
        "{not-json}\n"
        '{"feedback_id": 123, "source": "human"}\n',
        encoding="utf-8",
    )

    loaded = load_feedback_jsonl(path)
    assert len(loaded) == 1
    assert loaded[0].feedback_id == "ok"
