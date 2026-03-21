"""Feedback persistence helpers for trajectory-linked optimization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adk_deepagents.optimization.models import FeedbackRecord


def _feedback_to_json(feedback: FeedbackRecord) -> dict[str, Any]:
    return {
        "feedback_id": feedback.feedback_id,
        "timestamp": feedback.timestamp,
        "source": feedback.source,
        "score": feedback.score,
        "label": feedback.label,
        "rationale": feedback.rationale,
        "trace_id": feedback.trace_id,
        "span_id": feedback.span_id,
        "session_id": feedback.session_id,
        "metadata": feedback.metadata,
    }


def append_feedback_jsonl(feedback: FeedbackRecord, path: str | Path) -> None:
    """Append one feedback record as a JSONL line."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_feedback_to_json(feedback), ensure_ascii=True))
        handle.write("\n")


def load_feedback_jsonl(path: str | Path) -> list[FeedbackRecord]:
    """Load feedback records from JSONL.

    Invalid lines are ignored to keep ingestion resilient.
    """
    input_path = Path(path)
    if not input_path.exists():
        return []

    records: list[FeedbackRecord] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(payload, dict):
                continue

            feedback_id = payload.get("feedback_id")
            source = payload.get("source")
            if not isinstance(feedback_id, str) or source not in {"human", "auto", "judge"}:
                continue

            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}

            score = payload.get("score")
            if score is not None and not isinstance(score, (int, float)):
                score = None

            label = payload.get("label") if isinstance(payload.get("label"), str) else None
            rationale = (
                payload.get("rationale") if isinstance(payload.get("rationale"), str) else None
            )
            trace_id = payload.get("trace_id") if isinstance(payload.get("trace_id"), str) else None
            span_id = payload.get("span_id") if isinstance(payload.get("span_id"), str) else None
            session_id = (
                payload.get("session_id") if isinstance(payload.get("session_id"), str) else None
            )
            timestamp = (
                payload.get("timestamp") if isinstance(payload.get("timestamp"), str) else ""
            )

            records.append(
                FeedbackRecord(
                    feedback_id=feedback_id,
                    timestamp=timestamp,
                    source=source,
                    score=float(score) if isinstance(score, (int, float)) else None,
                    label=label,
                    rationale=rationale,
                    trace_id=trace_id,
                    span_id=span_id,
                    session_id=session_id,
                    metadata=metadata,
                )
            )

    return records
