"""OTEL trace import helpers for optimization workflows.

This module ingests OTEL JSON exports and maps spans into a compact,
framework-level trajectory schema used by adk-deepagents optimizers.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from adk_deepagents.optimization.models import EventKind, Trajectory, TrajectoryEvent

_EVENT_KIND_MAP: dict[str, EventKind] = {
    "agent_turn": "agent_turn",
    "model_call": "model_call",
    "model_response": "model_response",
    "tool_call": "tool_call",
    "tool_result": "tool_result",
    "delegation": "delegation",
    "approval": "approval",
    "feedback": "feedback",
    "unknown": "unknown",
}


def load_otel_json(path: str | Path) -> dict[str, Any]:
    """Load an OTEL JSON export payload from disk."""
    raw = Path(path).read_text(encoding="utf-8")
    text = raw.strip()
    if not text:
        return {"resourceSpans": []}

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        resource_spans: list[Any] = []
        for line in raw.splitlines():
            item_text = line.strip()
            if not item_text:
                continue
            try:
                item = json.loads(item_text)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            spans = item.get("resourceSpans")
            if isinstance(spans, list):
                resource_spans.extend(spans)
        return {"resourceSpans": resource_spans}

    if isinstance(payload, dict):
        return payload
    raise ValueError("OTEL JSON payload must be an object at top level.")


def _iter_spans(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    resource_spans = payload.get("resourceSpans", [])
    if not isinstance(resource_spans, list):
        return
    for resource in resource_spans:
        if not isinstance(resource, dict):
            continue
        scope_spans = resource.get("scopeSpans", [])
        if not isinstance(scope_spans, list):
            continue
        for scope in scope_spans:
            if not isinstance(scope, dict):
                continue
            spans = scope.get("spans", [])
            if not isinstance(spans, list):
                continue
            for span in spans:
                if isinstance(span, dict):
                    yield span


def _attr_value_to_python(value: dict[str, Any]) -> Any:
    if "stringValue" in value:
        return value.get("stringValue")
    if "boolValue" in value:
        return bool(value.get("boolValue"))
    if "intValue" in value:
        raw = value.get("intValue")
        if isinstance(raw, str):
            try:
                return int(raw)
            except ValueError:
                return raw
        return raw
    if "doubleValue" in value:
        return value.get("doubleValue")
    if "arrayValue" in value:
        arr = value.get("arrayValue")
        if isinstance(arr, dict):
            values = arr.get("values", [])
            if isinstance(values, list):
                return [_attr_value_to_python(v) for v in values if isinstance(v, dict)]
        return arr
    if "kvlistValue" in value:
        kv = value.get("kvlistValue")
        if isinstance(kv, dict):
            vals = kv.get("values", [])
            if isinstance(vals, list):
                out: dict[str, Any] = {}
                for item in vals:
                    if not isinstance(item, dict):
                        continue
                    key = item.get("key")
                    val = item.get("value")
                    if isinstance(key, str) and isinstance(val, dict):
                        out[key] = _attr_value_to_python(val)
                return out
        return kv
    return value


def _parse_attributes(span: dict[str, Any]) -> dict[str, Any]:
    attrs = span.get("attributes", [])
    if not isinstance(attrs, list):
        return {}
    out: dict[str, Any] = {}
    for item in attrs:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        value = item.get("value")
        if isinstance(key, str) and isinstance(value, dict):
            out[key] = _attr_value_to_python(value)
    return out


def _infer_kind(name: str, attrs: dict[str, Any]) -> EventKind:
    lowered = name.lower()

    op_name = attrs.get("gen_ai.operation.name")
    if isinstance(op_name, str):
        op = op_name.lower()
        if op == "invoke_agent":
            return cast(EventKind, "agent_turn")
        if op == "execute_tool":
            tool_response = attrs.get("gcp.vertex.agent.tool_response")
            if isinstance(tool_response, str) and tool_response not in {"", "{}", "N/A"}:
                return cast(EventKind, "tool_result")
            return cast(EventKind, "tool_call")
        if op == "generate_content":
            return cast(EventKind, "model_response")

    if lowered == "invocation":
        return cast(EventKind, "agent_turn")
    if lowered == "call_llm":
        return cast(EventKind, "model_call")
    if lowered.startswith("generate_content"):
        return cast(EventKind, "model_response")

    if "tool" in lowered and "call" in lowered:
        return cast(EventKind, "tool_call")
    if "tool" in lowered and ("result" in lowered or "response" in lowered):
        return cast(EventKind, "tool_result")
    if "model" in lowered and "call" in lowered:
        return cast(EventKind, "model_call")
    if "model" in lowered and ("response" in lowered or "output" in lowered):
        return cast(EventKind, "model_response")
    if "delegate" in lowered or "subagent" in lowered or "task" in lowered:
        return cast(EventKind, "delegation")
    if "approval" in lowered or "confirm" in lowered:
        return cast(EventKind, "approval")

    span_kind = attrs.get("adk.kind") or attrs.get("kind")
    if isinstance(span_kind, str):
        val = span_kind.lower()
        mapped = _EVENT_KIND_MAP.get(val)
        if mapped is not None:
            return mapped
    return cast(EventKind, "unknown")


def _coerce_event_kind(value: str) -> EventKind:
    return _EVENT_KIND_MAP.get(value, "unknown")


def _pick_str(attrs: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = attrs.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def import_otel_traces(payload: dict[str, Any]) -> list[Trajectory]:
    """Convert OTEL export payload into normalized trajectories by trace id."""
    grouped: dict[str, list[TrajectoryEvent]] = defaultdict(list)

    for span in _iter_spans(payload):
        trace_id = span.get("traceId")
        span_id = span.get("spanId")
        if not isinstance(trace_id, str) or not isinstance(span_id, str):
            continue

        attrs = _parse_attributes(span)
        name = span.get("name") if isinstance(span.get("name"), str) else ""
        kind = _infer_kind(str(name), attrs)

        event = TrajectoryEvent(
            kind=kind,
            timestamp=str(span.get("startTimeUnixNano", "")),
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=(
                span.get("parentSpanId") if isinstance(span.get("parentSpanId"), str) else None
            ),
            name=name or None,
            agent_name=_pick_str(attrs, "adk.agent.name", "gen_ai.agent.name"),
            session_id=_pick_str(
                attrs,
                "adk.session.id",
                "gcp.vertex.agent.session_id",
                "gen_ai.conversation.id",
            ),
            tool_name=_pick_str(attrs, "adk.tool.name", "gen_ai.tool.name"),
            model=_pick_str(attrs, "adk.model.name", "gen_ai.request.model"),
            status=(
                span.get("status", {}).get("message")
                if isinstance(span.get("status"), dict)
                else None
            ),
            input_text=_pick_str(
                attrs,
                "adk.input",
                "gcp.vertex.agent.llm_request",
                "gcp.vertex.agent.tool_call_args",
            ),
            output_text=_pick_str(
                attrs,
                "adk.output",
                "gcp.vertex.agent.llm_response",
                "gcp.vertex.agent.tool_response",
            ),
            attributes=attrs,
        )
        grouped[trace_id].append(event)

    trajectories: list[Trajectory] = []
    for trace_id, events in grouped.items():
        events.sort(key=lambda e: e.timestamp)
        root_name = events[0].name if events else None
        session_id = next((e.session_id for e in events if e.session_id), None)
        agent_name = next((e.agent_name for e in events if e.agent_name), None)
        trajectories.append(
            Trajectory(
                trace_id=trace_id,
                started_at=(events[0].timestamp if events else None),
                ended_at=(events[-1].timestamp if events else None),
                root_span_name=root_name,
                session_id=session_id,
                agent_name=agent_name,
                events=events,
            )
        )

    trajectories.sort(key=lambda t: t.started_at or "")
    return trajectories


def _event_to_json(event: TrajectoryEvent) -> dict[str, Any]:
    return {
        "kind": event.kind,
        "timestamp": event.timestamp,
        "trace_id": event.trace_id,
        "span_id": event.span_id,
        "parent_span_id": event.parent_span_id,
        "name": event.name,
        "agent_name": event.agent_name,
        "session_id": event.session_id,
        "tool_name": event.tool_name,
        "model": event.model,
        "status": event.status,
        "input_text": event.input_text,
        "output_text": event.output_text,
        "attributes": event.attributes,
    }


def _trajectory_to_json(trajectory: Trajectory) -> dict[str, Any]:
    return {
        "trace_id": trajectory.trace_id,
        "started_at": trajectory.started_at,
        "ended_at": trajectory.ended_at,
        "root_span_name": trajectory.root_span_name,
        "session_id": trajectory.session_id,
        "agent_name": trajectory.agent_name,
        "events": [_event_to_json(event) for event in trajectory.events],
    }


def save_trajectories_jsonl(trajectories: list[Trajectory], path: str | Path) -> None:
    """Write trajectories to JSONL for downstream optimization jobs."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(json.dumps(_trajectory_to_json(trajectory), ensure_ascii=True))
            handle.write("\n")


def load_trajectories_jsonl(path: str | Path) -> list[Trajectory]:
    """Load normalized trajectories from JSONL.

    Invalid rows are ignored to keep ingestion robust.
    """
    input_path = Path(path)
    if not input_path.exists():
        return []

    trajectories: list[Trajectory] = []
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
            trace_id = payload.get("trace_id")
            if not isinstance(trace_id, str):
                continue

            raw_events = payload.get("events", [])
            events: list[TrajectoryEvent] = []
            if isinstance(raw_events, list):
                for item in raw_events:
                    if not isinstance(item, dict):
                        continue
                    kind = item.get("kind")
                    timestamp = item.get("timestamp")
                    span_id = item.get("span_id")
                    if not isinstance(kind, str) or not isinstance(timestamp, str):
                        continue
                    if not isinstance(span_id, str):
                        continue

                    parent_span_id = item.get("parent_span_id")
                    if not isinstance(parent_span_id, str):
                        parent_span_id = None

                    attrs = item.get("attributes")
                    attributes = attrs if isinstance(attrs, dict) else {}
                    event_kind = _coerce_event_kind(kind)

                    name = item.get("name") if isinstance(item.get("name"), str) else None
                    agent_name = (
                        item.get("agent_name") if isinstance(item.get("agent_name"), str) else None
                    )
                    session_id = (
                        item.get("session_id") if isinstance(item.get("session_id"), str) else None
                    )
                    tool_name = (
                        item.get("tool_name") if isinstance(item.get("tool_name"), str) else None
                    )
                    model = item.get("model") if isinstance(item.get("model"), str) else None
                    status = item.get("status") if isinstance(item.get("status"), str) else None
                    input_text = (
                        item.get("input_text") if isinstance(item.get("input_text"), str) else None
                    )
                    output_text = (
                        item.get("output_text")
                        if isinstance(item.get("output_text"), str)
                        else None
                    )

                    events.append(
                        TrajectoryEvent(
                            kind=event_kind,
                            timestamp=timestamp,
                            trace_id=trace_id,
                            span_id=span_id,
                            parent_span_id=parent_span_id,
                            name=name,
                            agent_name=agent_name,
                            session_id=session_id,
                            tool_name=tool_name,
                            model=model,
                            status=status,
                            input_text=input_text,
                            output_text=output_text,
                            attributes=attributes,
                        )
                    )

            trajectories.append(
                Trajectory(
                    trace_id=trace_id,
                    started_at=(
                        payload.get("started_at")
                        if isinstance(payload.get("started_at"), str)
                        else None
                    ),
                    ended_at=(
                        payload.get("ended_at")
                        if isinstance(payload.get("ended_at"), str)
                        else None
                    ),
                    root_span_name=(
                        payload.get("root_span_name")
                        if isinstance(payload.get("root_span_name"), str)
                        else None
                    ),
                    session_id=(
                        payload.get("session_id")
                        if isinstance(payload.get("session_id"), str)
                        else None
                    ),
                    agent_name=(
                        payload.get("agent_name")
                        if isinstance(payload.get("agent_name"), str)
                        else None
                    ),
                    events=events,
                )
            )

    return trajectories
