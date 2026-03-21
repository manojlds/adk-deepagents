"""Unit tests for OTEL trajectory import helpers."""

from __future__ import annotations

import json

from adk_deepagents.optimization.otel import (
    import_otel_traces,
    load_otel_json,
    load_trajectories_jsonl,
)


def test_import_otel_traces_groups_by_trace_id() -> None:
    payload = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "t1",
                                "spanId": "s1",
                                "name": "model call",
                                "startTimeUnixNano": "1",
                                "attributes": [
                                    {
                                        "key": "adk.session.id",
                                        "value": {"stringValue": "session-a"},
                                    }
                                ],
                            },
                            {
                                "traceId": "t2",
                                "spanId": "s2",
                                "name": "tool result",
                                "startTimeUnixNano": "2",
                                "attributes": [],
                            },
                        ]
                    }
                ]
            }
        ]
    }

    trajectories = import_otel_traces(payload)
    assert len(trajectories) == 2
    ids = [t.trace_id for t in trajectories]
    assert ids == ["t1", "t2"]


def test_import_otel_traces_infers_tool_kind_and_fields() -> None:
    payload = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "trace-1",
                                "spanId": "span-1",
                                "name": "tool call",
                                "startTimeUnixNano": "123",
                                "attributes": [
                                    {
                                        "key": "adk.tool.name",
                                        "value": {"stringValue": "read_file"},
                                    },
                                    {
                                        "key": "adk.agent.name",
                                        "value": {"stringValue": "build"},
                                    },
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }

    trajectories = import_otel_traces(payload)
    assert len(trajectories) == 1
    event = trajectories[0].events[0]
    assert event.kind == "tool_call"
    assert event.tool_name == "read_file"
    assert event.agent_name == "build"


def test_import_otel_traces_maps_genai_semconv_fields() -> None:
    payload = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "trace-semconv",
                                "spanId": "span-semconv",
                                "name": "invocation",
                                "startTimeUnixNano": "123",
                                "attributes": [
                                    {
                                        "key": "gen_ai.operation.name",
                                        "value": {"stringValue": "execute_tool"},
                                    },
                                    {
                                        "key": "gen_ai.tool.name",
                                        "value": {"stringValue": "read_file"},
                                    },
                                    {
                                        "key": "gen_ai.agent.name",
                                        "value": {"stringValue": "general-purpose"},
                                    },
                                    {
                                        "key": "gen_ai.conversation.id",
                                        "value": {"stringValue": "session-1"},
                                    },
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }

    trajectories = import_otel_traces(payload)
    assert len(trajectories) == 1
    event = trajectories[0].events[0]
    assert event.kind in {"tool_call", "tool_result"}
    assert event.tool_name == "read_file"
    assert event.agent_name == "general-purpose"
    assert event.session_id == "session-1"


def test_import_otel_traces_maps_invocation_and_call_llm_names() -> None:
    payload = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "trace-a",
                                "spanId": "span-a",
                                "name": "invocation",
                                "startTimeUnixNano": "1",
                                "attributes": [],
                            },
                            {
                                "traceId": "trace-a",
                                "spanId": "span-b",
                                "name": "call_llm",
                                "startTimeUnixNano": "2",
                                "attributes": [],
                            },
                        ]
                    }
                ]
            }
        ]
    }

    trajectories = import_otel_traces(payload)
    assert len(trajectories) == 1
    kinds = [event.kind for event in trajectories[0].events]
    assert kinds == ["agent_turn", "model_call"]


def test_import_otel_traces_tolerates_missing_shapes() -> None:
    trajectories = import_otel_traces({"resourceSpans": [{"scopeSpans": [{"spans": [None]}]}]})
    assert trajectories == []


def test_import_otel_traces_parses_scalar_attributes() -> None:
    payload = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "trace-attrs",
                                "spanId": "span-attrs",
                                "name": "unknown span",
                                "startTimeUnixNano": "1",
                                "attributes": [
                                    {"key": "flag", "value": {"boolValue": True}},
                                    {"key": "count", "value": {"intValue": "7"}},
                                    {"key": "ratio", "value": {"doubleValue": 0.5}},
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }

    trajectories = import_otel_traces(payload)
    assert len(trajectories) == 1
    attrs = trajectories[0].events[0].attributes
    assert attrs["flag"] is True
    assert attrs["count"] == 7
    assert attrs["ratio"] == 0.5


def test_import_otel_output_json_round_trip_smoke(tmp_path) -> None:
    payload = {"resourceSpans": []}
    p = tmp_path / "otel.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    assert p.exists()


def test_load_trajectories_jsonl_reads_valid_rows(tmp_path) -> None:
    path = tmp_path / "trajectories.jsonl"
    path.write_text(
        '{"trace_id":"t1","events":[]}\n{not-json}\n',
        encoding="utf-8",
    )

    trajectories = load_trajectories_jsonl(path)
    assert len(trajectories) == 1
    assert trajectories[0].trace_id == "t1"


def test_load_otel_json_parses_jsonl_lines(tmp_path) -> None:
    path = tmp_path / "otel.json"
    path.write_text(
        '{"resourceSpans":[{"scopeSpans":[{"spans":[{"traceId":"t1","spanId":"s1"}]}]}]}\n'
        '{"resourceSpans":[{"scopeSpans":[{"spans":[{"traceId":"t2","spanId":"s2"}]}]}]}\n',
        encoding="utf-8",
    )

    payload = load_otel_json(path)
    spans = payload.get("resourceSpans")
    assert isinstance(spans, list)
    assert len(spans) == 2
