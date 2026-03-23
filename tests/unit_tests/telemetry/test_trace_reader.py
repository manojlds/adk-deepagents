"""Tests for telemetry/trace_reader.py – reading OTEL JSON traces."""

from __future__ import annotations

import json
from pathlib import Path

from adk_deepagents.telemetry.trace_reader import (
    _extract_value,
    _ns_to_ms,
    _parse_attributes,
    read_traces_dir,
    read_traces_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_span(
    trace_id: str,
    span_id: str,
    name: str,
    *,
    parent_span_id: str | None = None,
    start_ns: int = 1_000_000_000,
    end_ns: int = 2_000_000_000,
    attributes: list[dict] | None = None,
    status_code: int = 0,
) -> dict:
    span: dict = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "startTimeUnixNano": start_ns,
        "endTimeUnixNano": end_ns,
        "status": {"code": status_code} if status_code else {},
        "attributes": attributes or [],
    }
    if parent_span_id:
        span["parentSpanId"] = parent_span_id
    return span


def _make_batch(spans: list[dict]) -> str:
    batch = {
        "resourceSpans": [
            {
                "resource": {"attributes": []},
                "scopeSpans": [
                    {
                        "scope": {"name": "gcp.vertex.agent"},
                        "spans": spans,
                    }
                ],
            }
        ]
    }
    return json.dumps(batch)


def _str_attr(key: str, value: str) -> dict:
    return {"key": key, "value": {"stringValue": value}}


def _int_attr(key: str, value: int) -> dict:
    return {"key": key, "value": {"intValue": value}}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_extract_value_types():
    assert _extract_value({"stringValue": "hello"}) == "hello"
    assert _extract_value({"boolValue": True}) is True
    assert _extract_value({"intValue": 42}) == 42
    assert _extract_value({"intValue": "42"}) == 42
    assert _extract_value({"doubleValue": 3.14}) == 3.14
    assert _extract_value({"bytesValue": b"abc"}) == b"abc"
    assert _extract_value({"arrayValue": {"values": [{"stringValue": "a"}, {"intValue": 1}]}}) == [
        "a",
        1,
    ]
    assert _extract_value(
        {"kvlistValue": {"values": [{"key": "k", "value": {"stringValue": "v"}}]}}
    ) == {"k": "v"}
    assert _extract_value({}) is None


def test_parse_attributes():
    attrs = [
        _str_attr("name", "alice"),
        _int_attr("age", 30),
    ]
    result = _parse_attributes(attrs)
    assert result == {"name": "alice", "age": 30}


def test_ns_to_ms():
    assert _ns_to_ms(1_000_000_000, 2_000_000_000) == 1000.0


# ---------------------------------------------------------------------------
# read_traces_file
# ---------------------------------------------------------------------------


def test_read_traces_file_empty(tmp_path: Path):
    f = tmp_path / "empty.json"
    f.write_text("")
    assert read_traces_file(f) == []


def test_read_traces_file_missing(tmp_path: Path):
    assert read_traces_file(tmp_path / "nonexistent.json") == []


def test_read_traces_file_malformed_json(tmp_path: Path):
    f = tmp_path / "bad.json"
    f.write_text("not valid json\n{also bad}\n")
    assert read_traces_file(f) == []


def test_read_traces_file_no_invocation_span(tmp_path: Path):
    span = _make_span("aaa", "01", "some_other_span")
    f = tmp_path / "traces.json"
    f.write_text(_make_batch([span]))
    assert read_traces_file(f) == []


def test_read_traces_file_basic(tmp_path: Path):
    tid = "aaa"
    spans = [
        _make_span(tid, "01", "invocation"),
        _make_span(
            tid,
            "02",
            "invoke_agent",
            parent_span_id="01",
            attributes=[
                _str_attr("gen_ai.agent.name", "test_agent"),
                _str_attr("gen_ai.conversation.id", "session_123"),
            ],
        ),
        _make_span(tid, "03", "call_llm", parent_span_id="02"),
        _make_span(
            tid,
            "04",
            "generate_content",
            parent_span_id="03",
            start_ns=1_100_000_000,
            end_ns=1_500_000_000,
            attributes=[
                _str_attr("gen_ai.request.model", "gemini-2.5-flash"),
                _int_attr("gen_ai.usage.input_tokens", 100),
                _int_attr("gen_ai.usage.output_tokens", 50),
                {
                    "key": "gen_ai.response.finish_reasons",
                    "value": {
                        "arrayValue": {
                            "values": [{"stringValue": "stop"}],
                        }
                    },
                },
            ],
        ),
        _make_span(
            tid,
            "05",
            "execute_tool",
            parent_span_id="02",
            start_ns=1_600_000_000,
            end_ns=1_800_000_000,
            attributes=[
                _str_attr("gen_ai.tool.name", "read_file"),
                _str_attr(
                    "gcp.vertex.agent.tool_call_args",
                    json.dumps({"path": "/test.txt"}),
                ),
                _str_attr(
                    "gcp.vertex.agent.tool_response",
                    json.dumps({"content": "hello"}),
                ),
            ],
        ),
    ]
    f = tmp_path / "traces.json"
    f.write_text(_make_batch(spans))

    trajectories = read_traces_file(f)
    assert len(trajectories) == 1

    traj = trajectories[0]
    assert traj.trace_id == tid
    assert traj.agent_name == "test_agent"
    assert traj.session_id == "session_123"
    assert len(traj.steps) == 1

    step = traj.steps[0]
    assert step.model_call is not None
    assert step.model_call.model == "gemini-2.5-flash"
    assert step.model_call.input_tokens == 100
    assert len(step.tool_calls) == 1
    assert step.tool_calls[0].name == "read_file"


def test_read_traces_file_suffixed_span_names(tmp_path: Path):
    """Span names may include labels like model/agent names after a space."""
    tid = "aaa_suffix"
    spans = [
        _make_span(tid, "01", "invocation"),
        _make_span(
            tid,
            "02",
            "invoke_agent demo_cli",
            parent_span_id="01",
            attributes=[_str_attr("gen_ai.conversation.id", "session_abc")],
        ),
        _make_span(tid, "03", "call_llm", parent_span_id="02"),
        _make_span(
            tid,
            "04",
            "generate_content openai/glm-5",
            parent_span_id="03",
            attributes=[
                _str_attr("gen_ai.request.model", "openai/glm-5"),
                _int_attr("gen_ai.usage.input_tokens", 7),
                _int_attr("gen_ai.usage.output_tokens", 3),
            ],
        ),
    ]
    f = tmp_path / "traces.json"
    f.write_text(_make_batch(spans))

    trajectories = read_traces_file(f)
    assert len(trajectories) == 1

    traj = trajectories[0]
    assert traj.trace_id == tid
    assert traj.agent_name == "demo_cli"
    assert traj.session_id == "session_abc"
    assert len(traj.steps) == 1
    assert traj.steps[0].agent_name == "demo_cli"
    assert traj.steps[0].model_call is not None
    assert traj.steps[0].model_call.model == "openai/glm-5"


def test_read_traces_file_multiple_steps(tmp_path: Path):
    tid = "bbb"
    spans = [
        _make_span(tid, "01", "invocation"),
        _make_span(
            tid,
            "02",
            "invoke_agent",
            parent_span_id="01",
            attributes=[_str_attr("gen_ai.agent.name", "multi")],
        ),
        _make_span(tid, "03", "call_llm", parent_span_id="02"),
        _make_span(
            tid,
            "04",
            "generate_content",
            parent_span_id="03",
            start_ns=1_000_000_000,
            end_ns=1_100_000_000,
            attributes=[
                _str_attr("gen_ai.request.model", "m1"),
                _int_attr("gen_ai.usage.input_tokens", 10),
                _int_attr("gen_ai.usage.output_tokens", 5),
            ],
        ),
        _make_span(
            tid,
            "05",
            "execute_tool",
            parent_span_id="02",
            start_ns=1_200_000_000,
            end_ns=1_300_000_000,
            attributes=[_str_attr("gen_ai.tool.name", "tool_a")],
        ),
        _make_span(tid, "06", "call_llm", parent_span_id="02"),
        _make_span(
            tid,
            "07",
            "generate_content",
            parent_span_id="06",
            start_ns=1_400_000_000,
            end_ns=1_500_000_000,
            attributes=[
                _str_attr("gen_ai.request.model", "m2"),
                _int_attr("gen_ai.usage.input_tokens", 20),
                _int_attr("gen_ai.usage.output_tokens", 10),
            ],
        ),
        _make_span(
            tid,
            "08",
            "execute_tool",
            parent_span_id="02",
            start_ns=1_600_000_000,
            end_ns=1_700_000_000,
            attributes=[_str_attr("gen_ai.tool.name", "tool_b")],
        ),
    ]
    f = tmp_path / "traces.json"
    f.write_text(_make_batch(spans))

    trajectories = read_traces_file(f)
    assert len(trajectories) == 1
    assert len(trajectories[0].steps) == 2
    assert trajectories[0].steps[0].model_call is not None
    assert trajectories[0].steps[0].model_call.model == "m1"
    assert len(trajectories[0].steps[0].tool_calls) == 1
    assert trajectories[0].steps[0].tool_calls[0].name == "tool_a"
    assert trajectories[0].steps[1].model_call is not None
    assert trajectories[0].steps[1].model_call.model == "m2"
    assert len(trajectories[0].steps[1].tool_calls) == 1
    assert trajectories[0].steps[1].tool_calls[0].name == "tool_b"


def test_read_traces_file_tool_error(tmp_path: Path):
    tid = "ccc"
    spans = [
        _make_span(tid, "01", "invocation"),
        _make_span(
            tid,
            "02",
            "invoke_agent",
            parent_span_id="01",
            attributes=[_str_attr("gen_ai.agent.name", "err_agent")],
        ),
        _make_span(
            tid,
            "03",
            "execute_tool",
            parent_span_id="02",
            attributes=[
                _str_attr("gen_ai.tool.name", "bad_tool"),
                _str_attr("error.type", "FileNotFoundError"),
            ],
        ),
    ]
    f = tmp_path / "traces.json"
    f.write_text(_make_batch(spans))

    trajectories = read_traces_file(f)
    assert len(trajectories) == 1
    step = trajectories[0].steps[0]
    assert len(step.tool_calls) == 1
    assert step.tool_calls[0].error == "FileNotFoundError"


def test_read_traces_file_status_ok(tmp_path: Path):
    tid = "ddd"
    spans = [
        _make_span(tid, "01", "invocation", status_code=1),
        _make_span(tid, "02", "invoke_agent", parent_span_id="01"),
    ]
    f = tmp_path / "traces.json"
    f.write_text(_make_batch(spans))

    trajectories = read_traces_file(f)
    assert len(trajectories) == 1
    assert trajectories[0].status == "ok"


def test_read_traces_file_status_error(tmp_path: Path):
    tid = "eee"
    spans = [
        _make_span(tid, "01", "invocation", status_code=2),
        _make_span(tid, "02", "invoke_agent", parent_span_id="01"),
    ]
    f = tmp_path / "traces.json"
    f.write_text(_make_batch(spans))

    trajectories = read_traces_file(f)
    assert len(trajectories) == 1
    assert trajectories[0].status == "error"


def test_read_traces_file_multiple_traces(tmp_path: Path):
    spans_a = [
        _make_span("t1", "01", "invocation"),
        _make_span("t1", "02", "invoke_agent", parent_span_id="01"),
    ]
    spans_b = [
        _make_span("t2", "11", "invocation"),
        _make_span("t2", "12", "invoke_agent", parent_span_id="11"),
    ]
    f = tmp_path / "traces.json"
    f.write_text(_make_batch(spans_a) + "\n" + _make_batch(spans_b))

    trajectories = read_traces_file(f)
    assert len(trajectories) == 2
    trace_ids = {t.trace_id for t in trajectories}
    assert trace_ids == {"t1", "t2"}


# ---------------------------------------------------------------------------
# read_traces_dir
# ---------------------------------------------------------------------------


def test_read_traces_dir(tmp_path: Path):
    spans_a = [
        _make_span("x1", "01", "invocation"),
        _make_span("x1", "02", "invoke_agent", parent_span_id="01"),
    ]
    spans_b = [
        _make_span("x2", "11", "invocation"),
        _make_span("x2", "12", "invoke_agent", parent_span_id="11"),
    ]
    (tmp_path / "file1.json").write_text(_make_batch(spans_a))
    (tmp_path / "file2.json").write_text(_make_batch(spans_b))

    trajectories = read_traces_dir(tmp_path)
    assert len(trajectories) == 2
    trace_ids = {t.trace_id for t in trajectories}
    assert trace_ids == {"x1", "x2"}


def test_read_traces_dir_missing(tmp_path: Path):
    assert read_traces_dir(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# LLM request / response parsing
# ---------------------------------------------------------------------------


def test_read_traces_file_llm_request_response_parsed(tmp_path: Path):
    tid = "fff"
    llm_request = json.dumps({"model": "gemini", "messages": []})
    llm_response = json.dumps({"choices": [{"text": "hi"}]})
    spans = [
        _make_span(tid, "01", "invocation"),
        _make_span(
            tid,
            "02",
            "invoke_agent",
            parent_span_id="01",
            attributes=[_str_attr("gen_ai.agent.name", "parse_agent")],
        ),
        _make_span(tid, "03", "call_llm", parent_span_id="02"),
        _make_span(
            tid,
            "04",
            "generate_content",
            parent_span_id="03",
            attributes=[
                _str_attr("gen_ai.request.model", "gemini"),
                _int_attr("gen_ai.usage.input_tokens", 10),
                _int_attr("gen_ai.usage.output_tokens", 5),
                _str_attr("gcp.vertex.agent.llm_request", llm_request),
                _str_attr("gcp.vertex.agent.llm_response", llm_response),
            ],
        ),
    ]
    f = tmp_path / "traces.json"
    f.write_text(_make_batch(spans))

    trajectories = read_traces_file(f)
    assert len(trajectories) == 1
    mc = trajectories[0].steps[0].model_call
    assert mc is not None
    assert mc.request == {"model": "gemini", "messages": []}
    assert mc.response == {"choices": [{"text": "hi"}]}


def test_read_traces_file_llm_request_response_fallback_to_call_llm(tmp_path: Path):
    tid = "ggg"
    llm_request = json.dumps(
        {
            "model": "gemini",
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        }
    )
    llm_response = json.dumps(
        {
            "content": {
                "role": "model",
                "parts": [
                    {"text": "thinking", "thought": True},
                    {"text": "hello there"},
                ],
            }
        }
    )
    spans = [
        _make_span(tid, "01", "invocation"),
        _make_span(
            tid,
            "02",
            "invoke_agent",
            parent_span_id="01",
            attributes=[_str_attr("gen_ai.agent.name", "parse_agent")],
        ),
        _make_span(
            tid,
            "03",
            "call_llm",
            parent_span_id="02",
            attributes=[
                _str_attr("gcp.vertex.agent.llm_request", llm_request),
                _str_attr("gcp.vertex.agent.llm_response", llm_response),
            ],
        ),
        _make_span(
            tid,
            "04",
            "generate_content",
            parent_span_id="03",
            attributes=[
                _str_attr("gen_ai.request.model", "gemini"),
                _int_attr("gen_ai.usage.input_tokens", 10),
                _int_attr("gen_ai.usage.output_tokens", 5),
            ],
        ),
    ]
    f = tmp_path / "traces.json"
    f.write_text(_make_batch(spans))

    trajectories = read_traces_file(f)
    assert len(trajectories) == 1
    mc = trajectories[0].steps[0].model_call
    assert mc is not None
    assert mc.request == {
        "model": "gemini",
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
    }
    assert mc.response == {
        "content": {
            "role": "model",
            "parts": [
                {"text": "thinking", "thought": True},
                {"text": "hello there"},
            ],
        }
    }
