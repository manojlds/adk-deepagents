"""Read OTEL collector JSON trace files and reconstruct Trajectory objects."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from adk_deepagents.optimization.trajectory import (
    AgentStep,
    ModelCall,
    ToolCall,
    Trajectory,
)

logger = logging.getLogger(__name__)

_STATUS_MAP = {0: "unset", 1: "ok", 2: "error"}


def _span_base_name(name: str) -> str:
    """Return the canonical span operation name without suffix labels."""
    return name.split(" ", 1)[0] if name else ""


def _span_suffix(name: str, base_name: str) -> str | None:
    """Return a span name suffix when formatted as ``"{base_name} <suffix>"``."""
    prefix = f"{base_name} "
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :].strip()
    return suffix or None


def _extract_value(v: dict) -> Any:
    """Extract concrete value from an OTLP AnyValue object."""
    if "stringValue" in v:
        return v["stringValue"]
    if "boolValue" in v:
        return v["boolValue"]
    if "intValue" in v:
        return int(v["intValue"])
    if "doubleValue" in v:
        return v["doubleValue"]
    if "bytesValue" in v:
        return v["bytesValue"]
    if "arrayValue" in v:
        return [_extract_value(item) for item in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return {kv["key"]: _extract_value(kv["value"]) for kv in v["kvlistValue"].get("values", [])}
    return None


def _parse_attributes(attrs: list[dict]) -> dict[str, Any]:
    """Convert OTLP attributes array to a flat dict."""
    result: dict[str, Any] = {}
    for attr in attrs:
        key = attr.get("key")
        value = attr.get("value")
        if key is not None and value is not None:
            result[key] = _extract_value(value)
    return result


def _ns_to_ms(start_ns: int, end_ns: int) -> float:
    """Convert nanosecond timestamps to millisecond duration."""
    return (end_ns - start_ns) / 1_000_000


def _safe_json_loads(raw: Any) -> dict[str, Any] | None:
    """Parse a JSON string, returning None on failure."""
    if not isinstance(raw, str):
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    """Best-effort integer coercion for token counters."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _build_trajectory(trace_id: str, spans: list[dict]) -> Trajectory | None:
    """Build a Trajectory from a list of spans belonging to one trace."""
    span_by_id: dict[str, dict] = {}
    invocation_span: dict | None = None
    invoke_agent_spans: list[dict] = []
    generate_content_spans: list[dict] = []
    execute_tool_spans: list[dict] = []

    for span in spans:
        sid = span.get("spanId", "")
        span_by_id[sid] = span
        name = span.get("name", "")
        base_name = _span_base_name(name)

        if base_name == "invocation":
            invocation_span = span
        elif base_name == "invoke_agent":
            invoke_agent_spans.append(span)
        elif base_name == "generate_content":
            generate_content_spans.append(span)
        elif base_name == "execute_tool":
            execute_tool_spans.append(span)

    if invocation_span is None:
        logger.debug("Trace %s has no invocation span, skipping", trace_id)
        return None

    start_ns = int(invocation_span.get("startTimeUnixNano", 0))
    end_ns = int(invocation_span.get("endTimeUnixNano", 0))
    status_code = invocation_span.get("status", {}).get("code", 0)
    status = _STATUS_MAP.get(status_code, "unset")

    # Determine agent_name and session_id from invoke_agent spans.
    agent_name: str | None = None
    session_id: str | None = None
    for ia_span in invoke_agent_spans:
        ia_attrs = _parse_attributes(ia_span.get("attributes", []))
        if agent_name is None:
            agent_name = ia_attrs.get("gen_ai.agent.name") or _span_suffix(
                ia_span.get("name", ""),
                "invoke_agent",
            )
        if session_id is None:
            session_id = ia_attrs.get("gen_ai.conversation.id")

    # Map each invoke_agent span_id so we can group children.
    invoke_agent_by_id: dict[str, dict] = {s["spanId"]: s for s in invoke_agent_spans}

    def _find_invoke_agent_ancestor(span: dict) -> str | None:
        """Walk parent chain to find the owning invoke_agent span."""
        visited: set[str] = set()
        current = span
        while current:
            pid = current.get("parentSpanId", "")
            if not pid or pid in visited:
                return None
            visited.add(pid)
            if pid in invoke_agent_by_id:
                return pid
            current = span_by_id.get(pid)
        return None

    def _find_ancestor_span(span: dict, *, base_name: str) -> dict | None:
        """Walk parent chain to find the nearest ancestor with a base span name."""
        visited: set[str] = set()
        current = span
        while current:
            pid = current.get("parentSpanId", "")
            if not pid or pid in visited:
                return None
            visited.add(pid)
            current = span_by_id.get(pid)
            if current is None:
                return None
            if _span_base_name(current.get("name", "")) == base_name:
                return current
        return None

    # Build ModelCall objects keyed by their span_id.
    model_calls_by_span: dict[str, ModelCall] = {}
    # Track which invoke_agent owns each generate_content span.
    gc_to_agent: dict[str, str] = {}

    for gc_span in sorted(
        generate_content_spans,
        key=lambda s: int(s.get("startTimeUnixNano", 0)),
    ):
        gc_attrs = _parse_attributes(gc_span.get("attributes", []))
        call_llm_span = _find_ancestor_span(gc_span, base_name="call_llm")
        call_llm_attrs = (
            _parse_attributes(call_llm_span.get("attributes", []))
            if call_llm_span is not None
            else {}
        )
        gc_start = int(gc_span.get("startTimeUnixNano", 0))
        gc_end = int(gc_span.get("endTimeUnixNano", 0))

        finish_reasons = gc_attrs.get("gen_ai.response.finish_reasons")
        if finish_reasons is None:
            finish_reasons = call_llm_attrs.get("gen_ai.response.finish_reasons")
        finish_reason: str | None = None
        if isinstance(finish_reasons, list) and finish_reasons:
            finish_reason = str(finish_reasons[0])
        elif isinstance(finish_reasons, str):
            finish_reason = finish_reasons

        request_payload = _safe_json_loads(gc_attrs.get("gcp.vertex.agent.llm_request"))
        if request_payload is None:
            request_payload = _safe_json_loads(call_llm_attrs.get("gcp.vertex.agent.llm_request"))

        response_payload = _safe_json_loads(gc_attrs.get("gcp.vertex.agent.llm_response"))
        if response_payload is None:
            response_payload = _safe_json_loads(call_llm_attrs.get("gcp.vertex.agent.llm_response"))

        model_name = gc_attrs.get("gen_ai.request.model") or call_llm_attrs.get(
            "gen_ai.request.model"
        )
        input_tokens = gc_attrs.get("gen_ai.usage.input_tokens")
        if input_tokens is None:
            input_tokens = call_llm_attrs.get("gen_ai.usage.input_tokens")
        output_tokens = gc_attrs.get("gen_ai.usage.output_tokens")
        if output_tokens is None:
            output_tokens = call_llm_attrs.get("gen_ai.usage.output_tokens")

        mc = ModelCall(
            model=model_name if isinstance(model_name, str) else "",
            input_tokens=_as_int(input_tokens),
            output_tokens=_as_int(output_tokens),
            duration_ms=_ns_to_ms(gc_start, gc_end),
            request=request_payload,
            response=response_payload,
            finish_reason=finish_reason,
        )
        gc_sid = gc_span.get("spanId", "")
        model_calls_by_span[gc_sid] = mc

        ia_id = _find_invoke_agent_ancestor(gc_span)
        if ia_id:
            gc_to_agent[gc_sid] = ia_id

    # Build ToolCall objects and associate with the nearest preceding model call.
    # Group tool calls by invoke_agent ancestor.
    tool_calls_by_agent: dict[str, list[tuple[int, ToolCall]]] = defaultdict(list)

    for et_span in execute_tool_spans:
        et_attrs = _parse_attributes(et_span.get("attributes", []))
        et_start = int(et_span.get("startTimeUnixNano", 0))
        et_end = int(et_span.get("endTimeUnixNano", 0))

        tool_args = _safe_json_loads(et_attrs.get("gcp.vertex.agent.tool_call_args"))
        tool_resp = _safe_json_loads(et_attrs.get("gcp.vertex.agent.tool_response"))

        tc = ToolCall(
            name=et_attrs.get("gen_ai.tool.name", ""),
            args=tool_args if tool_args is not None else {},
            response=tool_resp,
            duration_ms=_ns_to_ms(et_start, et_end),
            error=et_attrs.get("error.type"),
        )

        ia_id = _find_invoke_agent_ancestor(et_span)
        if ia_id:
            tool_calls_by_agent[ia_id].append((et_start, tc))

    # Build AgentStep objects: group generate_content spans by invoke_agent,
    # then attach tool calls that follow each model call.
    steps: list[AgentStep] = []

    # Group model calls by invoke_agent id, preserving order.
    mc_by_agent: dict[str, list[tuple[int, str, ModelCall]]] = defaultdict(list)
    for gc_span in sorted(
        generate_content_spans,
        key=lambda s: int(s.get("startTimeUnixNano", 0)),
    ):
        gc_sid = gc_span.get("spanId", "")
        ia_id = gc_to_agent.get(gc_sid)
        if ia_id and gc_sid in model_calls_by_span:
            gc_start = int(gc_span.get("startTimeUnixNano", 0))
            mc_by_agent[ia_id].append((gc_start, gc_sid, model_calls_by_span[gc_sid]))

    # For each invoke_agent, pair model calls with their subsequent tool calls.
    for ia_span in sorted(
        invoke_agent_spans,
        key=lambda s: int(s.get("startTimeUnixNano", 0)),
    ):
        ia_id = ia_span["spanId"]
        ia_attrs = _parse_attributes(ia_span.get("attributes", []))
        ia_agent_name = (
            ia_attrs.get("gen_ai.agent.name")
            or _span_suffix(ia_span.get("name", ""), "invoke_agent")
            or agent_name
            or ""
        )

        mc_list = mc_by_agent.get(ia_id, [])
        tc_list = sorted(tool_calls_by_agent.get(ia_id, []), key=lambda x: x[0])

        if not mc_list:
            # Agent invocation with no model calls — skip or add empty step.
            if tc_list:
                step = AgentStep(
                    agent_name=ia_agent_name,
                    tool_calls=[tc for _, tc in tc_list],
                )
                steps.append(step)
            continue

        for i, (mc_start, _mc_sid, mc) in enumerate(mc_list):
            # Tool calls belong to this model call if they start after this
            # model call but before the next model call in the same agent.
            next_mc_start = mc_list[i + 1][0] if i + 1 < len(mc_list) else None

            step_tools: list[ToolCall] = []
            for tc_start, tc in tc_list:
                if tc_start >= mc_start and (next_mc_start is None or tc_start < next_mc_start):
                    step_tools.append(tc)

            step = AgentStep(
                agent_name=ia_agent_name,
                model_call=mc,
                tool_calls=step_tools,
            )
            steps.append(step)

    return Trajectory(
        trace_id=trace_id,
        session_id=session_id,
        agent_name=agent_name,
        steps=steps,
        start_time_ns=start_ns,
        end_time_ns=end_ns,
        status=status,
    )


def read_traces_file(path: str | Path) -> list[Trajectory]:
    """Read an OTEL file exporter JSON file and return Trajectory objects.

    Groups spans by trace_id, then reconstructs the trajectory:
    1. Find the ``invocation`` root span
    2. Find ``invoke_agent`` spans to get agent_name and session_id
    3. Group ``generate_content`` and ``execute_tool`` spans under their parent
       ``invoke_agent``
    4. Build AgentStep objects (one model call + its tool calls per step)
    """
    file_path = Path(path)
    if not file_path.exists():
        logger.warning("Trace file does not exist: %s", file_path)
        return []

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Collect all spans grouped by traceId.
    spans_by_trace: dict[str, list[dict]] = defaultdict(list)

    for line_num, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            batch = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSON at %s:%d", file_path, line_num)
            continue

        for resource_span in batch.get("resourceSpans", []):
            for scope_span in resource_span.get("scopeSpans", []):
                for span in scope_span.get("spans", []):
                    trace_id = span.get("traceId", "")
                    if trace_id:
                        spans_by_trace[trace_id].append(span)

    trajectories: list[Trajectory] = []
    for trace_id, spans in spans_by_trace.items():
        traj = _build_trajectory(trace_id, spans)
        if traj is not None:
            trajectories.append(traj)

    return trajectories


def read_traces_dir(directory: str | Path) -> list[Trajectory]:
    """Read all .json files in a directory and return combined trajectories."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        logger.warning("Trace directory does not exist: %s", dir_path)
        return []

    trajectories: list[Trajectory] = []
    for json_file in sorted(dir_path.glob("*.json")):
        trajectories.extend(read_traces_file(json_file))

    return trajectories
