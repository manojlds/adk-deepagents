"""OpenTelemetry tracing setup for ADK runtime spans."""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

log = logging.getLogger("adk_deepagents.telemetry")

_INITIALIZED = False


def _resolve_traces_endpoint() -> str:
    explicit = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip()
    if explicit:
        return explicit

    base = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not base:
        return "http://127.0.0.1:4318/v1/traces"

    if base.endswith("/"):
        base = base[:-1]

    parsed = urlparse(base)
    if parsed.path.endswith("/v1/traces"):
        return base
    if parsed.path and parsed.path != "/":
        return f"{base}/v1/traces"
    return f"{base}/v1/traces"


def ensure_otel_tracing_configured() -> bool:
    """Configure OTLP HTTP trace export once per process.

    Returns True when tracing is configured and enabled.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return True

    enabled_raw = os.getenv("ADK_DEEPAGENTS_ENABLE_OTEL", "true").lower()
    if enabled_raw in {"0", "false", "no"}:
        return False

    endpoint = _resolve_traces_endpoint()
    if not endpoint:
        log.debug("OTEL not configured: missing OTEL_EXPORTER_OTLP_ENDPOINT")
        return False

    service_name = os.getenv("OTEL_SERVICE_NAME", "adk-deepagents")

    current_provider = trace.get_tracer_provider()
    if isinstance(current_provider, TracerProvider):
        _INITIALIZED = True
        return True

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _INITIALIZED = True
    log.info("Configured OTEL tracing exporter: %s", endpoint)
    return True
