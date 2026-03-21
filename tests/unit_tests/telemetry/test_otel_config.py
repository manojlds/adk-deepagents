"""Unit tests for OTEL tracing setup helpers."""

from __future__ import annotations

from adk_deepagents.telemetry import otel


def test_resolve_traces_endpoint_uses_explicit(monkeypatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://x:4318/custom")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://x:4318")
    assert otel._resolve_traces_endpoint() == "http://x:4318/custom"


def test_resolve_traces_endpoint_appends_v1_traces(monkeypatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    assert otel._resolve_traces_endpoint() == "http://127.0.0.1:4318/v1/traces"


def test_resolve_traces_endpoint_defaults_to_local_collector(monkeypatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert otel._resolve_traces_endpoint() == "http://127.0.0.1:4318/v1/traces"
