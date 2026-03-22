{ pkgs, ... }:

{
  packages = [
    pkgs.opentelemetry-collector-contrib
    pkgs.temporal-cli
    pkgs.uv
  ];

  env = {
    ADK_DEEPAGENTS_TEMPORAL_TARGET_HOST = "127.0.0.1:7233";
    ADK_DEEPAGENTS_TEMPORAL_NAMESPACE = "default";
    ADK_DEEPAGENTS_TEMPORAL_TASK_QUEUE = "adk-deepagents-tasks";
    ADK_DEEPAGENTS_TEMPORAL_WORKER_HEALTH_PORT = "17451";
    OTEL_EXPORTER_OTLP_ENDPOINT = "http://127.0.0.1:4318";
    OTEL_EXPORTER_OTLP_PROTOCOL = "http/protobuf";
    OTEL_SERVICE_NAME = "adk-deepagents";
  };

  processes.temporal-server.exec = ''
    mkdir -p .devenv/state/temporal
    temporal server start-dev \
      --ip 127.0.0.1 \
      --port 7233 \
      --namespace default \
      --db-filename .devenv/state/temporal/temporal.db
  '';

  processes.temporal-worker.exec = ''
    exec uv run python -m adk_deepagents.temporal.dev_worker \
      --health-port 17451
  '';

  processes.otel-collector.exec = ''
    mkdir -p .devenv/state/otel
    cat > .devenv/state/otel/collector-config.yaml <<'EOF'
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
          http:
            endpoint: 0.0.0.0:4318

    processors:
      batch: {}

    exporters:
      debug:
        verbosity: basic
      file:
        path: .devenv/state/otel/traces.json

    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [batch]
          exporters: [debug, file]
    EOF

    ${pkgs.opentelemetry-collector-contrib}/bin/otelcol-contrib \
      --config .devenv/state/otel/collector-config.yaml
  '';

  scripts.temporal-reset.exec = ''
    rm -rf .devenv/state/temporal
    mkdir -p .devenv/state/temporal
    echo "Temporal state reset: .devenv/state/temporal"
  '';

  scripts.otel-reset.exec = ''
    rm -rf .devenv/state/otel
    mkdir -p .devenv/state/otel
    echo "OTEL state reset: .devenv/state/otel"
  '';
}
