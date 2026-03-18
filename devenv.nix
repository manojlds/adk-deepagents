{ pkgs, ... }:

{
  packages = [
    pkgs.temporal-cli
    pkgs.uv
  ];

  env = {
    ADK_DEEPAGENTS_TEMPORAL_TARGET_HOST = "127.0.0.1:7233";
    ADK_DEEPAGENTS_TEMPORAL_NAMESPACE = "default";
    ADK_DEEPAGENTS_TEMPORAL_TASK_QUEUE = "adk-deepagents-tasks";
  };

  processes.temporal-server.exec = ''
    mkdir -p .devenv/state/temporal
    temporal server start-dev \
      --ip 127.0.0.1 \
      --port 7233 \
      --namespace default \
      --db-filename .devenv/state/temporal/temporal.db
  '';

  scripts.temporal-reset.exec = ''
    rm -rf .devenv/state/temporal
    mkdir -p .devenv/state/temporal
    echo "Temporal state reset: .devenv/state/temporal"
  '';
}
