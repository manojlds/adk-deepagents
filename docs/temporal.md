# Temporal Dynamic Task Backend

`adk-deepagents` can run dynamic `task()` delegation on [Temporal](https://temporal.io/)
instead of in-process `InMemoryRunner` child sessions.

This is useful when you want:

- durable delegated task state across process restarts
- external worker processes for sub-agent execution
- queue-based task routing and operational visibility

## Install

```bash
uv pip install "adk-deepagents[temporal]"
```

## Configure the agent

```python
from adk_deepagents import create_deep_agent
from adk_deepagents.types import DynamicTaskConfig, TemporalTaskConfig

agent = create_deep_agent(
    delegation_mode="dynamic",
    dynamic_task_config=DynamicTaskConfig(
        timeout_seconds=90,
        temporal=TemporalTaskConfig(
            target_host="127.0.0.1:7233",
            namespace="default",
            task_queue="adk-deepagents-tasks",
        ),
    ),
)
```

When `DynamicTaskConfig.temporal` is set, dynamic sub-agent turns are dispatched
through Temporal workflows. Runtime specializations created with
`register_subagent` are included in serialized task snapshots so workers can
honor custom descriptions/prompts/models/tool selections. The client also
tracks a stable spec hash and avoids re-sending unchanged full spec payloads
on every workflow update.

## Run Temporal services with devenv

This repository includes a `devenv.nix` profile with:

- `temporal-server`
- `temporal-worker`
- `otel-collector`

Quick start:

```bash
# Start Temporal server + worker
devenv up temporal-server temporal-worker

# Or start all local support services
devenv up
```

Defaults:

- gRPC: `127.0.0.1:7233`
- namespace: `default`
- task queue: `adk-deepagents-tasks`
- worker health probe: `127.0.0.1:17451`
- SQLite DB: `.devenv/state/temporal/temporal.db`

The `temporal-worker` process runs:

```bash
uv run python -m adk_deepagents.temporal.dev_worker
```

The devenv process passes `--health-port 17451` so local supervisors can do a
simple liveness probe.

and picks model in this order:

1. `ADK_DEEPAGENTS_TEMPORAL_WORKER_MODEL`
2. `ADK_DEEPAGENTS_MODEL`
3. `gemini-2.5-flash`

When these `ADK_DEEPAGENTS_TEMPORAL_*` variables are present, CLI/TUI runs
also auto-enable Temporal-backed dynamic tasks.

If you use vaibhav, one command starts the same stack:

```bash
vaibhav dev start adk-deepagents
```

## Run a custom Temporal worker

If you need custom tools/subagents/models, build a worker explicitly with
`create_temporal_worker(...)`:

```python
import asyncio

from adk_deepagents.temporal.worker import create_temporal_worker
from adk_deepagents.types import DynamicTaskConfig, TemporalTaskConfig


async def main():
    config = DynamicTaskConfig(
        temporal=TemporalTaskConfig(
            target_host="127.0.0.1:7233",
            namespace="default",
            task_queue="adk-deepagents-tasks",
        )
    )

    worker = await create_temporal_worker(
        default_model="gemini-2.5-flash",
        dynamic_task_config=config,
    )
    await worker.run()


asyncio.run(main())
```

## Integration tests

Non-LLM Temporal plumbing test (starts local Temporal test env automatically):

- `tests/integration_tests/test_temporal_workflow_local_env.py`

```bash
uv run pytest tests/integration_tests/test_temporal_workflow_local_env.py
```

Real LLM + Temporal integration test:

- `tests/integration_tests/llm/test_temporal_dynamic_task_delegation.py`

```bash
uv run pytest -m llm tests/integration_tests/llm/test_temporal_dynamic_task_delegation.py
```

The test first checks if Temporal is reachable at the configured host. If not,
it attempts to auto-start `devenv up temporal-server` for the test duration and
cleans it up afterwards. If neither is possible (e.g., no reachable server and
no `devenv`), the test is skipped with guidance.
