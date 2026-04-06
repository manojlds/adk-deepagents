# A2A Tasks Example

This example shows **A2A + dynamic task delegation** end to end:

- a worker deep agent exposed as an A2A server
- an orchestrator deep agent that uses `task()` with `DynamicTaskConfig.a2a`

## What this demonstrates

1. Exposing a deep agent via `to_a2a_app(...)`.
2. Delegating dynamic `task()` calls to an external A2A endpoint.
3. Reusing `task_id` across turns for delegated-session continuity.

## Prerequisites

```bash
uv sync --extra a2a
```

Set model credentials (for example via litellm):

```bash
export LITELLM_MODEL=openai/gpt-4o-mini
export OPENAI_API_KEY=your-key
```

## 1) Start the A2A worker server

```bash
export A2A_PORT=8000
uv run python -m examples.a2a_tasks.agent_server
```

This starts a Starlette A2A app via uvicorn at `http://127.0.0.1:8000`.

## 2) Start the orchestrator client

In another terminal:

```bash
export A2A_AGENT_URL=http://127.0.0.1:8000
uv run python -m examples.a2a_tasks.agent_client
```

## 3) Try delegated tasks

Prompt examples:

- `Use task to write /notes.txt with content hello from a2a and confirm.`
- `Now call task again with task_id task_1 and read /notes.txt.`
- `Use task to solve: what is 144 divided by 12?`

The second prompt should continue the same delegated context (`task_1`).

## Notes

- `task_id` maps to A2A `context_id` in this integration.
- If `temporal` and `a2a` are both configured, Temporal takes precedence.
