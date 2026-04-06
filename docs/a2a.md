# A2A Integration

`adk-deepagents` supports Agent2Agent (A2A) in two ways:

1. expose a deep agent as an A2A server endpoint
2. use an external A2A agent as the backend for dynamic `task()` delegation

## Install

```bash
uv pip install "adk-deepagents[a2a]"
```

This installs `a2a-sdk` (HTTP server/client support) used by the integrations.

## Expose a deep agent via A2A

Create your deep agent, then wrap it with `to_a2a_app(...)`:

```python
from adk_deepagents import create_deep_agent, to_a2a_app

agent = create_deep_agent(name="deep_a2a")
app = to_a2a_app(agent, host="0.0.0.0", port=8000)
```

Run with uvicorn:

```bash
uv run uvicorn my_agent_module:app --host 0.0.0.0 --port 8000
```

The app serves the A2A RPC endpoint and agent card routes expected by the SDK.

## Use A2A for dynamic task delegation

Configure dynamic delegation with `A2ATaskConfig`:

```python
from adk_deepagents import A2ATaskConfig, DeepAgentConfig, DynamicTaskConfig, create_deep_agent

agent = create_deep_agent(
    config=DeepAgentConfig(
        delegation_mode="dynamic",
        dynamic_task_config=DynamicTaskConfig(
            a2a=A2ATaskConfig(
                agent_url="http://127.0.0.1:8000",
                timeout_seconds=120.0,
            )
        ),
    ),
)
```

When enabled, `task()` calls dispatch delegated turns to the configured A2A
agent URL instead of creating an in-process child runtime.

## Notes

- `task_id` is mapped to A2A `context_id` for delegated turn continuity.
- Temporal (`DynamicTaskConfig.temporal`) and A2A (`DynamicTaskConfig.a2a`) are
  both external execution backends; if both are configured, Temporal is used.

## Structured artifact contract (optional)

For richer dynamic task parity, the remote A2A agent can return structured
payloads in artifact parts using this schema key:

- `schema`: `adk_deepagents.dynamic_task_result.v1`

Supported fields (top-level or under `payload`) are:

- `result` (string)
- `function_calls` (list of tool names)
- `files` (object)
- `todos` (array)
- `error` (string)

Example:

```json
{
  "schema": "adk_deepagents.dynamic_task_result.v1",
  "payload": {
    "result": "Done",
    "function_calls": ["glob", "read_file"],
    "files": {
      "/summary.md": {
        "content": "..."
      }
    },
    "todos": [
      {
        "content": "Ship changes",
        "status": "completed",
        "priority": "medium"
      }
    ]
  }
}
```

If no structured payload is present, `adk-deepagents` falls back to plain-text
artifact/message extraction for `result`.

## LLM integration test mode

The shared integration test helpers support an A2A bridge mode so existing LLM
tests can execute through `to_a2a_app(...)` instead of direct in-process
`Runner.run_async(...)` calls.

Enable it with:

```bash
ADK_DEEPAGENTS_LLM_TRANSPORT=a2a uv run pytest -m llm
```

This keeps the same test expectations while exercising the A2A request/response
path end-to-end in-process.
