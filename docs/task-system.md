# Task System Internals

This document describes exactly how sub-agent delegation is implemented in
`adk_deepagents`, including static `AgentTool` delegation and dynamic `task`
delegation.

## Where delegation is wired

Delegation is configured in `create_deep_agent()` (`adk_deepagents/graph.py`):

- `delegation_mode="static"` (default): build one `AgentTool` per sub-agent
- `delegation_mode="dynamic"`: add one dynamic `task` tool
- `delegation_mode="both"`: include both static sub-agent tools and dynamic `task`

Relevant config types:

- `SubAgentSpec` in `adk_deepagents/types.py`
- `DynamicTaskConfig` in `adk_deepagents/types.py`

## Static delegation path

Static delegation is implemented in `adk_deepagents/tools/task.py`:

1. `build_subagent_tools(...)` receives `subagents`, `default_model`, and `default_tools`.
2. A default `general_purpose` sub-agent is prepended unless already present.
3. Each `SubAgentSpec` becomes an `LlmAgent` wrapped in `AgentTool`.
4. Each pre-built `LlmAgent` is wrapped directly in `AgentTool`.

Key behavior:

- sub-agent names are sanitized to valid ADK identifiers
- sub-agent `skills` are resolved and appended to that sub-agent's tools
- sub-agent `interrupt_on` creates a per-sub-agent `before_tool_callback`

## Dynamic delegation path

Dynamic delegation is implemented in `adk_deepagents/tools/task_dynamic.py`.

`create_dynamic_task_tool(...)` returns an async tool named `task` with this
runtime contract:

- inputs: `description`, `prompt`, `subagent_type`, optional `task_id`, optional `model`
- output: dict with `status` plus `task_id`, `subagent_type`, `result`, `function_calls`

`status` is:

- `"completed"` when child execution finishes normally
- `"error"` for validation failures, timeout, runtime failures, and policy blocks

### Dynamic task lifecycle

For each tool call:

1. Validate input and load state trackers.
2. Resolve `subagent_type` against a runtime registry of specs/agents.
3. If `task_id` is provided, resume existing runtime session from in-memory registry.
4. If no `task_id`, spawn a new child `LlmAgent` and create a child ADK session.
5. Execute child run with timeout via `asyncio.wait_for(...)`.
6. Collect child text and function-call names from events.
7. Pull child `files`/`todos` from child session state and copy into parent state.
8. Return structured tool result (`completed`/`error`).

## Guardrails and policies

`DynamicTaskConfig` controls runtime behavior:

- `max_parallel`: max in-flight dynamic tasks per parent state
- `max_depth`: delegation depth cap (`_dynamic_delegation_depth`)
- `timeout_seconds`: per dynamic task run timeout
- `allow_model_override`: controls whether `model=` input is honored

Enforcement is inside `task(...)` before/around child execution.

## State model (persisted vs process-local)

### Persisted in ADK session state

Dynamic task metadata stored in parent session state:

- `_dynamic_tasks`
- `_dynamic_task_counter`
- `_dynamic_parent_session_id`
- `_dynamic_delegation_depth`
- `_dynamic_running_tasks`

These are JSON-serializable values only.

### Process-local registries (not persisted)

- `adk_deepagents/tools/task_dynamic.py`:
  - `_RUNTIME_REGISTRY` maps `<logical_parent_id>:<task_id>` -> child runtime object
- `adk_deepagents/backends/runtime.py`:
  - `_backend_factory_by_session` maps ADK `session_id` -> backend factory

These are intentionally in-memory to avoid sqlite serialization errors in
`adk web` / `adk api_server` session storage.

## Backend propagation in delegated tasks

To keep filesystem behavior consistent in child sessions:

1. Parent session's backend factory is looked up from runtime registry.
2. Child session id is registered with that same backend factory.
3. Filesystem tools resolve backend by:
   - explicit `_backend` in state, else
   - callable `_backend_factory` in state, else
   - runtime backend registry by session id, else
   - `StateBackend` fallback

This avoids storing function objects in session state while preserving backend
behavior across delegation.

## Timeout and failure semantics

Dynamic child failures are converted to tool-level error payloads, not uncaught
exceptions:

- timeout -> `status: "error"` with timeout message
- execution exception -> `status: "error"` with failure message
- partial child output (if any) is returned in `result`

This prevents ADK Web event stream crashes from bubbling timeout errors.

## Known constraints

- Process-local registries are per-process only (no cross-process resume).
- Registry cleanup is currently manual (`clear_session_backend(...)` exists; no
  automatic session-end hook yet).
- `_RUNTIME_REGISTRY` persists entries for resumable `task_id` behavior; it is
  not currently TTL-pruned.

## Test coverage

Primary tests covering this system:

- `tests/unit_tests/test_graph.py`
- `tests/integration_tests/test_agent_creation.py`
- `tests/unit_tests/test_task_dynamic.py`
- `tests/integration_tests/llm/test_dynamic_task_delegation.py`
- `tests/integration_tests/llm/test_deep_research_dynamic.py`
- `tests/integration_tests/test_callback_pipeline.py`
