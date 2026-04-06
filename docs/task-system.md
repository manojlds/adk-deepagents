# Task System Internals

This document describes exactly how sub-agent delegation is implemented in
`adk_deepagents`, including static `AgentTool` delegation and dynamic `task`
delegation.

## Where delegation is wired

Delegation is configured in `create_deep_agent()` (`adk_deepagents/graph.py`)
via `config=DeepAgentConfig(...)`:

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
- output: dict with `status` plus `task_id`, `subagent_type`, `result`,
  `function_calls`, and `recovered_runtime`

`create_register_subagent_tool(...)` returns an async tool named
`register_subagent` for runtime specialization:

- inputs: `name`, `description`, optional `system_prompt`, optional `model`,
  optional `tool_names`
- output: dict with `status` (`registered`/`error`) plus normalized
  `subagent_type` metadata

`status` is:

- `"completed"` when child execution finishes normally
- `"error"` for validation failures, timeout, runtime failures, and policy blocks

### Dynamic delegation with skills

When a dynamic child agent is created from a `SubAgentSpec`:

- it starts from the parent `default_tools` snapshot (filesystem/todo/custom tools,
  plus parent-level skills tools if `skills=[...]` is configured on the parent)
- if that sub-agent spec also sets `skills`, those skill tools are discovered and
  appended to the child tool list
- the dynamic `task` tool is not included in this default snapshot, so skill usage
  does not recursively spawn more dynamic tasks unless you explicitly pass `task`
  through `SubAgentSpec.tools`

In other words, skills can *instruct* the model to call `task`, but skills do not
automatically spawn dynamic tasks on their own.

### Dynamic task lifecycle

For each tool call:

1. Validate input and load state trackers.
2. Resolve `subagent_type` against a runtime registry of specs/agents.
3. If `task_id` is provided, resume existing runtime session from in-memory
   registry when available; otherwise recover a fresh runtime from persisted
   task metadata and history snapshots.
4. If no `task_id`, spawn a new child `LlmAgent` and create a child ADK session.
5. Execute child run with timeout via `asyncio.wait_for(...)`.
6. Collect child text and function-call names from events.
7. Pull child `files`/`todos` from child session state and copy into parent state.
8. Return structured tool result (`completed`/`error`).

If `task` is called with an unknown `subagent_type`, a runtime sub-agent is
created automatically using default tools and persisted in session state for
future turns.

## Guardrails and policies

`DynamicTaskConfig` controls runtime behavior:

- `max_parallel`: max in-flight dynamic tasks per parent state
- `concurrency_policy`: behavior when parallel capacity is full (`error` or `wait`)
- `queue_timeout_seconds`: max wait time for a free slot when policy is `wait`
- `max_depth`: delegation depth cap (`_dynamic_delegation_depth`)
- `timeout_seconds`: per dynamic task run timeout
- `allow_model_override`: controls whether `model=` input is honored
- `temporal`: optional `TemporalTaskConfig`; when set, each task turn is
  dispatched to Temporal workflows/workers instead of in-process child sessions
- `a2a`: optional `A2ATaskConfig`; when set, delegated turns are sent to an
  external A2A agent endpoint instead of in-process child sessions

Enforcement is inside `task(...)` before/around child execution.

When `concurrency_policy="wait"`, overflow task calls are queued in-process and
wait for a free concurrency slot instead of immediately returning a
"concurrency limit exceeded" error.

The parent model is also informed of the active dynamic task limits via
before-model system prompt injection (max parallelism, policy, queue timeout)
so it can plan task waves proactively.

## Temporal execution path (optional)

When `DynamicTaskConfig.temporal` is configured:

1. `task()` builds a JSON-serializable task snapshot.
2. `adk_deepagents.temporal.client.run_task_via_temporal()` dispatches that
   snapshot to a deterministic workflow ID (`{prefix}:{parent_id}:{task_id}`).
3. Workflow update `run_turn` executes activity `run_dynamic_task`.
4. The activity runs one delegated sub-agent turn, then returns
   `{result, function_calls, files, todos, timed_out, error}`.

When Temporal is enabled, the parent process does not need to instantiate
in-process child runtimes for execution and can resume `task_id` turns from
persisted metadata alone. Runtime sub-agent specializations registered via
`register_subagent` are serialized into task snapshots and forwarded to the
worker for execution parity. Temporal update calls carry a spec hash and only
send full spec payloads when the specialization changes.

See [Temporal Backend](temporal.md) for setup details.

## A2A execution path (optional)

When `DynamicTaskConfig.a2a` is configured:

1. `task()` sends the delegated prompt to the configured A2A endpoint.
2. `task_id` is passed as A2A `context_id` to preserve turn continuity.
3. The dynamic tool consumes streamed/non-streamed A2A responses and returns
   the final text as `result`.

When the remote endpoint emits structured artifact payloads with schema
`adk_deepagents.dynamic_task_result.v1`, dynamic execution also maps
`function_calls`, `files`, `todos`, and `error` into the returned tool payload.

See [A2A Integration](a2a.md) for setup details.

For integration testing, helper fixtures also support an in-process A2A bridge
mode (`ADK_DEEPAGENTS_LLM_TRANSPORT=a2a`) so the same LLM scenarios can run
through A2A transport rather than direct in-memory runner calls.

## State model (persisted vs process-local)

### Persisted in ADK session state

Dynamic task metadata stored in parent session state:

- `_dynamic_tasks`
- `_dynamic_task_counter`
- `_dynamic_parent_session_id`
- `_dynamic_delegation_depth`
- `_dynamic_running_tasks`

`_dynamic_tasks` now stores enough JSON-serializable metadata to recover a
delegated task runtime after process restarts (sub-agent type/depth,
task-scoped files/todos snapshots, optional model override, and compact turn
history used for resume prompts).

### Process-local registries (not persisted)

- `adk_deepagents/tools/task_dynamic.py`:
  - `_RUNTIME_REGISTRY` maps `<logical_parent_id>:<task_id>` -> child runtime object
- `adk_deepagents/backends/runtime.py`:
  - `_backend_factory_by_session` maps ADK `session_id` -> backend factory

These remain in-memory to avoid sqlite serialization errors in `adk web` /
`adk api_server` session storage. When `_RUNTIME_REGISTRY` entries are missing,
the `task` tool now rehydrates a runtime from persisted `_dynamic_tasks`
metadata.

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

- Runtime recovery rebuilds delegated context from compact prompt/result
  history, not raw full-fidelity child event streams.
- Registry cleanup is currently manual (`clear_session_backend(...)` exists; no
  automatic session-end hook yet).
- `_RUNTIME_REGISTRY` persists entries for active/resumable `task_id` behavior;
  it is not currently TTL-pruned.

## Test coverage

Primary tests covering this system:

- `tests/unit_tests/test_graph.py`
- `tests/integration_tests/test_agent_creation.py`
- `tests/unit_tests/test_task_dynamic.py`
- `tests/integration_tests/llm/test_dynamic_task_delegation.py`
- `tests/integration_tests/llm/test_temporal_dynamic_task_delegation.py`
- `tests/integration_tests/test_temporal_workflow_local_env.py`
- `tests/integration_tests/llm/test_deep_research_dynamic.py`
- `tests/integration_tests/test_callback_pipeline.py`
- `tests/unit_tests/temporal/`
