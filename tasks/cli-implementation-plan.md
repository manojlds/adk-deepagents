# ADK DeepAgents CLI Implementation Plan

## Objective

Build a first-class `adk-deepagents` CLI that mirrors the core user experience of LangChain DeepAgents CLI while staying idiomatic to this repository and Google ADK primitives.

The CLI should support:

1. Interactive chat mode with streaming output.
2. Non-interactive mode for scripting/CI.
3. Persistent threads/sessions with resume/list/delete.
4. Human-in-the-loop approvals for sensitive tools.
5. Agent profiles, model management, and persistent config.
6. Skills and memory discovery from global and project paths.

## Why This Is Needed

Today, `adk-deepagents` has strong agent primitives (`create_deep_agent`, tools, callbacks, backends, dynamic tasking), but no dedicated CLI package. Interactive behavior is currently spread across examples, which is useful for demos but insufficient for day-to-day usage.

Creating a dedicated CLI unlocks:

1. A production-like user workflow for coding/research tasks.
2. Better validation of HITL/session behavior in realistic loops.
3. A direct migration path for users evaluating parity with DeepAgents.

## Current Foundations In This Repo

These components are already implemented and should be reused rather than rebuilt:

1. `adk_deepagents/graph.py`: sync/async factories, execution/browser/tool wiring.
2. `adk_deepagents/callbacks/before_tool.py`: HITL pause/resume behavior and `resume_approval` helper.
3. `adk_deepagents/tools/filesystem.py`: file system tool surface expected by coding agents.
4. `adk_deepagents/execution/local.py`: local shell execution tool.
5. `adk_deepagents/backends/runtime.py`: session-safe backend factory registration.
6. `adk_deepagents/memory.py` + callback stack: memory loading and injection.
7. `adk_deepagents/tools/task.py` and `tools/task_dynamic.py`: sub-agent delegation.

## Product Scope

### In Scope (MVP)

1. Default interactive terminal mode with streaming assistant text.
2. `-n` single-task mode (script-friendly).
3. Session persistence using ADK `SqliteSessionService`.
4. Thread commands: list, resume, delete.
5. HITL approval prompts for configured tools.
6. Agent profile selection (`--agent`) and model override (`--model`).
7. Persistent config file for defaults.
8. Global/project memory + skills directory resolution.

### In Scope (V1+)

1. Rich slash commands (`/model`, `/threads`, `/help`, `/clear`, `/quit`, `/tokens`).
2. Tool call formatting and concise status rendering.
3. Optional diff rendering for file writes/edits.
4. Shell allow-list behavior for non-interactive mode.

### Out Of Scope (Initial Delivery)

1. Full Textual UI parity (virtualized message store, advanced widget framework).
2. Remote sandbox provider matrix parity with DeepAgents CLI.
3. Advanced tracing integrations (LangSmith-style deep linking).
4. Plugin ecosystem for third-party CLI extensions.

## Design Principles

1. Reuse existing `adk_deepagents` APIs; avoid duplicate tool orchestration logic.
2. Keep non-interactive mode deterministic and script-safe.
3. Prefer explicit user consent for destructive operations.
4. Persist conversations by default to make long-lived workflows practical.
5. Keep architecture modular so a richer TUI can be added later without replacing runtime plumbing.

## High-Level Architecture

## CLI Package Layout

Create a new package: `adk_deepagents/cli/`

Proposed modules:

1. `__main__.py`: module entry (`python -m adk_deepagents.cli`).
2. `main.py`: argparse definitions and top-level command dispatch.
3. `config.py`: load/save config (`~/.adk-deepagents/config.toml`) and defaults.
4. `paths.py`: resolve app-data dirs, agent profile dirs, project root, skills/memory paths.
5. `agent_factory.py`: CLI-aware wrapper around `create_deep_agent`/`create_deep_agent_async`.
6. `session_store.py`: session CRUD helpers over ADK `SqliteSessionService`.
7. `interactive.py`: interactive streaming loop and slash command router.
8. `non_interactive.py`: `-n` mode, stdin merge, quiet/no-stream behavior.
9. `event_renderer.py`: terminal rendering for text, tool calls, responses, errors.
10. `approval.py`: HITL request extraction and approve/reject resume flow.
11. `models.py`: typed dataclasses/Pydantic models for CLI config/state.

## Runtime Components

1. Runner: use `google.adk.runners.Runner` (not only `InMemoryRunner`) so we can inject `SqliteSessionService`.
2. Session service: `google.adk.sessions.sqlite_session_service.SqliteSessionService`.
3. Backend default for CLI: `FilesystemBackend(root_dir=<cwd>, virtual_mode=True)`.
4. Model source: user CLI flags + persistent config fallback.

## Configuration And Data Layout

Use `~/.adk-deepagents/` as CLI home:

1. `config.toml`: global config.
2. `sessions.db`: persistent ADK session store.
3. `<agent_name>/AGENTS.md`: per-agent global memory.
4. `<agent_name>/skills/`: per-agent user skills.
5. Optional future files: `recent_models.json`, `warnings.toml`.

Suggested `config.toml` shape:

```toml
[models]
default = "gemini-2.5-flash"
recent = "gemini-2.5-pro"

[warnings]
suppress = []

[cli]
auto_approve = false
```

## Command Surface

Initial command/flag matrix:

1. Default command: interactive mode.
2. `-n <prompt>`: non-interactive single task.
3. `-q`: quiet output (non-interactive only).
4. `--no-stream`: buffer output before printing (non-interactive only).
5. `-a, --agent <name>`: agent profile.
6. `-M, --model <model>`: model override.
7. `-r, --resume [thread_id]`: resume latest or specific thread.
8. `--auto-approve`: disable HITL prompts.
9. `threads list [--agent <name>] [--limit N]`.
10. `threads delete <thread_id>`.
11. `list`: list available agent profiles.
12. `reset --agent <name>`: reset profile memory file to default template.

## Thread And Session Strategy

Leverage ADK sessions as threads:

1. A thread ID maps to ADK `session.id`.
2. On new session, create with metadata in state, for example:
   - `_cli_agent_name`
   - `_cli_model`
   - `_cli_created_at`
3. `threads list` reads sessions, enriches with metadata, sorts by `last_update_time`.
4. Resume behavior:
   - `-r` with no ID: pick most recent session for selected profile.
   - `-r <id>`: validate existence and load.
5. Delete behavior: remove session via session service API.

## Human-In-The-Loop Approval Design

The CLI must bridge ADK confirmation requests and user decisions:

1. Watch streamed events for function calls named `adk_request_confirmation`.
2. Parse payload to extract:
   - original tool call name
   - tool args
   - function call ID to resume
3. Prompt user with `Approve / Reject / Auto-approve` options.
4. Send a new user message containing `function_response` for the confirmation call.
5. Continue `runner.run_async(...)` to resume the interrupted tool execution.

Approval policy defaults:

1. Interactive: prompt unless `--auto-approve`.
2. Non-interactive: default reject for confirmation-requiring tools unless explicitly allowed by flags.
3. Future: command allow-list for shell to support safer automation.

## Interactive Flow

Interactive loop behavior:

1. Start/load session.
2. Read user input line.
3. Handle local slash commands if input starts with `/`.
4. Otherwise send as ADK user content.
5. Stream events and render:
   - assistant text chunks
   - tool call headers
   - tool outputs (compact)
   - confirmation prompts
6. Return to prompt after turn completion.

Initial slash command set:

1. `/help`: command help.
2. `/threads`: list and quick-resume by selecting ID.
3. `/model`: show current model and allow switch.
4. `/clear`: create a new thread and switch to it.
5. `/quit` or `/q`: exit.

## Non-Interactive Flow

Non-interactive mode requirements:

1. Accept prompt via `-n`, stdin pipe, or both (pipe content prepended).
2. Stream or buffer assistant output depending on `--no-stream`.
3. Output only assistant text in quiet mode.
4. Exit codes:
   - `0` success
   - non-zero on runtime/config/model/session failures
5. Confirmation handling:
   - reject-by-default unless approved by flags/policy
   - return clear stderr diagnostics

## Agent Factory Contract

`agent_factory.py` responsibilities:

1. Resolve model from flag > profile config > global default.
2. Resolve memory paths in precedence order.
3. Resolve skills directories in precedence order.
4. Build backend appropriate for CLI mode.
5. Configure `interrupt_on` defaults for high-risk tools.
6. Call `create_deep_agent(...)` or `create_deep_agent_async(...)` as required by selected execution/browser config.
7. Return `(agent, cleanup)` where `cleanup` handles MCP teardown when needed.

## Memory And Skills Resolution

Recommended lookup order:

1. Global profile memory: `~/.adk-deepagents/<agent>/AGENTS.md`.
2. Project memory: `<project_root>/.deepagents/AGENTS.md`.
3. Optional project root memory: `<project_root>/AGENTS.md`.

Skills directory precedence (later can override earlier conflicts):

1. Built-in skills (if shipped in package).
2. Global profile skills: `~/.adk-deepagents/<agent>/skills/`.
3. Shared global alias: `~/.agents/skills/`.
4. Project local: `<project_root>/.deepagents/skills/`.
5. Project alias: `<project_root>/.agents/skills/`.

## Implementation Phases

## Phase 0: RFC + Scaffolding

Deliverables:

1. This planning doc.
2. CLI module skeleton with import-safe placeholders.
3. `pyproject.toml` script entry.

Acceptance criteria:

1. `adk-deepagents --help` works.

## Phase 1: Config + Paths + Profiles

Deliverables:

1. Config load/save with defaults.
2. Profile directory bootstrap logic.
3. Agent list/reset commands.

Acceptance criteria:

1. First-run creates expected directories/files.
2. `list` and `reset` commands pass unit tests.

## Phase 2: Persistent Sessions

Deliverables:

1. SQLite session service wiring.
2. Session/thread CRUD helpers.
3. `threads list/delete` commands.

Acceptance criteria:

1. Can create, list, resume, and delete sessions across process restarts.

## Phase 3: Non-Interactive Mode (MVP Execution)

Deliverables:

1. `-n` mode execution.
2. stdin merge behavior.
3. quiet/no-stream flags.

Acceptance criteria:

1. Non-interactive mode works in shell pipelines.
2. Stable exit codes for success/failure.

## Phase 4: Interactive Streaming Mode

Deliverables:

1. Streaming chat REPL.
2. Basic event rendering for text/tools/errors.
3. Slash command core set.

Acceptance criteria:

1. Multi-turn interactive sessions are usable and persistent.

## Phase 5: HITL Approval Loop

Deliverables:

1. Detect confirmation requests in event stream.
2. Prompt/resume path for approve/reject.
3. `--auto-approve` mode.

Acceptance criteria:

1. `write_file`/`edit_file`/`execute` approval flow works end-to-end.

## Phase 6: Model Management And UX Polish

Deliverables:

1. `--model` + `/model` switching.
2. Persistent default model support.
3. Cleaner tool call rendering.

Acceptance criteria:

1. Model switching works without losing thread continuity.

## Phase 7: Docs And Hardening

Deliverables:

1. `docs/cli.md` usage guide.
2. README updates.
3. Integration and regression tests.

Acceptance criteria:

1. Documented commands match implemented behavior.
2. CI passes lint/type/test gates.

## Testing Strategy

## Unit Tests

Add under `tests/unit_tests/cli/`:

1. Arg parsing and flag validation.
2. Config file load/save with defaults and malformed TOML handling.
3. Path resolution and project root detection.
4. Session store utilities (list/filter/sort/metadata extraction).
5. Event rendering formatting helpers.
6. Approval payload parsing and response generation.

## Integration Tests

Add under `tests/integration_tests/cli/`:

1. Non-interactive turn execution over real `Runner` + SQLite sessions.
2. Resume thread and continue multi-turn conversation.
3. HITL pause/resume flow with controlled confirmation input.
4. CLI command invocations (`threads list/delete`, profile switching).

## Manual Smoke Matrix

1. Linux/macOS interactive REPL.
2. `-n` and pipe mode.
3. `--auto-approve` behavior.
4. Profile-specific memory/skills loading.
5. Async cleanup behavior for MCP-enabled configs.

## Security And Safety Considerations

1. Keep destructive tool approval on by default in interactive mode.
2. Reject confirmations by default in non-interactive mode unless policy allows.
3. Avoid shell auto-approval unless explicit allow-list is configured.
4. Use `FilesystemBackend(..., virtual_mode=True)` rooted at selected workspace to reduce accidental file traversal.
5. Redact obvious sensitive values in rendered logs when possible.

## Risks And Mitigations

1. Risk: ADK confirmation resume wiring is subtle.
   Mitigation: implement and test approval flow in isolation before polishing UI.
2. Risk: session metadata conventions drift.
   Mitigation: centralize metadata keys in one module with constants.
3. Risk: async cleanup leaks (MCP/browser).
   Mitigation: enforce structured cleanup in both interactive and non-interactive runners.
4. Risk: scope creep toward full Textual parity.
   Mitigation: ship modular REPL first, treat rich TUI as a separate follow-up project.

## Milestone-Based Delivery Plan

1. Milestone A: CLI skeleton + config + session persistence.
2. Milestone B: Non-interactive mode + stable command UX.
3. Milestone C: Interactive streaming + slash commands.
4. Milestone D: HITL end-to-end approvals + auto-approve policy.
5. Milestone E: Model/profile polish + docs + release hardening.

## Definition Of Done

The CLI initiative is done when all of the following are true:

1. `adk-deepagents` command is discoverable, documented, and installable.
2. Users can run both interactive and non-interactive workflows reliably.
3. Session persistence and thread management commands are stable.
4. HITL approvals work end-to-end for configured tools.
5. Profile + model defaults persist between runs.
6. Test coverage exists for parser/config/session/HITL critical paths.
7. Core docs (`README`, `docs/cli.md`) reflect real behavior.

## Open Questions

1. Should we adopt Textual early, or lock in the backend contract with a plain REPL first?
2. Which exact tools should be interrupted by default in CLI mode?
3. Should non-interactive mode ever auto-approve by default for a safe command subset?
4. Do we want per-project `.adk-deepagents.toml` overrides in v1?
5. Should thread IDs be ADK UUIDs or user-facing shortened aliases?
