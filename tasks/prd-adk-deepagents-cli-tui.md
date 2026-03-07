# PRD: adk-deepagents CLI TUI (v1)

## 1. Introduction / Overview

`adk-deepagents` currently provides a functional CLI and REPL, but users still need to remember flags,
subcommands, and operational details to run local agent workflows effectively.

This PRD defines a **terminal UI (TUI) for the existing adk-deepagents CLI** that provides a polished,
keyboard-first experience for local developers. The TUI should take inspiration from modern coding CLIs
(e.g., deepagents, pi, amp, claude code) while staying grounded in this repository’s runtime model and
CLI capabilities.

The v1 focus is an interactive front-end over existing CLI workflows: compose runs, watch live progress,
inspect tool activity/artifacts, and resume/replay prior sessions.

---

## 2. Goals

- Reduce CLI memorization burden so a new user can complete an end-to-end local run from the TUI.
- Provide a single interface for run setup, live execution visibility, and post-run inspection.
- Make debugging easier by surfacing tool calls, arguments, outputs/errors, and generated artifacts.
- Preserve and expose session/thread continuity (history + resume/replay).
- Keep architecture incremental so future enhancements can be added without rewriting core runtime logic.

---

## 3. User Stories

### US-001: TUI Shell and Navigation
**Description:** As a local developer, I want a structured TUI shell with predictable navigation so that I can operate the CLI without memorizing commands.

**Acceptance Criteria:**
- [ ] A TUI shell launches from a dedicated entrypoint (e.g., `adk-deepagents tui`) and renders core panels.
- [ ] A command palette (or slash-command style launcher) is available for key actions.
- [ ] Focus can be moved across major panels entirely via keyboard.
- [ ] A visible status bar shows active profile, model, cwd/project, and thread id.
- [ ] Manual terminal verification confirms core navigation works on a clean local setup.
- [ ] Tests/typecheck/lint pass.

### US-002: Guided Run Composer
**Description:** As a new user, I want a guided flow to configure and start runs so that I can execute tasks without remembering CLI flags.

**Acceptance Criteria:**
- [ ] The run composer allows editing prompt/task text, model, agent/profile, and key run options.
- [ ] The UI validates required fields before launch and shows actionable validation messages.
- [ ] The composer can save and load run presets.
- [ ] Starting a run transitions directly into the live run view with the created/resolved thread id.
- [ ] Manual terminal verification confirms a first-time user can launch an end-to-end run without raw CLI flags.
- [ ] Tests/typecheck/lint pass.

### US-003: Live Run Timeline and Logs
**Description:** As an operator, I want live run progress and output in the TUI so that I can track what the agent is doing in real time.

**Acceptance Criteria:**
- [ ] Streaming assistant output is rendered incrementally in the run view.
- [ ] Tool invocation events are shown in a timeline with timestamps and status (started/completed/error).
- [ ] Errors are highlighted and linked to relevant run steps.
- [ ] The run view supports pausing/scrolling through output without losing new events.
- [ ] Manual terminal verification confirms expected output formatting and behavior for long runs.
- [ ] Tests/typecheck/lint pass.

### US-004: Tool Call Inspector and Artifact Viewer
**Description:** As a developer debugging runs, I want to inspect tool calls and artifacts so that I can quickly understand failures and side effects.

**Acceptance Criteria:**
- [ ] Selecting a timeline event opens a detail pane with tool name, arguments, result summary, and error info.
- [ ] File-oriented tool calls can open related artifacts (created/edited files, diffs, or outputs where available).
- [ ] The inspector supports truncation handling for large payloads with explicit “expand” controls.
- [ ] Sensitive values are masked according to existing CLI/tool policies where applicable.
- [ ] Manual terminal verification confirms inspector usability on successful and failing tool calls.
- [ ] Tests/typecheck/lint pass.

### US-005: Session History and Resume/Replay
**Description:** As a returning user, I want to browse and resume/replay prior runs so that I can continue work or reproduce behavior quickly.

**Acceptance Criteria:**
- [ ] A sessions view lists recent threads with metadata (thread id, timestamps, model/profile, status).
- [ ] Users can resume an existing thread as active context.
- [ ] Users can replay a prior prompt/run configuration into a new run.
- [ ] Session listing and selection behavior align with existing thread persistence semantics.
- [ ] Manual terminal verification confirms resume and replay workflows across app restarts.
- [ ] Tests/typecheck/lint pass.

### US-006: Local Settings and Environment Management
**Description:** As a developer, I want to manage local run defaults and environment hints in the TUI so that setup friction is reduced.

**Acceptance Criteria:**
- [ ] A settings view exposes editable defaults (model, agent/profile, safety/approval-related toggles as applicable).
- [ ] The UI shows environment/config source precedence (flag-like explicit value vs env vs persisted default).
- [ ] Changes can be persisted and reflected in subsequent runs.
- [ ] Invalid settings are blocked with clear recovery guidance.
- [ ] Manual terminal verification confirms persisted settings survive process restart.
- [ ] Tests/typecheck/lint pass.

### US-007: Keyboard Shortcuts and Help
**Description:** As a power user, I want discoverable shortcuts so that I can use the TUI efficiently.

**Acceptance Criteria:**
- [ ] Global shortcuts exist for command palette, panel switching, search/filter, and quit/help.
- [ ] A help overlay lists shortcuts and major workflows.
- [ ] Shortcut conflicts are prevented or resolved consistently.
- [ ] The first-launch experience includes concise “how to use this TUI” guidance.
- [ ] Manual terminal verification confirms full workflow can be completed keyboard-only.
- [ ] Tests/typecheck/lint pass.

---

## 4. Functional Requirements

- **FR-1:** The system must provide a dedicated TUI entrypoint integrated with the existing CLI package.
- **FR-2:** The system must present a multi-panel interface supporting chat/task input, run timeline, and detail inspector.
- **FR-3:** The system must provide command-palette or slash-command style action invocation.
- **FR-4:** The system must support guided run configuration (prompt, model, profile/agent, run options).
- **FR-5:** The system must validate run configuration before execution and block invalid submissions.
- **FR-6:** The system must stream run output and tool events in near-real-time.
- **FR-7:** The system must display tool call details (inputs, outputs, errors) for selected events.
- **FR-8:** The system must expose related artifacts for file/tool operations where available.
- **FR-9:** The system must list persisted sessions/threads and allow selecting one as active.
- **FR-10:** The system must support resume and replay workflows from session history.
- **FR-11:** The system must provide a settings surface for default model/profile and relevant local options.
- **FR-12:** The system must persist settings using existing project/CLI configuration mechanisms.
- **FR-13:** The system must include keyboard shortcuts and in-app help/documentation.
- **FR-14:** The system must preserve existing safety/approval semantics and clearly surface blocked/approval-required actions.
- **FR-15:** The system must degrade gracefully in narrow terminals with clear messaging.
- **FR-16:** The system must not break existing non-interactive and interactive CLI behavior.

---

## 5. Non-Goals (Out of Scope)

- Multi-user collaboration, identity, or permission management.
- Remote SaaS backend or cloud synchronization.
- Plugin marketplace/ecosystem in v1.
- Visual DAG/graph editor for agent workflows.
- In-app full code editor / IDE replacement.
- Browser-based web UI.
- Re-architecting core `adk_deepagents` runtime primitives that already work in CLI mode.
- Building features tied to unrelated workflows that are not part of adk-deepagents CLI use cases.

---

## 6. Design Considerations

- Use a **hybrid interaction model**: conversational/task input + structured panels + command palette.
- Keep visual design terminal-native, dense, and keyboard-first.
- Prioritize clarity over decoration: status indicators, error emphasis, deterministic navigation.
- Ensure information hierarchy supports two modes equally well:
  - “I want to run something quickly”
  - “I need to debug what just happened deeply”

---

## 7. Technical Considerations

- Reuse existing modules under `adk_deepagents/cli` and avoid duplicating core run/session logic.
- Standardize on **Textual** (built on **Rich**) as the v1 TUI framework.
- Preserve Python 3.11+ compatibility and existing project quality gates.
- Keep module boundaries aligned with current package structure.
- Ensure TUI rendering/event handling can consume existing streaming events from the runner.
- Maintain compatibility with current thread/session persistence and model/profile resolution behavior.
- Implement keyboard-first interactions using Textual key bindings, modal screens, and focus management.
- Keep rendering and state updates efficient for long-running/streaming sessions (incremental updates, lazy detail loading).
- Plan for incremental rollout (feature flags or staged command exposure) if needed.

---

## 8. Success Metrics

- A new local user can complete one full run from TUI setup to result without manually recalling CLI flags.
- Median time-to-first-successful-run in onboarding tests is reduced versus current CLI baseline.
- Debugging tasks require fewer context switches (fewer direct DB/file inspections outside TUI).
- Resume/replay workflows are used successfully in real local sessions with low error rates.
- User feedback indicates improved confidence in understanding tool activity and failures.

---

## 9. Open Questions

- Should v1 default to opening the run composer, sessions list, or a dashboard/home screen?
- What portability constraints (terminal support, OS quirks) should be documented for Textual-based behavior?
- How much artifact content should be loaded eagerly vs on demand for performance?
- What is the minimal preset format needed for future compatibility?
- Should replay clone prior config exactly, or allow selective parameter overrides before launch?
- What telemetry (if any) is acceptable for measuring success locally without introducing privacy concerns?

---

## 10. Implementation Starter Plan

### 10.1 Initial Package/Module Skeleton

```text
adk_deepagents/cli/tui/
  __init__.py
  app.py                    # Textual App + bootstrapping
  keymap.py                 # global keybindings and help metadata
  state.py                  # app/session/run state models
  actions.py                # command palette action registry
  screens/
    shell.py                # main shell layout + panel composition
    composer.py             # guided run composer modal/screen
    sessions.py             # session history, resume, replay
    settings.py             # local defaults and env precedence
    help.py                 # shortcuts and workflow help overlay
  widgets/
    status_bar.py           # active model/profile/cwd/thread indicators
    timeline.py             # streaming run events
    inspector.py            # selected tool call/event details
    artifacts.py            # artifact browser/viewer
  services/
    runner.py               # adapter to existing run/stream APIs
    sessions.py             # thread/session listing + resume/replay
    settings.py             # persistence + precedence resolution
    artifacts.py            # artifact lookup/loading/truncation policy
```

Suggested tests:

```text
tests/unit_tests/cli/tui/
  test_keymap.py
  test_state.py
  test_actions.py
  test_services_runner.py
  test_services_sessions.py
  test_services_settings.py

tests/integration_tests/cli/
  test_tui_smoke.py
  test_tui_run_flow.py
  test_tui_resume_replay.py
```

### 10.2 Milestone Plan (Incremental)

- **M0 — Shell Foundation (US-001)**
  - Add `adk-deepagents tui` entrypoint, base Textual app, panel layout, status bar, and focus navigation.
  - Deliverable: keyboard-only navigation across shell panels + smoke test.

- **M1 — Guided Run Composer (US-002)**
  - Implement composer screen/modal, validation, and run start wiring.
  - Deliverable: user can configure and start a run without CLI flags.

- **M2 — Live Timeline/Logs (US-003)**
  - Stream assistant output and tool lifecycle events into timeline widget.
  - Deliverable: near-real-time updates with pause/scroll behavior.

- **M3 — Inspector + Artifacts (US-004)**
  - Add detail inspector and artifact viewer with truncation/expand behavior.
  - Deliverable: selected timeline event shows call args/results/errors and related artifacts.

- **M4 — Sessions Resume/Replay (US-005)**
  - Implement session list, thread selection, resume/replay flows.
  - Deliverable: workflows work across restarts with persisted history.

- **M5 — Settings/Environment (US-006)**
  - Build settings screen with precedence display and persistence validation.
  - Deliverable: saved defaults applied to future runs and survive process restart.

- **M6 — Shortcuts/Help + Hardening (US-007)**
  - Finalize shortcut map, help overlay, first-launch guidance, and conflict checks.
  - Deliverable: keyboard-driven UX completion and docs/update notes.

### 10.3 Definition of Done for Starter Execution

- Each milestone maps cleanly to one or more user stories and acceptance criteria.
- For every milestone merge: `uv run ruff format --check .`, `uv run ruff check .`,
  `uv run ty check`, and `uv run pytest -m "not llm"` pass.
- Add manual verification notes per milestone in PR descriptions to capture terminal behavior
  not easily covered by automated tests.
