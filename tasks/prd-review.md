# PRD: Close All Review Gaps (REVIEW.md §4.1–§4.10)

## 1. Introduction/Overview

The `adk-deepagents` project has reached ~88% feature parity with the original LangChain `deepagents` library. A comprehensive review (REVIEW.md) identified 10 remaining gaps. This PRD covers implementing **all 10 gaps** to bring feature parity to ≥95% and establishing an integration test suite that validates end-to-end agent behavior using a real LLM via OpenCode Zen.

**Problem:** The remaining gaps include missing backends (StoreBackend), missing tool features (image support), incorrect defaults (model-aware summarization), incomplete features (HITL, async backends, SubAgentSpec), missing extensibility (custom middleware), and no integration tests — all of which limit production readiness.

---

## 2. Goals

- Increase feature parity score from ~88% to ≥95%.
- Implement all 10 gaps documented in REVIEW.md §4.1–§4.10.
- All new code passes unit tests, `ruff check`, `ruff format`, and `ty check`.
- Integration tests validate end-to-end agent behavior using a real LLM (GLM-5 via OpenCode Zen).
- At least 4 integration test scenarios covering filesystem round-trip, sub-agent delegation, summarization trigger, and memory loading.

---

## 3. User Stories

### US-001: StoreBackend (Cross-Thread Persistence)

**Description:** As a developer building multi-session agents, I want a `StoreBackend` that persists files across different conversation sessions so that agents can share data across threads.

**Acceptance Criteria:**
- [ ] `StoreBackend` class exists in `adk_deepagents/backends/store.py`
- [ ] Implements the full `Backend` ABC (ls, read, write, edit, grep, glob)
- [ ] Uses ADK's artifact storage or an external key-value store for persistence
- [ ] Supports namespace-based isolation (files are scoped by namespace prefix)
- [ ] Files written in session A can be read in session B (cross-thread)
- [ ] `CompositeBackend` can route to `StoreBackend` by path prefix
- [ ] Unit tests in `tests/unit_tests/test_store_backend.py` cover all operations
- [ ] Tests/typecheck/lint passes

### US-002: Image Support in `read_file`

**Description:** As a developer building multimodal agents, I want `read_file` to detect image files and return base64-encoded multimodal content so that the agent can process images.

**Acceptance Criteria:**
- [ ] `read_file` in `tools/filesystem.py` detects image extensions (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`)
- [ ] Image files are read via `backend.download_files()` (binary read)
- [ ] Image content is base64-encoded and returned as a dict with `{"type": "image", "media_type": "image/png", "data": "<base64>"}`
- [ ] Non-image files continue to return text content as before
- [ ] Unit tests cover both image and non-image paths
- [ ] Tests/typecheck/lint passes

### US-003: Model-Aware Summarization Defaults

**Description:** As a developer using models with varying context windows, I want summarization triggers to be computed based on the actual model's context window so that summarization fires at the right time.

**Acceptance Criteria:**
- [ ] `SummarizationConfig` has an optional `context_window: int | None` field
- [ ] `_resolve_context_window()` in `summarization.py` checks config first, then falls back to a model-name lookup table (e.g., `{"gemini-2.5-flash": 1_048_576, "gemini-2.5-pro": 1_048_576, "gpt-4o": 128_000}`)
- [ ] If model is not in lookup and config is not set, falls back to `DEFAULT_CONTEXT_WINDOW`
- [ ] Trigger and keep values are computed as fractions of the resolved context window (85% trigger, 10% keep — matching deepagents defaults)
- [ ] Unit tests verify correct resolution for known models, unknown models, and explicit config
- [ ] Tests/typecheck/lint passes

### US-004: Human-in-the-Loop (True Pause/Resume)

**Description:** As a developer building approval workflows, I want the agent to truly pause execution when a tool requires human approval and resume only after approval is granted so that humans can review and approve/reject tool actions.

**Acceptance Criteria:**
- [ ] `before_tool_callback` in `callbacks/before_tool.py` returns a response that halts the agent's execution loop (not just sets state)
- [ ] The halting mechanism is documented — either raising a custom exception, returning a specific ADK signal, or using ADK's native interrupt/yield mechanism
- [ ] Pending approval state includes: tool name, arguments, and a unique approval ID
- [ ] A `resume_approval(approval_id, approved: bool, modified_args: dict | None)` helper function or pattern is provided
- [ ] When approved, the tool executes with original (or modified) arguments
- [ ] When rejected, the agent receives a rejection message and continues without executing the tool
- [ ] If ADK does not support true interrupts, document the limitation clearly and implement the best available approximation (e.g., polling-based or callback-based)
- [ ] Unit tests cover approve, reject, and modified-args paths
- [ ] Tests/typecheck/lint passes

### US-005: Async Backend Methods

**Description:** As a developer using async tools, I want backend methods to have async equivalents so that async ADK tools can call backends without blocking the event loop.

**Acceptance Criteria:**
- [ ] `Backend` ABC in `backends/protocol.py` defines async methods: `als_info`, `aread`, `awrite`, `aedit`, `agrep_raw`, `aglob_info`
- [ ] Default implementations wrap sync methods via `asyncio.to_thread()` (matching deepagents' pattern)
- [ ] `StateBackend` overrides async methods with direct async implementations (no thread needed — it's in-memory)
- [ ] `FilesystemBackend` uses the default `asyncio.to_thread()` wrappers (I/O-bound)
- [ ] `CompositeBackend` delegates to the resolved backend's async methods
- [ ] Unit tests verify async methods produce the same results as sync methods
- [ ] Tests/typecheck/lint passes

### US-006: Custom Middleware Extensibility

**Description:** As a developer extending `adk-deepagents`, I want to pass custom callback functions to `create_deep_agent()` so that I can add logging, monitoring, or custom prompt injection without modifying the library.

**Acceptance Criteria:**
- [ ] `create_deep_agent()` in `graph.py` accepts an optional `extra_callbacks` parameter
- [ ] `extra_callbacks` is a dict with optional keys: `before_agent`, `before_model`, `before_tool`, `after_tool`
- [ ] Each extra callback is composed with the built-in callback — the built-in runs first, then the extra callback
- [ ] If the built-in callback returns a value that short-circuits (e.g., `before_tool_callback` returning a dict to skip the tool), the extra callback is NOT called
- [ ] Type signature: `extra_callbacks: dict[str, Callable] | None = None`
- [ ] Unit tests verify composition order and short-circuit behavior
- [ ] Tests/typecheck/lint passes

### US-007: SubAgentSpec Missing Fields

**Description:** As a developer defining specialized sub-agents, I want `SubAgentSpec` to support `skills` and `interrupt_on` fields so that sub-agents can independently discover skills and have their own HITL configuration.

**Acceptance Criteria:**
- [ ] `SubAgentSpec` in `types.py` adds optional fields: `skills: list[str]` and `interrupt_on: dict[str, bool]`
- [ ] When `skills` is set on a sub-agent, skill tools are added to that sub-agent (not shared from parent)
- [ ] When `interrupt_on` is set, the sub-agent gets its own `before_tool_callback` for HITL
- [ ] `create_deep_agent()` in `graph.py` processes these new fields when building sub-agents
- [ ] Also support passing pre-built `LlmAgent` instances directly in the `subagents` list (equivalent to deepagents' `CompiledSubAgent`)
- [ ] Unit tests verify sub-agent creation with skills, interrupt_on, and pre-built agents
- [ ] Tests/typecheck/lint passes

### US-008: Skills Silent Failure Fix

**Description:** As a developer, I want a clear error message when I pass `skills=[...]` but the `adk-skills-agent` library is not installed so that I know what dependency I'm missing.

**Acceptance Criteria:**
- [ ] In `graph.py`, when `skills` parameter is provided and `adk-skills-agent` import fails, raise `ImportError` with message: `"adk-skills-agent is required for skills support. Install it with: pip install adk-skills-agent"`
- [ ] When `skills` is NOT provided, the import failure is still silently ignored (library is optional)
- [ ] Unit test verifies the error is raised when skills are requested but import fails
- [ ] Unit test verifies no error when skills are not requested and import fails
- [ ] Tests/typecheck/lint passes

### US-009: `write_file` Error Code Semantics

**Description:** As a developer, I want `write_file` to return `"already_exists"` instead of `"invalid_path"` when a file already exists so that error messages are semantically correct.

**Acceptance Criteria:**
- [ ] `FileOperationError` (or equivalent error type in `backends/protocol.py` or `types.py`) includes `"already_exists"` as a valid error code
- [ ] `StateBackend.write()` returns `error="already_exists"` when a file already exists
- [ ] `FilesystemBackend.write()` returns `error="already_exists"` when a file already exists
- [ ] Existing unit tests are updated to expect the new error code
- [ ] Tests/typecheck/lint passes

### US-010: Integration Tests

**Description:** As a developer, I want end-to-end integration tests that create a real agent and verify tool usage against a real LLM so that I have confidence the system works in production.

**Acceptance Criteria:**
- [ ] Integration tests live in `tests/integration_tests/`
- [ ] Tests use `LiteLlm` from `google.adk.models.lite_llm` configured to hit OpenCode Zen endpoint (`https://opencode.ai/zen/v1/chat/completions`) with model `opencode/glm-5`
- [ ] API key is read from `OPENCODE_API_KEY` environment variable
- [ ] Tests are marked with `@pytest.mark.integration` so they can be run separately via `uv run pytest -m integration`
- [ ] `pyproject.toml` is updated with the `integration` marker definition
- [ ] `litellm` is added to dev dependencies in `pyproject.toml`
- [ ] At minimum, these 4 scenarios are tested:
  - **Filesystem round-trip:** Agent creates a file via `write_file`, reads it back via `read_file`, edits it via `edit_file`, verifies final content
  - **Sub-agent delegation:** Main agent delegates a task to a sub-agent, verifies the sub-agent ran and returned a result
  - **Summarization trigger:** Send enough messages to trigger summarization, verify conversation history is preserved (not lost)
  - **Memory loading:** Agent loads an AGENTS.md file via memory config, verify the content appears in the agent's system prompt
- [ ] Tests are skipped with a clear message if `OPENCODE_API_KEY` is not set
- [ ] Tests/typecheck/lint passes

---

## 4. Functional Requirements

- **FR-1:** The system must provide a `StoreBackend` class that persists file data across sessions using a pluggable storage mechanism (ADK artifacts or key-value store).
- **FR-2:** The `read_file` tool must detect image file extensions and return base64-encoded content with appropriate MIME type metadata.
- **FR-3:** The summarization system must resolve context window size based on (in order): explicit config → model name lookup table → default constant (200,000).
- **FR-4:** The HITL system must halt agent execution when a tool requires approval, expose pending approval state, and support resume with approve/reject/modify.
- **FR-5:** The `Backend` ABC must define async equivalents for all sync methods, with default `asyncio.to_thread()` wrappers.
- **FR-6:** `create_deep_agent()` must accept an `extra_callbacks` parameter that composes user-provided callbacks with built-in callbacks.
- **FR-7:** `SubAgentSpec` must support `skills` and `interrupt_on` fields, and `create_deep_agent()` must support pre-built `LlmAgent` instances in the `subagents` list.
- **FR-8:** When `skills` is explicitly provided to `create_deep_agent()` but `adk-skills-agent` is not installed, the system must raise an `ImportError` with an actionable message.
- **FR-9:** `write()` on `StateBackend` and `FilesystemBackend` must return `error="already_exists"` (not `"invalid_path"`) when the target file already exists.
- **FR-10:** The project must include integration tests using a real LLM (GLM-5 via OpenCode Zen with LiteLLM) that validate agent creation, tool usage, sub-agent delegation, summarization, and memory loading.

---

## 5. Non-Goals (Out of Scope)

- **Not implementing a full database-backed store:** `StoreBackend` will use ADK artifacts or a simple in-memory cross-session dict, not a full SQL/NoSQL backend.
- **Not supporting all image formats:** Only common web formats (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`) are supported.
- **Not building a production HITL UI:** Only the backend mechanism is implemented; no web UI for approvals.
- **Not adding async overrides for `SandboxBackend`:** Only `StateBackend`, `FilesystemBackend`, and `CompositeBackend` get async methods.
- **Not changing the public API of existing tools:** `read_file`, `write_file`, etc. retain their current signatures.
- **Not adding real-time streaming tests:** Integration tests use request-response, not streaming.

---

## 6. Design Considerations

### StoreBackend Storage
Use ADK's `ArtifactService` if available, otherwise implement a simple dict-based store that persists across sessions via a shared reference. The key insight is that `StoreBackend` outlives any single session — it's a shared namespace store.

### Image Support
Follow deepagents' pattern: detect by extension, read binary via `download_files()`, base64-encode, return structured dict. ADK natively supports `types.Part` with `inline_data` for multimodal content.

### HITL Implementation
Research ADK's `before_tool_callback` return semantics thoroughly. If returning a dict/Content from the callback truly skips the tool AND stops the agent loop, that's the mechanism. If ADK continues to the next turn, we may need to use a state-based polling pattern where the agent checks `state["_pending_approval"]` on each turn.

### Integration Test Model Choice
OpenCode Zen provides GLM-5 via an OpenAI-compatible `/chat/completions` endpoint. Use `LiteLlm(model="openai/glm-5")` with `api_base="https://opencode.ai/zen/v1"` and `OPENCODE_API_KEY`. GLM-5 supports tool calling which is required for integration tests.

> **Note:** The model ID `opencode/glm-5` uses the format `opencode/<model-id>` in OpenCode config, but when accessed via LiteLLM's OpenAI-compatible provider, use `openai/glm-5` with the Zen base URL. Verify the exact model ID and endpoint format during implementation.

---

## 7. Technical Considerations

### Dependencies
- `litellm` must be added to dev dependencies for integration tests.
- `pyproject.toml` needs an `integration` pytest marker.
- No new runtime dependencies are required for the gap implementations.

### Async Compatibility
- Async backend methods must be compatible with `pytest-asyncio` (asyncio_mode = "auto").
- `asyncio.to_thread()` requires Python 3.9+ (project requires 3.11+, so this is fine).

### Backwards Compatibility
- `write_file` error code change from `"invalid_path"` to `"already_exists"` is a breaking change for anyone parsing error codes. This is acceptable since the project is pre-1.0.
- All new parameters to `create_deep_agent()` are optional with `None` defaults.
- `SubAgentSpec` uses `total=False` so new fields are optional.

### ADK Constraints
- `after_tool_callback` does NOT receive the tool's return value — this is an ADK limitation that affects large result eviction. Already mitigated via inline truncation (no change needed).
- HITL implementation depends on ADK's callback return semantics — may need research spike during implementation.

---

## 8. Success Metrics

- **Feature parity score** increases from ~88% to ≥95% (as assessed against REVIEW.md criteria).
- **All implemented gaps** have passing unit tests (target: 30+ new unit tests across all gaps).
- **Lint/typecheck clean:** `ruff check .` and `ty check` produce no errors.
- **Integration tests pass:** At least 4 integration test scenarios pass against GLM-5 via OpenCode Zen.
- **No regressions:** All existing 231 unit tests continue to pass.

---

## 9. Open Questions

1. **GLM-5 tool calling support:** Does GLM-5 on OpenCode Zen reliably support function/tool calling? If not, which alternative free or low-cost model on Zen should be used for integration tests? (Candidates: `kimi-k2.5-free`, `minimax-m2.5-free`, `big-pickle`)
2. **ADK HITL interrupt mechanism:** Does ADK's `before_tool_callback` returning a Content/dict truly halt the agent loop, or does it just skip the tool and continue to the next LLM turn? This determines whether true pause/resume is possible.
3. **ADK ArtifactService cross-session:** Can ADK's `ArtifactService` be used as a shared store across sessions, or is it session-scoped? This determines `StoreBackend` implementation approach.
4. **LiteLLM + OpenCode Zen auth:** Does `LiteLlm(model="openai/glm-5", api_base="https://opencode.ai/zen/v1")` work with `OPENCODE_API_KEY` passed as the API key, or does it need a different auth header?
