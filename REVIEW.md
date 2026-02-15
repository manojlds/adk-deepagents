# ADK Deep Agents — Review Against LangChain deepagents

> **Date:** 2026-02-15
> **Scope:** Feature parity review of `adk-deepagents` against
> [langchain deepagents](https://github.com/langchain-ai/deepagents),
> example correctness, and adherence to ADK core primitives.
> **Result:** 231 tests passing, all lint clean. 7 critical/major issues fixed,
> several minor gaps documented below.

---

## Table of Contents

1. [Architecture Mapping Summary](#1-architecture-mapping-summary)
2. [Feature Parity Checklist](#2-feature-parity-checklist)
3. [Critical Issues Found & Fixed](#3-critical-issues-found--fixed)
4. [Remaining Gaps & Recommendations](#4-remaining-gaps--recommendations)
5. [Example-by-Example Review](#5-example-by-example-review)
6. [ADK Primitives Alignment](#6-adk-primitives-alignment)
7. [Test Coverage Assessment](#7-test-coverage-assessment)

---

## 1. Architecture Mapping Summary

The project correctly maps deepagents' middleware-based architecture to ADK
callbacks and tools:

| deepagents Concept | ADK Equivalent (this project) | Status |
|---|---|---|
| `create_deep_agent()` factory | `create_deep_agent()` → `LlmAgent` | ✅ |
| Middleware `wrap_model_call` | `before_model_callback` | ✅ |
| Middleware `wrap_tool_call` | `before_tool_callback` / `after_tool_callback` | ✅ |
| Middleware `before_agent` | `before_agent_callback` | ✅ |
| `BackendProtocol` | `Backend` ABC in `backends/protocol.py` | ✅ |
| `StateBackend` | `StateBackend` → `session.state["files"]` | ✅ |
| `FilesystemBackend` | `FilesystemBackend` (pathlib-based) | ✅ |
| `CompositeBackend` | `CompositeBackend` (path-prefix routing) | ✅ |
| `StoreBackend` (cross-thread) | ❌ Not implemented | Gap |
| `SandboxBackend` (abstract) | `SandboxBackend` ABC exists | ✅ |
| `LocalShellBackend` | `execution/local.py` (subprocess) | ✅ |
| Sandbox execution | Heimdall MCP via `execution/heimdall.py` | ✅ |
| `TodoListMiddleware` | `tools/todos.py` (state-backed) | ✅ |
| `FilesystemMiddleware` | `tools/filesystem.py` (6 tools) | ✅ |
| `SubAgentMiddleware` | `tools/task.py` → `AgentTool` | ✅ |
| `MemoryMiddleware` | `memory.py` + `before_agent_callback` | ✅ |
| `SkillsMiddleware` | `skills/integration.py` → adk-skills lib | ✅ |
| `SummarizationMiddleware` | `summarization.py` + `before_model_callback` | ✅ |
| `PatchToolCallsMiddleware` | `before_agent_callback` + `before_model_callback` | ✅ |
| `AnthropicPromptCachingMiddleware` | Deliberately dropped (Gemini-native) | ✅ By design |
| `HumanInTheLoopMiddleware` | `before_tool_callback` | ✅ |
| LangGraph checkpointer | ADK `SessionService` | ✅ (delegated to ADK) |
| LangGraph store | Not implemented | Gap |

---

## 2. Feature Parity Checklist

### Core Tools

| Tool | deepagents | adk-deepagents | Notes |
|---|---|---|---|
| `write_todos` | ✅ | ✅ | State-backed |
| `read_todos` | ✅ | ✅ | State-backed |
| `read_file` | ✅ (limit=2000, images) | ✅ (limit=2000) | **Fixed:** was 100. Image support missing |
| `write_file` | ✅ (create-only) | ✅ (create-only) | Matches |
| `edit_file` | ✅ (string replace) | ✅ (string replace) | Matches |
| `ls` | ✅ | ✅ | Matches |
| `glob` | ✅ | ✅ | Uses wcmatch with fallback |
| `grep` | ✅ | ✅ | ripgrep + Python fallback |
| `execute` | ✅ (sandbox) | ✅ (local + Heimdall) | Different security model |
| `task` | ✅ (single tool, route by type) | ✅ (one AgentTool per sub-agent) | ADK-native pattern |

### Middleware / Callback Stack

| Feature | deepagents | adk-deepagents | Notes |
|---|---|---|---|
| Todo prompt injection | ✅ | ✅ | Via `before_model_callback` |
| Memory loading (AGENTS.md) | ✅ | ✅ | Via `before_agent_callback` |
| Memory prompt injection | ✅ | ✅ | Via `before_model_callback` |
| Skills discovery & tools | ✅ | ✅ | Via adk-skills library |
| Filesystem prompt injection | ✅ | ✅ | Via `before_model_callback` |
| Sub-agent prompt injection | ✅ | ✅ | Via `before_model_callback` |
| Summarization (LLM-based) | ✅ | ✅ | Structured prompt matches |
| Summarization (inline fallback) | ✅ | ✅ | Text truncation fallback |
| Tool arg truncation | ✅ | ✅ | `TruncateArgsConfig` |
| History offloading to backend | ✅ | ✅ | Append-based with timestamps |
| Dangling tool call patching | ✅ | ✅ | Detect in before_agent, inject in before_model |
| Large result eviction | ✅ (after_tool) | ⚠️ Partial | ADK after_tool has no return value; inline truncation used instead |
| Prompt caching (Anthropic) | ✅ | N/A | Deliberately dropped |
| Human-in-the-loop | ✅ | ✅ | Via `before_tool_callback` |

### Backends

| Backend | deepagents | adk-deepagents | Notes |
|---|---|---|---|
| StateBackend (ephemeral) | ✅ | ✅ | Session state dict |
| FilesystemBackend (local) | ✅ | ✅ | pathlib-based, virtual mode |
| CompositeBackend (routing) | ✅ | ✅ | Longest-prefix match |
| StoreBackend (cross-thread) | ✅ | ❌ | Not implemented |
| SandboxBackend (abstract) | ✅ | ✅ | ABC exists |

### Advanced Features

| Feature | deepagents | adk-deepagents | Notes |
|---|---|---|---|
| Image support in read_file | ✅ (base64) | ❌ | Not implemented |
| Model-aware summarization defaults | ✅ | ⚠️ Partial | Uses fixed defaults |
| Configurable context window | ✅ | ⚠️ | Config exists but `_resolve_context_window()` returns constant |
| Structured output | ✅ (`response_format`) | ✅ (`output_schema`) | ADK-native |
| Multi-model support | ✅ (any LangChain model) | ✅ (any ADK model) | ADK-native, litellm for non-Gemini |
| Async factory for MCP | N/A | ✅ | `create_deep_agent_async()` |

---

## 3. Critical Issues Found & Fixed

### 3.1 Backend Factory Not Wired Into Tool State — **FIXED**

**Severity:** Critical (all filesystem tools broken at runtime)

**Problem:** `create_deep_agent()` computed a `backend_factory` but never
stored it in session state. The filesystem tools in `tools/filesystem.py`
call `_get_backend(tool_context)` which looks for `state["_backend"]` or
`state["_backend_factory"]` — neither was ever set. Every `ls`, `read_file`,
`write_file`, `edit_file`, `glob`, `grep` call would throw:

> "No backend configured. Set state['_backend'] or state['_backend_factory']."

**Fix:**
- `graph.py`: Always pass `backend_factory` to `make_before_agent_callback`
  (was only passed when `memory` was provided).
- `before_agent.py`: Store `backend_factory` in `state["_backend_factory"]`
  at the start of the callback, before memory loading.

**Files changed:** `graph.py`, `callbacks/before_agent.py`

---

### 3.2 General-Purpose Sub-Agent Not Always Available — **FIXED**

**Severity:** Major (parity gap with deepagents)

**Problem:** Sub-agent tools were only created when `subagents is not None`.
If the user called `create_deep_agent()` without passing `subagents`, no
sub-agents existed — including the general-purpose one that deepagents
always provides.

**Fix:** Always call `build_subagent_tools()` with an empty list when
`subagents=None`, ensuring the general-purpose sub-agent is always created.

**Files changed:** `graph.py`

---

### 3.3 Sub-Agent Name Sanitization Mismatch — **FIXED**

**Severity:** Major (LLM told wrong tool names)

**Problem:** Sub-agent descriptions used raw `spec["name"]` (e.g.,
`"general-purpose"`) but ADK requires sanitized names (e.g.,
`"general_purpose"`). The LLM would see docs for `general-purpose` but
the actual callable tool is `general_purpose` — causing tool-not-found
failures.

**Fix:** Apply `_sanitize_agent_name()` when building `subagent_descriptions`
so documentation names match actual ADK tool names.

**Files changed:** `graph.py`

---

### 3.4 Async Heimdall Factory Added Unwanted Local Exec Tool — **FIXED**

**Severity:** Major (security footgun)

**Problem:** `create_deep_agent_async()` set `effective_execution = "local"`
after resolving Heimdall MCP tools. This caused the sync factory to add a
local subprocess `execute` tool ON TOP of the sandboxed Heimdall tools —
undermining the entire sandboxing model.

**Fix:** Changed to `effective_execution = "_resolved"` which signals
`has_execution=True` for prompt injection without matching `"local"` (no
subprocess tool) or `"heimdall"` (no warning).

**Files changed:** `graph.py`

---

### 3.5 Deep Research Example Tool Name Mismatches — **FIXED**

**Severity:** Major (example broken — LLM calls nonexistent tools)

**Problem:** `RESEARCHER_INSTRUCTIONS` in `prompts.py` referenced:
- `internet_search` → actual tool is `web_search`
- `think_tool` → actual tool is `think`

This would cause the researcher sub-agent to repeatedly fail to call the
correct tools.

**Fix:** Updated all 4 occurrences in `prompts.py` to match actual function
names.

**Files changed:** `examples/deep_research/prompts.py`

---

### 3.6 `read_file` Default Limit Too Small — **FIXED**

**Severity:** Moderate (behavioral difference from deepagents)

**Problem:** `read_file` defaulted to `limit=100` lines. deepagents uses
`limit=2000`. 100 lines is too small for most coding tasks (logs, configs,
source files would all be truncated).

**Fix:** Changed default to `limit=2000` matching deepagents.

**Files changed:** `tools/filesystem.py`

---

### 3.7 Sandboxed Coder ADK CLI Path Broken — **FIXED**

**Severity:** Moderate (advertised CLI command doesn't work)

**Problem:** The sync `root_agent = build_agent()` used
`execution="heimdall"` which emits a warning and adds no execution tools.
The ADK CLI (`adk run examples/sandboxed_coder/`) would start an agent
with no execution capability.

**Fix:** Changed sync `build_agent()` to use `execution="local"` with
clear documentation that `build_agent_async()` provides full Heimdall
sandboxing.

**Files changed:** `examples/sandboxed_coder/agent.py`

---

## 4. Remaining Gaps & Recommendations

### 4.1 StoreBackend (Cross-Thread Persistence) — Not Implemented

**deepagents feature:** `StoreBackend` uses LangGraph's `BaseStore` for
persistent storage across threads and conversations. Files organized via
namespaces, supports multi-agent isolation.

**Recommendation:** Implement using ADK's artifact storage or an external
database. Priority depends on whether cross-thread persistence is needed.

**Effort:** M (4–8 hours)

---

### 4.2 Image Support in `read_file` — Not Implemented

**deepagents feature:** `read_file` detects image extensions
(`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`) and returns base64-encoded
content that the LLM can process visually.

**Recommendation:** Add extension detection in `read_file` and return
base64 data with MIME type metadata. ADK's tool infrastructure can carry
structured dicts with image data.

**Effort:** M (2–4 hours)

---

### 4.3 Summarization Config Partially Ignored

**Problem:** `_resolve_context_window()` always returns
`DEFAULT_CONTEXT_WINDOW` (200,000) regardless of config. If a user passes
a custom context window size, it's ignored.

**Recommendation:** Add a `context_window` field to `SummarizationConfig`
and resolve it in `_resolve_context_window()`. Also support token-based
triggers (not just fraction) for full parity.

**Effort:** S (1–2 hours)

---

### 4.4 Memory Path Normalization in Content Builder Example

**Problem:** `memory=["./AGENTS.md"]` with a `FilesystemBackend(virtual_mode=True)`
relies on path normalization working correctly through `download_files()`.
While it likely works (`./ → /AGENTS.md → <root>/AGENTS.md`), it's fragile.

**Recommendation:** Use absolute virtual paths in examples:
`memory=["/AGENTS.md"]`. This is clearer and avoids normalization edge cases.

**Effort:** S (5 minutes)

---

### 4.5 `write_file` Error Code Semantics

**Problem:** `StateBackend.write()` returns `error="invalid_path"` when a
file already exists. The `write_file` tool maps `"invalid_path"` to a
"File already exists" message. This works but is semantically confusing —
`"invalid_path"` could mean other things.

**Recommendation:** Add a dedicated `"already_exists"` error type to the
`FileOperationError` literal, or document the convention clearly.

**Effort:** S (30 minutes)

---

### 4.6 Skills Silent Failure

**Problem:** In `graph.py`, if the `adk-skills-agent` import fails, skills
are silently skipped (`except ImportError: pass`). A user who passes
`skills=[...]` without installing the dependency gets no feedback.

**Recommendation:** Log a warning (as `skills/integration.py` already does)
or raise when `skills` is explicitly provided but the library is missing.

**Effort:** S (15 minutes)

---

### 4.7 No Integration Tests

**Problem:** The `tests/integration_tests/` directory exists but is empty.
All 231 tests are unit tests with mocked ADK primitives. There are no
end-to-end tests that create a real agent, send a message, and verify
tool usage.

**Recommendation:** Add at least:
1. Agent creation → filesystem tool round-trip (write → read → edit → verify)
2. Sub-agent delegation → result synthesis
3. Summarization trigger → verify history is preserved
4. Memory loading → verify prompt injection

These require a `GOOGLE_API_KEY` and should be marked with
`@pytest.mark.skipif` for CI.

**Effort:** L (1–2 days)

---

## 5. Example-by-Example Review

### 5.1 Quickstart (`examples/quickstart/`)

**Status:** ✅ Correct after fixes

- Minimal working example: `create_deep_agent()` with default settings
- Uses `InMemoryRunner` with async event loop
- Interactive REPL pattern is clear and idiomatic
- **After fix 3.1:** Backend is now properly wired, filesystem tools work
- **After fix 3.2:** General-purpose sub-agent is available

**Verdict:** Good introductory example. Works as advertised.

---

### 5.2 Content Builder (`examples/content_builder/`)

**Status:** ✅ Correct (minor path recommendation)

- Demonstrates skills, memory, sub-agents, and FilesystemBackend together
- `AGENTS.md` provides clear role and workflow context
- Skills (`blog-writing`, `social-media`) are well-structured SKILL.md files
  with actionable guidelines
- Researcher sub-agent spec is clear
- `FilesystemBackend(root_dir=".", virtual_mode=True)` is appropriate

**Minor issue:** `memory=["./AGENTS.md"]` should ideally be
`memory=["/AGENTS.md"]` for consistency with the virtual path convention.

**Verdict:** Good showcase of the full feature set. The AGENTS.md and
SKILL.md files are realistic and useful.

---

### 5.3 Deep Research (`examples/deep_research/`)

**Status:** ✅ Correct after fixes

- Faithful port of the deepagents deep_research example
- Orchestrator + parallel researcher sub-agents pattern
- Configurable model (Gemini, OpenAI, Anthropic via litellm)
- Web search with Tavily → DuckDuckGo fallback and full page extraction
- Strategic thinking tool for reflection
- Summarization configured for long research sessions
- Comprehensive README with architecture diagram and usage examples

**After fix 3.5:** Tool names in prompts now match actual function names.

**Strengths:**
- `prompts.py` is well-structured: workflow, delegation strategy, researcher
  instructions are all clearly separated
- `tools.py` web search implementation is robust (Tavily + DuckDuckGo + page
  fetch) with no external dependencies beyond optional `tavily-python`
- Citation consolidation instructions are detailed and actionable
- Multi-model support via `build_agent(model=...)` is clean

**Verdict:** The strongest example. Demonstrates parallel sub-agent delegation,
custom tools, summarization, and multi-model support in a realistic scenario.

---

### 5.4 Sandboxed Coder (`examples/sandboxed_coder/`)

**Status:** ✅ Correct after fixes

- Demonstrates Heimdall MCP sandboxed execution with skills
- Code review skill is detailed and actionable
- Prompts cover coding workflow, execution, testing, and quality
- Async runner properly manages MCP lifecycle (connect → use → cleanup)

**After fix 3.7:** Sync CLI path now uses local execution as working
fallback.

**Strengths:**
- `prompts.py` provides comprehensive coding assistant instructions
- `skills/code-review/SKILL.md` is a realistic, well-structured skill with
  a practical checklist
- README includes clear architecture diagram and security model table
- Both sync (CLI) and async (full Heimdall) paths documented

**Verdict:** Good showcase of execution + skills integration. The dual
sync/async pattern is well-documented.

---

## 6. ADK Primitives Alignment

The project correctly uses ADK core primitives throughout:

| ADK Primitive | Usage | Assessment |
|---|---|---|
| `LlmAgent` | Main agent and sub-agents | ✅ Correct |
| `FunctionTool` (implicit) | All tool functions via auto-wrapping | ✅ Idiomatic |
| `AgentTool` | Sub-agent delegation | ✅ Correct — each sub-agent is a separate AgentTool |
| `ToolContext` | Backend resolution, state access in tools | ✅ Correct |
| `CallbackContext` | State access in callbacks | ✅ Correct |
| `before_agent_callback` | Memory loading, dangling tool patching | ✅ Correct |
| `before_model_callback` | Prompt injection, summarization | ✅ Correct |
| `before_tool_callback` | Human-in-the-loop | ✅ Correct |
| `after_tool_callback` | Large result eviction | ⚠️ Limited (ADK doesn't pass return value) |
| `LlmRequest` manipulation | System instruction + contents modification | ✅ Correct |
| `session.state` | File storage, todos, memory, summarization state | ✅ Correct |
| `InMemoryRunner` | Example interactive loops | ✅ Correct |
| `MCPToolset` | Heimdall MCP integration | ✅ Correct |
| `types.Content` / `types.Part` | Message construction for summaries, patches | ✅ Correct |

### ADK-Specific Design Decisions (correct)

1. **No middleware stack:** deepagents uses ordered middleware; ADK uses
   callbacks. The project correctly composes all middleware logic into
   4 callbacks (`before_agent`, `before_model`, `before_tool`, `after_tool`).

2. **Sub-agents as AgentTool, not transfer:** deepagents uses a single
   `task` tool that routes by `subagent_type`. ADK's `AgentTool` pattern
   creates one tool per sub-agent, which is more ADK-native and gives the
   LLM clearer tool names.

3. **State instead of LangGraph Command:** deepagents uses `Command` objects
   for atomic state updates. ADK uses `tool_context.state` mutations, which
   is the correct ADK pattern.

4. **Prompt caching dropped:** Anthropic-specific `PromptCachingMiddleware`
   is not needed for Gemini. Correct decision.

5. **Backend abstraction retained:** The `Backend` ABC is orthogonal to ADK
   and provides real value for pluggable storage. Correct decision.

---

## 7. Test Coverage Assessment

### Current State

- **231 unit tests**, all passing
- **0 integration tests** (directory exists but empty)
- Lint clean (ruff), type check configured (ty)

### Coverage by Module

| Module | Tests | Assessment |
|---|---|---|
| `backends/state.py` | ✅ `test_state_backend.py` | Good coverage of all operations |
| `backends/filesystem.py` | ✅ `test_filesystem_backend.py` | Good coverage |
| `backends/composite.py` | ✅ `test_composite_backend.py` | Good coverage |
| `backends/protocol.py` | ✅ `test_protocol.py` | ABC validation |
| `tools/filesystem.py` | ✅ `test_filesystem_tools.py` | All 6 tools tested |
| `tools/todos.py` | ✅ `test_todos.py` | Read/write round-trip |
| `skills/integration.py` | ✅ `test_skills_integration.py` | Import errors, config, injection |
| `callbacks/before_agent.py` | ✅ `test_before_agent.py` | Memory loading, dangling calls |
| `callbacks/before_model.py` | ✅ `test_before_model.py` | Prompt injection |
| `callbacks/before_tool.py` | ✅ `test_before_tool.py` | Interrupt logic |
| `callbacks/after_tool.py` | ✅ `test_after_tool.py` | Eviction logic |
| `graph.py` | ✅ `test_graph.py` | Factory function, all parameters |
| `memory.py` | ✅ `test_memory.py` | Load and format |
| `summarization.py` | ✅ `test_summarization.py` | Comprehensive (tokens, partition, LLM summary, truncation, offloading) |
| `execution/local.py` | ✅ `test_local.py` | Subprocess execution |
| `execution/heimdall.py` | ✅ `test_heimdall.py` | MCP connection (mocked) |
| `execution/bridge.py` | ✅ `test_bridge.py` | Script routing |
| `tools/task.py` | ⚠️ Indirect only | Tested via `test_graph.py`, no dedicated tests |
| Examples | ❌ Not tested | No example smoke tests |
| Integration (end-to-end) | ❌ Empty | No real agent interaction tests |

### Recommended Additions

1. **`test_task.py`:** Dedicated tests for `build_subagent_tools()`,
   `_sanitize_agent_name()`, general-purpose sub-agent creation
2. **Example smoke tests:** Import each example's `root_agent` to verify
   it constructs without errors
3. **Integration tests:** End-to-end with real LLM (gated by API key)
