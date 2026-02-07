# ADK Deep Agents: Implementation Plan

> Re-implementing LangChain's [deepagents](https://github.com/langchain-ai/deepagents) using
> [Google ADK](https://google.github.io/adk-docs/) primitives with close feature compatibility.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Comparison](#2-architecture-comparison)
3. [Concept Mapping: deepagents → Google ADK](#3-concept-mapping)
4. [Module-by-Module Implementation Plan](#4-module-by-module-implementation-plan)
5. [Package Structure](#5-package-structure)
6. [Implementation Phases](#6-implementation-phases)
7. [Key Design Decisions](#7-key-design-decisions)
8. [Examples to Port](#8-examples-to-port)
9. [Testing Strategy](#9-testing-strategy)
10. [Open Questions & Risks](#10-open-questions--risks)

---

## 1. Executive Summary

**deepagents** is LangChain's opinionated, production-ready agent harness built on LangGraph. It
provides out-of-the-box capabilities: planning (todo lists), filesystem operations, shell execution,
sub-agent delegation, conversation summarization, memory (AGENTS.md), and skills (SKILL.md). The
core abstraction is `create_deep_agent()` which wires a middleware stack that decorates a
LangGraph-based agent.

**Google ADK** is Google's agent framework providing `LlmAgent`, workflow agents
(`SequentialAgent`, `ParallelAgent`, `LoopAgent`), sessions/state, callbacks
(`before_model_callback`, `after_model_callback`, `before_tool_callback`, `after_tool_callback`),
function tools with `ToolContext`, multi-agent delegation via `sub_agents` and `AgentTool`, and
built-in tools (Google Search, code execution).

This plan details how to re-implement every deepagents feature using native ADK primitives, mapping
the **middleware pattern** to **callbacks + tools**, the **backend abstraction** to
**session state + artifacts**, and the **sub-agent system** to ADK's **AgentTool delegation**.

---

## 2. Architecture Comparison

### deepagents Architecture

```
create_deep_agent()
├── Model (ChatAnthropic / any LangChain model)
├── Middleware Stack (ordered, composable)
│   ├── TodoListMiddleware      → write_todos / read_todos tools + system prompt
│   ├── MemoryMiddleware        → loads AGENTS.md → system prompt injection
│   ├── SkillsMiddleware        → loads SKILL.md → progressive disclosure in system prompt
│   ├── FilesystemMiddleware    → ls/read/write/edit/glob/grep/execute tools
│   ├── SubAgentMiddleware      → `task` tool that spawns ephemeral sub-agents
│   ├── SummarizationMiddleware → context window management via summarization
│   ├── PromptCachingMiddleware → Anthropic-specific prompt caching
│   └── PatchToolCallsMiddleware→ fixes dangling tool calls
├── Backends (pluggable storage)
│   ├── StateBackend            → ephemeral (LangGraph state)
│   ├── FilesystemBackend       → local filesystem
│   ├── StoreBackend            → LangGraph persistent store
│   ├── CompositeBackend        → route by path prefix
│   ├── LocalShellBackend       → local shell execution
│   └── SandboxBackend          → remote sandbox execution
└── LangGraph Runtime (checkpointer, store, streaming)
```

### Target ADK Architecture

```
create_deep_agent()
├── LlmAgent (Gemini / any supported model)
├── Callbacks (middleware equivalent)
│   ├── before_agent_callback   → memory loading, skills loading, dangling tool patching
│   ├── before_model_callback   → system prompt injection (memory, skills, filesystem, subagent docs)
│   ├── after_model_callback    → (reserved for future use)
│   ├── before_tool_callback    → path validation, interrupt/approval
│   └── after_tool_callback     → large result eviction, state updates
├── Tools (function tools via ToolContext)
│   ├── write_todos / read_todos
│   ├── ls / read_file / write_file / edit_file / glob / grep
│   ├── execute (shell)
│   └── task (sub-agent spawner via AgentTool)
├── State Management
│   ├── session.state           → ephemeral file storage, todos, metadata
│   └── artifacts               → persistent file storage
├── Sub-agents (via AgentTool, not transfer)
│   ├── general-purpose         → default sub-agent with all main agent tools
│   └── custom sub-agents       → user-defined specialists
└── ADK Runtime (Runner, SessionService, streaming)
```

---

## 3. Concept Mapping

| deepagents Concept | Google ADK Equivalent | Notes |
|---|---|---|
| `create_deep_agent()` | `create_deep_agent()` factory → `LlmAgent` | Our top-level factory |
| `AgentMiddleware.wrap_model_call` | `before_model_callback` / `after_model_callback` | Callbacks on `LlmAgent` |
| `AgentMiddleware.wrap_tool_call` | `before_tool_callback` / `after_tool_callback` | Callbacks on `LlmAgent` |
| `AgentMiddleware.before_agent` | `before_agent_callback` | Called before agent runs |
| `BackendProtocol` | Abstract `Backend` class | Our own abstraction |
| `StateBackend` | `session.state` dict | ADK session state |
| `FilesystemBackend` | Custom backend wrapping `os`/`pathlib` | Direct filesystem access |
| `CompositeBackend` | Custom routing backend | Same pattern, different primitives |
| `SandboxBackendProtocol` | Custom backend wrapping subprocess/Docker | Shell execution |
| `FilesystemMiddleware` tools | ADK `FunctionTool`s with `ToolContext` | Tools access state via `ToolContext` |
| `TodoListMiddleware` | ADK `FunctionTool`s + `session.state["todos"]` | State-backed todo tools |
| `SubAgentMiddleware` (`task` tool) | `AgentTool` wrapping sub-`LlmAgent`s | Key architectural difference |
| `SummarizationMiddleware` | `before_model_callback` + custom logic | Manual implementation needed |
| `MemoryMiddleware` | `before_model_callback` + file loading | System prompt injection |
| `SkillsMiddleware` | `before_model_callback` + SKILL.md parsing | Progressive disclosure |
| `PatchToolCallsMiddleware` | `before_agent_callback` | Message history repair |
| `HumanInTheLoopMiddleware` | `before_tool_callback` returning dict to skip | Approval gating |
| `LangGraph checkpointer` | `SessionService` (database-backed) | Persistence |
| `LangGraph store` | ADK artifacts / external DB | Long-term storage |
| `response_format` (structured output) | `output_schema` on `LlmAgent` | JSON schema |
| `interrupt_on` | `before_tool_callback` conditional logic | Tool-specific interrupts |
| Model string `"provider:model"` | ADK model string `"gemini-2.5-flash"` | Model resolution |

---

## 4. Module-by-Module Implementation Plan

### 4.1 Core: `adk_deepagents/__init__.py` + `create_deep_agent()`

**File:** `adk_deepagents/graph.py`

The main factory function that mirrors `deepagents.graph.create_deep_agent()`.

```python
def create_deep_agent(
    model: str = "gemini-2.5-flash",
    tools: list[Callable] | None = None,
    *,
    instruction: str | None = None,
    subagents: list[SubAgentSpec] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    output_schema: type | None = None,
    backend: Backend | BackendFactory | None = None,
    interrupt_on: dict[str, bool] | None = None,
    session_service: SessionService | None = None,
    name: str = "deep_agent",
) -> LlmAgent:
```

**Implementation:**
1. Resolve `backend` (default to `StateBackend`)
2. Build tool list: todo tools + filesystem tools + execute tool + user tools
3. Build sub-agent list: general-purpose + user sub-agents, wrapped as `AgentTool`
4. Compose callbacks from middleware-equivalent functions
5. Build instruction string: base prompt + memory + skills + filesystem docs + subagent docs
6. Create and return configured `LlmAgent`

### 4.2 Backends: `adk_deepagents/backends/`

#### 4.2.1 Protocol: `backends/protocol.py`

Port the `BackendProtocol` abstract class with the same interface:

- `ls_info(path) -> list[FileInfo]`
- `read(file_path, offset, limit) -> str`
- `write(file_path, content) -> WriteResult`
- `edit(file_path, old_string, new_string, replace_all) -> EditResult`
- `grep_raw(pattern, path, glob) -> list[GrepMatch] | str`
- `glob_info(pattern, path) -> list[FileInfo]`
- `upload_files(files) -> list[FileUploadResponse]`
- `download_files(paths) -> list[FileDownloadResponse]`
- `execute(command) -> ExecuteResponse` (for `SandboxBackendProtocol`)

Port all dataclasses: `FileInfo`, `GrepMatch`, `WriteResult`, `EditResult`, `FileDownloadResponse`,
`FileUploadResponse`, `ExecuteResponse`.

#### 4.2.2 State Backend: `backends/state.py`

Maps to **ADK session state** (`session.state`). Files are stored as:

```python
session.state["files"] = {
    "/path/to/file.txt": {
        "content": ["line1", "line2"],
        "created_at": "2025-01-01T00:00:00",
        "modified_at": "2025-01-01T00:00:00",
    }
}
```

**Key difference:** In deepagents, `StateBackend` receives a `ToolRuntime` and reads from
`runtime.state`. In ADK, tools receive `ToolContext` which provides `tool_context.state` for
session state access. The `StateBackend` constructor takes the state dict directly.

#### 4.2.3 Filesystem Backend: `backends/filesystem.py`

Direct port - reads/writes to the local filesystem using `pathlib` and `os`. No ADK-specific
changes needed.

#### 4.2.4 Composite Backend: `backends/composite.py`

Direct port - routes operations by path prefix to different backends.

#### 4.2.5 Local Shell Backend: `backends/local_shell.py`

Port for local shell execution via `subprocess`. Implements `SandboxBackendProtocol.execute()`.

#### 4.2.6 Backend Utilities: `backends/utils.py`

Port utility functions: `format_content_with_line_numbers`, `create_file_data`,
`perform_string_replacement`, `grep_matches_from_files`, `truncate_if_too_long`, etc.

### 4.3 Tools: `adk_deepagents/tools/`

All tools are implemented as Python functions that ADK auto-wraps as `FunctionTool`. Tools that
need state access include a `tool_context: ToolContext` parameter.

#### 4.3.1 Todo Tools: `tools/todos.py`

```python
from google.adk.tools import ToolContext

def write_todos(todos: list[dict], tool_context: ToolContext) -> dict:
    """Write/update the todo list."""
    tool_context.state["todos"] = todos
    return {"status": "success", "count": len(todos)}

def read_todos(tool_context: ToolContext) -> dict:
    """Read the current todo list."""
    return {"todos": tool_context.state.get("todos", [])}
```

#### 4.3.2 Filesystem Tools: `tools/filesystem.py`

Each tool resolves the backend from `tool_context.state["_backend"]` (or a factory), validates
the path, and delegates to the backend:

- `ls(path, tool_context)` → `backend.ls_info(path)`
- `read_file(file_path, tool_context, offset=0, limit=100)` → `backend.read(...)`
- `write_file(file_path, content, tool_context)` → `backend.write(...)`
- `edit_file(file_path, old_string, new_string, tool_context, replace_all=False)` → `backend.edit(...)`
- `glob(pattern, tool_context, path="/")` → `backend.glob_info(...)`
- `grep(pattern, tool_context, path=None, glob=None, output_mode="files_with_matches")` → `backend.grep_raw(...)`
- `execute(command, tool_context)` → `backend.execute(command)`

**State update pattern:** Since ADK tools return dicts (not LangGraph `Command` objects), file
state updates happen through `tool_context.state` mutations:

```python
def write_file(file_path: str, content: str, tool_context: ToolContext) -> dict:
    backend = _get_backend(tool_context)
    result = backend.write(validated_path, content)
    if result.error:
        return {"status": "error", "message": result.error}
    # Update state for StateBackend
    if result.files_update:
        files = tool_context.state.get("files", {})
        files.update(result.files_update)
        tool_context.state["files"] = files
    return {"status": "success", "path": result.path}
```

#### 4.3.3 Sub-agent Task Tool: `tools/task.py`

In deepagents, the `task` tool creates ephemeral sub-agents and invokes them inline. In ADK, we
use `AgentTool` to wrap pre-configured `LlmAgent` instances.

**Approach:** The sub-agents are created at `create_deep_agent()` time and added to the main
agent's `tools` list as `AgentTool` instances. Each `AgentTool` has a unique name matching the
sub-agent spec.

```python
from google.adk.tools import AgentTool

def build_subagent_tools(
    subagents: list[SubAgentSpec],
    default_model: str,
    default_tools: list,
    backend: Backend,
) -> list[AgentTool]:
    tools = []
    for spec in subagents:
        sub_agent = LlmAgent(
            name=spec["name"],
            model=spec.get("model", default_model),
            instruction=spec.get("system_prompt", DEFAULT_SUBAGENT_PROMPT),
            description=spec["description"],
            tools=spec.get("tools", default_tools),
        )
        tools.append(AgentTool(agent=sub_agent))
    return tools
```

**Key difference from deepagents:** In deepagents, there's a single `task` tool that takes
`subagent_type` as a parameter. In ADK, each sub-agent becomes its own `AgentTool` (e.g.,
`general-purpose`, `researcher`). The LLM calls them by name directly. This is actually more
natural in ADK and avoids the routing logic inside a single tool.

**Alternative:** If we want to preserve the single `task` tool pattern exactly, we can implement
a custom `task` function tool that internally invokes sub-agents via `Runner`:

```python
def task(description: str, subagent_type: str, tool_context: ToolContext) -> dict:
    """Launch an ephemeral subagent."""
    subagent = _subagent_registry[subagent_type]
    runner = InMemoryRunner(agent=subagent)
    session = runner.session_service.create_session()
    # Send the task description as user message and collect result
    result = runner.run(session_id=session.id, user_message=description)
    return {"result": result}
```

We should support **both** patterns (individual `AgentTool`s and a unified `task` tool) and let
users choose via configuration.

### 4.4 Callbacks (Middleware Equivalent): `adk_deepagents/callbacks/`

ADK callbacks replace the deepagents middleware stack. Each middleware becomes one or more callback
functions composed together.

#### 4.4.1 Before Agent Callback: `callbacks/before_agent.py`

Composes:
1. **PatchToolCalls** - Scan message history for dangling tool calls and patch them
2. **Memory Loading** - Load AGENTS.md files from backend into state
3. **Skills Loading** - Discover and parse SKILL.md files into state

```python
def make_before_agent_callback(
    memory_sources: list[str] | None,
    skills_sources: list[str] | None,
    backend_factory: BackendFactory,
) -> Callable:
    def before_agent_callback(callback_context: CallbackContext) -> Content | None:
        state = callback_context.state
        backend = backend_factory(state)

        # 1. Patch dangling tool calls
        _patch_dangling_tool_calls(callback_context)

        # 2. Load memory
        if memory_sources and "memory_contents" not in state:
            contents = {}
            for path in memory_sources:
                content = _load_file(backend, path)
                if content:
                    contents[path] = content
            state["memory_contents"] = contents

        # 3. Load skills
        if skills_sources and "skills_metadata" not in state:
            skills = _discover_skills(backend, skills_sources)
            state["skills_metadata"] = skills

        return None  # Continue with normal agent execution
    return before_agent_callback
```

#### 4.4.2 Before Model Callback: `callbacks/before_model.py`

Composes system prompt injections (replaces `wrap_model_call` from all middleware):
1. **Memory** - Inject AGENTS.md content into system prompt
2. **Skills** - Inject skills listing with progressive disclosure
3. **Filesystem** - Inject filesystem tool documentation
4. **Sub-agents** - Inject sub-agent documentation and usage guidelines
5. **Summarization trigger** - Check if summarization is needed, modify messages

```python
def make_before_model_callback(
    memory_sources: list[str] | None,
    skills_sources: list[str] | None,
    has_execution: bool,
    subagent_descriptions: list[dict],
    summarization_config: SummarizationConfig | None,
) -> Callable:
    def before_model_callback(
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        state = callback_context.state

        # Build dynamic system instruction additions
        additions = []

        # Memory injection
        if memory_sources:
            memory_contents = state.get("memory_contents", {})
            additions.append(_format_memory(memory_contents, memory_sources))

        # Skills injection
        if skills_sources:
            skills = state.get("skills_metadata", [])
            additions.append(_format_skills(skills, skills_sources))

        # Filesystem docs
        additions.append(FILESYSTEM_SYSTEM_PROMPT)
        if has_execution:
            additions.append(EXECUTION_SYSTEM_PROMPT)

        # Sub-agent docs
        if subagent_descriptions:
            additions.append(_format_subagent_docs(subagent_descriptions))

        # Modify the system instruction in llm_request
        combined = "\n\n".join(additions)
        _append_to_system_instruction(llm_request, combined)

        # Summarization check
        if summarization_config:
            _maybe_summarize(callback_context, llm_request, summarization_config)

        return None  # Proceed with LLM call
    return before_model_callback
```

#### 4.4.3 After Tool Callback: `callbacks/after_tool.py`

Handles large result eviction (from `FilesystemMiddleware.wrap_tool_call`):

```python
def make_after_tool_callback(
    backend_factory: BackendFactory,
    token_limit: int = 20000,
) -> Callable:
    def after_tool_callback(
        callback_context: CallbackContext,
        tool_context: ToolContext,
        tool_result: dict,
    ) -> dict | None:
        # Check if result exceeds token threshold
        result_str = str(tool_result)
        if len(result_str) > token_limit * 4:  # ~4 chars per token
            backend = backend_factory(callback_context.state)
            file_path = f"/large_tool_results/{tool_context.function_call_id}"
            backend.write(file_path, result_str)
            preview = _create_content_preview(result_str)
            return {
                "status": "result_too_large",
                "saved_to": file_path,
                "preview": preview,
            }
        return None  # Use result as-is
    return after_tool_callback
```

#### 4.4.4 Before Tool Callback: `callbacks/before_tool.py`

Handles human-in-the-loop approval (from `HumanInTheLoopMiddleware`):

```python
def make_before_tool_callback(
    interrupt_on: dict[str, bool] | None,
) -> Callable | None:
    if not interrupt_on:
        return None

    def before_tool_callback(
        callback_context: CallbackContext,
        tool_context: ToolContext,
        tool_args: dict,
    ) -> dict | None:
        tool_name = tool_context.function_call_id  # or however ADK identifies the tool
        if tool_name in interrupt_on:
            # Signal that approval is needed
            # This requires integration with ADK's event/interrupt mechanism
            callback_context.state["_pending_approval"] = {
                "tool": tool_name,
                "args": tool_args,
            }
            return {"status": "awaiting_approval", "tool": tool_name}
        return None
    return before_tool_callback
```

### 4.5 Summarization: `adk_deepagents/summarization.py`

This is the most complex middleware to port because ADK doesn't have built-in summarization.

**Approach:** Implement in `before_model_callback`:

1. Count tokens in conversation history
2. If above threshold, partition messages into summarize vs. keep
3. Offload old messages to backend
4. Call a lightweight model to generate summary
5. Replace old messages with summary message

```python
@dataclass
class SummarizationConfig:
    model: str = "gemini-2.5-flash"
    trigger: tuple[str, float] = ("fraction", 0.85)
    keep: tuple[str, int] = ("messages", 6)
    history_path_prefix: str = "/conversation_history"
```

**Challenge:** ADK's `before_model_callback` receives `LlmRequest` which contains the messages.
We need to modify the request's contents list to replace old messages with a summary. This requires
understanding ADK's internal message format (which uses `google.genai.types.Content` objects rather
than LangChain messages).

### 4.6 Memory: `adk_deepagents/memory.py`

Port of `MemoryMiddleware`. Loads AGENTS.md files and injects into system prompt.

```python
MEMORY_SYSTEM_PROMPT = """<agent_memory>
{agent_memory}
</agent_memory>

<memory_guidelines>
...  # Same guidelines as deepagents
</memory_guidelines>
"""

def load_memory(backend: Backend, sources: list[str]) -> dict[str, str]:
    """Load memory files from backend."""
    contents = {}
    for path in sources:
        result = backend.download_files([path])
        if result[0].content:
            contents[path] = result[0].content.decode("utf-8")
    return contents

def format_memory(contents: dict[str, str], sources: list[str]) -> str:
    """Format memory for system prompt injection."""
    if not contents:
        return MEMORY_SYSTEM_PROMPT.format(agent_memory="(No memory loaded)")
    sections = [f"{path}\n{contents[path]}" for path in sources if contents.get(path)]
    return MEMORY_SYSTEM_PROMPT.format(agent_memory="\n\n".join(sections))
```

### 4.7 Skills: `adk_deepagents/skills.py`

Port of `SkillsMiddleware`. Discovers SKILL.md files, parses YAML frontmatter, and injects
skill metadata into system prompt with progressive disclosure.

All parsing logic (`_parse_skill_metadata`, `_validate_skill_name`, `_list_skills`) ports
directly since it's framework-agnostic YAML/file handling.

### 4.8 Prompt Constants: `adk_deepagents/prompts.py`

Port all system prompt templates from deepagents:

- `BASE_AGENT_PROMPT`
- `TASK_TOOL_DESCRIPTION` / `TASK_SYSTEM_PROMPT`
- `FILESYSTEM_SYSTEM_PROMPT` / `EXECUTION_SYSTEM_PROMPT`
- `MEMORY_SYSTEM_PROMPT`
- `SKILLS_SYSTEM_PROMPT`
- Tool descriptions (`LIST_FILES_TOOL_DESCRIPTION`, `READ_FILE_TOOL_DESCRIPTION`, etc.)

---

## 5. Package Structure

```
adk_deepagents/
├── __init__.py                     # Public API: create_deep_agent, SubAgentSpec, etc.
├── graph.py                        # create_deep_agent() factory
├── prompts.py                      # All system prompt templates
├── types.py                        # SubAgentSpec, SummarizationConfig, etc.
├── backends/
│   ├── __init__.py
│   ├── protocol.py                 # Backend ABC + dataclasses
│   ├── state.py                    # StateBackend (session.state)
│   ├── filesystem.py               # FilesystemBackend (local FS)
│   ├── composite.py                # CompositeBackend (path routing)
│   ├── local_shell.py              # LocalShellBackend (subprocess)
│   └── utils.py                    # Shared utilities
├── tools/
│   ├── __init__.py
│   ├── todos.py                    # write_todos, read_todos
│   ├── filesystem.py               # ls, read_file, write_file, edit_file, glob, grep, execute
│   └── task.py                     # task tool (sub-agent spawner)
├── callbacks/
│   ├── __init__.py
│   ├── before_agent.py             # Memory/skills loading, dangling tool patching
│   ├── before_model.py             # System prompt injection (memory, skills, fs, subagents)
│   ├── after_tool.py               # Large result eviction
│   └── before_tool.py              # Human-in-the-loop approval
├── summarization.py                # Context window management
├── memory.py                       # AGENTS.md loading and formatting
└── skills.py                       # SKILL.md discovery, parsing, formatting

tests/
├── unit_tests/
│   ├── backends/
│   │   ├── test_state_backend.py
│   │   ├── test_filesystem_backend.py
│   │   ├── test_composite_backend.py
│   │   └── test_protocol.py
│   ├── tools/
│   │   ├── test_todos.py
│   │   ├── test_filesystem_tools.py
│   │   └── test_task.py
│   ├── callbacks/
│   │   ├── test_before_agent.py
│   │   ├── test_before_model.py
│   │   └── test_after_tool.py
│   ├── test_memory.py
│   ├── test_skills.py
│   └── test_summarization.py
├── integration_tests/
│   ├── test_deep_agent.py
│   ├── test_subagents.py
│   └── test_filesystem_integration.py
└── conftest.py

examples/
├── quickstart/
│   └── agent.py                    # Minimal working example
├── content_builder/
│   ├── agent.py
│   ├── AGENTS.md
│   ├── skills/
│   └── subagents.yaml
└── deep_research/
    ├── agent.py
    └── research_agent/
```

---

## 6. Implementation Phases

### Phase 1: Core Foundation (Backend + Basic Tools)

**Goal:** A working agent with filesystem tools on session state.

1. Set up `pyproject.toml` with `google-adk` dependency
2. Implement `backends/protocol.py` - port all dataclasses and ABC
3. Implement `backends/utils.py` - port utility functions
4. Implement `backends/state.py` - `StateBackend` backed by a dict (session.state)
5. Implement `tools/todos.py` - todo tools using `ToolContext`
6. Implement `tools/filesystem.py` - all filesystem tools using `ToolContext`
7. Implement basic `graph.py` - `create_deep_agent()` with just tools, no callbacks
8. Write unit tests for backends and tools

**Deliverable:** `create_deep_agent()` returns an `LlmAgent` with filesystem + todo tools.

### Phase 2: Sub-agents

**Goal:** Sub-agent delegation via `AgentTool`.

1. Define `SubAgentSpec` TypedDict (matching deepagents `SubAgent`)
2. Implement `tools/task.py` - sub-agent builder using `AgentTool`
3. Implement general-purpose sub-agent (with full tool set)
4. Port sub-agent system prompt templates
5. Wire sub-agents into `create_deep_agent()`
6. Write integration tests for sub-agent delegation

**Deliverable:** Main agent can delegate to sub-agents, including parallel delegation.

### Phase 3: Callbacks (Middleware Stack)

**Goal:** Full middleware-equivalent callback system.

1. Implement `callbacks/before_model.py` - system prompt injection
2. Implement `callbacks/before_agent.py` - tool call patching
3. Implement `callbacks/after_tool.py` - large result eviction
4. Implement `callbacks/before_tool.py` - HITL approval
5. Wire all callbacks into `create_deep_agent()`
6. Port prompt templates to `prompts.py`

**Deliverable:** Agent has dynamic system prompts with filesystem docs and subagent docs.

### Phase 4: Memory + Skills

**Goal:** AGENTS.md and SKILL.md support.

1. Implement `memory.py` - load and format AGENTS.md files
2. Implement `skills.py` - discover, parse, and format SKILL.md files
3. Add memory loading to `before_agent_callback`
4. Add memory/skills injection to `before_model_callback`
5. Implement `backends/filesystem.py` - local filesystem backend
6. Write tests for memory and skills loading

**Deliverable:** Agent loads memory and skills from filesystem or state.

### Phase 5: Summarization + Advanced

**Goal:** Context window management and production features.

1. Implement `summarization.py` - token counting, message partitioning, summary generation
2. Integrate summarization into `before_model_callback`
3. Implement `backends/composite.py` - path-based routing
4. Implement `backends/local_shell.py` - local shell execution
5. Add execution tool support
6. Comprehensive integration tests

**Deliverable:** Full feature parity with deepagents core.

### Phase 6: Examples + Documentation

**Goal:** Working examples and docs.

1. Port `quickstart` example
2. Port `content-builder-agent` example
3. Port `deep_research` example
4. Write README with usage guide
5. Write migration guide (deepagents → adk-deepagents)

---

## 7. Key Design Decisions

### 7.1 Sub-agent Pattern: `AgentTool` vs Custom `task` Tool

**Decision:** Use `AgentTool` as the primary pattern, with optional `task` tool for compatibility.

**Rationale:** `AgentTool` is the native ADK way to delegate. It wraps each sub-agent as a
callable tool where the parent retains control (unlike `sub_agents` transfer which hands off
entirely). This matches deepagents' `task` tool semantics where the main agent receives the
result and synthesizes it.

However, `AgentTool` creates one tool per sub-agent, whereas deepagents has a single `task` tool
with a `subagent_type` parameter. We'll support both:

- **Default:** Individual `AgentTool`s (more idiomatic ADK)
- **Compat mode:** Single `task` tool that routes to sub-agents internally

### 7.2 State Management: Session State vs Artifacts

**Decision:** Use `session.state` for ephemeral file storage (todo list, in-memory files) and
artifacts for persistent file storage.

**Rationale:** `session.state` is a dict that persists within a session, matching `StateBackend`
semantics. Artifacts are for larger, persistent content that may span sessions.

### 7.3 System Prompt Injection: Static vs Dynamic

**Decision:** Use `before_model_callback` for dynamic system prompt injection.

**Rationale:** deepagents builds system prompts dynamically based on available tools, backend
capabilities, loaded memory, and discovered skills. ADK's `instruction` parameter is static.
The `before_model_callback` lets us modify the `LlmRequest` to inject dynamic content into the
system instruction before each LLM call, matching deepagents' `wrap_model_call` pattern exactly.

### 7.4 Model Agnosticism

**Decision:** Default to Gemini but support any ADK-compatible model.

**Rationale:** deepagents defaults to Claude Sonnet 4.5 but supports any LangChain model. Our
ADK version defaults to Gemini 2.5 Flash but accepts any model string that ADK supports. The
Anthropic-specific `PromptCachingMiddleware` is dropped (Gemini handles caching differently).

### 7.5 Backend Abstraction Retention

**Decision:** Keep the `BackendProtocol` abstraction.

**Rationale:** The backend abstraction is one of deepagents' best design decisions. It decouples
file operations from storage implementation, enabling the same agent to work with in-memory state,
local filesystem, remote sandboxes, or databases. This is orthogonal to ADK and should be preserved.

---

## 8. Examples to Port

### 8.1 Quickstart

```python
from adk_deepagents import create_deep_agent
from google.adk.runners import InMemoryRunner

agent = create_deep_agent()
runner = InMemoryRunner(agent=agent)
session = runner.session_service.create_session(app_name="deep_agent")

# Interactive loop
result = runner.run(session_id=session.id, user_message="List files in /")
```

### 8.2 Content Builder (port of `examples/content-builder-agent/`)

```python
from adk_deepagents import create_deep_agent
from adk_deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    memory=["./AGENTS.md"],
    skills=["./skills/"],
    tools=[web_search, generate_cover],
    subagents=[researcher_spec],
    backend=FilesystemBackend(root_dir="."),
)
```

### 8.3 Deep Research (port of `examples/deep_research/`)

```python
from adk_deepagents import create_deep_agent

agent = create_deep_agent(
    model="gemini-2.5-pro",
    tools=[tavily_search, think_tool],
    instruction=RESEARCH_INSTRUCTIONS,
    subagents=[{"name": "research-agent", "description": "...", "system_prompt": "..."}],
)
```

---

## 9. Testing Strategy

### Unit Tests

- **Backends:** Test each backend operation (read/write/edit/glob/grep/ls) with known state
- **Tools:** Test each tool function with mocked `ToolContext` and backend
- **Callbacks:** Test each callback with mocked `CallbackContext` and `LlmRequest`
- **Memory/Skills:** Test file loading, YAML parsing, prompt formatting

### Integration Tests

- **End-to-end agent:** Create agent, send message, verify tool usage and response
- **Sub-agent delegation:** Verify main agent delegates to sub-agent and synthesizes result
- **Filesystem round-trip:** Write file → read file → edit file → verify content
- **Summarization:** Send enough messages to trigger summarization, verify history is preserved

### Framework

- `pytest` with `pytest-asyncio` for async tests
- Mock ADK's `Runner` and `SessionService` for unit tests
- Real ADK runner for integration tests (requires API key)

---

## 10. Open Questions & Risks

### Questions

1. **ADK message format manipulation:** `before_model_callback` receives `LlmRequest`. How
   exactly do we append to the system instruction? Need to verify the exact API for modifying
   request contents.

2. **Conversation history access in callbacks:** For summarization, we need to read and modify
   the full conversation history. Does `CallbackContext` provide access to all session events?

3. **AgentTool context isolation:** When using `AgentTool`, does the sub-agent get its own
   session state or share the parent's? deepagents explicitly isolates sub-agent state
   (excluding certain keys). We need to verify and potentially replicate this isolation.

4. **Streaming support:** deepagents supports LangGraph streaming (`astream`). ADK has its own
   streaming mechanism. Need to verify ADK streaming works with our callback-heavy setup.

5. **Token counting:** deepagents uses `count_tokens_approximately` from LangChain. ADK may
   have its own token counting API, or we may need to implement our own.

### Risks

1. **Callback limitations:** ADK callbacks may not offer the same level of control as LangGraph
   middleware (e.g., modifying tool lists dynamically, intercepting and replacing tool results).
   Mitigation: Prototype the most complex callback (summarization) early.

2. **State update atomicity:** deepagents uses LangGraph's `Command` objects for atomic state
   updates. ADK's `tool_context.state` mutations may not have the same guarantees. Mitigation:
   Test concurrent state mutations thoroughly.

3. **Sub-agent context passing:** deepagents carefully controls what state passes to sub-agents
   (excluding messages, todos, etc.). AgentTool may pass different context. Mitigation: Test
   sub-agent state isolation early and implement manual filtering if needed.

4. **Model-specific features:** deepagents includes Anthropic-specific middleware (prompt
   caching). Some features may not have Gemini equivalents. Mitigation: Make these optional
   and implement Gemini-specific optimizations where available.

---

## Appendix: Feature Parity Checklist

| Feature | deepagents | adk-deepagents | Status |
|---|---|---|---|
| Todo list (write_todos/read_todos) | Yes | Planned | Phase 1 |
| File operations (ls/read/write/edit) | Yes | Planned | Phase 1 |
| Glob search | Yes | Planned | Phase 1 |
| Grep search | Yes | Planned | Phase 1 |
| Shell execution | Yes | Planned | Phase 5 |
| Sub-agent delegation | Yes | Planned | Phase 2 |
| General-purpose sub-agent | Yes | Planned | Phase 2 |
| Custom sub-agents | Yes | Planned | Phase 2 |
| Parallel sub-agent execution | Yes | Planned | Phase 2 |
| Conversation summarization | Yes | Planned | Phase 5 |
| Memory (AGENTS.md) | Yes | Planned | Phase 4 |
| Skills (SKILL.md) | Yes | Planned | Phase 4 |
| Human-in-the-loop approval | Yes | Planned | Phase 3 |
| Large result eviction | Yes | Planned | Phase 3 |
| Dangling tool call patching | Yes | Planned | Phase 3 |
| Dynamic system prompts | Yes | Planned | Phase 3 |
| State backend (ephemeral) | Yes | Planned | Phase 1 |
| Filesystem backend (local) | Yes | Planned | Phase 4 |
| Composite backend (routing) | Yes | Planned | Phase 5 |
| Structured output | Yes | Planned | Phase 1 |
| Model agnosticism | Yes (LangChain) | Yes (ADK) | Phase 1 |
| Streaming | Yes (LangGraph) | Yes (ADK) | Phase 1 |
| Checkpointing/persistence | Yes (LangGraph) | Yes (SessionService) | Phase 5 |
| Prompt caching (Anthropic) | Yes | N/A (model-specific) | - |
