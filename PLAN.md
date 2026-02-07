# ADK Deep Agents: Implementation Plan

> Re-implementing LangChain's [deepagents](https://github.com/langchain-ai/deepagents) using
> [Google ADK](https://google.github.io/adk-docs/) primitives with close feature compatibility.

### External Library Integration

This implementation leverages two companion libraries rather than rebuilding from scratch:

- **[adk-skills](https://github.com/manojlds/adk-skills)** (`adk-skills-agent`) — Provides
  full [Agent Skills](https://agentskills.io) support for ADK, replacing the custom
  `SkillsMiddleware` in deepagents. Handles SKILL.md discovery, parsing, validation,
  on-demand activation via `use_skill` tool, script execution via `run_script` tool,
  reference loading via `read_reference` tool, and prompt injection.

- **[Heimdall MCP](https://github.com/manojlds/heimdall)** (`@heimdall-ai/heimdall`) — An
  MCP server providing sandboxed Python and Bash execution via Pyodide (WebAssembly) and
  just-bash. Replaces the `LocalShellBackend`/`SandboxBackend` in deepagents with a
  security-first execution environment.

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

Two external libraries are used instead of re-implementing from scratch:
- **[adk-skills](https://github.com/manojlds/adk-skills)** for Agent Skills (SKILL.md) support,
  replacing deepagents' `SkillsMiddleware` with a mature, spec-compliant implementation.
- **[Heimdall MCP](https://github.com/manojlds/heimdall)** for sandboxed code execution,
  replacing deepagents' `LocalShellBackend`/`SandboxBackend` with WebAssembly-sandboxed
  Python/Bash execution via the MCP protocol.

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
│   ├── before_agent_callback   → memory loading, dangling tool patching
│   ├── before_model_callback   → system prompt injection (memory, filesystem, subagent docs)
│   ├── after_model_callback    → (reserved for future use)
│   ├── before_tool_callback    → path validation, interrupt/approval
│   └── after_tool_callback     → large result eviction, state updates
├── Tools (function tools via ToolContext)
│   ├── write_todos / read_todos
│   ├── ls / read_file / write_file / edit_file / glob / grep
│   ├── use_skill / run_script / read_reference  ← adk-skills library
│   └── task (sub-agent spawner via AgentTool)
├── Code Execution (via Heimdall MCP)
│   ├── execute_python          → sandboxed Python (Pyodide/WASM)
│   ├── execute_bash            → sandboxed Bash (just-bash)
│   └── workspace filesystem    → shared persistent workspace
├── Skills (via adk-skills library)
│   ├── SkillsRegistry          → discovery, parsing, validation
│   ├── use_skill tool          → on-demand skill activation
│   ├── run_script tool         → execute skill scripts (via Heimdall sandbox)
│   └── read_reference tool     → load skill reference docs
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
| `SandboxBackendProtocol` | **Heimdall MCP** (Pyodide + just-bash sandbox) | Sandboxed execution via MCP |
| `FilesystemMiddleware` tools | ADK `FunctionTool`s with `ToolContext` | Tools access state via `ToolContext` |
| `TodoListMiddleware` | ADK `FunctionTool`s + `session.state["todos"]` | State-backed todo tools |
| `SubAgentMiddleware` (`task` tool) | `AgentTool` wrapping sub-`LlmAgent`s | Key architectural difference |
| `SummarizationMiddleware` | `before_model_callback` + custom logic | Manual implementation needed |
| `MemoryMiddleware` | `before_model_callback` + file loading | System prompt injection |
| `SkillsMiddleware` | **adk-skills** `SkillsRegistry` + tools | Full replacement via library |
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
    skills_config: SkillsConfig | None = None,
    memory: list[str] | None = None,
    output_schema: type | None = None,
    backend: Backend | BackendFactory | None = None,
    execution: str | dict | None = None,       # "heimdall", "local", or MCP config dict
    interrupt_on: dict[str, bool] | None = None,
    session_service: SessionService | None = None,
    name: str = "deep_agent",
) -> LlmAgent:
```

**Parameters:**
- `skills`: List of directory paths to discover Agent Skills from (via adk-skills library)
- `skills_config`: Optional `SkillsConfig` for adk-skills customization (DB, validation, etc.)
- `execution`: Code execution backend. `"heimdall"` for sandboxed execution via Heimdall MCP,
  `"local"` for local subprocess, or a dict of MCP server config for custom setup.

**Implementation:**
1. Resolve `backend` (default to `StateBackend`)
2. Build tool list: todo tools + filesystem tools + user tools
3. If `skills` provided: create `SkillsRegistry`, discover skills, add `use_skill` /
   `run_script` / `read_reference` tools (via adk-skills)
4. If `execution` provided: connect to Heimdall MCP or create local shell tool. If both
   `skills` and `execution="heimdall"` are set, bridge skill script execution through Heimdall
5. Build sub-agent list: general-purpose + user sub-agents, wrapped as `AgentTool`
6. Compose callbacks from middleware-equivalent functions
7. Build instruction string: base prompt + memory + filesystem docs + subagent docs
8. Create and return configured `LlmAgent`

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

#### 4.2.5 Execution Backends: `execution/`

Code execution is handled separately from file backends (see Section 4.8):

- **`execution/heimdall.py`**: Primary — Heimdall MCP (sandboxed Python + Bash via WASM)
- **`execution/local.py`**: Fallback — local `subprocess.run()` (less secure)
- **`execution/bridge.py`**: Routes adk-skills `run_script` through Heimdall

The old `LocalShellBackend` / `SandboxBackend` pattern is replaced by the `execution`
parameter on `create_deep_agent()`.

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

**Note:** The `execute` tool is NOT in filesystem tools — code execution is handled by the
`execution/` module (Heimdall MCP or local shell). See Section 4.8.

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

### 4.7 Skills: Integration with `adk-skills` Library

**NOT a port** — we delegate entirely to the
[adk-skills](https://github.com/manojlds/adk-skills) library (`adk-skills-agent` on PyPI),
which already provides a complete, spec-compliant implementation of Agent Skills for Google ADK.

#### What adk-skills provides

- **`SkillsRegistry`**: Discovers SKILL.md files from directories, parses YAML frontmatter,
  validates against the [agentskills.io](https://agentskills.io) spec, and caches full skill
  content on-demand. Supports both file-based and database-backed storage.

- **`use_skill` tool**: On-demand skill activation. The tool's description contains an
  `<available_skills>` XML block listing all discovered skills (progressive disclosure).
  When the LLM calls `use_skill(name="web-scraper")`, it receives the full SKILL.md
  instructions, base directory, and capability flags (has_scripts, has_references, has_assets).

- **`run_script` tool**: Execute Python/Bash scripts bundled with skills. Scripts live in
  `skills/<name>/scripts/`. Supports explicit activation and timeouts.

- **`read_reference` tool**: Load reference documents from `skills/<name>/references/` on
  demand, keeping context lean until needed.

- **Prompt injection**: `registry.inject_skills_prompt(instruction, format="xml")` appends
  `<available_skills>` to the system prompt. Alternative to tool-based discovery.

- **`SkillsAgent`**: High-level agent wrapper that integrates discovery, validation, tools,
  and prompt injection in one class.

- **Helper functions**: `with_skills(agent, dirs)`, `create_skills_agent(...)`,
  `inject_skills_prompt(...)` for ergonomic integration.

#### Integration pattern in `create_deep_agent()`

```python
from adk_skills_agent import SkillsRegistry

def create_deep_agent(
    ...
    skills: list[str] | None = None,    # directories to discover skills from
    skills_config: SkillsConfig | None = None,
    ...
) -> LlmAgent:
    tools = [...]  # other tools

    if skills:
        registry = SkillsRegistry(config=skills_config)
        registry.discover(skills)

        # Option A: Tool-based (default, matches deepagents' progressive disclosure)
        tools.append(registry.create_use_skill_tool())
        tools.append(registry.create_run_script_tool())
        tools.append(registry.create_read_reference_tool())

        # Option B: Prompt injection (alternative, injects into system prompt)
        # instruction = registry.inject_skills_prompt(instruction, format="xml")
        # tools.append(registry.create_use_skill_tool(include_skills_listing=False))
```

#### Mapping from deepagents SkillsMiddleware

| deepagents SkillsMiddleware | adk-skills equivalent |
|---|---|
| `_discover_skills()` | `registry.discover(directories)` |
| `_list_skills()` → system prompt | `registry.to_prompt_xml()` or tool description |
| `_parse_skill_metadata()` (YAML) | `registry.list_metadata()` |
| `use_skill` tool (progressive disclosure) | `registry.create_use_skill_tool()` |
| `run_script` tool | `registry.create_run_script_tool()` |
| SKILL.md validation | `registry.validate_all(strict=True)` |
| System prompt injection | `registry.inject_skills_prompt(instruction)` |

**Key advantage:** adk-skills is already ADK-native, supports both tool-based and prompt-injection
patterns, includes database-backed skills (SQLAlchemy), and follows the agentskills.io spec. No
porting effort needed — just wire it into `create_deep_agent()`.

#### Script execution via Heimdall

The `run_script` tool from adk-skills executes scripts locally by default. For sandboxed
execution, we bridge it to Heimdall MCP (see Section 4.9). The adk-skills `run_script` tool
can be configured with a custom executor that routes execution through Heimdall's
`execute_python` or `execute_bash` MCP tools instead of local subprocess.

### 4.8 Code Execution: Integration with Heimdall MCP

**NOT a port of `LocalShellBackend`/`SandboxBackend`** — we delegate code execution to
[Heimdall MCP](https://github.com/manojlds/heimdall) (`@heimdall-ai/heimdall` on npm), which
provides a sandboxed execution environment via the MCP protocol.

#### What Heimdall provides

Heimdall is a TypeScript MCP server that exposes:

- **`execute_python`**: Sandboxed Python execution via Pyodide (WebAssembly). Auto-detects
  imports and installs packages. No network access from user code (WASM security boundary).
  Supports numpy, pandas, scipy, matplotlib, etc.

- **`execute_bash`**: Bash command execution via just-bash (TypeScript simulation, no real
  processes). Supports 50+ built-in commands (grep, sed, awk, find, jq, curl, tar, etc.),
  pipes, redirections, variables, loops, and conditionals.

- **`write_file` / `read_file` / `list_files` / `delete_file`**: Virtual filesystem
  operations within a persistent workspace directory.

- **`install_packages`**: Install Python packages via micropip.

- **Shared workspace**: Bash and Python share the same `/workspace` filesystem, enabling
  cross-language workflows (e.g., Bash prepares CSV → Python analyzes with pandas).

#### Security model

| Feature | Guarantee |
|---|---|
| Python execution | WebAssembly sandbox (memory-isolated, no network) |
| Bash execution | TypeScript simulation (no real process spawning) |
| Filesystem | Workspace directory only (no host FS access) |
| Execution limits | Timeouts prevent infinite loops |
| Package installation | Pyodide's trusted mechanism only |

#### Integration via ADK MCP Tools

Google ADK has native MCP tool support. We connect Heimdall as an MCP server:

```python
from google.adk.tools.mcp_tool import MCPToolset

# Option 1: Stdio transport (local Heimdall server)
heimdall_tools = MCPToolset.from_server(
    command="npx",
    args=["@heimdall-ai/heimdall"],
)

# Option 2: SSE transport (remote Heimdall server)
heimdall_tools = MCPToolset.from_server(
    uri="http://localhost:3000/sse",
)
```

In `create_deep_agent()`:

```python
def create_deep_agent(
    ...
    execution: str | dict | None = None,  # "heimdall", "local", or MCP config dict
    ...
) -> LlmAgent:
    tools = [...]

    if execution == "heimdall":
        # Connect to Heimdall MCP server for sandboxed execution
        heimdall_tools = MCPToolset.from_server(
            command="npx",
            args=["@heimdall-ai/heimdall"],
            env={"HEIMDALL_WORKSPACE": workspace_path},
        )
        tools.extend(heimdall_tools)
    elif execution == "local":
        # Fallback: local shell execution (less secure)
        tools.append(local_execute_tool)
    elif isinstance(execution, dict):
        # Custom MCP server config
        heimdall_tools = MCPToolset.from_server(**execution)
        tools.extend(heimdall_tools)
```

#### Mapping from deepagents execution backends

| deepagents Backend | Heimdall MCP equivalent |
|---|---|
| `LocalShellBackend.execute(cmd)` | `execute_bash` MCP tool |
| `SandboxBackendProtocol.execute(cmd)` | `execute_bash` + `execute_python` MCP tools |
| Sandbox file read/write | `read_file` / `write_file` MCP tools |
| Sandbox package install | `install_packages` MCP tool |

#### Key advantages over deepagents execution

1. **Security-first**: Pyodide WASM sandbox + just-bash simulation vs. raw `subprocess.run()`
2. **Language support**: Both Python and Bash in a single sandbox
3. **Shared workspace**: Files persist across executions and are shared between languages
4. **No host access**: Workspace isolation prevents agent from accessing host filesystem
5. **MCP standard**: Standard protocol enables remote execution, scaling, and swappability
6. **Configurable limits**: File size, workspace size, and execution timeouts via env vars

#### Bridging adk-skills script execution to Heimdall

When both adk-skills and Heimdall are configured, the `run_script` tool from adk-skills can
route script execution through Heimdall for sandboxed execution:

```python
# Custom executor that routes through Heimdall
class HeimdallScriptExecutor:
    def __init__(self, heimdall_tools):
        self.execute_python = heimdall_tools["execute_python"]
        self.execute_bash = heimdall_tools["execute_bash"]

    async def execute(self, script_path: str, script_content: str) -> dict:
        if script_path.endswith(".py"):
            return await self.execute_python(code=script_content)
        elif script_path.endswith(".sh"):
            return await self.execute_bash(command=script_content)
```

### 4.9 Prompt Constants: `adk_deepagents/prompts.py`

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
│   └── utils.py                    # Shared utilities
├── tools/
│   ├── __init__.py
│   ├── todos.py                    # write_todos, read_todos
│   ├── filesystem.py               # ls, read_file, write_file, edit_file, glob, grep
│   └── task.py                     # task tool (sub-agent spawner)
├── execution/
│   ├── __init__.py
│   ├── heimdall.py                 # Heimdall MCP integration (sandboxed execution)
│   ├── local.py                    # Local shell fallback (subprocess, less secure)
│   └── bridge.py                   # Bridge adk-skills run_script → Heimdall executor
├── skills/
│   ├── __init__.py
│   └── integration.py              # adk-skills SkillsRegistry wiring into create_deep_agent
├── callbacks/
│   ├── __init__.py
│   ├── before_agent.py             # Memory loading, dangling tool patching
│   ├── before_model.py             # System prompt injection (memory, fs, subagents)
│   ├── after_tool.py               # Large result eviction
│   └── before_tool.py              # Human-in-the-loop approval
├── summarization.py                # Context window management
└── memory.py                       # AGENTS.md loading and formatting

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
│   ├── execution/
│   │   ├── test_heimdall.py
│   │   ├── test_local.py
│   │   └── test_bridge.py
│   ├── skills/
│   │   └── test_integration.py
│   ├── callbacks/
│   │   ├── test_before_agent.py
│   │   ├── test_before_model.py
│   │   └── test_after_tool.py
│   ├── test_memory.py
│   └── test_summarization.py
├── integration_tests/
│   ├── test_deep_agent.py
│   ├── test_subagents.py
│   ├── test_heimdall_execution.py
│   ├── test_skills_integration.py
│   └── test_filesystem_integration.py
└── conftest.py

examples/
├── quickstart/
│   └── agent.py                    # Minimal working example
├── content_builder/
│   ├── agent.py
│   ├── AGENTS.md
│   ├── skills/                     # Agent Skills (adk-skills format)
│   │   ├── blog-writing/
│   │   │   └── SKILL.md
│   │   └── social-media/
│   │       └── SKILL.md
│   └── subagents.yaml
├── deep_research/
│   ├── agent.py
│   └── research_agent/
└── sandboxed_coder/
    ├── agent.py                    # Agent with Heimdall execution
    └── skills/
        └── code-review/
            └── SKILL.md
```

---

## 6. Implementation Phases

### Phase 1: Core Foundation (Backend + Basic Tools) — ✅ COMPLETE

**Goal:** A working agent with filesystem tools on session state.

1. ✅ Set up `pyproject.toml` with dependencies: `google-adk>=1.0.0`, `wcmatch>=8.5`
2. ✅ Implement `backends/protocol.py` - Backend ABC, SandboxBackend ABC, all dataclasses
3. ✅ Implement `backends/utils.py` - 14 utility functions (path validation, line numbering,
   string replacement, grep/glob helpers, truncation, content preview)
4. ✅ Implement `backends/state.py` - `StateBackend` backed by a dict
5. ✅ Implement `tools/todos.py` - write_todos, read_todos using `ToolContext`
6. ✅ Implement `tools/filesystem.py` - 6 tools (ls, read_file, write_file, edit_file, glob, grep)
7. ✅ Implement `graph.py` - full `create_deep_agent()` factory with callbacks
8. ✅ Write unit tests - 46 tests (protocol, state backend, filesystem tools, todo tools)

**Deliverable:** ✅ `create_deep_agent()` returns an `LlmAgent` with filesystem + todo tools.

#### Implementation Notes (Phase 1)
- Root path `/` requires special handling in `filter_files_by_path()` and `glob_search_files()`
  — prefix check `startswith("/" + "/")` fails, so we short-circuit for root.
- `BackendFactory` type alias is `Callable[[dict], Backend]` — takes raw state dict, not
  a `ToolRuntime` as in deepagents.
- Tools resolve backend via `tool_context.state["_backend"]` or
  `tool_context.state["_backend_factory"]` (lazy creation pattern).
- `read_file` defaults to `limit=100` (matching deepagents behavior, not protocol's `2000`).
- `write_file` returns `"File already exists"` error (not generic `"invalid_path"` string).
- `_apply_files_update()` helper merges `files_update` dict into `tool_context.state["files"]`.

### Phase 2: Sub-agents — ✅ COMPLETE

**Goal:** Sub-agent delegation via `AgentTool`.

1. ✅ Define `SubAgentSpec` TypedDict in `types.py`
2. ✅ Implement `tools/task.py` - `build_subagent_tools()` using `AgentTool`
3. ✅ Implement general-purpose sub-agent (`GENERAL_PURPOSE_SUBAGENT`)
4. ✅ Port sub-agent prompt templates to `prompts.py`
5. ✅ Wire sub-agents into `create_deep_agent()`
6. ⬚ Write integration tests (pending — needs real ADK Runner)

**Deliverable:** ✅ Main agent can delegate to sub-agents.

#### Implementation Notes (Phase 2)
- **ADK agent names MUST be valid Python identifiers** (no hyphens, spaces, etc.).
  `_sanitize_agent_name()` replaces `-` with `_` and strips invalid chars.
  deepagents uses `"general-purpose"` → we use `"general_purpose"`.
- Each sub-agent becomes its own `AgentTool` (not a single routing `task` tool).
  The LLM calls them by name directly (e.g., `general_purpose(description="...")`).
- Sub-agents inherit the parent's tools by default unless `spec["tools"]` overrides.
- **State isolation is NOT yet tested** — Open Question #3 remains. Need to verify
  whether `AgentTool` sub-agents share parent state or get their own session.

### Phase 3: Callbacks (Middleware Stack) — ✅ COMPLETE (with known gaps)

**Goal:** Full middleware-equivalent callback system.

1. ✅ Implement `callbacks/before_model.py` - dynamic system prompt injection
2. ✅ Implement `callbacks/before_agent.py` - memory loading (once per session)
3. ⚠️ Implement `callbacks/after_tool.py` - **infrastructure only** (see below)
4. ⚠️ Implement `callbacks/before_tool.py` - **state-setting only** (see below)
5. ✅ Wire all callbacks into `create_deep_agent()`
6. ✅ Port prompt templates to `prompts.py`
7. ✅ 4 unit tests for before_model_callback

**Deliverable:** ✅ Agent has dynamic system prompts with filesystem/subagent/memory docs.

#### Implementation Notes (Phase 3)
- **System prompt injection works via `llm_request.config.system_instruction`**.
  Handles `str`, `types.Content`, and `None` cases. Open Question #1 is RESOLVED.
- `_append_to_system_instruction()` appends a `types.Part(text=...)` to existing
  `Content.parts` or concatenates strings with `"\n\n"`.
- **after_tool_callback gap:** ADK's `after_tool_callback` signature is
  `(tool, args, tool_context) -> dict | None`. It does NOT receive the tool's return value
  as a parameter. The large-result eviction must happen inside the tool functions themselves
  via `truncate_if_too_long()`. The callback currently only checks tool name exclusion.
  Consider moving eviction fully into the tools or finding an ADK mechanism to intercept
  tool results.
- **before_tool_callback gap:** Returns `{"status": "awaiting_approval"}` dict, which
  ADK interprets as a replacement result (skipping the tool). This effectively blocks the
  tool but doesn't actually pause for user approval. A proper HITL flow needs ADK's
  interrupt/event mechanism or an external approval service.
- **Dangling tool call patching NOT implemented** — `before_agent_callback` currently only
  loads memory. The `PatchToolCallsMiddleware` equivalent (scanning message history for
  orphaned tool calls) is deferred because `CallbackContext` message access needs research.

#### Known Gaps (addressed in Phase 6)
1. ~~`after_tool_callback` doesn't receive actual tool result~~ — **RESOLVED**: cooperative
   eviction via `_last_tool_result` state key + in-tool `truncate_if_too_long()`
2. `before_tool_callback` HITL blocks tool but doesn't pause for approval input
3. ~~Dangling tool call patching not implemented~~ — **RESOLVED**: `before_agent_callback`
   detects dangling calls via session events, `before_model_callback` injects synthetic
   `FunctionResponse` parts

### Phase 4: Memory + Skills + FilesystemBackend

**Status:** ✅ Complete.

**Goal:** AGENTS.md memory, Agent Skills, and local filesystem backend.

#### Completed:
- ✅ `memory.py` - `load_memory()` and `format_memory()` (65 lines, 4 tests)
- ✅ Memory loading wired into `before_agent_callback`
- ✅ Memory injection wired into `before_model_callback`
- ✅ `skills/integration.py` - full implementation with logging, config forwarding,
  per-operation error handling, registry/metadata state storage, and `inject_skills_into_prompt()`
- ✅ `backends/filesystem.py` - local filesystem backend with virtual mode sandboxing,
  ripgrep-accelerated grep, pathlib operations, upload/download support
- ✅ 35 FilesystemBackend tests + 7 skills integration tests (42 new tests, 104 total)

#### Implementation Notes:
- `skills/integration.py` stores `SkillsRegistry` in `state["_skills_registry"]` and
  metadata in `state["skills_metadata"]` for prompt injection
- `inject_skills_into_prompt()` provides non-tool-based alternative for skills listing
- `FilesystemBackend` returns `files_update=None` from write/edit (persists directly to disk)
- `FilesystemBackend.grep_raw()` tries ripgrep first, falls back to Python search
- Virtual mode uses `Path.relative_to()` to prevent directory escapes
- Non-virtual mode accesses absolute paths directly (for system-level access)

#### Implementation Details for `backends/filesystem.py`

```python
class FilesystemBackend(Backend):
    """Backend that reads/writes to the local filesystem."""

    def __init__(
        self,
        root_dir: str | Path | None = None,
        *,
        virtual_mode: bool = False,
        max_file_size_mb: float = 10,
    ) -> None:
        self._root = Path(root_dir).resolve() if root_dir else Path.cwd()
        self._virtual_mode = virtual_mode
        self._max_file_size = int(max_file_size_mb * 1024 * 1024)

    def _resolve_path(self, key: str) -> Path:
        """Resolve a virtual path to a real filesystem path."""
        if self._virtual_mode:
            # Strip leading / and resolve relative to root
            rel = key.lstrip("/")
            resolved = (self._root / rel).resolve()
            if not str(resolved).startswith(str(self._root)):
                raise ValueError(f"Path escapes root directory: {key}")
            return resolved
        return Path(key)
```

Key patterns from deepagents `FilesystemBackend`:
- `write()` returns `WriteResult(files_update=None)` — NOT `files_update={...}` like
  StateBackend, because the file is persisted directly to disk.
- `edit()` returns `EditResult(files_update=None)` — same reason.
- `read()` uses `format_content_with_line_numbers()` from utils.
- `ls_info()` uses `Path.iterdir()` for directory listing.
- `grep_raw()` tries `subprocess.run(["rg", "--json", "-F", pattern, ...])` first for
  performance, falls back to Python regex search.

**Deliverable:** Agent works with both in-memory state and local filesystem. Skills
discovered and activated via adk-skills library.

### Phase 5: Code Execution (Heimdall MCP) + Summarization + CompositeBackend

**Status:** ✅ Complete.

**Goal:** Sandboxed code execution, context window management, and path-based backend routing.

#### Completed:
- ✅ `execution/local.py` - local shell fallback (74 lines, wired into graph.py, 6 tests)
- ✅ `execution/heimdall.py` - Heimdall MCP integration via `MCPToolset` (192 lines, 4 tests)
  - `get_heimdall_tools()` - connect to Heimdall MCP server, filter tools
  - `get_heimdall_tools_from_config()` - support both stdio and SSE transports
  - Tool filtering: only exposes `execute_python`, `execute_bash`, `install_packages`,
    and workspace tools (`write_file`, `read_file`, `list_files`, `delete_file`)
- ✅ `execution/bridge.py` - adk-skills → Heimdall bridge (178 lines, 12 tests)
  - `HeimdallScriptExecutor` class that routes `.py` → `execute_python` and `.sh` → `execute_bash`
  - Language auto-detection from file extension
  - Timeout support and error handling
- ✅ `summarization.py` - full implementation (318 lines, 11 tests)
  - `count_tokens_approximate()` - character-based heuristic (~4 chars/token)
  - `count_content_tokens()` / `count_messages_tokens()` - Content message token counting
  - `partition_messages()` - split into (to_summarize, to_keep)
  - `format_messages_for_summary()` - human-readable formatting for summarization
  - `create_summary_content()` - wrap summary in `<conversation_summary>` tags
  - `offload_messages_to_backend()` - save old messages for reference
  - `maybe_summarize()` - main integration point for `before_model_callback`
- ✅ `backends/composite.py` - path-based routing (180 lines, 14 tests)
  - Routes file operations by longest-matching path prefix
  - Merges results from multiple backends for grep/glob
  - Supports all Backend interface methods
- ✅ `graph.py` updated with:
  - `summarization` parameter and `SummarizationConfig` support
  - `create_deep_agent_async()` variant for Heimdall MCP
  - Heimdall/dict execution config warning in sync factory
  - Both (a) async factory and (b) pre-resolved tools patterns supported
- ✅ `callbacks/before_model.py` updated with summarization integration
- ✅ `__init__.py` exports: `create_deep_agent_async`, `SummarizationConfig`
- ✅ `backends/__init__.py` exports: `CompositeBackend`

#### Implementation Notes:
- **Async lifecycle:** Both `create_deep_agent_async()` and pre-resolved tools are supported.
  The sync `create_deep_agent()` warns if `execution="heimdall"` is passed.
- **Summarization approach:** Uses inline message reformatting (not LLM call) to avoid
  async complexity. Old messages are formatted as readable text, truncated to 15% of
  context window, and wrapped in `<conversation_summary>` tags.
- **`llm_request.contents`** provides full conversation history and can be modified
  in-place in `before_model_callback` (confirmed working).
- **CompositeBackend** sorts routes by prefix length (longest first) for specificity.
- **Heimdall cleanup:** The async factory returns `(agent, cleanup_fn)` where `cleanup_fn`
  must be awaited to close the MCP server connection.

#### Test Coverage:
- 6 tests for `execution/local.py`
- 4 tests for `execution/heimdall.py` (mock MCP toolset)
- 12 tests for `execution/bridge.py`
- 11 tests for `summarization.py`
- 14 tests for `backends/composite.py`
- 18 tests for `graph.py` factory
- 5 tests for `callbacks/before_agent.py`
- 4 tests for `callbacks/after_tool.py`
- 7 tests for `callbacks/before_tool.py`

**Deliverable:** Agent can execute Python/Bash in Heimdall sandbox, summarize long
conversations, and route file operations to different backends by path.

### Phase 6: Examples + Documentation + Gap Closure

**Status:** ✅ Complete.

**Goal:** Working examples, close remaining code gaps, comprehensive docs.

#### Completed:
- ✅ `examples/quickstart/agent.py` - full working example with InMemoryRunner
  and interactive loop
- ✅ `examples/content_builder/` - content builder with skills, memory, sub-agents
  - `agent.py` — creates agent with FilesystemBackend, skills, researcher sub-agent
  - `AGENTS.md` — content creation memory
  - `skills/blog-writing/SKILL.md` — blog writing guidelines
  - `skills/social-media/SKILL.md` — social media content templates
- ✅ `examples/deep_research/agent.py` - parallel sub-agent research workflow
  - Three sub-agents: web_researcher, analyst, writer
  - SummarizationConfig for long research sessions
- ✅ `examples/sandboxed_coder/` - Heimdall MCP execution with skills
  - `agent.py` — async agent with sandboxed execution
  - `skills/code-review/SKILL.md` — code review checklist
- ✅ **Dangling tool call patching** — two-phase approach:
  - `before_agent_callback`: scans session events for orphaned function_calls,
    stores dangling info in `state["_dangling_tool_calls"]`
  - `before_model_callback`: reads dangling info from state, injects synthetic
    `FunctionResponse` content into `llm_request.contents`
  - 7 unit tests covering detection, patching, and edge cases
- ✅ **After-tool large result eviction** — cooperative pattern:
  - Tools can store raw result under `state["_last_tool_result"]` before returning
  - `after_tool_callback` checks size, saves to backend, returns preview
  - Includes `TOO_LARGE_TOOL_MSG` template with preview and file path
  - 4 new unit tests for eviction behavior

5. **Write README.md** with:
   - Installation instructions
   - Quick start guide
   - API reference for `create_deep_agent()` parameters
   - Backend comparison table (State vs Filesystem vs Composite)
   - Skills integration guide
   - Execution backend guide (Heimdall vs local)

6. **Write migration guide** (deepagents → adk-deepagents):
   - Parameter mapping table
   - Middleware → callback migration
   - Backend migration
   - Breaking changes and differences

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

### 7.6 Skills: Delegate to adk-skills, Don't Reimplement

**Decision:** Use `adk-skills-agent` library for all skills functionality.

**Rationale:** deepagents' `SkillsMiddleware` implements custom SKILL.md parsing, validation,
and progressive disclosure. The `adk-skills` library already provides all of this for Google ADK,
plus additional features:

- **Spec-compliant**: Follows the [agentskills.io](https://agentskills.io) standard exactly
- **Two integration patterns**: Tool-based (progressive disclosure via `use_skill`) and
  prompt injection (skills in system prompt) — matching both deepagents patterns
- **Database support**: Optional SQLAlchemy-backed skill storage for production deployments
- **Validation**: Built-in skill validation against the spec
- **Reference loading**: On-demand `read_reference` tool for skill documentation
- **Already ADK-native**: Produces tools compatible with `google.adk.agents.Agent`

Re-implementing this would duplicate significant effort with no benefit.

### 7.7 Code Execution: Heimdall MCP over Local Shell

**Decision:** Use Heimdall MCP as the primary execution backend, with local shell as fallback.

**Rationale:** deepagents' `LocalShellBackend` uses raw `subprocess.run()`, which gives the
agent unrestricted access to the host system — a significant security concern. Heimdall provides:

1. **WebAssembly sandbox**: Python runs in Pyodide (WASM) with no host filesystem or network
   access. Bash runs via just-bash (TypeScript simulation, no real processes).
2. **MCP protocol**: Standard protocol enabling remote execution, horizontal scaling, and
   swappability. ADK has native MCP tool support via `MCPToolset`.
3. **Shared workspace**: Python and Bash share a persistent `/workspace` directory, enabling
   cross-language workflows common in research and data processing.
4. **Configurable limits**: File size, workspace size, and execution timeouts via env vars.

The trade-off is that Heimdall requires Node.js and has some limitations (no native C
extensions in Python, no real networking). For use cases that need full host access, the
`execution="local"` fallback preserves raw `subprocess` behavior.

### 7.8 Execution + Skills Bridge

**Decision:** Bridge adk-skills' `run_script` through Heimdall when both are configured.

**Rationale:** Skills may include scripts in `scripts/` directories. By default, adk-skills
runs these locally. When Heimdall is configured, we route script execution through the sandbox
for consistent security. This is implemented as a custom executor class that the adk-skills
`run_script` tool delegates to.

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

Uses adk-skills for blog-writing and social-media skills:

```python
from adk_deepagents import create_deep_agent
from adk_deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    memory=["./AGENTS.md"],
    skills=["./skills/"],              # adk-skills discovers blog-writing, social-media
    tools=[web_search, generate_cover],
    subagents=[researcher_spec],
    backend=FilesystemBackend(root_dir="."),
)

# The agent can now:
# 1. Activate blog-writing skill via use_skill("blog-writing") → gets full instructions
# 2. Activate social-media skill → gets platform-specific guidelines
# 3. Read skill references → use_reference("blog-writing", "seo-guide.md")
# 4. Delegate research to sub-agent
# 5. Write files to local filesystem
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

### 8.4 Sandboxed Coder (new example, Heimdall + adk-skills)

Demonstrates sandboxed code execution with Heimdall MCP and code-review skill:

```python
from adk_deepagents import create_deep_agent

agent = create_deep_agent(
    model="gemini-2.5-flash",
    instruction="You are a coding assistant. Write and test code in the sandbox.",
    skills=["./skills/"],              # code-review skill
    execution="heimdall",              # Sandboxed Python + Bash via Heimdall MCP
)

# The agent can now:
# 1. Write Python code and execute it in the Pyodide sandbox
# 2. Run bash commands (grep, find, jq, etc.) in just-bash sandbox
# 3. Install Python packages (numpy, pandas, etc.)
# 4. Read/write files in the shared /workspace
# 5. Activate code-review skill for review guidelines
# 6. Cross-language workflows: bash prepares data → python analyzes
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

1. ✅ **RESOLVED — ADK message format manipulation:** `before_model_callback` receives
   `LlmRequest`. The system instruction lives at `llm_request.config.system_instruction`
   and can be a `str`, `types.Content`, or `None`. Our `_append_to_system_instruction()`
   helper handles all three cases — appending text parts to `Content.parts` or concatenating
   strings with `"\n\n"`.

2. **Conversation history access in callbacks:** For summarization, we need to read and modify
   the full conversation history. Does `CallbackContext` provide access to all session events?
   **Hypothesis:** `llm_request.contents` (a `list[types.Content]`) contains the conversation
   messages passed to the model. We can modify this list in `before_model_callback` to remove
   old messages and insert a summary. Needs verification with real ADK Runner.

3. **AgentTool context isolation:** When using `AgentTool`, does the sub-agent get its own
   session state or share the parent's? deepagents explicitly isolates sub-agent state
   (excluding certain keys like `messages`, `todos`, `structured_response`, `skills_metadata`,
   `memory_contents`). **Status:** NOT YET TESTED. Need to verify with real ADK Runner and
   potentially implement manual state filtering if sub-agents share parent state.

4. **Streaming support:** deepagents supports LangGraph streaming (`astream`). ADK has its own
   streaming mechanism. Need to verify ADK streaming works with our callback-heavy setup.
   **Status:** Not addressed. Low risk — ADK's Runner handles streaming transparently.

5. **Token counting:** deepagents uses `count_tokens_approximately` from LangChain. ADK may
   have its own token counting API. **Status:** Plan to use `len(text) / 4` heuristic
   (matching deepagents) with optional `google.genai.Client.count_tokens()` for precision.

6. ⚠️ **PARTIALLY RESOLVED — ADK MCP tool lifecycle:** `MCPToolset.from_server()` is async
   and returns `(tools, exit_stack)`. The `exit_stack` must be managed for cleanup. This means
   `create_deep_agent()` needs an async variant or users must pre-resolve MCP tools. See
   Phase 5 "Design Decision: Async MCP Lifecycle" for proposed solutions.

7. **adk-skills run_script executor extensibility:** Can we cleanly inject a custom executor
   into adk-skills' `run_script` tool to route execution through Heimdall? Or do we need to
   wrap/replace the tool? **Status:** Not researched. Need to examine adk-skills source code.

### Risks

1. ⚠️ **Callback limitations — PARTIALLY CONFIRMED:** ADK's `after_tool_callback` does NOT
   receive the tool's return value as a parameter (signature: `tool, args, tool_context`).
   This means we cannot intercept and replace large tool results in the callback. Mitigation:
   Large result truncation is handled inside the tool functions themselves via
   `truncate_if_too_long()`. This is a deviation from deepagents' approach but achieves the
   same outcome.

2. **State update atomicity:** deepagents uses LangGraph's `Command` objects for atomic state
   updates. ADK's `tool_context.state` mutations may not have the same guarantees. Mitigation:
   Test concurrent state mutations thoroughly. **Note:** In practice, tools run sequentially
   within a single agent turn, so atomicity is less of a concern than in deepagents' parallel
   middleware execution model.

3. **Sub-agent context passing:** deepagents carefully controls what state passes to sub-agents
   (excluding messages, todos, etc.). AgentTool may pass different context. Mitigation: Test
   sub-agent state isolation early and implement manual filtering if needed.
   **Status:** Not tested. HIGH PRIORITY for Phase 4.

4. **Model-specific features:** deepagents includes Anthropic-specific middleware (prompt
   caching). Some features may not have Gemini equivalents. Mitigation: Make these optional
   and implement Gemini-specific optimizations where available.
   **Status:** Anthropic prompt caching deliberately dropped. Not needed for Gemini.

5. **Heimdall MCP availability:** Heimdall requires Node.js >= 18. In environments where
   Node.js is unavailable, the `execution="heimdall"` option won't work. Mitigation: Always
   support `execution="local"` fallback and make Heimdall optional.
   **Status:** `execution="local"` fallback is implemented and working.

6. **Heimdall execution limitations:** Pyodide doesn't support all Python packages (those
   requiring native C extensions without WASM ports). Bash is simulated (not real shell).
   Mitigation: Document limitations clearly and offer local shell fallback.

7. **adk-skills version compatibility:** We depend on adk-skills' public API. Breaking changes
   in the library could affect our integration. Mitigation: Pin to a specific version range
   and contribute upstream if needed.

---

## Appendix: Feature Parity Checklist

| Feature | deepagents | adk-deepagents | Implementation | Status |
|---|---|---|---|---|
| Todo list (write_todos/read_todos) | Yes | ✅ Done | Custom tools + session.state | Phase 1 |
| File operations (ls/read/write/edit) | Yes | ✅ Done | Custom tools + Backend abstraction | Phase 1 |
| Glob search | Yes | ✅ Done | Custom tool + `wcmatch` | Phase 1 |
| Grep search | Yes | ✅ Done | Custom tool + literal match | Phase 1 |
| Shell execution (Bash) | Yes (subprocess) | ✅ Done | `execution/local.py` + Heimdall MCP | Phase 5 |
| Python execution | N/A | ✅ Done | **Heimdall MCP** `execute_python` | Phase 5 |
| Sandboxed execution | Partial | ✅ Done | **Heimdall MCP** (WASM sandbox) | Phase 5 |
| Sub-agent delegation | Yes | ✅ Done | ADK `AgentTool` + `_sanitize_agent_name()` | Phase 2 |
| General-purpose sub-agent | Yes | ✅ Done | `general_purpose` AgentTool | Phase 2 |
| Custom sub-agents | Yes | ✅ Done | ADK `AgentTool` from `SubAgentSpec` | Phase 2 |
| Parallel sub-agent execution | Yes | ✅ Done | ADK `AgentTool` (native) | Phase 2 |
| Conversation summarization | Yes | ✅ Done | `summarization.py` + `before_model_callback` | Phase 5 |
| Memory (AGENTS.md) | Yes | ✅ Done | `memory.py` + callback injection | Phase 4 |
| Skills (SKILL.md) discovery | Yes | ✅ Done | **adk-skills** `SkillsRegistry.discover()` | Phase 4 |
| Skills progressive disclosure | Yes | ✅ Done | **adk-skills** `use_skill` tool | Phase 4 |
| Skills prompt injection | Yes | ✅ Done | `inject_skills_into_prompt()` | Phase 4 |
| Skills script execution | Yes | ✅ Done | **adk-skills** `run_script` → `execution/bridge.py` | Phase 5 |
| Skills reference loading | N/A | ✅ Done | **adk-skills** `read_reference` tool | Phase 4 |
| Skills validation | N/A | ✅ Done | **adk-skills** `registry.validate_all()` via config | Phase 4 |
| Skills database storage | N/A | ⚠️ Deferred | **adk-skills** SQLAlchemy backend (external dep) | Phase 4 |
| Human-in-the-loop approval | Yes | ⚠️ Partial | Sets state, blocks tool, but no pause | Phase 3 |
| Large result eviction | Yes | ✅ Done | In-tool `truncate_if_too_long()` + cooperative `after_tool_callback` eviction | Phase 3/6 |
| Dangling tool call patching | Yes | ✅ Done | `before_agent_callback` detects + `before_model_callback` injects synthetic responses | Phase 6 |
| Dynamic system prompts | Yes | ✅ Done | `before_model_callback` injection | Phase 3 |
| State backend (ephemeral) | Yes | ✅ Done | `StateBackend` on session state dict | Phase 1 |
| Filesystem backend (local) | Yes | ✅ Done | `FilesystemBackend` with virtual mode | Phase 4 |
| Composite backend (routing) | Yes | ✅ Done | `backends/composite.py` | Phase 5 |
| Structured output | Yes | ✅ Done | ADK `output_schema` param | Phase 1 |
| Model agnosticism | Yes (LangChain) | ✅ Done | ADK model string | Phase 1 |
| Streaming | Yes (LangGraph) | ✅ Done | ADK Runner streaming (native) | Phase 1 |
| Checkpointing/persistence | Yes (LangGraph) | ❌ Pending | ADK `SessionService` | Phase 5 |
| Prompt caching (Anthropic) | Yes | N/A | Model-specific, not needed | - |
| Shared Python/Bash workspace | N/A | ✅ Done | **Heimdall MCP** shared `/workspace` | Phase 5 |
| Package installation (sandbox) | N/A | ✅ Done | **Heimdall MCP** `install_packages` | Phase 5 |
