# DRS Project Context

## Architecture
adk-deepagents is a Python library that re-implements deepagents using Google ADK primitives.
It provides planning, filesystem operations, shell execution, sub-agent delegation,
conversation summarization, memory management, and skills integration — all wired
together through a single `create_deep_agent()` factory function.

### Key Components
- **`create_deep_agent()`**: Main factory function (`graph.py`) that assembles an ADK
  `LlmAgent` with tools, callbacks, sub-agents, memory, and skills.
- **Backends**: Pluggable storage layer (`Backend` ABC) with `StateBackend` (ADK session
  state) and `FilesystemBackend` (local disk) implementations plus a `SandboxBackend`
  variant for shell execution.
- **Callbacks**: Lifecycle hooks (`before_agent`, `before_model`, `before_tool`,
  `after_tool`) that inject memory, manage tool state, and enforce interrupt-on policies.
- **Tools**: File operations (`read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep`),
  todo management (`read_todos`, `write_todos`), and sub-agent delegation (`AgentTool`).
- **Execution**: Local subprocess execution and Heimdall MCP integration (sandboxed
  Pyodide/WASM Python and Bash).
- **Skills**: Optional integration with `adk-skills-agent` for Agent Skills discovery and use.
- **Memory**: AGENTS.md loading and system prompt injection.
- **Summarization**: Conversation summarization configuration.

## Technology Stack
- **Language**: Python 3.11+
- **Framework**: Google ADK (`google-adk>=1.0.0`)
- **Glob matching**: wcmatch
- **Optional skills**: adk-skills-agent
- **Build system**: Hatchling
- **Package manager**: uv
- **Testing**: Pytest + pytest-asyncio (asyncio_mode = "auto")
- **Linting/Formatting**: Ruff (line length 100, target py311)
- **Type checking**: ty

## Trust Boundaries

### Trusted Inputs
- **ADK agent code**: The library is embedded in trusted Python applications using
  Google ADK.
- **Configuration**: `create_deep_agent()` parameters are set by the application developer.
- **Backend implementations**: Chosen and configured by the application developer.
- **Memory files**: AGENTS.md files are authored by developers and loaded from trusted paths.

### Semi-Trusted Inputs
- **Skill directories**: Paths provided to skills discovery are caller-supplied and validated.
- **Tool inputs**: File paths and patterns from the LLM are validated by backend methods.
- **Sub-agent specs**: Sub-agent definitions come from the application developer.

### NOT Web-Facing
- This is a Python library embedded in ADK agent applications.
- No public web endpoints or direct untrusted network inputs.
- The library does not accept arbitrary user input directly — it processes LLM tool calls
  within the ADK framework.

## Security Context

### Standard Practices (NOT Security Issues)
- Reading/writing files through the Backend abstraction layer
- Loading AGENTS.md memory files from configured paths
- Glob/grep operations scoped to the backend's filesystem
- Sub-agent delegation through ADK's AgentTool mechanism
- YAML/Markdown parsing for skill metadata (handled by adk-skills)
- Local subprocess execution when explicitly enabled by the developer

### What Actually Matters
- Path traversal in file operations (backends must validate paths)
- Shell injection in local execution tools
- Resource limits and timeouts for code execution
- Secrets leaking through tool outputs or error messages
- Validation of skill metadata when skills integration is enabled
- Proper isolation between sub-agent contexts

## Review Guidelines

### Focus Areas
- Correctness of tool implementations and backend operations
- Safe defaults for execution backends (local, Heimdall)
- Proper async patterns and ADK callback contracts
- Type safety with TypedDict specs and Protocol/ABC adherence
- Error handling that doesn't expose internal state
- Clean separation between backend implementations

### Avoid Over-Flagging
- Don't flag standard ADK callback patterns as issues
- Internal file operations through the Backend ABC are by design — not a vulnerability
- Sub-agent tool inheritance is intentional architecture, not a privilege escalation
- Optional imports (`adk-skills-agent`) with silent fallback are expected
- `StateBackend` storing data in ADK session state is the intended mechanism
