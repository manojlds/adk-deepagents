"""System prompt templates.

Ported from deepagents middleware prompts, adapted for ADK.
"""

# ---------------------------------------------------------------------------
# Base agent prompt
# ---------------------------------------------------------------------------

BASE_AGENT_PROMPT = (
    "In order to complete the objective that the user asks of you, "
    "you have access to a number of standard tools."
)

# ---------------------------------------------------------------------------
# Filesystem tools prompt
# ---------------------------------------------------------------------------

FILESYSTEM_SYSTEM_PROMPT = """\
## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`

You have access to a filesystem which you can interact with using these tools.
All file paths must start with a /.

- ls: list files in a directory (requires absolute path)
- read_file: read a file from the filesystem
- write_file: write to a file in the filesystem (creates new files only)
- edit_file: edit a file in the filesystem using string replacement
- glob: find files matching a pattern (e.g., "**/*.py")
- grep: search for text within files"""

# ---------------------------------------------------------------------------
# Execution tools prompt
# ---------------------------------------------------------------------------

EXECUTION_SYSTEM_PROMPT = """\
## Execute Tool `execute`

You have access to an `execute` tool for running shell commands in a sandboxed environment.
Use this tool to run commands, scripts, tests, builds, and other shell operations.

- execute: run a shell command in the sandbox (returns output and exit code)"""

# ---------------------------------------------------------------------------
# Todo tools prompt
# ---------------------------------------------------------------------------

TODO_SYSTEM_PROMPT = """\
## Todo Tools `write_todos`, `read_todos`

You have access to a todo list for tracking tasks and progress.

- write_todos: create or update the todo list with a list of items
- read_todos: read the current todo list"""

# ---------------------------------------------------------------------------
# Conversation compaction prompt
# ---------------------------------------------------------------------------

COMPACT_CONVERSATION_SYSTEM_PROMPT = """\
## Context Compaction Tool `compact_conversation`

You can call `compact_conversation` to summarize older conversation messages
and refresh context budget.

Use it when:
- You are switching to a new task and old context is no longer important
- You finished a large investigation and want to reduce context bloat
- The user explicitly asks to compact/summarize conversation history

The tool takes no arguments."""

# ---------------------------------------------------------------------------
# Sub-agent / task tool prompt
# ---------------------------------------------------------------------------

TASK_TOOL_DESCRIPTION = """\
Launch a sub-agent to handle a specific task autonomously.

The task tool launches specialized agents that autonomously handle tasks.
Each sub-agent has specific capabilities and tools available to it.

Usage notes:
- Provide clear, detailed prompts so the sub-agent can work autonomously
- Sub-agents have access to their own set of tools
- Use sub-agents for complex, multi-step operations that benefit from isolation
- Launch multiple sub-agents in parallel when tasks are independent
- The sub-agent's result is returned to you for synthesis"""

TASK_SYSTEM_PROMPT = """\
## Sub-agent Delegation

You can delegate work to specialized sub-agents using the tools below.
Each sub-agent runs independently with its own tools and context.

**Lifecycle:**
1. **Spawn** — You call the sub-agent tool with a task description
2. **Run** — The sub-agent works autonomously using its own tools
3. **Return** — The result is returned to you
4. **Reconcile** — You synthesize the result into your response

**Tips:**
- Parallelize independent tasks by calling multiple sub-agent tools at once
- Give sub-agents clear, self-contained instructions
- Sub-agents do NOT see your conversation history"""

TASK_RUNTIME_SUBAGENT_PROMPT = """\
If the `register_subagent` tool is available, you can define specialist
sub-agents at runtime before delegating with the `task` tool.

- Use `register_subagent` when you need a new specialist role
- Then call `task` with `subagent_type` set to that registered name
- If you call `task` with a new `subagent_type`, a runtime specialist is
  created automatically using default tools"""

TASK_CONCURRENCY_SYSTEM_PROMPT = """\
## Dynamic Task Concurrency Limits

When delegating with `task`, follow these runtime limits:

- `max_parallel={max_parallel}`
- `concurrency_policy={concurrency_policy}`
- `queue_timeout_seconds={queue_timeout_seconds}`

Delegation guidance:
- Launch task calls in waves of at most `{max_parallel}` concurrent calls
- Wait for one wave to complete before starting the next wave
- Reuse `task_id` when continuing existing delegated work
- Avoid over-spawning; queue timeouts waste turns and tokens"""

# ---------------------------------------------------------------------------
# Memory prompt
# ---------------------------------------------------------------------------

MEMORY_SYSTEM_PROMPT = """\
<agent_memory>
{agent_memory}
</agent_memory>

<memory_guidelines>
You have access to persistent memory stored in AGENTS.md files.
These contain information about the project, your role, and accumulated knowledge.

When to update memory:
- When the user explicitly asks you to remember something
- When you discover important role descriptions or project context
- When you receive feedback about your behavior or approach
- When you find tool-specific information that would help future sessions
- When you notice patterns in how the user works

When NOT to update memory:
- Transient information (current task details, temporary data)
- One-time tasks that won't recur
- Simple factual questions
- Small talk or greetings
- Information already in memory

Never store API keys, passwords, or credentials in memory.
</memory_guidelines>"""

# ---------------------------------------------------------------------------
# Default sub-agent prompts
# ---------------------------------------------------------------------------

DEFAULT_SUBAGENT_PROMPT = (
    "In order to complete the objective that the user asks of you, "
    "you have access to a number of standard tools."
)

DEFAULT_GENERAL_PURPOSE_DESCRIPTION = (
    "General-purpose agent for researching complex questions, searching for files "
    "and content, and executing multi-step tasks. When you are searching for a "
    "keyword or file and are not confident that you will find the right match in "
    "the first few tries use this agent to perform the search for you. This agent "
    "has access to all tools as the main agent."
)

# ---------------------------------------------------------------------------
# Tool descriptions (for dynamic injection)
# ---------------------------------------------------------------------------

LIST_FILES_TOOL_DESCRIPTION = (
    "List files and directories at the given path. "
    "Returns name, type (file/dir), size, and modification time."
)

READ_FILE_TOOL_DESCRIPTION = (
    "Read the contents of a file with optional pagination. "
    "Returns content with line numbers. Use offset and limit for large files."
)

WRITE_FILE_TOOL_DESCRIPTION = (
    "Create a new file with the given content. "
    "Cannot overwrite existing files — use edit_file for modifications."
)

EDIT_FILE_TOOL_DESCRIPTION = (
    "Edit a file by replacing a specific string with a new string. "
    "The old_string must uniquely identify the location to edit, "
    "unless replace_all is set to True."
)

GLOB_TOOL_DESCRIPTION = (
    "Find files matching a glob pattern. "
    "Supports ** for recursive matching and {} for alternatives."
)

GREP_TOOL_DESCRIPTION = (
    "Search for a text pattern within files. "
    "Returns matching files, line numbers and content, or match counts."
)
