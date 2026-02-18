# Human-in-the-Loop Approval

## Overview

adk-deepagents supports requiring human approval before specific tools execute. This is built on ADK's `ToolConfirmation` system and allows you to gate dangerous operations (file writes, code execution, etc.) behind explicit human review.

## How It Works

The approval flow uses ADK's `request_confirmation()` mechanism to truly pause the agent execution loop:

### Step 1: First Invocation

When a tool that requires approval is called, the `before_tool_callback`:

1. Checks if the tool name is in `tools_requiring_approval`
2. Sees that `tool_context.tool_confirmation` is `None` (no prior confirmation)
3. Calls `tool_context.request_confirmation()` with the tool name, args, and a unique `approval_id`
4. Sets `tool_context.actions.skip_summarization = True`
5. Returns a dict with `status: "awaiting_approval"`

The agent **halts** and waits for a human response.

### Step 2: Resume

The caller (CLI, web UI, test harness) creates a `ToolConfirmation` and feeds it back to ADK:

- **`confirmed=True`**: The tool proceeds with the original (or modified) arguments
- **`confirmed=False`**: The tool is skipped and a rejection message is returned

```
┌─────────┐     ┌──────────┐     ┌───────────┐     ┌─────────┐
│  Agent   │────▶│  Tool    │────▶│ Callback  │────▶│  Human  │
│          │     │  Call    │     │ (pause)   │     │ Review  │
└─────────┘     └──────────┘     └───────────┘     └────┬────┘
                                                        │
                                      ┌─────────────────┘
                                      ▼
                               ┌─────────────┐
                               │ Approve /   │
                               │ Reject /    │
                               │ Modify Args │
                               └─────────────┘
```

## interrupt_on Parameter

The `interrupt_on` parameter is a `dict[str, bool]` mapping tool names to whether they require approval:

```python
from adk_deepagents import create_deep_agent

agent = create_deep_agent(
    interrupt_on={
        "write_file": True,
        "edit_file": True,
        "execute": True,
    },
)
```

Only tools with `True` values require approval. Tools not in the dict or set to `False` proceed normally.

## make_before_tool_callback

The `make_before_tool_callback` function in `adk_deepagents.callbacks.before_tool` creates the callback:

```python
from adk_deepagents.callbacks.before_tool import make_before_tool_callback

callback = make_before_tool_callback(
    interrupt_on={"write_file": True, "execute": True},
)
```

Returns `None` if no tools need approval (empty dict or all `False`).

The callback signature is:

```python
def before_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> dict | None:
```

- Returns `None` to proceed with tool execution
- Returns a dict to skip execution and use the dict as the tool result

## resume_approval Helper

The `resume_approval` function creates `ToolConfirmation` objects for resuming interrupted tool calls:

```python
from adk_deepagents.callbacks.before_tool import resume_approval

# Approve
confirmation = resume_approval(approved=True)

# Reject
confirmation = resume_approval(approved=False)

# Approve with modified arguments
confirmation = resume_approval(
    approved=True,
    modified_args={"file_path": "/safe/path.txt"},
)
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `approved` | `bool` | Whether the human approved the tool call |
| `modified_args` | `dict \| None` | Replacement arguments (only used when approved) |

The returned `ToolConfirmation` has:
- `confirmed` — matches `approved`
- `payload` — `{"modified_args": {...}}` if modified_args provided, else `None`

## Modified Arguments

When approving with modified arguments, the callback calls `args.update(modified)` to merge the modifications into the original arguments. This allows you to:

- Change a file path to a safer location
- Modify a command before execution
- Adjust parameters while still approving the action

```python
# Original tool call: write_file(file_path="/etc/config", content="...")
# Approve but redirect to a safe path:
confirmation = resume_approval(
    approved=True,
    modified_args={"file_path": "/workspace/config"},
)
```

## Sub-Agent HITL

Each sub-agent can have its own `interrupt_on` configuration, independent of the parent agent. This is set in the `SubAgentSpec`:

```python
from adk_deepagents import SubAgentSpec, create_deep_agent

# Parent agent has no HITL
# But the coder sub-agent requires approval for writes
coder = SubAgentSpec(
    name="coder",
    description="Coding agent with write approval.",
    interrupt_on={"write_file": True},
)

agent = create_deep_agent(
    subagents=[coder],
    # No interrupt_on on parent
)
```

The sub-agent gets its own `before_tool_callback` created by `make_before_tool_callback`.

## Examples

### Basic: Interrupt write_file and execute

```python
from adk_deepagents import create_deep_agent

agent = create_deep_agent(
    instruction="You are a coding assistant.",
    execution="local",
    interrupt_on={
        "write_file": True,
        "execute": True,
    },
)
```

### Approving a Tool Call

```python
from adk_deepagents.callbacks.before_tool import resume_approval

# When the agent pauses for approval, create a confirmation:
confirmation = resume_approval(approved=True)

# Pass this confirmation back to ADK when resuming the agent.
# The exact mechanism depends on your runner (CLI, web UI, etc.).
```

### Rejecting a Tool Call

```python
from adk_deepagents.callbacks.before_tool import resume_approval

confirmation = resume_approval(approved=False)

# The tool will be skipped and the agent receives:
# {"status": "rejected", "tool": "write_file",
#  "message": "Tool 'write_file' was rejected by the human reviewer."}
```

### Approving with Modified Arguments

```python
from adk_deepagents.callbacks.before_tool import resume_approval

# The agent tried to write to /etc/hosts — redirect to workspace
confirmation = resume_approval(
    approved=True,
    modified_args={
        "file_path": "/workspace/hosts",
    },
)
```

### Sub-Agent Specific HITL

```python
from adk_deepagents import SubAgentSpec, create_deep_agent

# The researcher can run freely
researcher = SubAgentSpec(
    name="researcher",
    description="Research agent — no approval needed.",
)

# The deployer needs approval for everything dangerous
deployer = SubAgentSpec(
    name="deployer",
    description="Deployment agent — requires approval for execution.",
    interrupt_on={
        "execute": True,
        "write_file": True,
        "edit_file": True,
    },
)

agent = create_deep_agent(
    subagents=[researcher, deployer],
    execution="local",
)
```
