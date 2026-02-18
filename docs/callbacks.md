# Callback System

## Overview

adk-deepagents uses ADK's four callback points to implement middleware functionality. Each callback is a function created by a factory (`make_*_callback`) and composed with optional user-provided callbacks via `_compose_callbacks`.

The callbacks handle memory loading, system prompt injection, human-in-the-loop approval, large result eviction, conversation summarization, and dangling tool call patching.

## Four Callback Points

| Callback | When | Module |
|---|---|---|
| `before_agent` | Once when the agent starts | `callbacks/before_agent.py` |
| `before_model` | Before every LLM call | `callbacks/before_model.py` |
| `before_tool` | Before each tool execution | `callbacks/before_tool.py` |
| `after_tool` | After each tool execution | `callbacks/after_tool.py` |

## Built-in Callbacks

### before_agent_callback

Created by `make_before_agent_callback`. Runs once at agent startup.

**What it does:**

1. **Store backend factory** — Saves `backend_factory` into `state["_backend_factory"]` so filesystem tools can resolve it later
2. **Patch dangling tool calls** — Scans session events for orphaned `function_call` parts (calls without matching `function_response`). Stores dangling call info in `state["_dangling_tool_calls"]` for the `before_model_callback` to inject synthetic responses
3. **Load memory** — If `memory_sources` are configured and not yet loaded, creates a backend and loads the AGENTS.md files into `state["memory_contents"]`

```python
from adk_deepagents.callbacks.before_agent import make_before_agent_callback

callback = make_before_agent_callback(
    memory_sources=["./AGENTS.md"],
    backend_factory=my_factory,
)
```

### before_model_callback

Created by `make_before_model_callback`. Runs before every LLM call.

**What it does:**

1. **Patch dangling tool responses** — Reads `state["_dangling_tool_calls"]` and injects synthetic `function_response` parts into `llm_request.contents` for any orphaned calls. The synthetic responses have `status: "cancelled"` and explain the call was not completed.
2. **Inject system prompts** — Appends documentation blocks to the system instruction:
   - `TODO_SYSTEM_PROMPT` — Todo tools documentation
   - `FILESYSTEM_SYSTEM_PROMPT` — Filesystem tools documentation
   - `EXECUTION_SYSTEM_PROMPT` — Execution tools documentation (if `has_execution`)
   - `MEMORY_SYSTEM_PROMPT` — Memory contents (if `memory_sources` configured)
   - `TASK_SYSTEM_PROMPT` + sub-agent list (if `subagent_descriptions` provided)
3. **Trigger summarization** — Calls `maybe_summarize()` if `summarization_config` is set

```python
from adk_deepagents.callbacks.before_model import make_before_model_callback

callback = make_before_model_callback(
    memory_sources=["./AGENTS.md"],
    has_execution=True,
    subagent_descriptions=[{"name": "researcher", "description": "..."}],
    summarization_config=my_config,
    backend_factory=my_factory,
)
```

### before_tool_callback

Created by `make_before_tool_callback`. Runs before each tool execution.

**What it does:**

- **Human-in-the-loop approval** — If the tool name is in `tools_requiring_approval`, calls `tool_context.request_confirmation()` to pause the agent. On resume, checks the `ToolConfirmation` to approve (with optional arg modification) or reject.

Returns `None` if no tools need approval. See [Human-in-the-Loop](./human-in-the-loop.md) for details.

### after_tool_callback

Created by `make_after_tool_callback`. Runs after each tool execution.

**What it does:**

- **Large result eviction** — For custom/external tools that opt in via the cooperative `_last_tool_result` pattern in `tool_context.state`:
  1. Checks if the raw result exceeds the token threshold (`TOOL_RESULT_TOKEN_LIMIT`)
  2. Saves the full result to the backend at `/large_tool_results/{tool_name}_{call_id}`
  3. Returns a preview with the saved file path

Built-in filesystem tools (`ls`, `glob`, `grep`, `read_file`, `edit_file`, `write_file`) are excluded from eviction — they handle truncation inline via `truncate_if_too_long()`.

```python
from adk_deepagents.callbacks.after_tool import make_after_tool_callback

callback = make_after_tool_callback(
    backend_factory=my_factory,
    token_limit=25000,
)
```

## extra_callbacks Parameter

The `extra_callbacks` parameter on `create_deep_agent` lets you compose custom callbacks after the built-in ones:

```python
from adk_deepagents import create_deep_agent

agent = create_deep_agent(
    extra_callbacks={
        "before_agent": my_before_agent,
        "before_model": my_before_model,
        "before_tool": my_before_tool,
        "after_tool": my_after_tool,
    },
)
```

Only include the callbacks you need — missing keys are ignored.

## _compose_callbacks

The `_compose_callbacks` function composes a built-in callback with an extra callback:

```python
composed = _compose_callbacks(builtin, extra)
```

**Semantics:**

1. The built-in callback runs **first**
2. If it returns a **non-`None`** value (short-circuit), the extra callback is **not called** and the built-in result is returned
3. If it returns `None`, the extra callback is called with the same arguments
4. If either side is `None`, the other is returned as-is

This ensures built-in behavior (like HITL approval) takes priority, while still allowing user extensions.

## _append_to_system_instruction

The `_append_to_system_instruction` helper handles all variants of the system instruction:

| Existing Type | Behavior |
|---|---|
| `None` | Sets as new string |
| `str` | Concatenates with `\n\n` separator |
| `Content` | Appends a new `Part(text=...)` |
| Other | Converts to string and concatenates |

## Dangling Tool Call Patching

Dangling tool calls are orphaned `function_call` parts in model messages that have no corresponding `function_response`. This can happen when:

- The agent is interrupted mid-execution
- A tool execution crashes
- A session is resumed from a checkpoint

Without patching, these cause LLM API errors because most models require every `function_call` to have a matching response.

**Patching flow:**

1. `before_agent_callback` scans session events, collects call IDs and response IDs
2. Dangling calls (calls without responses) are stored in `state["_dangling_tool_calls"]`
3. `before_model_callback` reads this state and injects synthetic responses into `llm_request.contents`
4. Each synthetic response has `role: "user"` with a `FunctionResponse` containing `status: "cancelled"`

## Examples

### Adding a Logging Callback

```python
import logging

from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from adk_deepagents import create_deep_agent

logger = logging.getLogger(__name__)


def my_before_agent(callback_context: CallbackContext) -> types.Content | None:
    logger.info("Agent starting, session: %s", callback_context.session.id)
    return None


agent = create_deep_agent(
    extra_callbacks={"before_agent": my_before_agent},
)
```

### Custom before_model for Prompt Augmentation

```python
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse

from adk_deepagents import create_deep_agent
from adk_deepagents.callbacks.before_model import _append_to_system_instruction


def augment_prompt(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    # Add custom context to every LLM call
    user_prefs = callback_context.state.get("user_preferences", "")
    if user_prefs:
        _append_to_system_instruction(
            llm_request,
            f"## User Preferences\n{user_prefs}",
        )
    return None


agent = create_deep_agent(
    extra_callbacks={"before_model": augment_prompt},
)
```

### Custom before_tool for Rate Limiting

```python
import time
from typing import Any

from google.adk.tools import BaseTool, ToolContext

from adk_deepagents import create_deep_agent

_last_call: dict[str, float] = {}
MIN_INTERVAL = 1.0  # seconds


def rate_limit(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> dict | None:
    tool_name = getattr(tool, "name", "")
    now = time.time()
    last = _last_call.get(tool_name, 0)
    if now - last < MIN_INTERVAL:
        return {"error": f"Rate limited: {tool_name}. Wait {MIN_INTERVAL}s between calls."}
    _last_call[tool_name] = now
    return None


agent = create_deep_agent(
    extra_callbacks={"before_tool": rate_limit},
)
```
