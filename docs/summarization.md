# Conversation Summarization

## Overview

Long agent conversations can exceed the model's context window. adk-deepagents provides automatic conversation summarization that monitors token usage, summarizes older messages when a threshold is reached, and offloads the full history to the backend for reference. This keeps the agent functional during extended sessions without losing critical context.

The summarization system is implemented in `adk_deepagents.summarization` and integrates with the `before_model_callback` via the `maybe_summarize()` function.

## SummarizationConfig

Configure summarization with the `SummarizationConfig` dataclass:

```python
from adk_deepagents import SummarizationConfig

config = SummarizationConfig(
    model="gemini-2.5-flash",
    trigger=("fraction", 0.85),
    keep=("messages", 6),
    history_path_prefix="/conversation_history",
    use_llm_summary=True,
    truncate_args=None,
    context_window=None,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"gemini-2.5-flash"` | Model for LLM-based summary generation |
| `trigger` | `tuple[str, float]` | `("fraction", 0.85)` | When to trigger summarization |
| `keep` | `tuple[str, int]` | `("messages", 6)` | How many recent messages to keep verbatim |
| `history_path_prefix` | `str` | `"/conversation_history"` | Path prefix for offloaded history files |
| `use_llm_summary` | `bool` | `True` | Use LLM to generate summaries (vs inline fallback) |
| `truncate_args` | `TruncateArgsConfig \| None` | `None` | Optional tool argument truncation |
| `context_window` | `int \| None` | `None` | Explicit context window size in tokens |

## Token Counting

Token counting uses a character-based heuristic (~4 characters per token):

```python
from adk_deepagents.summarization import (
    count_tokens_approximate,
    count_content_tokens,
    count_messages_tokens,
)

count_tokens_approximate("Hello world")  # → 2 (11 chars // 4)
count_content_tokens(content_message)    # Sums tokens across all parts
count_messages_tokens(all_messages)      # Sums across all messages
```

The constant `NUM_CHARS_PER_TOKEN = 4` is used throughout.

## Trigger Mechanism

Summarization triggers when the total token count of conversation messages exceeds a fraction of the context window:

```
trigger_threshold = context_window × trigger_fraction
```

The context window is resolved in this order:

1. **Explicit** — `SummarizationConfig.context_window` if set
2. **Model lookup** — from `MODEL_CONTEXT_WINDOWS` dict
3. **Default** — `200,000` tokens (`DEFAULT_CONTEXT_WINDOW`)

Known model context windows:

| Model | Tokens |
|---|---|
| `gemini-2.5-flash` | 1,048,576 |
| `gemini-2.5-pro` | 1,048,576 |
| `gemini-2.0-flash` | 1,048,576 |
| `gemini-1.5-pro` | 2,097,152 |
| `gpt-4o` | 128,000 |
| `gpt-4o-mini` | 128,000 |
| `claude-3.5-sonnet` | 200,000 |

## Message Partitioning

The `partition_messages` function splits the conversation into two lists:

```python
from adk_deepagents.summarization import partition_messages

to_summarize, to_keep = partition_messages(messages, keep_count=6)
```

- **`to_summarize`** — Older messages that will be replaced by a summary
- **`to_keep`** — The most recent `keep_count` messages, kept verbatim

If there are fewer messages than `keep_count`, everything is kept and nothing is summarized.

## TruncateArgsConfig

Before summarization triggers, you can truncate large `write_file` and `edit_file` arguments in older messages. This frees context window space without losing the record of which tools were called.

```python
from adk_deepagents import TruncateArgsConfig

truncate = TruncateArgsConfig(
    trigger=("fraction", 0.7),       # Start truncating at 70% of context window
    keep=("messages", 20),           # Leave the 20 most recent messages untouched
    max_length=2000,                 # Truncate args longer than 2000 chars
    truncation_text="...(argument truncated)",
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `trigger` | `tuple \| None` | `None` | When to start truncating (same format as `SummarizationConfig.trigger`) |
| `keep` | `tuple` | `("messages", 20)` | Recent messages to leave untouched |
| `max_length` | `int` | `2000` | Max character length before truncation |
| `truncation_text` | `str` | `"...(argument truncated)"` | Text appended after the 20-char prefix |

Only arguments to tools in `TRUNCATABLE_TOOLS` (`write_file`, `edit_file`) are truncated. A truncated argument keeps the first 20 characters followed by the truncation text.

## Summary Generation

### LLM-Based Summary

When `use_llm_summary=True` (default), `generate_llm_summary` calls the configured model with a structured prompt (`LLM_SUMMARY_PROMPT`). The summary follows this structure:

```markdown
## SESSION INTENT
What is the user's primary goal or request?

## SUMMARY
Important context, decisions, strategies, and reasoning.

## ARTIFACTS
Files created, modified, or accessed with descriptions of changes.

## NEXT STEPS
Remaining tasks and what to do next.
```

The messages are formatted into readable text, trimmed to `max_input_tokens` (4000 by default), and sent to the summary model.

### Inline Fallback

If the LLM summary fails or `use_llm_summary=False`, the system falls back to inline text concatenation:

1. Messages are formatted as `[role]: content` text
2. The result is truncated to 15% of the context window
3. No additional API call is made

## History Offloading

The `offload_messages_to_backend` function saves the summarized messages to the backend for future reference:

```python
path = offload_messages_to_backend(
    messages=to_summarize,
    backend=backend,
    history_path_prefix="/conversation_history",
    chunk_index=0,
)
# Returns: "/conversation_history/session_history.md"
```

The function uses an **append-based running log** — each summarization event appends a new section with a timestamp:

```markdown
## Summarized at 2025-06-15T10:30:00+00:00

[user]: Please analyze the codebase...

[model]: I'll start by reading the main files...
```

The offloaded history path is included in the summary content so the agent can refer back to it.

## Integration

The `maybe_summarize` function is called from `before_model_callback` on every LLM call:

```python
from adk_deepagents.summarization import maybe_summarize

was_summarized = maybe_summarize(
    callback_context,
    llm_request,
    context_window=1_048_576,
    trigger_fraction=0.85,
    keep_messages=6,
    backend_factory=backend_factory,
    history_path_prefix="/conversation_history",
    use_llm_summary=True,
    summary_model="gemini-2.5-flash",
    truncate_args_config=None,
)
```

**Steps performed by `maybe_summarize`:**

1. Truncate tool arguments in older messages (if `truncate_args_config` is set)
2. Count current tokens and check against trigger threshold
3. Partition messages into `to_summarize` and `to_keep`
4. Offload old messages to backend (if `backend_factory` is available)
5. Generate summary (LLM-based or inline fallback)
6. Replace old messages with summary content in `llm_request.contents`
7. Update `SummarizationState` in session state

## SummarizationState

Summarization progress is tracked in `state["_summarization_state"]`:

```python
{
    "summaries_performed": 2,
    "total_tokens_summarized": 150000,
    "last_summary": "## SESSION INTENT\nThe user wants to..."  # First 500 chars
}
```

## Examples

### Basic Summarization

```python
from adk_deepagents import SummarizationConfig, create_deep_agent

agent = create_deep_agent(
    summarization=SummarizationConfig(),  # All defaults
)
```

### Custom Trigger and Keep

```python
from adk_deepagents import SummarizationConfig, create_deep_agent

agent = create_deep_agent(
    summarization=SummarizationConfig(
        trigger=("fraction", 0.75),  # Trigger at 75% of context window
        keep=("messages", 10),       # Keep 10 most recent messages
    ),
)
```

### With Argument Truncation

```python
from adk_deepagents import SummarizationConfig, TruncateArgsConfig, create_deep_agent

agent = create_deep_agent(
    summarization=SummarizationConfig(
        truncate_args=TruncateArgsConfig(
            trigger=("fraction", 0.6),
            keep=("messages", 20),
            max_length=1000,
        ),
    ),
)
```

### Without LLM Summary

```python
from adk_deepagents import SummarizationConfig, create_deep_agent

agent = create_deep_agent(
    summarization=SummarizationConfig(
        use_llm_summary=False,  # Faster, no extra API call
    ),
)
```

### Custom Context Window

```python
from adk_deepagents import SummarizationConfig, create_deep_agent

agent = create_deep_agent(
    model="openai/gpt-4o",
    summarization=SummarizationConfig(
        model="gemini-2.5-flash",      # Use Gemini for summaries (cheaper)
        context_window=128_000,        # Override for gpt-4o
        trigger=("fraction", 0.80),
    ),
)
```
