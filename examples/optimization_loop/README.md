# Self-Improving Agent — Optimization Loop Example

This example demonstrates the full optimization cycle: an agent that improves
its own prompt by replaying test scenarios, evaluating results with an LLM
judge, and applying the suggested improvements.

## How it works

```
┌─────────────────────────────────────────────────────────┐
│ 1. SEED RUNS                                            │
│    Run the agent on test prompts → baseline trajectories │
├─────────────────────────────────────────────────────────┤
│ 2. OPTIMIZATION LOOP (repeats N iterations)             │
│    ┌─────────────┐  ┌──────────────┐  ┌──────────────┐ │
│    │ Replay each │→ │ LLM Judge    │→ │ Reflector    │ │
│    │ prompt with │  │ scores the   │  │ suggests     │ │
│    │ current     │  │ replay on    │  │ prompt/skill │ │
│    │ config      │  │ task quality,│  │ improvements │ │
│    │             │  │ efficiency,  │  │              │ │
│    │             │  │ tool usage   │  │              │ │
│    └─────────────┘  └──────────────┘  └──────────────┘ │
│                                            ↓            │
│                                   Auto-apply prompt     │
│                                   changes, re-iterate   │
├─────────────────────────────────────────────────────────┤
│ 3. RESULTS                                              │
│    Score progression, optimized instruction,             │
│    all suggestions (applied + manual)                    │
└─────────────────────────────────────────────────────────┘
```

## Running

```bash
# With Google AI (Gemini)
GOOGLE_API_KEY=... uv run python examples/optimization_loop/run.py

# With OpenAI-compatible API
OPENAI_API_KEY=... ADK_DEEPAGENTS_MODEL=openai/gpt-4o-mini \
  uv run python examples/optimization_loop/run.py

# With a custom endpoint
OPENAI_API_KEY=... OPENAI_API_BASE=https://your-api.com/v1 \
  ADK_DEEPAGENTS_MODEL=openai/your-model \
  uv run python examples/optimization_loop/run.py
```

## What gets optimized

The loop can auto-apply:
- **`instruction_append`** — adds guidance to the agent's system prompt
- **`instruction_replace`** — rewrites the system prompt
- **`skill_add` / `skill_remove`** — modifies which skill directories are loaded

It also produces manual suggestions for:
- **`tool_definition_note`** — notes about tool usage improvements (not auto-applied)

## Files

- `run.py` — Main example script
- `skills/content-writing/SKILL.md` — Sample skill for content creation
- `README.md` — This file
