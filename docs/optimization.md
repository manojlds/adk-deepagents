# Optimization & Trajectory Evaluation

adk-deepagents includes an optimization pipeline that captures agent execution
trajectories, evaluates them with an LLM judge, replays them under different
configurations, and iteratively improves agent prompts and skills.

## Overview

```
Chat with agent → OTEL captures traces → Trajectories stored
                                              ↓
                          ┌───────────────────────────────────┐
                          │     Optimization Loop             │
                          │                                   │
                          │  Replay each trajectory           │
                          │       ↓                           │
                          │  LLM Judge scores replay          │
                          │       ↓                           │
                          │  Compare to baseline              │
                          │       ↓                           │
                          │  Reflector suggests improvements  │
                          │       ↓                           │
                          │  Auto-apply prompt/skill changes  │
                          │       ↓                           │
                          │  Iterate until convergence        │
                          └───────────────────────────────────┘
```

## TUI Workflow

The TUI (`adk-deepagents --tui`) provides the full workflow interactively.

### Step 1: Chat and collect trajectories

Chat with the agent normally. OTEL traces are captured automatically when
the OTEL collector is configured (via `devenv` or `OTEL_TRACES_FILE` env var).

```
you> Write a blog post about Python for beginners
assistant> [writes content using tools]
```

### Step 2: Review trajectories

List all captured trajectories:

```
/trajectories
```

Or open the trajectory review overlay for a visual picker:

```
/trajectories review
```

The review overlay supports keyboard shortcuts:

| Key | Action |
|-----|--------|
| `Enter` | Show trajectory details |
| `v` | Evaluate with LLM judge |
| `p` | Replay with current config |
| `g` | Toggle golden mark |
| `0`–`5` | Quick score (0→0.0, 5→1.0) |
| `r` | Refresh from OTEL traces |
| `e` | Export dataset |
| `Esc` | Close overlay |

### Step 3: Mark golden trajectories

Mark high-quality trajectories as "golden" — these serve as the baseline
for optimization:

```
/trajectories mark <trace_id_prefix>
```

In the review overlay, press `g` on a highlighted trajectory.

### Step 4: Evaluate trajectories

Run the LLM judge on a trajectory to get a structured score:

```
/trajectories evaluate <trace_id_prefix>
```

The evaluator scores on three criteria (default rubric):

| Criterion | Weight | Description |
|-----------|--------|-------------|
| `task_completion` | 0.6 | Did the agent complete the requested task? |
| `efficiency` | 0.2 | Was the agent efficient (steps, tokens, retries)? |
| `tool_usage_quality` | 0.2 | Were tools used correctly and appropriately? |

Output includes per-criterion scores, strengths, issues, and an overall
weighted score that is saved to the trajectory store.

### Step 5: Replay trajectories

Re-run a trajectory's original task with the current agent configuration:

```
/trajectories replay <trace_id_prefix>
```

The replay:
- Extracts the original user prompt(s) from the trajectory
- Runs a fresh agent session with the current config
- Auto-approves tool confirmation requests
- Saves the replay as a new trajectory linked to the original

### Step 6: Run the optimization loop

Run the full autoresearch-style optimization loop:

```
/optimize loop
```

With options:

```
/optimize loop --golden-only --max-iter 3
```

| Flag | Default | Description |
|------|---------|-------------|
| `--golden-only` | off | Only optimize over golden trajectories |
| `--max-iter N` | 2 | Maximum optimization iterations |

The loop:
1. Replays each trajectory with the current agent config
2. Evaluates each replay with the LLM judge
3. Compares replay scores to baseline scores
4. Uses a reflector LLM to suggest improvements
5. Auto-applies prompt and skill changes
6. Repeats until convergence or max iterations

Progress is displayed in real-time:

```
Starting optimization loop: 3 trajectories, max 2 iterations
--- Iteration 1 ---
  Avg score: 0.720
  Avg delta: -0.280
  Regressions: 1
  • instruction_append [auto]: Add SEO guidelines to improve content quality
--- Iteration 2 ---
  Avg score: 0.850
  Avg delta: +0.130
  Regressions: 0
Optimization complete: converged
```

## Slash Command Reference

### Trajectory management

| Command | Description |
|---------|-------------|
| `/trajectories` | List all trajectories (auto-imports from OTEL) |
| `/trajectories golden` | List golden trajectories only |
| `/trajectories show <id> [--detail]` | Show trajectory flow |
| `/trajectories mark <id>` | Mark as golden |
| `/trajectories unmark <id>` | Remove golden mark |
| `/trajectories rate <id> <0-1>` | Set score manually |
| `/trajectories feedback <id> <0-1> [comment]` | Add feedback entry |
| `/trajectories tag <id> <key> <value>` | Set a tag |
| `/trajectories untag <id> <key>` | Remove a tag |
| `/trajectories export [path]` | Export dataset (summary or JSONL) |
| `/trajectories review` | Open trajectory review overlay |
| `/trajectories evaluate <id>` | Run LLM judge |
| `/trajectories replay <id>` | Replay with current config |

### Optimization

| Command | Description |
|---------|-------------|
| `/optimize loop` | Run optimization loop on all trajectories |
| `/optimize loop --golden-only` | Loop on golden trajectories only |
| `/optimize loop --max-iter N` | Set iteration count |
| `/optimize gepa [path] [--run]` | Export GEPA dataset / run external optimizer |

## Python API

All optimization features are also available as a Python API for
programmatic use, testing, or integration into custom workflows.

### Evaluator

```python
from adk_deepagents.optimization import evaluate_trajectory, EvaluationRubric

# Evaluate with default rubric
feedback = await evaluate_trajectory(trajectory, model="gemini-2.5-flash")

print(feedback.rating)   # 0.0–1.0
print(feedback.comment)  # Summary from the judge
print(feedback.metadata) # Per-criterion scores, strengths, issues
```

Custom rubric:

```python
from adk_deepagents.optimization.evaluator import (
    EvaluationCriterion,
    EvaluationRubric,
    evaluate_trajectory,
)

rubric = EvaluationRubric(
    criteria=[
        EvaluationCriterion(name="accuracy", description="...", weight=0.5),
        EvaluationCriterion(name="style", description="...", weight=0.5),
    ],
    name="custom_v1",
)

feedback = await evaluate_trajectory(trajectory, model="gemini-2.5-flash", rubric=rubric)
```

### Replay

```python
from adk_deepagents import create_deep_agent
from adk_deepagents.optimization import BuiltAgent, ReplayConfig, replay_trajectory

def builder() -> BuiltAgent:
    agent = create_deep_agent(model="gemini-2.5-flash", name="my_agent")
    return BuiltAgent(agent=agent)

# Single-turn replay
result = await replay_trajectory(
    trajectory,
    agent_builder=builder,
    config=ReplayConfig(tool_approval="auto_approve"),
)

print(result.output_text)
print(result.replay_trajectory)
```

Multi-turn with user simulator:

```python
async def user_simulator(original_task, previous_messages, agent_output):
    """Generate contextual follow-up based on agent's response."""
    if len(previous_messages) >= 3:
        return ""  # End conversation
    return "Looks good, now expand the introduction."

config = ReplayConfig(
    tool_approval="auto_approve",
    user_simulator=user_simulator,
)

result = await replay_trajectory(trajectory, agent_builder=builder, config=config)
print(result.per_turn_outputs)  # Output from each turn
```

### Optimization Loop

```python
from adk_deepagents import create_deep_agent
from adk_deepagents.optimization import (
    BuiltAgent,
    OptimizationCandidate,
    ReplayConfig,
    TrajectoryStore,
    run_optimization_loop,
)

store = TrajectoryStore("./trajectories")
golden = store.list_trajectories(is_golden=True)

base = OptimizationCandidate(agent_kwargs={
    "name": "my_agent",
    "instruction": "You are a helpful assistant.",
    "execution": "local",
})

def agent_factory(candidate):
    agent = create_deep_agent(**{**candidate.agent_kwargs, "model": "gemini-2.5-flash"})
    return BuiltAgent(agent=agent)

result = await run_optimization_loop(
    trajectories=golden,
    base_candidate=base,
    agent_builder_factory=agent_factory,
    evaluator_model="gemini-2.5-flash",
    replay_config=ReplayConfig(tool_approval="auto_approve"),
    store=store,
    max_iterations=3,
    apply_mode="prompt_and_skills",
)

print(result.stopped_reason)
print(result.best_candidate.agent_kwargs["instruction"])
```

### ReplayConfig Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tool_approval` | `"auto_approve"` / `"auto_reject"` / `"original"` | `"auto_approve"` | How to handle tool confirmation requests |
| `user_simulator` | `Callable` or `None` | `None` | LLM-based follow-up generator for multi-turn |
| `max_approval_rounds` | `int` | `10` | Safety limit on approval round-trips per turn |

Tool approval policies:

- **`auto_approve`** — approve all tool calls (tests full agent capability)
- **`auto_reject`** — reject all (tests how agent handles denied tools)
- **`original`** — approve only tools that were used in the original trajectory

## Example

See `examples/optimization_loop/` for a complete end-to-end example with
single-turn prompts, multi-turn user simulation, and the full optimization
loop. Run it with:

```bash
GOOGLE_API_KEY=... uv run python examples/optimization_loop/run.py

# Or with OpenAI-compatible API
OPENAI_API_KEY=... ADK_DEEPAGENTS_MODEL=openai/gpt-4o-mini \
  uv run python examples/optimization_loop/run.py
```
