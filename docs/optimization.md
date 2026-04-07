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

## Library Workflow

Optimization is a library-first workflow:

1. Capture trajectories from agent runs (for example from OTEL traces).
2. Store and curate trajectories (including golden labels and feedback).
3. Replay trajectories with current agent configuration.
4. Evaluate replays with the LLM judge.
5. Run iterative optimization with `run_optimization_loop`.

Use the Python API sections below for concrete evaluator, replay, and loop usage.

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
