"""Data extraction meta-agent optimization.

Demonstrates the full optimization loop with:
1. A benchmark with deterministic JSON verification (no LLM judge for scoring)
2. Train/test split with 3-step gating
3. Self-growing regression suite
4. Score history tracking
5. Learnings persistence
6. LLM judge for trajectory quality analysis (used by the reflector)

Usage:
    GOOGLE_API_KEY=... uv run python -m examples.data_extraction.run

    # With a different model:
    ADK_DEEPAGENTS_MODEL=openai/gpt-4o-mini \
      OPENAI_API_KEY=... \
      uv run python -m examples.data_extraction.run
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


MODEL = os.environ.get(
    "ADK_DEEPAGENTS_MODEL",
    os.environ.get("LITELLM_MODEL", "gemini-2.5-flash"),
)

MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "3"))

EXAMPLE_DIR = Path(__file__).parent
TASKS_DIR = EXAMPLE_DIR / "tasks"
WORKSPACE_DIR = EXAMPLE_DIR / "workspace"

INITIAL_INSTRUCTION = (
    "You are a data extraction assistant. "
    "Read input files and extract structured data as JSON. "
    "Write the result to the specified output file."
)


def _resolve_model():
    """Resolve model, using LiteLlm if not a native Gemini model."""
    if MODEL.startswith("gemini"):
        return MODEL
    try:
        from google.adk.models.lite_llm import LiteLlm

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENCODE_API_KEY", "")
        api_base = os.environ.get("OPENAI_API_BASE", "https://opencode.ai/zen/v1")
        return LiteLlm(model=MODEL, api_key=api_key, api_base=api_base)
    except ImportError:
        return MODEL


async def run_example():
    """Run the full data extraction optimization example."""
    from adk_deepagents import create_deep_agent
    from adk_deepagents.optimization import (
        BuiltAgent,
        EvaluationCriterion,
        EvaluationRubric,
        OptimizationCandidate,
        ReplayConfig,
        ScoreHistory,
        TrajectoryStore,
        run_optimization_loop,
    )
    from adk_deepagents.optimization.gate import GateConfig, RegressionSuite, run_gate
    from adk_deepagents.optimization.history import HistoryEntry
    from adk_deepagents.optimization.learnings import LearningEntry, LearningsStore
    from examples.data_extraction.benchmark import DataExtractionBenchmark

    model = _resolve_model()
    print(f"Model: {MODEL}")
    print(f"Tasks dir: {TASKS_DIR}")
    print(f"Max iterations: {MAX_ITERATIONS}")

    # Set up workspace
    WORKSPACE_DIR.mkdir(exist_ok=True)

    # Set up benchmarks (train/test split)
    train_benchmark = DataExtractionBenchmark(TASKS_DIR, split="train")
    test_benchmark = DataExtractionBenchmark(TASKS_DIR, split="test")

    print(f"Train tasks: {train_benchmark.list_task_ids()}")
    print(f"Test tasks: {test_benchmark.list_task_ids()}")

    # Set up optimization infrastructure
    store = TrajectoryStore(WORKSPACE_DIR / "trajectories")
    history = ScoreHistory(WORKSPACE_DIR / "history.jsonl")
    learnings = LearningsStore(WORKSPACE_DIR / "learnings.jsonl")
    suite = RegressionSuite.load(WORKSPACE_DIR / "suite.json")

    # Base agent configuration
    base_kwargs: dict = {
        "name": "data_extractor",
        "instruction": INITIAL_INSTRUCTION,
    }

    base_candidate = OptimizationCandidate(agent_kwargs=base_kwargs)

    # Agent builder
    def agent_builder_factory(candidate: OptimizationCandidate) -> BuiltAgent:
        kwargs = {**candidate.agent_kwargs, "model": model}
        agent = create_deep_agent(**kwargs)
        return BuiltAgent(agent=agent)

    # Agent factory for benchmarks (returns LlmAgent directly)
    def make_agent_factory(candidate: OptimizationCandidate):
        def factory():
            kwargs = {**candidate.agent_kwargs, "model": model}
            return create_deep_agent(**kwargs)

        return factory

    # Evaluation rubric
    rubric = EvaluationRubric(
        criteria=[
            EvaluationCriterion(
                name="extraction_completeness",
                description=(
                    "Did the agent extract ALL fields from the input? "
                    "Consider whether every piece of data mentioned in the "
                    "instruction was captured in the output."
                ),
                weight=0.4,
            ),
            EvaluationCriterion(
                name="format_correctness",
                description=(
                    "Is the output valid JSON matching the expected schema? "
                    "Consider field names, data types, and structure."
                ),
                weight=0.3,
            ),
            EvaluationCriterion(
                name="value_accuracy",
                description=(
                    "Are extracted values correct? Consider numbers, dates, "
                    "names, and whether data was parsed accurately from "
                    "messy input."
                ),
                weight=0.3,
            ),
        ],
        name="data_extraction_v1",
    )

    # -----------------------------------------------------------------------
    # Step 1: Baseline run
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Baseline benchmark run (train split)")
    print("=" * 60)

    baseline_result = await train_benchmark.run(make_agent_factory(base_candidate))
    print(f"  Pass rate: {baseline_result.pass_rate:.1%}")
    print(f"  Mean reward: {baseline_result.mean_reward:.3f}")
    for tid, tr in baseline_result.task_results.items():
        status = "✓" if tr.passed else "✗"
        print(f"  [{status}] {tid}: reward={tr.reward:.3f}")
        if tr.trajectory:
            store.save(tr.trajectory)

    # Record baseline
    history.append(
        HistoryEntry(
            iteration=0,
            val_score=baseline_result.mean_reward,
            pass_rate=baseline_result.pass_rate,
            description="baseline",
        )
    )

    # -----------------------------------------------------------------------
    # Step 2: Optimization loop
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Optimization loop")
    print("=" * 60)

    # Collect seed trajectories for the reflector
    seed_trajs = [
        tr.trajectory for tr in baseline_result.task_results.values() if tr.trajectory is not None
    ]

    def on_iteration(iteration_result):
        it = iteration_result
        print(f"\n--- Iteration {it.iteration} ---")
        if it.average_score is not None:
            print(f"  Average score: {it.average_score:.3f}")
        if it.average_delta is not None:
            print(f"  Average delta: {it.average_delta:+.3f}")
        print(f"  Suggestions: {len(it.suggestions)}")
        for s in it.suggestions:
            tag = " [auto]" if s.auto_applicable else " [manual]"
            print(f"    • {s.kind}{tag}: {s.rationale[:80]}")

    result = await run_optimization_loop(
        trajectories=seed_trajs,
        base_candidate=base_candidate,
        agent_builder_factory=agent_builder_factory,
        evaluator_model=model,
        rubric=rubric,
        replay_config=ReplayConfig(tool_approval="auto_approve"),
        store=store,
        max_iterations=MAX_ITERATIONS,
        apply_mode="prompt_and_skills",
        on_iteration=on_iteration,
    )

    # -----------------------------------------------------------------------
    # Step 3: Gate the best candidate
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Gate the best candidate")
    print("=" * 60)

    gate_result = await run_gate(
        make_agent_factory(result.best_candidate),
        train_runner=train_benchmark,
        test_runner=test_benchmark,
        suite=suite,
        history=history,
        config=GateConfig(
            regression_threshold=0.8,
            require_improvement=False,  # first gated run — no prior best
            auto_promote=True,
        ),
    )

    print(f"  Gate passed: {gate_result.passed}")
    for step in gate_result.step_results:
        status = "✓" if step.passed else "✗"
        print(f"  [{status}] Step {step.step} ({step.name}): {step.detail}")
    if gate_result.promoted_tasks:
        print(f"  Promoted to regression suite: {gate_result.promoted_tasks}")

    # Record gated result
    test_score = 0.0
    if gate_result.test_result:
        test_score = gate_result.test_result.mean_reward
    history.append(
        HistoryEntry(
            iteration=MAX_ITERATIONS + 1,
            val_score=test_score,
            pass_rate=(gate_result.test_result.pass_rate if gate_result.test_result else 0.0),
            accepted=gate_result.passed,
            description="final_gated",
        )
    )

    # Record learnings
    learnings.append(
        LearningEntry(
            iteration=MAX_ITERATIONS + 1,
            category="successful_change" if gate_result.passed else "failed_attempt",
            summary=(f"Final candidate {'passed' if gate_result.passed else 'failed'} gate"),
        )
    )

    # -----------------------------------------------------------------------
    # Step 4: Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Stopped: {result.stopped_reason}")
    print(f"Iterations: {len(result.iterations)}")

    print(f"\n{history.summary()}")

    original = INITIAL_INSTRUCTION
    optimized = result.best_candidate.agent_kwargs.get("instruction", "")
    print(f"\nOriginal instruction ({len(original)} chars):")
    print(f"  {original[:200]}")

    if optimized != original:
        print(f"\nOptimized instruction ({len(optimized)} chars):")
        for line in optimized.split("\n"):
            print(f"  {line}")
    else:
        print("\n(Instruction was not changed)")

    print(f"\nRegression suite: {len(suite)} tasks")
    print(f"Learnings: {len(learnings)} entries")
    print(f"Trajectories: {len(store.list_ids())} stored")
    print(f"Workspace: {WORKSPACE_DIR}")


def main():
    print("=" * 60)
    print("Data Extraction — Meta-Agent Optimization Example")
    print("=" * 60)
    asyncio.run(run_example())


if __name__ == "__main__":
    main()
