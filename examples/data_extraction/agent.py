"""Data extraction agent harness.

This is the meta-agent's edit surface. A coding agent (OpenCode, Claude Code,
Amp, etc.) modifies the EDITABLE section below to improve data extraction
quality.

The fixed boundary section contains the benchmark runner, gating system, and
scoring — do not modify unless the human explicitly asks.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import LlmAgent

from adk_deepagents import DeepAgentConfig, SubAgentSpec, create_deep_agent
from adk_deepagents.optimization.benchmark import BenchmarkResult
from adk_deepagents.optimization.gate import GateConfig, RegressionSuite, run_gate
from adk_deepagents.optimization.history import HistoryEntry, ScoreHistory

load_dotenv()

# ============================================================================
# EDITABLE HARNESS — prompt, tools, agent construction
# ============================================================================

_MODEL_NAME = os.environ.get(
    "ADK_DEEPAGENTS_MODEL",
    os.environ.get("LITELLM_MODEL", "gemini-2.5-flash"),
)


def _resolve_model() -> str | Any:
    """Resolve model from environment, supporting LiteLLM providers."""
    if _MODEL_NAME.startswith("gemini"):
        return _MODEL_NAME

    try:
        from google.adk.models.lite_llm import LiteLlm
    except ImportError:
        return _MODEL_NAME

    kwargs: dict[str, Any] = {"model": _MODEL_NAME}
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENCODE_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key

    api_base = os.environ.get("OPENAI_API_BASE")
    if api_base:
        kwargs["api_base"] = api_base

    return LiteLlm(**kwargs)


MODEL = _resolve_model()

SYSTEM_PROMPT = """\
You are a data extraction assistant. You read input files containing
semi-structured text (invoices, receipts, logs, etc.) and extract
structured JSON data.

When given a task:
1. Read the input file using read_file
2. Build the output schema directly from the task instruction: use the exact field names requested, use arrays of objects for repeated records (for example contacts/items/products), and include every required top-level field even when a value is missing
3. Extract the data carefully, paying attention to numbers and dates
4. Write valid JSON to the specified output file using write_file
"""

# Additional tools beyond the built-in filesystem tools (ls, read_file,
# write_file, edit_file, glob, grep). Add custom tool functions here.
EXTRA_TOOLS: list = []

# Sub-agent specifications for delegation. Uncomment or add specs to
# enable task delegation.
SUBAGENTS: list[SubAgentSpec | LlmAgent] | None = None

# Advanced configuration. Uncomment to enable features.
CONFIG = DeepAgentConfig(
    # output_schema=None,        # Pydantic model for structured output
    # summarization=None,        # Context window management
    # http_tools=False,          # Enable fetch_url / http_request
    error_handling=True,
)


def create_agent(*, model=None):
    """Build the data extraction agent.

    The meta-agent can modify SYSTEM_PROMPT, EXTRA_TOOLS, SUBAGENTS,
    and CONFIG above — or restructure this function entirely.
    """
    return create_deep_agent(
        name="data_extractor",
        model=model or MODEL,
        instruction=SYSTEM_PROMPT,
        tools=EXTRA_TOOLS or None,
        subagents=SUBAGENTS,
        config=CONFIG,
    )


# ============================================================================
# FIXED BOUNDARY — benchmark runner, gating, scoring
# Do not modify below this line unless the human explicitly asks.
# ============================================================================

_EXAMPLE_DIR = Path(__file__).parent
_TASKS_DIR = _EXAMPLE_DIR / "tasks"
_WORKSPACE_DIR = _EXAMPLE_DIR / "workspace"


def _run_benchmark(*, split: str = "train", task_ids: list[str] | None = None) -> BenchmarkResult:
    """Run the benchmark synchronously. Returns BenchmarkResult."""
    from examples.data_extraction.benchmark import DataExtractionBenchmark

    benchmark = DataExtractionBenchmark(_TASKS_DIR, split=split)

    async def _run():
        return await benchmark.run(create_agent, task_ids=task_ids)

    return asyncio.run(_run())


def _run_gate(*, iteration: int = 0) -> dict:
    """Run the 3-step gate. Returns a dict with gate results."""
    from examples.data_extraction.benchmark import DataExtractionBenchmark

    _WORKSPACE_DIR.mkdir(exist_ok=True)
    suite = RegressionSuite.load(_WORKSPACE_DIR / "suite.json")
    history = ScoreHistory(_WORKSPACE_DIR / "history.jsonl")

    train_bench = DataExtractionBenchmark(_TASKS_DIR, split="train")
    test_bench = DataExtractionBenchmark(_TASKS_DIR, split="test")

    async def _gate():
        return await run_gate(
            create_agent,
            train_runner=train_bench,
            test_runner=test_bench,
            suite=suite,
            history=history,
            config=GateConfig(
                regression_threshold=0.8,
                require_improvement=True,
                auto_promote=True,
            ),
            iteration=iteration,
        )

    result = asyncio.run(_gate())
    return {
        "passed": result.passed,
        "steps": [
            {"step": s.step, "name": s.name, "passed": s.passed, "detail": s.detail}
            for s in result.step_results
        ],
        "promoted": result.promoted_tasks,
        "test_score": result.test_result.mean_reward if result.test_result else None,
    }


def _print_results(result: BenchmarkResult, label: str = "Benchmark") -> None:
    """Pretty-print benchmark results."""
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    print(f"  Pass rate: {result.pass_rate:.1%}")
    print(f"  Mean reward: {result.mean_reward:.3f}")
    print(f"  Total cost: ${result.total_cost_usd:.4f}")
    print()
    for tid, tr in sorted(result.task_results.items()):
        status = "✓" if tr.passed else "✗"
        error = f"  ERROR: {tr.error}" if tr.error else ""
        print(f"  [{status}] {tid}: reward={tr.reward:.3f}{error}")


if __name__ == "__main__":
    """CLI for the coding agent to run benchmarks and gates.

    Usage:
        # Run train benchmark
        uv run python examples/data_extraction/agent.py benchmark

        # Run test benchmark
        uv run python examples/data_extraction/agent.py benchmark --split test

        # Run specific tasks
        uv run python examples/data_extraction/agent.py benchmark --tasks invoice_simple,receipt_multiline

        # Run the 3-step gate
        uv run python examples/data_extraction/agent.py gate

        # Record a result
        uv run python examples/data_extraction/agent.py record --score 0.75 --pass-rate 0.8 --desc "added JSON schema hint"
    """
    import argparse

    parser = argparse.ArgumentParser(description="Data extraction agent harness")
    sub = parser.add_subparsers(dest="command")

    bench_p = sub.add_parser("benchmark", help="Run the benchmark")
    bench_p.add_argument("--split", default="train", choices=["train", "test"])
    bench_p.add_argument("--tasks", default=None, help="Comma-separated task IDs")

    gate_p = sub.add_parser("gate", help="Run the 3-step acceptance gate")
    gate_p.add_argument("--iteration", type=int, default=0)

    record_p = sub.add_parser("record", help="Record a result to history")
    record_p.add_argument("--score", type=float, required=True)
    record_p.add_argument("--pass-rate", type=float, required=True)
    record_p.add_argument("--iteration", type=int, default=0)
    record_p.add_argument("--desc", default="")
    record_p.add_argument("--accepted", action="store_true", default=True)
    record_p.add_argument("--rejected", action="store_true", default=False)

    history_p = sub.add_parser("history", help="Show score history")

    args = parser.parse_args()

    if args.command == "benchmark":
        task_ids = args.tasks.split(",") if args.tasks else None
        result = _run_benchmark(split=args.split, task_ids=task_ids)
        _print_results(result, f"Benchmark ({args.split} split)")

    elif args.command == "gate":
        gate_result = _run_gate(iteration=args.iteration)
        print(f"\nGate passed: {gate_result['passed']}")
        for step in gate_result["steps"]:
            status = "✓" if step["passed"] else "✗"
            print(f"  [{status}] Step {step['step']} ({step['name']}): {step['detail']}")
        if gate_result["promoted"]:
            print(f"  Promoted: {gate_result['promoted']}")
        if gate_result["test_score"] is not None:
            print(f"  Test score: {gate_result['test_score']:.3f}")

    elif args.command == "record":
        _WORKSPACE_DIR.mkdir(exist_ok=True)
        history = ScoreHistory(_WORKSPACE_DIR / "history.jsonl")
        from datetime import datetime

        history.append(
            HistoryEntry(
                iteration=args.iteration,
                val_score=args.score,
                pass_rate=args.pass_rate,
                accepted=not args.rejected,
                description=args.desc,
                timestamp=datetime.now(UTC).isoformat(),
            )
        )
        print(f"Recorded: score={args.score}, pass_rate={args.pass_rate}")

    elif args.command == "history":
        _WORKSPACE_DIR.mkdir(exist_ok=True)
        history = ScoreHistory(_WORKSPACE_DIR / "history.jsonl")
        print(history.summary())

    else:
        parser.print_help()
