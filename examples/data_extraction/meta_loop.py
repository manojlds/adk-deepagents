"""Meta-agent outer loop — drives a coding agent to improve the data extraction harness.

Runs the benchmark/gate/score-tracking loop in Python, and shells out to a
coding agent CLI (OpenCode, Claude Code, Amp, etc.) for the "diagnose failures
and edit agent.py" step.

Supported coding agents:
  - opencode  → ``opencode run "prompt"``
  - claude    → ``claude -p "prompt" --allowedTools "Read,Write,Edit,Bash"``
  - amp       → ``amp run "prompt"``
  - custom    → any command that accepts a prompt and edits files

Usage:
    # With OpenCode (default)
    uv run python -m examples.data_extraction.meta_loop

    # With Claude Code
    META_AGENT=claude uv run python -m examples.data_extraction.meta_loop

    # With Amp
    META_AGENT=amp uv run python -m examples.data_extraction.meta_loop

    # With a custom command template
    META_AGENT_CMD='my-agent --prompt "{prompt}"' uv run python -m examples.data_extraction.meta_loop

    # Control iterations
    MAX_ITERATIONS=10 uv run python -m examples.data_extraction.meta_loop
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXAMPLE_DIR = Path(__file__).parent
TASKS_DIR = EXAMPLE_DIR / "tasks"
WORKSPACE_DIR = EXAMPLE_DIR / "workspace"
AGENT_PY = EXAMPLE_DIR / "agent.py"

MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "10"))
META_AGENT = os.environ.get("META_AGENT", "opencode")
META_AGENT_CMD = os.environ.get("META_AGENT_CMD", "")
META_AGENT_MODEL = os.environ.get("META_AGENT_MODEL", "")

# ---------------------------------------------------------------------------
# Coding agent CLI wrappers
# ---------------------------------------------------------------------------

_AGENT_COMMANDS: dict[str, list[str]] = {
    "opencode": ["opencode", "run"],
    "claude": [
        "claude",
        "-p",
        "{prompt}",
        "--allowedTools",
        "Read,Write,Edit,Bash,Glob,Grep",
    ],
    "amp": ["amp", "run"],
}


def _build_agent_command(prompt: str) -> list[str]:
    """Build the shell command to invoke the coding agent with a prompt."""
    if META_AGENT_CMD:
        # Custom command template: replace {prompt}
        cmd_str = META_AGENT_CMD.replace("{prompt}", prompt)
        return ["bash", "-c", cmd_str]

    if META_AGENT == "claude":
        cmd = list(_AGENT_COMMANDS["claude"])
        # Replace the {prompt} placeholder
        cmd = [part.replace("{prompt}", prompt) for part in cmd]
        if META_AGENT_MODEL:
            cmd.extend(["--model", META_AGENT_MODEL])
        return cmd

    if META_AGENT in _AGENT_COMMANDS:
        cmd = list(_AGENT_COMMANDS[META_AGENT])
        if META_AGENT_MODEL:
            cmd.extend(["-m", META_AGENT_MODEL])
        cmd.append(prompt)
        return cmd

    # Fallback: treat META_AGENT as a command name
    cmd = [META_AGENT]
    if META_AGENT_MODEL:
        cmd.extend(["-m", META_AGENT_MODEL])
    cmd.append(prompt)
    return cmd


def _invoke_coding_agent(prompt: str) -> tuple[int, str]:
    """Invoke the coding agent and return (exit_code, output)."""
    cmd = _build_agent_command(prompt)
    print(f"\n{'─' * 60}")
    print(f"  Invoking {META_AGENT}...")
    print(f"{'─' * 60}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(EXAMPLE_DIR),
            capture_output=False,
            text=True,
            timeout=600,  # 10 minute timeout per invocation
        )
        return result.returncode, ""
    except subprocess.TimeoutExpired:
        print("  ⚠ Coding agent timed out (10 min)")
        return 1, "timeout"
    except FileNotFoundError:
        print(f"  ✗ Command not found: {cmd[0]}")
        print(f"    Install {META_AGENT} or set META_AGENT_CMD")
        return 1, "not_found"


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _run_benchmark(*, split: str = "train") -> dict:
    """Run the benchmark and return parsed results."""
    from examples.data_extraction.benchmark import DataExtractionBenchmark

    benchmark = DataExtractionBenchmark(TASKS_DIR, split=split)

    # Import agent.py fresh each time (it may have been edited)
    agent_module = _reload_agent_module()
    if agent_module is None:
        return {"error": "Failed to import agent.py", "pass_rate": 0.0, "mean_reward": 0.0}

    create_agent = agent_module.create_agent

    async def _run():
        return await benchmark.run(create_agent)

    result = asyncio.run(_run())

    task_details = {}
    for tid, tr in result.task_results.items():
        task_details[tid] = {
            "reward": round(tr.reward, 3),
            "passed": tr.passed,
            "error": tr.error,
        }

    return {
        "pass_rate": round(result.pass_rate, 3),
        "mean_reward": round(result.mean_reward, 3),
        "total_cost": round(result.total_cost_usd, 4),
        "tasks": task_details,
    }


def _reload_agent_module():
    """Reload agent.py to pick up coding agent's edits."""
    import importlib

    module_name = "examples.data_extraction.agent"

    # Remove from cache so edits are picked up
    if module_name in sys.modules:
        del sys.modules[module_name]

    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        print(f"  ✗ Failed to import agent.py: {exc}")
        return None


def _format_results(results: dict, label: str = "train") -> str:
    """Format benchmark results as a human-readable string."""
    lines = []
    lines.append(
        f"Benchmark ({label}): pass_rate={results['pass_rate']:.1%}, "
        f"mean_reward={results['mean_reward']:.3f}"
    )

    for tid, detail in sorted(results.get("tasks", {}).items()):
        status = "✓" if detail["passed"] else "✗"
        error = f"  ERROR: {detail['error']}" if detail.get("error") else ""
        lines.append(f"  [{status}] {tid}: reward={detail['reward']:.3f}{error}")

    return "\n".join(lines)


def _read_task_samples() -> str:
    """Read a few sample tasks for the coding agent's context."""
    lines = []
    sample_tasks = ["invoice_simple", "receipt_multiline"]

    for task_id in sample_tasks:
        task_dir = TASKS_DIR / task_id
        input_file = task_dir / "input.txt"
        expected_file = task_dir / "expected.json"
        task_yaml = task_dir / "task.yaml"

        if task_yaml.exists():
            lines.append(f"\n### Task: {task_id}")
            lines.append(f"**Instruction:**\n```\n{task_yaml.read_text()[:500]}\n```")
        if input_file.exists():
            content = input_file.read_text()[:400]
            lines.append(f"**Input (preview):**\n```\n{content}\n```")
        if expected_file.exists():
            content = expected_file.read_text()[:400]
            lines.append(f"**Expected output (preview):**\n```json\n{content}\n```")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_proposer_prompt(
    *,
    iteration: int,
    train_results: dict,
    prev_results: dict | None,
    history_summary: str,
    learnings_context: str,
) -> str:
    """Build the prompt sent to the coding agent."""
    results_text = _format_results(train_results, "train")

    # Identify failures
    failed_tasks = [
        f"  - {tid}: reward={d['reward']}"
        for tid, d in sorted(train_results.get("tasks", {}).items())
        if not d["passed"]
    ]
    failed_text = "\n".join(failed_tasks) if failed_tasks else "  (none — all tasks passed!)"

    # Delta from previous iteration
    delta_text = ""
    if prev_results is not None:
        delta = train_results["mean_reward"] - prev_results["mean_reward"]
        delta_text = f"\nDelta from previous iteration: {delta:+.3f}"

    task_samples = ""
    if iteration == 1:
        task_samples = f"\n## Sample Tasks (for context)\n{_read_task_samples()}"

    return textwrap.dedent(f"""\
        You are improving a data extraction agent. Your job is to edit the
        EDITABLE section of `agent.py` (specifically `SYSTEM_PROMPT`) to
        improve the agent's score on the benchmark.

        ## Current Results (iteration {iteration})

        {results_text}{delta_text}

        ## Failed Tasks

        {failed_text}

        ## Rules

        1. Read `agent.py` — only edit the EDITABLE section above the FIXED BOUNDARY.
        2. Make ONE focused change to `SYSTEM_PROMPT` that addresses the most
           common failure pattern.
        3. Do NOT hardcode task-specific logic (no "if invoice" conditionals).
        4. Every change must be general — it should help on unseen tasks too.
        5. Prefer prompt improvements over adding tools or sub-agents.
        6. After editing, do NOT run the benchmark yourself — just edit and exit.

        ## Change Hierarchy (try in order)
        1. Add formatting/schema hints to the prompt
        2. Add step-by-step reasoning instructions
        3. Add output validation rules
        4. Add date/number parsing guidance

        {history_summary}

        {learnings_context}
        {task_samples}

        Now read `agent.py`, analyze the failures above, and make ONE
        targeted edit to SYSTEM_PROMPT to improve the score. Then exit.
    """)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    """Run the meta-agent optimization loop."""
    from adk_deepagents.optimization.history import HistoryEntry, ScoreHistory
    from adk_deepagents.optimization.learnings import LearningEntry, LearningsStore

    print("=" * 60)
    print("Data Extraction — Coding Agent Meta-Loop")
    print("=" * 60)
    print(f"  Coding agent: {META_AGENT}")
    print(f"  Max iterations: {MAX_ITERATIONS}")
    print(f"  Agent harness: {AGENT_PY}")
    print(f"  Tasks: {TASKS_DIR}")

    # Check coding agent is available
    agent_cmd = META_AGENT if not META_AGENT_CMD else META_AGENT_CMD.split()[0]
    if not META_AGENT_CMD and shutil.which(META_AGENT) is None:
        if META_AGENT in _AGENT_COMMANDS:
            agent_cmd = _AGENT_COMMANDS[META_AGENT][0]
        if shutil.which(agent_cmd) is None:
            print(f"\n  ✗ '{agent_cmd}' not found in PATH.")
            print(f"    Install {META_AGENT} or set META_AGENT_CMD='your-cmd {{prompt}}'")
            sys.exit(1)

    # Set up workspace
    WORKSPACE_DIR.mkdir(exist_ok=True)
    history = ScoreHistory(WORKSPACE_DIR / "history.jsonl")
    learnings = LearningsStore(WORKSPACE_DIR / "learnings.jsonl")

    # Back up original agent.py
    backup_path = WORKSPACE_DIR / "agent.py.original"
    if not backup_path.exists():
        shutil.copy2(AGENT_PY, backup_path)

    # -----------------------------------------------------------------------
    # Step 1: Baseline
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE RUN")
    print("=" * 60)

    baseline = _run_benchmark(split="train")
    if "error" in baseline:
        print(f"  ✗ {baseline['error']}")
        sys.exit(1)

    print(f"\n{_format_results(baseline, 'train')}")

    history.append(
        HistoryEntry(
            iteration=0,
            val_score=baseline["mean_reward"],
            pass_rate=baseline["pass_rate"],
            description="baseline",
            timestamp=datetime.now(UTC).isoformat(),
        )
    )

    best_score = baseline["mean_reward"]
    best_pass_rate = baseline["pass_rate"]
    prev_results = baseline
    consecutive_no_improve = 0

    # -----------------------------------------------------------------------
    # Step 2: Optimization loop
    # -----------------------------------------------------------------------
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'=' * 60}")
        print(f"ITERATION {iteration}/{MAX_ITERATIONS}")
        print(f"{'=' * 60}")
        print(f"  Best so far: score={best_score:.3f}, pass_rate={best_pass_rate:.1%}")
        print(f"  Consecutive no-improve: {consecutive_no_improve}")

        # Save agent.py before edit
        pre_edit = AGENT_PY.read_text(encoding="utf-8")

        # Build prompt for coding agent
        prompt = _build_proposer_prompt(
            iteration=iteration,
            train_results=prev_results,
            prev_results=prev_results if iteration > 1 else None,
            history_summary=history.summary(last_n=5) if len(history) > 0 else "",
            learnings_context=learnings.to_prompt_context(max_entries=5),
        )

        # Invoke coding agent
        exit_code, error = _invoke_coding_agent(prompt)

        if exit_code != 0:
            print(f"  ⚠ Coding agent exited with code {exit_code}")
            if error == "not_found":
                break
            consecutive_no_improve += 1
            learnings.append(
                LearningEntry(
                    iteration=iteration,
                    category="failed_attempt",
                    summary=f"Coding agent failed (exit={exit_code}, error={error})",
                )
            )
            continue

        # Check if agent.py was actually modified
        post_edit = AGENT_PY.read_text(encoding="utf-8")
        if pre_edit == post_edit:
            print("  ⚠ agent.py was not modified — skipping benchmark")
            consecutive_no_improve += 1
            continue

        # Validate agent.py still imports cleanly
        agent_module = _reload_agent_module()
        if agent_module is None:
            print("  ✗ agent.py has import errors — reverting")
            AGENT_PY.write_text(pre_edit, encoding="utf-8")
            learnings.append(
                LearningEntry(
                    iteration=iteration,
                    category="failed_attempt",
                    summary="Edit broke agent.py imports — reverted",
                )
            )
            consecutive_no_improve += 1
            continue

        # Run benchmark with the edited agent
        print("\n  Running benchmark with edited agent...")
        new_results = _run_benchmark(split="train")

        if "error" in new_results:
            print(f"  ✗ Benchmark failed: {new_results['error']} — reverting")
            AGENT_PY.write_text(pre_edit, encoding="utf-8")
            consecutive_no_improve += 1
            continue

        print(f"\n{_format_results(new_results, 'train')}")

        # Keep / discard
        new_score = new_results["mean_reward"]
        new_pass_rate = new_results["pass_rate"]
        passed_improved = new_pass_rate > best_pass_rate
        score_improved = new_score > best_score
        same_pass_simpler = new_pass_rate == best_pass_rate and len(post_edit) <= len(pre_edit)

        accepted = passed_improved or score_improved or same_pass_simpler

        delta = new_score - prev_results["mean_reward"]
        print(f"\n  Score: {prev_results['mean_reward']:.3f} → {new_score:.3f} (Δ {delta:+.3f})")
        print(f"  Pass rate: {prev_results['pass_rate']:.1%} → {new_pass_rate:.1%}")

        if accepted:
            print("  ✓ KEEP — improvement detected")
            best_score = max(best_score, new_score)
            best_pass_rate = max(best_pass_rate, new_pass_rate)
            consecutive_no_improve = 0
            learnings.append(
                LearningEntry(
                    iteration=iteration,
                    category="successful_change",
                    summary=f"Score improved {delta:+.3f}",
                    score_before=prev_results["mean_reward"],
                    score_after=new_score,
                )
            )
            prev_results = new_results
        else:
            print("  ✗ DISCARD — no improvement, reverting")
            AGENT_PY.write_text(pre_edit, encoding="utf-8")
            consecutive_no_improve += 1
            learnings.append(
                LearningEntry(
                    iteration=iteration,
                    category="failed_attempt",
                    summary=f"Score delta {delta:+.3f} — reverted",
                    score_before=prev_results["mean_reward"],
                    score_after=new_score,
                )
            )

        history.append(
            HistoryEntry(
                iteration=iteration,
                val_score=new_score,
                pass_rate=new_pass_rate,
                accepted=accepted,
                description="keep" if accepted else "discard",
                timestamp=datetime.now(UTC).isoformat(),
            )
        )

        # Stop after 5 consecutive no-improvement iterations
        if consecutive_no_improve >= 5:
            print("\n  ⚠ 5 consecutive iterations with no improvement — stopping")
            break

    # -----------------------------------------------------------------------
    # Step 3: Final test-split evaluation
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("FINAL EVALUATION (test split)")
    print("=" * 60)

    test_results = _run_benchmark(split="test")
    if "error" not in test_results:
        print(f"\n{_format_results(test_results, 'test')}")
        history.append(
            HistoryEntry(
                iteration=MAX_ITERATIONS + 1,
                val_score=test_results["mean_reward"],
                pass_rate=test_results["pass_rate"],
                accepted=True,
                description="final_test_eval",
                timestamp=datetime.now(UTC).isoformat(),
            )
        )

    # -----------------------------------------------------------------------
    # Step 4: Report
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)
    print(f"\n{history.summary()}")
    print(f"\nLearnings: {len(learnings)} entries")

    successful = learnings.successful_changes()
    if successful:
        print("\nSuccessful changes:")
        for entry in successful:
            print(f"  iter {entry.iteration}: {entry.summary}")

    failed = learnings.failed_attempts()
    if failed:
        print(f"\nFailed attempts: {len(failed)}")

    print(f"\nFinal agent.py: {AGENT_PY}")
    print(f"Original backup: {backup_path}")
    print(f"Workspace: {WORKSPACE_DIR}")


if __name__ == "__main__":
    main()
