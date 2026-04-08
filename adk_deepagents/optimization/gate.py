"""Three-step acceptance gate for optimization candidates.

Inspired by auto-harness's gating system:

1. **Regression suite** (train split) — previously-passing tasks must keep
   passing.  Pass rate must meet a configurable threshold (default 80%).
2. **Held-out benchmark** (test split) — val_score must be >= the historical
   best (monotonic improvement).
3. **Suite promotion** — re-run previously-failing train tasks; promote
   newly-passing ones into the regression suite for future protection.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from google.adk.agents import LlmAgent

from adk_deepagents.optimization.benchmark import BenchmarkResult, BenchmarkRunner
from adk_deepagents.optimization.history import ScoreHistory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regression suite
# ---------------------------------------------------------------------------


@dataclass
class RegressionTask:
    """A task in the regression suite."""

    task_id: str
    added_at_iteration: int = 0
    last_reward: float = 0.0
    last_passed: bool = False


@dataclass
class RegressionSuite:
    """Self-growing regression suite backed by JSON.

    Tasks are added automatically when they pass for the first time
    (via gate step 3).  Once in the suite, they must keep passing
    (enforced by gate step 1).
    """

    path: Path
    tasks: dict[str, RegressionTask] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> RegressionSuite:
        """Load a suite from disk, creating an empty one if it doesn't exist."""
        p = Path(path)
        if not p.exists():
            return cls(path=p)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            tasks = {}
            for tid, tdata in data.get("tasks", {}).items():
                tasks[tid] = RegressionTask(
                    task_id=tid,
                    added_at_iteration=tdata.get("added_at_iteration", 0),
                    last_reward=tdata.get("last_reward", 0.0),
                    last_passed=tdata.get("last_passed", False),
                )
            return cls(path=p, tasks=tasks)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load regression suite from %s: %s", p, exc)
            return cls(path=p)

    def save(self) -> None:
        """Persist suite to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "tasks": {
                tid: {
                    "task_id": t.task_id,
                    "added_at_iteration": t.added_at_iteration,
                    "last_reward": t.last_reward,
                    "last_passed": t.last_passed,
                }
                for tid, t in self.tasks.items()
            }
        }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def promote(
        self,
        task_ids: list[str],
        *,
        iteration: int = 0,
        rewards: dict[str, float] | None = None,
    ) -> int:
        """Add newly-passing tasks to the suite.

        Returns the number of tasks promoted.
        """
        promoted = 0
        for tid in task_ids:
            if tid not in self.tasks:
                reward = rewards.get(tid, 1.0) if rewards else 1.0
                self.tasks[tid] = RegressionTask(
                    task_id=tid,
                    added_at_iteration=iteration,
                    last_reward=reward,
                    last_passed=True,
                )
                promoted += 1
        if promoted > 0:
            self.save()
        return promoted

    def update_results(self, results: dict[str, float]) -> None:
        """Update last_reward and last_passed for suite tasks."""
        for tid, reward in results.items():
            if tid in self.tasks:
                self.tasks[tid].last_reward = reward
                self.tasks[tid].last_passed = reward > 0.0

    def task_ids(self) -> list[str]:
        """Return all task IDs in the suite."""
        return list(self.tasks.keys())

    def __len__(self) -> int:
        return len(self.tasks)


# ---------------------------------------------------------------------------
# Gate types
# ---------------------------------------------------------------------------


@dataclass
class GateConfig:
    """Configuration for the acceptance gate."""

    regression_threshold: float = 0.8
    """Minimum pass rate on the regression suite (step 1)."""

    require_improvement: bool = True
    """If True, test-split score must beat or match historical best (step 2)."""

    auto_promote: bool = True
    """Automatically promote newly-passing tasks to the suite (step 3)."""

    skip_if_empty_suite: bool = True
    """Skip step 1 (regression) if the suite is empty."""


@dataclass
class GateStepResult:
    """Result of a single gate step."""

    step: int
    name: str
    passed: bool
    detail: str = ""
    score: float | None = None


@dataclass
class GateResult:
    """Result of running the full 3-step gate."""

    passed: bool
    step_results: list[GateStepResult] = field(default_factory=list)
    promoted_tasks: list[str] = field(default_factory=list)
    test_result: BenchmarkResult | None = None


# ---------------------------------------------------------------------------
# Gate runner
# ---------------------------------------------------------------------------


async def run_gate(
    agent_factory: Callable[[], LlmAgent | Awaitable[LlmAgent]],
    *,
    train_runner: BenchmarkRunner,
    test_runner: BenchmarkRunner | None = None,
    suite: RegressionSuite,
    history: ScoreHistory,
    config: GateConfig | None = None,
    iteration: int = 0,
) -> GateResult:
    """Run the 3-step acceptance gate.

    Parameters
    ----------
    agent_factory:
        Callable that creates a fresh agent instance.
    train_runner:
        Benchmark runner for the train split (steps 1 and 3).
    test_runner:
        Benchmark runner for the test split (step 2).  If ``None``,
        step 2 is skipped.
    suite:
        Regression suite (read + modified in-place for promotions).
    history:
        Score history (read-only — checked for best val_score).
    config:
        Gate configuration.
    iteration:
        Current optimization iteration number.

    Returns
    -------
    GateResult
        Whether the gate passed, per-step results, and promoted tasks.
    """
    if config is None:
        config = GateConfig()

    step_results: list[GateStepResult] = []

    # -----------------------------------------------------------------------
    # Step 1: Regression suite (train split)
    # -----------------------------------------------------------------------
    suite_task_ids = suite.task_ids()

    if not suite_task_ids and config.skip_if_empty_suite:
        step_results.append(
            GateStepResult(
                step=1,
                name="regression_suite",
                passed=True,
                detail="Suite is empty — skipping regression check.",
            )
        )
    elif suite_task_ids:
        logger.info("Gate step 1: running %d regression suite tasks", len(suite_task_ids))
        suite_result = await train_runner.run(
            agent_factory,
            task_ids=suite_task_ids,
        )
        suite_pass_rate = suite_result.pass_rate

        suite.update_results({tid: r.reward for tid, r in suite_result.task_results.items()})

        step1_passed = suite_pass_rate >= config.regression_threshold
        step_results.append(
            GateStepResult(
                step=1,
                name="regression_suite",
                passed=step1_passed,
                detail=(
                    f"Suite pass rate: {suite_pass_rate:.1%} "
                    f"(threshold: {config.regression_threshold:.1%}, "
                    f"{len(suite_task_ids)} tasks)"
                ),
                score=suite_pass_rate,
            )
        )

        if not step1_passed:
            failed = [tid for tid, r in suite_result.task_results.items() if not r.passed]
            logger.info(
                "Gate FAILED at step 1: suite pass rate %.1f%% < %.1f%%. Failed tasks: %s",
                suite_pass_rate * 100,
                config.regression_threshold * 100,
                failed,
            )
            return GateResult(passed=False, step_results=step_results)
    else:
        step_results.append(
            GateStepResult(
                step=1,
                name="regression_suite",
                passed=True,
                detail="Suite is empty and skip_if_empty_suite is disabled.",
            )
        )

    # -----------------------------------------------------------------------
    # Step 2: Held-out benchmark (test split)
    # -----------------------------------------------------------------------
    test_benchmark_result: BenchmarkResult | None = None

    if test_runner is None:
        step_results.append(
            GateStepResult(
                step=2,
                name="test_benchmark",
                passed=True,
                detail="No test runner configured — skipping held-out eval.",
            )
        )
    else:
        logger.info("Gate step 2: running held-out benchmark")
        test_benchmark_result = await test_runner.run(agent_factory)
        test_score = test_benchmark_result.mean_reward

        best = history.best_val_score()
        if best is not None and config.require_improvement:
            step2_passed = test_score >= best
            step_results.append(
                GateStepResult(
                    step=2,
                    name="test_benchmark",
                    passed=step2_passed,
                    detail=(
                        f"Test score: {test_score:.3f} "
                        f"(best: {best:.3f}, "
                        f"{'≥' if step2_passed else '<'} threshold)"
                    ),
                    score=test_score,
                )
            )
            if not step2_passed:
                logger.info(
                    "Gate FAILED at step 2: test score %.3f < best %.3f",
                    test_score,
                    best,
                )
                return GateResult(
                    passed=False,
                    step_results=step_results,
                    test_result=test_benchmark_result,
                )
        else:
            step_results.append(
                GateStepResult(
                    step=2,
                    name="test_benchmark",
                    passed=True,
                    detail=(
                        f"Test score: {test_score:.3f} (no prior best — accepting as baseline)"
                    ),
                    score=test_score,
                )
            )

    # -----------------------------------------------------------------------
    # Step 3: Suite promotion (train split, previously-failing tasks)
    # -----------------------------------------------------------------------
    promoted: list[str] = []

    if config.auto_promote:
        all_train_ids = set(train_runner.list_task_ids())
        suite_ids = set(suite.task_ids())
        non_suite_ids = list(all_train_ids - suite_ids)

        if non_suite_ids:
            logger.info(
                "Gate step 3: re-running %d non-suite tasks for promotion",
                len(non_suite_ids),
            )
            promo_result = await train_runner.run(
                agent_factory,
                task_ids=non_suite_ids,
            )
            newly_passing = [tid for tid, r in promo_result.task_results.items() if r.passed]
            if newly_passing:
                rewards = {tid: r.reward for tid, r in promo_result.task_results.items()}
                count = suite.promote(
                    newly_passing,
                    iteration=iteration,
                    rewards=rewards,
                )
                promoted = newly_passing
                logger.info("Gate step 3: promoted %d tasks to suite", count)

        step_results.append(
            GateStepResult(
                step=3,
                name="suite_promotion",
                passed=True,
                detail=(
                    f"Promoted {len(promoted)} task(s) to regression suite. "
                    f"Suite now has {len(suite)} task(s)."
                ),
            )
        )
    else:
        step_results.append(
            GateStepResult(
                step=3,
                name="suite_promotion",
                passed=True,
                detail="Auto-promotion disabled.",
            )
        )

    return GateResult(
        passed=True,
        step_results=step_results,
        promoted_tasks=promoted,
        test_result=test_benchmark_result,
    )
