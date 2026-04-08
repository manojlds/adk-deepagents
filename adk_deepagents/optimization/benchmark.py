"""Benchmark runner abstraction for agent optimization.

Provides a pluggable interface for running agents against task suites and
collecting per-task results with trajectories for evaluation and gating.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from google.adk.agents import LlmAgent

from adk_deepagents.optimization.trajectory import Trajectory

logger = logging.getLogger(__name__)


@dataclass
class TaskSpec:
    """Specification for a single benchmark task."""

    task_id: str
    instruction: str
    workspace_files: dict[str, str] = field(default_factory=dict)
    """Files to pre-populate in the agent's workspace. Maps path -> content."""

    expected_output: dict[str, Any] | None = None
    """Expected output for deterministic verification."""

    verify_command: list[str] | None = None
    """Shell command whose exit code determines pass/fail."""

    timeout_seconds: float = 120.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of running a single benchmark task."""

    task_id: str
    reward: float
    passed: bool
    trajectory: Trajectory | None = None
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    output: Any = None
    verify_output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated result of running a full benchmark."""

    task_results: dict[str, TaskResult] = field(default_factory=dict)
    pass_rate: float = 0.0
    mean_reward: float = 0.0
    total_cost_usd: float = 0.0
    split: str = "train"
    duration_ms: float = 0.0

    @staticmethod
    def from_task_results(
        results: dict[str, TaskResult],
        *,
        split: str = "train",
        duration_ms: float = 0.0,
    ) -> BenchmarkResult:
        """Construct a BenchmarkResult with computed aggregates."""
        if not results:
            return BenchmarkResult(split=split, duration_ms=duration_ms)

        rewards = [r.reward for r in results.values()]
        passed = sum(1 for r in results.values() if r.passed)
        total_cost = sum(r.cost_usd for r in results.values())

        return BenchmarkResult(
            task_results=results,
            pass_rate=passed / len(results) if results else 0.0,
            mean_reward=sum(rewards) / len(rewards) if rewards else 0.0,
            total_cost_usd=total_cost,
            split=split,
            duration_ms=duration_ms,
        )


class BenchmarkRunner(ABC):
    """Abstract benchmark runner.

    Subclass this to integrate any benchmark (local tasks, Harbor, tau-bench, etc.)
    with the optimization loop.
    """

    @abstractmethod
    async def run(
        self,
        agent_factory: Callable[[], LlmAgent | Awaitable[LlmAgent]],
        *,
        task_ids: list[str] | None = None,
    ) -> BenchmarkResult:
        """Run the benchmark and return aggregated results.

        Parameters
        ----------
        agent_factory:
            Callable that creates a fresh agent instance for each task.
        task_ids:
            Optional subset of tasks to run. If None, run all tasks.
        """

    @abstractmethod
    def list_task_ids(self) -> list[str]:
        """Return all available task IDs."""


class LocalBenchmarkRunner(BenchmarkRunner):
    """Run tasks defined as TaskSpec objects against a deep agent.

    Each task:
    1. Creates a fresh agent via the factory
    2. Pre-populates workspace files in a StateBackend
    3. Sends the task instruction to the agent
    4. Optionally verifies output against expected data
    5. Captures the execution trajectory

    Parameters
    ----------
    tasks:
        List of task specifications.
    verify_fn:
        Optional custom verifier. Called as ``verify_fn(task, agent_output, state)``
        and should return a float reward in [0.0, 1.0].
    pass_threshold:
        Minimum reward to consider a task passed.
    max_concurrency:
        Maximum number of tasks to run concurrently.
    """

    def __init__(
        self,
        tasks: list[TaskSpec],
        *,
        verify_fn: (
            Callable[[TaskSpec, str, dict[str, Any]], float]
            | Callable[[TaskSpec, str, dict[str, Any]], Awaitable[float]]
            | None
        ) = None,
        pass_threshold: float = 0.5,
        max_concurrency: int = 4,
    ) -> None:
        self._tasks = {t.task_id: t for t in tasks}
        self._task_order = [t.task_id for t in tasks]
        self._verify_fn = verify_fn
        self._pass_threshold = pass_threshold
        self._max_concurrency = max_concurrency

    def list_task_ids(self) -> list[str]:
        return list(self._task_order)

    async def run(
        self,
        agent_factory: Callable[[], LlmAgent | Awaitable[LlmAgent]],
        *,
        task_ids: list[str] | None = None,
    ) -> BenchmarkResult:
        ids = task_ids or self._task_order
        tasks_to_run = [self._tasks[tid] for tid in ids if tid in self._tasks]

        if not tasks_to_run:
            return BenchmarkResult()

        semaphore = asyncio.Semaphore(self._max_concurrency)
        start_ns = time.time_ns()

        async def _run_one(task: TaskSpec) -> TaskResult:
            async with semaphore:
                return await self._run_task(task, agent_factory)

        raw_results = await asyncio.gather(
            *(_run_one(t) for t in tasks_to_run),
            return_exceptions=True,
        )

        results: dict[str, TaskResult] = {}
        for i, raw in enumerate(raw_results):
            task = tasks_to_run[i]
            if isinstance(raw, BaseException):
                logger.warning("Task %s failed with exception: %s", task.task_id, raw)
                results[task.task_id] = TaskResult(
                    task_id=task.task_id,
                    reward=0.0,
                    passed=False,
                    error=str(raw),
                )
            else:
                results[task.task_id] = raw

        end_ns = time.time_ns()
        return BenchmarkResult.from_task_results(
            results,
            duration_ms=(end_ns - start_ns) / 1_000_000,
        )

    async def _run_task(
        self,
        task: TaskSpec,
        agent_factory: Callable[[], LlmAgent | Awaitable[LlmAgent]],
    ) -> TaskResult:
        """Run a single task against a fresh agent."""
        from google.adk.runners import InMemoryRunner
        from google.genai import types

        from adk_deepagents.optimization.replay import _build_replay_trajectory
        from adk_deepagents.optimization.trajectory import AgentStep, ToolCall

        start_ns = time.time_ns()

        # Build agent
        agent_or_coro = agent_factory()
        if asyncio.iscoroutine(agent_or_coro) or asyncio.isfuture(agent_or_coro):
            agent: LlmAgent = await agent_or_coro  # type: ignore[assignment]
        else:
            agent: LlmAgent = agent_or_coro  # type: ignore[assignment]

        # Set up state with workspace files
        state: dict[str, Any] = {"files": {}}
        for path, content in task.workspace_files.items():
            state["files"][path] = {
                "content": content.splitlines(),
                "created_at": "2025-01-01T00:00:00",
                "modified_at": "2025-01-01T00:00:00",
            }

        runner = InMemoryRunner(agent=agent, app_name="benchmark")
        session = await runner.session_service.create_session(
            app_name="benchmark",
            user_id="benchmark",
            state=state,
        )

        # Run agent
        user_message = types.Content(
            role="user",
            parts=[types.Part(text=task.instruction)],
        )

        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []

        async for event in runner.run_async(
            session_id=session.id,
            user_id="benchmark",
            new_message=user_message,
        ):
            if not event.content or not event.content.parts:
                continue
            for part in event.content.parts:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    text_chunks.append(text)
                fc = getattr(part, "function_call", None)
                if fc and getattr(fc, "name", None):
                    tool_calls.append(
                        ToolCall(
                            name=fc.name,
                            args=dict(fc.args) if fc.args else {},
                            response=None,
                            duration_ms=0.0,
                        )
                    )

        end_ns = time.time_ns()
        output_text = "".join(text_chunks)

        # Get final state for verification
        final_session = await runner.session_service.get_session(
            app_name="benchmark",
            user_id="benchmark",
            session_id=session.id,
        )
        final_state = dict(final_session.state) if final_session else state

        # Verify
        reward: float = 0.0
        verify_output = ""
        if self._verify_fn is not None:
            try:
                result = self._verify_fn(task, output_text, final_state)
                reward = (
                    float(await result)  # type: ignore[arg-type]
                    if asyncio.iscoroutine(result)
                    else float(result)  # type: ignore[arg-type]
                )
            except Exception as exc:
                logger.warning("Verifier failed for %s: %s", task.task_id, exc)
                verify_output = f"Verifier error: {exc}"
        elif task.expected_output is not None:
            reward, verify_output = _default_verify(task, final_state)

        # Build trajectory
        trajectory = _build_replay_trajectory(
            source_trace_id=f"benchmark-{task.task_id}",
            session_id=session.id,
            agent_name=agent.name or "benchmark_agent",
            prompts=[task.instruction],
            per_turn_outputs=[output_text],
            all_steps=[
                AgentStep(
                    agent_name=agent.name or "benchmark_agent",
                    tool_calls=tool_calls,
                ),
            ],
            start_ns=start_ns,
            end_ns=end_ns,
        )

        return TaskResult(
            task_id=task.task_id,
            reward=reward,
            passed=reward >= self._pass_threshold,
            trajectory=trajectory,
            duration_ms=(end_ns - start_ns) / 1_000_000,
            output=output_text,
            verify_output=verify_output,
        )


def _default_verify(task: TaskSpec, final_state: dict[str, Any]) -> tuple[float, str]:
    """Default verifier: check if expected_output fields are in agent state files."""
    if task.expected_output is None:
        return 0.0, "No expected output defined"

    # Look for output in state files
    files = final_state.get("files", {})
    output_data: dict[str, Any] | None = None

    # Try common output file paths
    for candidate_path in ["/workspace/output.json", "/output.json", "output.json"]:
        file_entry = files.get(candidate_path)
        if file_entry is not None:
            content = file_entry.get("content", [])
            raw = "\n".join(content) if isinstance(content, list) else str(content)
            try:
                output_data = json.loads(raw)
                break
            except (json.JSONDecodeError, ValueError):
                continue

    if output_data is None:
        return 0.0, "No output.json found in agent workspace"

    return _score_json_match(task.expected_output, output_data)


def _score_json_match(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> tuple[float, str]:
    """Score how well actual JSON matches expected JSON.

    Returns (score, details) where score is 0.0-1.0.
    """
    if not expected:
        return 1.0, "Empty expected output"

    total_fields = 0
    matched_fields = 0
    partial_fields = 0
    details: list[str] = []

    for key, expected_val in expected.items():
        total_fields += 1
        actual_val = actual.get(key)

        if actual_val is None:
            details.append(f"MISSING: {key}")
            continue

        if _values_match(expected_val, actual_val):
            matched_fields += 1
        else:
            partial_fields += 1
            details.append(
                f"WRONG: {key} (expected {_preview(expected_val)}, got {_preview(actual_val)})"
            )

    if total_fields == 0:
        return 1.0, "No fields to check"

    score = (matched_fields + partial_fields * 0.3) / total_fields
    summary = f"{matched_fields}/{total_fields} exact, {partial_fields} partial"
    if details:
        summary += "\n" + "\n".join(details)
    return score, summary


def _values_match(expected: Any, actual: Any) -> bool:
    """Check if two values match with tolerance."""
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(expected - actual) < 0.01

    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False
        return all(_values_match(e, a) for e, a in zip(expected, actual, strict=True))

    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False
        return all(_values_match(expected[k], actual[k]) for k in expected)

    return expected == actual


def _preview(value: Any, max_len: int = 50) -> str:
    """Short preview of a value for error messages."""
    s = str(value)
    return s[:max_len] + "..." if len(s) > max_len else s
