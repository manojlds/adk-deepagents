"""Autoresearch-style optimization loop.

Replays trajectories, evaluates them with an LLM judge, compares to
baselines, and iteratively suggests/applies improvements to prompts,
skills, and tool definitions.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from adk_deepagents.optimization.evaluator import (
    EvaluationRubric,
    evaluate_trajectory,
)
from adk_deepagents.optimization.replay import (
    BuiltAgent,
    ReplayConfig,
    ReplayResult,
    replay_trajectory,
)
from adk_deepagents.optimization.store import TrajectoryStore
from adk_deepagents.optimization.trajectory import FeedbackEntry, Trajectory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate & suggestion types
# ---------------------------------------------------------------------------


@dataclass
class OptimizationCandidate:
    """A configuration snapshot for the agent being optimized.

    ``agent_kwargs`` are passed to ``create_deep_agent()`` or equivalent.
    """

    agent_kwargs: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0


@dataclass
class ImprovementSuggestion:
    """A single suggestion for improving the agent."""

    kind: Literal[
        "instruction_append",
        "instruction_replace",
        "skill_add",
        "skill_remove",
        "tool_definition_note",
    ]
    target: str
    proposal: str
    rationale: str
    evidence_trace_ids: list[str] = field(default_factory=list)
    auto_applicable: bool = False


# ---------------------------------------------------------------------------
# Per-example and per-iteration result types
# ---------------------------------------------------------------------------


@dataclass
class ExampleResult:
    """Result of replaying and evaluating a single trajectory."""

    source_trajectory: Trajectory
    replay: ReplayResult
    feedback: FeedbackEntry
    baseline_score: float | None = None
    delta: float | None = None


@dataclass
class IterationResult:
    """Aggregated result of one optimization iteration."""

    iteration: int
    candidate: OptimizationCandidate
    examples: list[ExampleResult] = field(default_factory=list)
    average_score: float | None = None
    average_delta: float | None = None
    regressions: int = 0
    suggestions: list[ImprovementSuggestion] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Final result of the optimization loop."""

    best_candidate: OptimizationCandidate
    iterations: list[IterationResult] = field(default_factory=list)
    stopped_reason: str = ""


# ---------------------------------------------------------------------------
# Structured output for the reflector
# ---------------------------------------------------------------------------


class SuggestionOutput(BaseModel):
    """A single improvement suggestion from the reflector."""

    kind: str = Field(
        description=(
            "One of: instruction_append, instruction_replace, "
            "skill_add, skill_remove, tool_definition_note"
        )
    )
    target: str = Field(
        description="What to change (e.g., 'instruction', a skill path, a tool name)"
    )
    proposal: str = Field(description="The specific change to make")
    rationale: str = Field(description="Why this change would help")


class ReflectorOutput(BaseModel):
    """Structured output from the reflector agent."""

    analysis: str = Field(description="Brief analysis of the iteration results")
    suggestions: list[SuggestionOutput] = []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VALID_SUGGESTION_KINDS = {
    "instruction_append",
    "instruction_replace",
    "skill_add",
    "skill_remove",
    "tool_definition_note",
}

_AUTO_APPLICABLE_KINDS = {"instruction_append", "instruction_replace"}


def _resolve_baseline_score(trajectory: Trajectory) -> float | None:
    """Determine the baseline score for a trajectory."""
    if trajectory.score is not None:
        return trajectory.score

    evaluator_feedback = [fb for fb in trajectory.feedback if fb.rating is not None]
    if evaluator_feedback:
        return sum(fb.rating for fb in evaluator_feedback) / len(evaluator_feedback)  # type: ignore[misc]

    if trajectory.is_golden:
        return 1.0

    return None


def _apply_suggestion(
    candidate: OptimizationCandidate,
    suggestion: ImprovementSuggestion,
) -> OptimizationCandidate:
    """Apply an auto-applicable suggestion to produce a new candidate."""
    new_kwargs = dict(candidate.agent_kwargs)

    if suggestion.kind == "instruction_append":
        current = new_kwargs.get("instruction", "") or ""
        new_kwargs["instruction"] = (
            current + "\n\n" + suggestion.proposal if current else suggestion.proposal
        )

    elif suggestion.kind == "instruction_replace":
        new_kwargs["instruction"] = suggestion.proposal

    elif suggestion.kind == "skill_add":
        skills = list(new_kwargs.get("skills", []) or [])
        if suggestion.target not in skills:
            skills.append(suggestion.target)
        new_kwargs["skills"] = skills

    elif suggestion.kind == "skill_remove":
        skills = list(new_kwargs.get("skills", []) or [])
        if suggestion.target in skills:
            skills.remove(suggestion.target)
        new_kwargs["skills"] = skills

    return OptimizationCandidate(
        agent_kwargs=new_kwargs,
        iteration=candidate.iteration + 1,
    )


def _build_reflector_payload(
    candidate: OptimizationCandidate,
    iteration: IterationResult,
) -> str:
    """Build the payload for the reflector agent."""
    lines: list[str] = []

    lines.append("## Current Agent Configuration")
    instruction = candidate.agent_kwargs.get("instruction", "(none)")
    lines.append(f"- Instruction: {instruction}")
    skills = candidate.agent_kwargs.get("skills", [])
    if skills:
        lines.append(f"- Skills: {', '.join(str(s) for s in skills)}")
    lines.append("")

    lines.append(f"## Iteration {iteration.iteration} Results")
    if iteration.average_score is not None:
        lines.append(f"- Average score: {iteration.average_score:.3f}")
    if iteration.average_delta is not None:
        lines.append(f"- Average delta from baseline: {iteration.average_delta:.3f}")
    lines.append(f"- Regressions: {iteration.regressions}")
    lines.append("")

    lines.append("## Per-Example Details")
    for i, ex in enumerate(iteration.examples, 1):
        baseline_str = f"{ex.baseline_score:.2f}" if ex.baseline_score is not None else "N/A"
        score_str = f"{ex.feedback.rating:.2f}" if ex.feedback.rating is not None else "N/A"
        delta_str = f"{ex.delta:+.2f}" if ex.delta is not None else "N/A"
        lines.append(f"\n### Example {i} (trace: {ex.source_trajectory.trace_id[:12]})")
        lines.append(f"- Baseline: {baseline_str} → Replay: {score_str} (Δ {delta_str})")
        lines.append(f"- Evaluator summary: {ex.feedback.comment}")

        criteria = ex.feedback.metadata.get("criteria", [])
        if criteria:
            for c in criteria:
                lines.append(f"  - {c['name']}: {c['score']:.2f} — {c['reasoning']}")

        issues = ex.feedback.metadata.get("issues", [])
        if issues:
            lines.append("- Issues:")
            for issue in issues:
                lines.append(f"  - {issue}")

    return "\n".join(lines)


_REFLECTOR_INSTRUCTION = """\
You are an optimization advisor for AI agents. You analyze the results of
an optimization iteration and suggest concrete improvements.

Given:
1. The current agent configuration (instruction, skills)
2. Results from replaying trajectories with the current config
3. Per-example evaluation scores, deltas, and evaluator feedback

Your job is to:
- Analyze patterns in the failures and regressions
- Suggest specific, actionable improvements

Suggestion types:
- **instruction_append**: Add text to the current instruction
- **instruction_replace**: Replace the entire custom instruction
- **skill_add**: Add a skill directory path
- **skill_remove**: Remove a skill directory path
- **tool_definition_note**: Note about tool usage (not auto-applied)

Guidelines:
- Focus on the biggest score improvements possible
- Be specific — don't suggest vague improvements
- If scores are already high (>0.9) and no regressions, suggest nothing
- Prefer instruction_append over instruction_replace when possible
- Keep instruction changes concise and targeted
"""


async def _suggest_improvements(
    *,
    candidate: OptimizationCandidate,
    iteration: IterationResult,
    model: str = "gemini-2.5-flash",
) -> list[ImprovementSuggestion]:
    """Use an LLM to suggest improvements based on iteration results."""
    from google.adk.agents import LlmAgent
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    payload = _build_reflector_payload(candidate, iteration)

    agent = LlmAgent(
        name="optimization_reflector",
        model=model,
        instruction=_REFLECTOR_INSTRUCTION,
        output_schema=ReflectorOutput,
    )

    runner = InMemoryRunner(agent=agent, app_name="optimization_reflect")
    session = await runner.session_service.create_session(
        app_name="optimization_reflect",
        user_id="optimizer",
    )

    user_message = types.Content(
        role="user",
        parts=[
            types.Part(
                text=(f"Analyze these optimization results and suggest improvements:\n\n{payload}")
            )
        ],
    )

    text_chunks: list[str] = []
    async for event in runner.run_async(
        session_id=session.id,
        user_id="optimizer",
        new_message=user_message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    text_chunks.append(text)

    raw_output = "".join(text_chunks)

    try:
        data = json.loads(raw_output)
        reflector_output = ReflectorOutput.model_validate(data)
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Failed to parse reflector output: %s", exc)
        return []

    suggestions: list[ImprovementSuggestion] = []
    for s in reflector_output.suggestions:
        kind = s.kind if s.kind in _VALID_SUGGESTION_KINDS else "tool_definition_note"
        suggestions.append(
            ImprovementSuggestion(
                kind=kind,  # type: ignore[arg-type]
                target=s.target,
                proposal=s.proposal,
                rationale=s.rationale,
                auto_applicable=kind in _AUTO_APPLICABLE_KINDS,
            )
        )

    return suggestions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_optimization_loop(
    *,
    trajectories: Sequence[Trajectory],
    base_candidate: OptimizationCandidate,
    agent_builder_factory: Callable[
        [OptimizationCandidate],
        BuiltAgent | Awaitable[BuiltAgent],
    ],
    evaluator_model: str = "gemini-2.5-flash",
    rubric: EvaluationRubric | None = None,
    replay_config: ReplayConfig | None = None,
    store: TrajectoryStore | None = None,
    max_iterations: int = 3,
    convergence_delta: float = 0.02,
    apply_mode: Literal["prompt_only", "prompt_and_skills", "suggest_only"] = "prompt_only",
    on_iteration: Callable[[IterationResult], None] | None = None,
) -> OptimizationResult:
    """Run an autoresearch-style optimization loop.

    Parameters
    ----------
    trajectories:
        Source trajectories to replay and evaluate. Use golden or
        high-quality scored trajectories for best results.
    base_candidate:
        Initial agent configuration (``agent_kwargs``).
    agent_builder_factory:
        Callable that takes an ``OptimizationCandidate`` and returns a
        ``BuiltAgent`` (sync or async). Called once per replay.
    evaluator_model:
        Model for the LLM judge and reflector.
    rubric:
        Evaluation rubric for the judge.
    replay_config:
        Replay configuration (tool approval, user simulator).
        Defaults to ``ReplayConfig()`` (auto-approve all tools).
    store:
        Optional trajectory store for persisting replay results.
    max_iterations:
        Maximum optimization iterations.
    convergence_delta:
        Stop if average improvement is below this threshold.
    apply_mode:
        What kinds of suggestions to auto-apply:
        ``"prompt_only"`` — only instruction changes,
        ``"prompt_and_skills"`` — instruction and skills changes,
        ``"suggest_only"`` — no auto-application.
    on_iteration:
        Optional callback invoked after each iteration completes.

    Returns
    -------
    OptimizationResult
        The best candidate found and all iteration results.
    """
    if not trajectories:
        return OptimizationResult(
            best_candidate=base_candidate,
            stopped_reason="no_trajectories",
        )

    current_candidate = base_candidate
    all_iterations: list[IterationResult] = []
    best_candidate = base_candidate
    best_avg_score: float | None = None

    for iteration_num in range(1, max_iterations + 1):
        logger.info("Optimization iteration %d/%d", iteration_num, max_iterations)

        examples: list[ExampleResult] = []

        for traj in trajectories:
            # 1. Resolve baseline score
            baseline = _resolve_baseline_score(traj)

            # 2. Replay
            def _make_builder(
                cand: OptimizationCandidate,
            ) -> Callable[[], BuiltAgent | Awaitable[BuiltAgent]]:
                def builder() -> BuiltAgent | Awaitable[BuiltAgent]:
                    return agent_builder_factory(cand)

                return builder

            try:
                replay_result = await replay_trajectory(
                    traj,
                    agent_builder=_make_builder(current_candidate),
                    config=replay_config,
                )
            except Exception as exc:
                logger.warning("Replay failed for %s: %s", traj.trace_id, exc)
                continue

            # 3. Evaluate the replay
            # Build a minimal trajectory from replay events for evaluation.
            eval_trajectory = replay_result.replay_trajectory
            if eval_trajectory is None:
                eval_trajectory = Trajectory(
                    trace_id=f"replay-{traj.trace_id[:8]}-{iteration_num}",
                    session_id=replay_result.replay_session_id,
                    agent_name=traj.agent_name,
                    steps=traj.steps,
                    status="ok" if replay_result.output_text else "error",
                )

            try:
                feedback = await evaluate_trajectory(
                    eval_trajectory,
                    model=evaluator_model,
                    rubric=rubric,
                )
            except Exception as exc:
                logger.warning(
                    "Evaluation failed for replay of %s: %s",
                    traj.trace_id,
                    exc,
                )
                continue

            # 4. Compute delta
            delta: float | None = None
            if baseline is not None and feedback.rating is not None:
                delta = feedback.rating - baseline

            example = ExampleResult(
                source_trajectory=traj,
                replay=replay_result,
                feedback=feedback,
                baseline_score=baseline,
                delta=delta,
            )
            examples.append(example)

            # 5. Persist if store is available
            if store is not None and eval_trajectory.trace_id:
                store.save(eval_trajectory)
                if feedback.rating is not None:
                    store.set_score(eval_trajectory.trace_id, feedback.rating)
                store.add_feedback(eval_trajectory.trace_id, feedback)
                store.set_tag(
                    eval_trajectory.trace_id,
                    "optimization_parent_trace_id",
                    traj.trace_id,
                )
                store.set_tag(
                    eval_trajectory.trace_id,
                    "optimization_iteration",
                    str(iteration_num),
                )
                store.set_tag(
                    eval_trajectory.trace_id,
                    "optimization_role",
                    "replay",
                )

        # 6. Aggregate iteration metrics
        scores = [ex.feedback.rating for ex in examples if ex.feedback.rating is not None]
        deltas = [ex.delta for ex in examples if ex.delta is not None]
        regressions = sum(1 for d in deltas if d < 0)

        avg_score = sum(scores) / len(scores) if scores else None
        avg_delta = sum(deltas) / len(deltas) if deltas else None

        # 7. Generate suggestions
        iteration_result = IterationResult(
            iteration=iteration_num,
            candidate=current_candidate,
            examples=examples,
            average_score=avg_score,
            average_delta=avg_delta,
            regressions=regressions,
        )

        try:
            suggestions = await _suggest_improvements(
                candidate=current_candidate,
                iteration=iteration_result,
                model=evaluator_model,
            )
        except Exception as exc:
            logger.warning("Reflector failed at iteration %d: %s", iteration_num, exc)
            suggestions = []

        iteration_result.suggestions = suggestions
        all_iterations.append(iteration_result)

        if on_iteration is not None:
            on_iteration(iteration_result)

        # Track best
        if avg_score is not None and (best_avg_score is None or avg_score > best_avg_score):
            best_avg_score = avg_score
            best_candidate = current_candidate

        # 8. Check convergence
        if not suggestions:
            return OptimizationResult(
                best_candidate=best_candidate,
                iterations=all_iterations,
                stopped_reason="no_suggestions",
            )

        if avg_delta is not None and abs(avg_delta) < convergence_delta:
            return OptimizationResult(
                best_candidate=best_candidate,
                iterations=all_iterations,
                stopped_reason="converged",
            )

        # 9. Apply suggestions
        if apply_mode == "suggest_only":
            logger.info(
                "Iteration %d: %d suggestion(s) (suggest_only mode)",
                iteration_num,
                len(suggestions),
            )
            continue

        applicable_kinds = {"instruction_append", "instruction_replace"}
        if apply_mode == "prompt_and_skills":
            applicable_kinds |= {"skill_add", "skill_remove"}

        next_candidate = current_candidate
        applied = 0
        for suggestion in suggestions:
            if suggestion.kind in applicable_kinds:
                next_candidate = _apply_suggestion(next_candidate, suggestion)
                applied += 1

        if applied == 0:
            return OptimizationResult(
                best_candidate=best_candidate,
                iterations=all_iterations,
                stopped_reason="no_applicable_suggestions",
            )

        logger.info(
            "Iteration %d: applied %d suggestion(s), avg_score=%.3f",
            iteration_num,
            applied,
            avg_score or 0.0,
        )
        current_candidate = next_candidate

    return OptimizationResult(
        best_candidate=best_candidate,
        iterations=all_iterations,
        stopped_reason="max_iterations",
    )
