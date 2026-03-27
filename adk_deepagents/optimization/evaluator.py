"""LLM-based trajectory evaluator (judge).

Uses a lightweight ADK agent with structured output to score trajectories
on configurable criteria.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from adk_deepagents.optimization.trajectory import FeedbackEntry, Trajectory

logger = logging.getLogger(__name__)

# Max characters for tool args/response previews in the judge payload.
_MAX_TOOL_PREVIEW = 400
_MAX_RESPONSE_PREVIEW = 800


# ---------------------------------------------------------------------------
# Rubric types
# ---------------------------------------------------------------------------


@dataclass
class EvaluationCriterion:
    """A single evaluation criterion with a weight."""

    name: str
    description: str
    weight: float = 1.0


@dataclass
class EvaluationRubric:
    """A set of criteria used by the evaluator agent."""

    criteria: list[EvaluationCriterion] = field(default_factory=list)
    judge_instructions: str = ""
    name: str = "default_v1"


def default_rubric() -> EvaluationRubric:
    """Return the default evaluation rubric."""
    return EvaluationRubric(
        criteria=[
            EvaluationCriterion(
                name="task_completion",
                description=(
                    "Did the agent successfully complete the task requested"
                    " by the user? Consider whether the final output"
                    " addresses the original prompt."
                ),
                weight=0.6,
            ),
            EvaluationCriterion(
                name="efficiency",
                description=(
                    "Was the agent efficient? Consider number of steps,"
                    " token usage, unnecessary retries, and whether it"
                    " avoided redundant work."
                ),
                weight=0.2,
            ),
            EvaluationCriterion(
                name="tool_usage_quality",
                description=(
                    "Did the agent use tools appropriately? Consider"
                    " correct tool selection, proper arguments, error"
                    " handling, and whether it avoided tool errors."
                ),
                weight=0.2,
            ),
        ],
        name="default_v1",
    )


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class CriterionAssessment(BaseModel):
    """LLM assessment for a single evaluation criterion."""

    name: str
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class TrajectoryJudgment(BaseModel):
    """Structured output from the evaluator agent."""

    summary: str
    strengths: list[str] = []
    issues: list[str] = []
    criteria: list[CriterionAssessment] = []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with an ellipsis indicator."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...(truncated)"


def extract_original_prompt(trajectory: Trajectory) -> str | None:
    """Extract the original user prompt from a trajectory's model call requests.

    Supports both Vertex-style ``contents`` and chat-style ``messages``
    formats.  Returns None if no prompt can be extracted.
    """
    for step in trajectory.steps:
        if step.model_call is None or step.model_call.request is None:
            continue

        request = step.model_call.request

        # Vertex-style: {"contents": [{"role": "user", "parts": [...]}]}
        contents = request.get("contents")
        if isinstance(contents, list):
            for content in contents:
                if not isinstance(content, dict):
                    continue
                if content.get("role") != "user":
                    continue
                parts = content.get("parts", [])
                if isinstance(parts, list):
                    texts = []
                    for part in parts:
                        if isinstance(part, dict) and "text" in part:
                            texts.append(part["text"])
                    if texts:
                        return " ".join(texts)

        # Chat-style: {"messages": [{"role": "user", "content": ...}]}
        messages = request.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") != "user":
                    continue
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()

    return None


def _extract_final_response(trajectory: Trajectory) -> str | None:
    """Extract the final assistant response text from a trajectory."""
    for step in reversed(trajectory.steps):
        if step.model_call is None or step.model_call.response is None:
            continue

        response = step.model_call.response

        # Vertex-style
        candidates = response.get("candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                content = candidate.get("content", {})
                parts = content.get("parts", []) if isinstance(content, dict) else []
                for part in parts:
                    if isinstance(part, dict) and "text" in part:
                        return _truncate(part["text"], _MAX_RESPONSE_PREVIEW)

        # Chat-style
        choices = response.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                msg = choice.get("message", {})
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return _truncate(content.strip(), _MAX_RESPONSE_PREVIEW)

    return None


def _trajectory_to_judge_payload(trajectory: Trajectory) -> str:
    """Serialize a trajectory into a compact text payload for the judge."""
    lines: list[str] = []

    prompt = extract_original_prompt(trajectory)
    lines.append(f"## Original User Prompt\n{prompt or '(not available)'}\n")

    lines.append("## Execution Summary")
    lines.append(f"- Status: {trajectory.status}")
    lines.append(f"- Duration: {trajectory.duration_ms:.0f}ms")
    lines.append(f"- Total steps: {len(trajectory.steps)}")
    lines.append(f"- Input tokens: {trajectory.total_input_tokens}")
    lines.append(f"- Output tokens: {trajectory.total_output_tokens}")

    error_count = sum(1 for step in trajectory.steps for tc in step.tool_calls if tc.error)
    if error_count:
        lines.append(f"- Tool errors: {error_count}")
    lines.append("")

    lines.append("## Step-by-Step Trajectory")
    for i, step in enumerate(trajectory.steps, 1):
        lines.append(f"\n### Step {i} (agent: {step.agent_name})")
        if step.model_call:
            mc = step.model_call
            lines.append(
                f"  Model: {mc.model} | "
                f"tokens: {mc.input_tokens}in/{mc.output_tokens}out | "
                f"{mc.duration_ms:.0f}ms | "
                f"finish: {mc.finish_reason or 'unknown'}"
            )

        for tc in step.tool_calls:
            args_str = _truncate(
                json.dumps(tc.args, ensure_ascii=False),
                _MAX_TOOL_PREVIEW,
            )
            resp_str = ""
            if tc.response is not None:
                resp_str = _truncate(
                    json.dumps(tc.response, ensure_ascii=False),
                    _MAX_TOOL_PREVIEW,
                )
            error_str = f" ERROR: {tc.error}" if tc.error else ""
            lines.append(
                f"  Tool: {tc.name}({args_str}) -> {resp_str}{error_str} [{tc.duration_ms:.0f}ms]"
            )

    final_response = _extract_final_response(trajectory)
    if final_response:
        lines.append(f"\n## Final Response\n{final_response}")

    return "\n".join(lines)


def _build_judge_instruction(rubric: EvaluationRubric) -> str:
    """Build the system instruction for the evaluator agent."""
    criteria_text = "\n".join(
        f"- **{c.name}** (weight {c.weight}): {c.description}" for c in rubric.criteria
    )

    extra = ""
    if rubric.judge_instructions:
        extra = f"\n\n## Additional Instructions\n{rubric.judge_instructions}"

    return (
        "You are an expert evaluator of AI agent trajectories.\n\n"
        "You will receive a detailed trajectory of an agent execution,"
        " including:\n"
        "- The original user prompt\n"
        "- Each step (model calls and tool calls)\n"
        "- The final response\n\n"
        "Evaluate the trajectory on these criteria:\n"
        f"{criteria_text}\n\n"
        "For each criterion, provide:\n"
        "- A score from 0.0 (worst) to 1.0 (best)\n"
        "- A brief reasoning explaining the score\n\n"
        "Also provide:\n"
        "- A one-sentence summary of the overall quality\n"
        "- A list of strengths (good things the agent did)\n"
        "- A list of issues (problems or inefficiencies)\n\n"
        "Be strict and evidence-based. Reference specific steps or"
        " tool calls in your reasoning. Do not inflate scores."
        f"{extra}"
    )


def _compute_weighted_score(
    judgment: TrajectoryJudgment,
    rubric: EvaluationRubric,
) -> float:
    """Compute the overall weighted score from per-criterion assessments."""
    criteria_by_name = {c.name: c for c in rubric.criteria}
    total_weight = 0.0
    weighted_sum = 0.0

    for assessment in judgment.criteria:
        criterion = criteria_by_name.get(assessment.name)
        if criterion is None:
            continue
        weight = criterion.weight
        weighted_sum += assessment.score * weight
        total_weight += weight

    if total_weight == 0.0:
        # Fallback: simple average of all assessment scores
        if not judgment.criteria:
            return 0.5
        return sum(a.score for a in judgment.criteria) / len(judgment.criteria)

    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def evaluate_trajectory(
    trajectory: Trajectory,
    *,
    model: str = "gemini-2.5-flash",
    rubric: EvaluationRubric | None = None,
) -> FeedbackEntry:
    """Evaluate a trajectory using an LLM judge and return a FeedbackEntry.

    Parameters
    ----------
    trajectory:
        The trajectory to evaluate.
    model:
        Model to use for the evaluator agent.
    rubric:
        Evaluation rubric. If ``None``, the default rubric is used.

    Returns
    -------
    FeedbackEntry
        A feedback entry with the computed score, summary comment,
        and structured metadata including per-criterion assessments.
    """
    from google.adk.agents import LlmAgent
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    resolved_rubric = rubric or default_rubric()
    judge_payload = _trajectory_to_judge_payload(trajectory)
    instruction = _build_judge_instruction(resolved_rubric)

    agent = LlmAgent(
        name="trajectory_evaluator",
        model=model,
        instruction=instruction,
        output_schema=TrajectoryJudgment,
    )

    runner = InMemoryRunner(agent=agent, app_name="trajectory_eval")
    session = await runner.session_service.create_session(
        app_name="trajectory_eval",
        user_id="evaluator",
    )

    user_message = types.Content(
        role="user",
        parts=[types.Part(text=(f"Evaluate this agent trajectory:\n\n{judge_payload}"))],
    )

    text_chunks: list[str] = []
    async for event in runner.run_async(
        session_id=session.id,
        user_id="evaluator",
        new_message=user_message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    text_chunks.append(text)

    raw_output = "".join(text_chunks)

    # Parse the structured output
    judgment: TrajectoryJudgment
    try:
        data = json.loads(raw_output)
        judgment = TrajectoryJudgment.model_validate(data)
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning(
            "Failed to parse evaluator structured output for %s: %s",
            trajectory.trace_id,
            exc,
        )
        judgment = TrajectoryJudgment(
            summary=f"Evaluation parse error: {exc}",
            criteria=[],
        )

    overall_score = _compute_weighted_score(judgment, resolved_rubric)

    return FeedbackEntry(
        source="evaluator",
        rating=overall_score,
        comment=judgment.summary,
        timestamp_ns=time.time_ns(),
        metadata={
            "rubric_name": resolved_rubric.name,
            "criteria": [c.model_dump() for c in judgment.criteria],
            "strengths": judgment.strengths,
            "issues": judgment.issues,
            "evaluated_trace_id": trajectory.trace_id,
            "trajectory_status": trajectory.status,
            "duration_ms": trajectory.duration_ms,
            "total_input_tokens": trajectory.total_input_tokens,
            "total_output_tokens": trajectory.total_output_tokens,
            "judge_model": model,
        },
    )
