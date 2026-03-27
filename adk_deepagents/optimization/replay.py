"""Session replay — re-execute a trajectory's original task.

Runs a fresh agent session with the same prompt(s) as the original trajectory
and captures the new execution.  Supports:

- **Single-turn and multi-turn** conversations
- **Tool approval handling** — auto-approves ``adk_request_confirmation``
  events (configurable per-tool)
- **User simulator** — optional callback for multi-turn conversations that
  generates contextual user responses instead of blindly replaying canned
  messages.  Falls back to the original user messages when not provided.
- **Full tool-call capture** with per-step grouping

Note: replay is a *re-execution of the task*, not a deterministic replay
of actions.  The agent may take a completely different path — that's by
design for optimization scoring.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from google.adk.agents import LlmAgent

from adk_deepagents.optimization.trajectory import (
    AgentStep,
    ModelCall,
    ToolCall,
    Trajectory,
)

logger = logging.getLogger(__name__)

_CONFIRMATION_FUNCTION_NAME = "adk_request_confirmation"


@dataclass
class BuiltAgent:
    """An agent instance with an optional async cleanup function."""

    agent: LlmAgent
    cleanup: Callable[[], Awaitable[None]] | None = None


@dataclass
class ReplayConfig:
    """Configuration for replay behaviour.

    Controls how tool approvals and multi-turn user responses are handled
    during non-deterministic replay.
    """

    tool_approval: Literal["auto_approve", "auto_reject", "original"] = "auto_approve"
    """How to handle ``adk_request_confirmation`` events during replay.

    - ``"auto_approve"`` (default): approve all tool calls immediately.
    - ``"auto_reject"``: reject all tool calls.
    - ``"original"``: look up the original trajectory's tool calls and
      approve only tools that were executed in the original run.
    """

    user_simulator: Callable[[str, list[str], str], str | Awaitable[str]] | None = None
    """Optional callback for multi-turn conversations.

    Called as ``simulator(original_task, conversation_so_far, agent_last_output)``
    and should return the next user message.  When ``None``, the original
    user messages from the trajectory are replayed verbatim.

    Signature::

        def simulate(
            original_task: str,
            previous_user_messages: list[str],
            agent_output: str,
        ) -> str: ...
    """

    max_approval_rounds: int = 10
    """Safety limit on approval round-trips per turn."""


@dataclass
class ReplayResult:
    """Result of replaying a trajectory."""

    source_trace_id: str
    replay_session_id: str
    prompts: list[str]
    output_text: str
    events: list[dict[str, Any]] = field(default_factory=list)
    replay_trajectory: Trajectory | None = None
    per_turn_outputs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt extraction
# ---------------------------------------------------------------------------


def _extract_user_messages_from_request(request: dict[str, Any]) -> list[str]:
    """Extract user message texts from a single model-call request payload."""
    texts: list[str] = []

    # Vertex-style: {"contents": [{"role": "user", "parts": [{"text": ...}]}]}
    contents = request.get("contents")
    if isinstance(contents, list):
        for content in contents:
            if not isinstance(content, dict):
                continue
            if content.get("role") != "user":
                continue
            parts = content.get("parts", [])
            if isinstance(parts, list):
                msg_texts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p]
                if msg_texts:
                    texts.append(" ".join(msg_texts))

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
                texts.append(content.strip())

    return texts


def extract_original_prompt(trajectory: Trajectory) -> str:
    """Extract the first user prompt from a trajectory.

    Supports both Vertex-style ``contents`` and chat-style ``messages`` formats.
    Raises ``ValueError`` if no prompt can be extracted.
    """
    prompts = extract_all_user_prompts(trajectory)
    if not prompts:
        raise ValueError(
            f"Could not extract original prompt from trajectory {trajectory.trace_id}. "
            "No user message found in model call requests."
        )
    return prompts[0]


def extract_all_user_prompts(trajectory: Trajectory) -> list[str]:
    """Extract all distinct user prompts from a trajectory.

    In a multi-turn conversation the model-call request for step *N*
    typically contains *all* prior user messages.  This function
    de-duplicates so that each unique user message appears exactly once,
    in order of first occurrence.

    Returns an empty list when no user messages are found.
    """
    seen: set[str] = set()
    ordered: list[str] = []

    for step in trajectory.steps:
        if step.model_call is None or step.model_call.request is None:
            continue
        for text in _extract_user_messages_from_request(step.model_call.request):
            if text not in seen:
                seen.add(text)
                ordered.append(text)

    return ordered


# ---------------------------------------------------------------------------
# Trajectory construction from replay events
# ---------------------------------------------------------------------------


def _should_approve_tool(
    tool_name: str,
    *,
    policy: Literal["auto_approve", "auto_reject", "original"],
    original_tool_names: set[str],
) -> bool:
    """Decide whether to approve a tool call during replay."""
    if policy == "auto_approve":
        return True
    if policy == "auto_reject":
        return False
    # "original": approve if the tool was used in the original trajectory.
    return tool_name in original_tool_names


def _build_confirmation_response(*, request_id: str, approved: bool):
    """Build a confirmation response message for the agent."""
    from google.adk.tools.tool_confirmation import ToolConfirmation
    from google.genai import types

    confirmation = ToolConfirmation(confirmed=approved, payload=None)
    response_payload = confirmation.model_dump(by_alias=True, exclude_none=True)

    part = types.Part.from_function_response(
        name=_CONFIRMATION_FUNCTION_NAME,
        response=response_payload,
    )
    if part.function_response is not None:
        part.function_response.id = request_id

    return types.Content(role="user", parts=[part])


def _extract_confirmation_from_event(event) -> list[dict[str, Any]]:
    """Extract adk_request_confirmation function calls from an event."""
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return []

    confirmations: list[dict[str, Any]] = []
    for part in parts:
        fc = getattr(part, "function_call", None)
        if fc is None:
            continue
        if getattr(fc, "name", None) != _CONFIRMATION_FUNCTION_NAME:
            continue
        request_id = getattr(fc, "id", None)
        if not isinstance(request_id, str) or not request_id.strip():
            continue

        args = getattr(fc, "args", None)
        args_dict = args if isinstance(args, dict) else {}
        original_call = args_dict.get("originalFunctionCall", {})
        original_call_dict = original_call if isinstance(original_call, dict) else {}
        tool_name = original_call_dict.get("name", "unknown_tool")

        confirmations.append(
            {
                "request_id": request_id,
                "tool_name": tool_name if isinstance(tool_name, str) else "unknown_tool",
            }
        )

    return confirmations


async def _collect_turn_events(
    runner,
    *,
    session_id: str,
    user_id: str,
    message,
    agent_name: str,
    config: ReplayConfig,
    original_tool_names: set[str],
) -> tuple[list[str], list[dict[str, Any]], list[AgentStep]]:
    """Run one turn, handling confirmations, and collect results."""
    text_chunks: list[str] = []
    event_summaries: list[dict[str, Any]] = []
    current_tool_calls: list[ToolCall] = []
    pending_fc: list[dict[str, Any]] = []

    pending_messages = [message]
    approval_rounds = 0

    while pending_messages:
        next_message = pending_messages.pop(0)
        pending_confirmations: list[dict[str, Any]] = []
        seen_confirmation_ids: set[str] = set()

        async for event in runner.run_async(
            session_id=session_id,
            user_id=user_id,
            new_message=next_message,
        ):
            if getattr(event, "author", None) == "user":
                continue

            if not event.content or not event.content.parts:
                continue

            # Check for confirmation requests.
            for conf in _extract_confirmation_from_event(event):
                if conf["request_id"] not in seen_confirmation_ids:
                    seen_confirmation_ids.add(conf["request_id"])
                    pending_confirmations.append(conf)

            for part in event.content.parts:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    text_chunks.append(text)

                fc = getattr(part, "function_call", None)
                if fc and getattr(fc, "name", None):
                    # Skip confirmation pseudo-calls.
                    if fc.name == _CONFIRMATION_FUNCTION_NAME:
                        continue
                    args = dict(fc.args) if fc.args else {}
                    event_summaries.append(
                        {
                            "type": "function_call",
                            "name": fc.name,
                            "args": args,
                        }
                    )
                    pending_fc.append({"name": fc.name, "args": args})

                fr = getattr(part, "function_response", None)
                if fr and getattr(fr, "name", None):
                    if fr.name == _CONFIRMATION_FUNCTION_NAME:
                        continue
                    resp = None
                    if hasattr(fr, "response") and fr.response:
                        resp = dict(fr.response) if hasattr(fr.response, "items") else fr.response
                    event_summaries.append(
                        {
                            "type": "function_response",
                            "name": fr.name,
                        }
                    )
                    matched = False
                    for i, pfc in enumerate(pending_fc):
                        if pfc["name"] == fr.name:
                            current_tool_calls.append(
                                ToolCall(
                                    name=pfc["name"],
                                    args=pfc["args"],
                                    response=resp,
                                    duration_ms=0.0,
                                )
                            )
                            pending_fc.pop(i)
                            matched = True
                            break
                    if not matched:
                        current_tool_calls.append(
                            ToolCall(
                                name=fr.name,
                                args={},
                                response=resp,
                                duration_ms=0.0,
                            )
                        )

        # Handle pending confirmations.
        for conf in pending_confirmations:
            if approval_rounds >= config.max_approval_rounds:
                logger.warning(
                    "Max approval rounds (%d) reached, auto-rejecting %s",
                    config.max_approval_rounds,
                    conf["tool_name"],
                )
                approved = False
            else:
                approved = _should_approve_tool(
                    conf["tool_name"],
                    policy=config.tool_approval,
                    original_tool_names=original_tool_names,
                )
            pending_messages.append(
                _build_confirmation_response(
                    request_id=conf["request_id"],
                    approved=approved,
                )
            )
            approval_rounds += 1
            event_summaries.append(
                {
                    "type": "tool_approval",
                    "tool_name": conf["tool_name"],
                    "approved": approved,
                }
            )

    # Flush remaining pending function_calls.
    for pfc in pending_fc:
        current_tool_calls.append(
            ToolCall(
                name=pfc["name"],
                args=pfc["args"],
                response=None,
                duration_ms=0.0,
                error="no_response",
            )
        )

    steps: list[AgentStep] = []
    output_text = "".join(text_chunks)
    if current_tool_calls or output_text:
        steps.append(
            AgentStep(
                agent_name=agent_name,
                model_call=None,
                tool_calls=current_tool_calls,
            )
        )

    return text_chunks, event_summaries, steps


def _build_replay_trajectory(
    *,
    source_trace_id: str,
    session_id: str,
    agent_name: str,
    prompts: list[str],
    per_turn_outputs: list[str],
    all_steps: list[AgentStep],
    start_ns: int,
    end_ns: int,
) -> Trajectory:
    """Construct a Trajectory from replay run data.

    Embeds each turn's prompt and output into the step's model call so that
    the evaluator can inspect them.
    """
    duration_ms = (end_ns - start_ns) / 1_000_000

    # Attach prompt/output to steps.  If we have more prompts than steps
    # (unlikely but safe), create additional steps.
    final_steps: list[AgentStep] = []
    prompt_idx = 0

    for step in all_steps:
        prompt_text = prompts[prompt_idx] if prompt_idx < len(prompts) else None
        output_text = per_turn_outputs[prompt_idx] if prompt_idx < len(per_turn_outputs) else ""

        request_payload: dict[str, Any] | None = None
        response_payload: dict[str, Any] | None = None

        if prompt_text is not None:
            request_payload = {"contents": [{"role": "user", "parts": [{"text": prompt_text}]}]}
            prompt_idx += 1

        if output_text:
            response_payload = {"candidates": [{"content": {"parts": [{"text": output_text}]}}]}

        final_steps.append(
            AgentStep(
                agent_name=step.agent_name,
                model_call=ModelCall(
                    model="",
                    input_tokens=0,
                    output_tokens=0,
                    duration_ms=duration_ms / max(len(all_steps), 1),
                    request=request_payload,
                    response=response_payload,
                    finish_reason="stop",
                ),
                tool_calls=step.tool_calls,
            )
        )

    # If there are remaining prompts with no matching steps, add them.
    while prompt_idx < len(prompts):
        prompt_text = prompts[prompt_idx]
        output_text = per_turn_outputs[prompt_idx] if prompt_idx < len(per_turn_outputs) else ""
        request_payload = {"contents": [{"role": "user", "parts": [{"text": prompt_text}]}]}
        response_payload = None
        if output_text:
            response_payload = {"candidates": [{"content": {"parts": [{"text": output_text}]}}]}
        final_steps.append(
            AgentStep(
                agent_name=agent_name,
                model_call=ModelCall(
                    model="",
                    input_tokens=0,
                    output_tokens=0,
                    duration_ms=0.0,
                    request=request_payload,
                    response=response_payload,
                    finish_reason="stop",
                ),
            )
        )
        prompt_idx += 1

    combined_output = " ".join(per_turn_outputs)
    return Trajectory(
        trace_id=f"replay-{source_trace_id[:12]}-{start_ns}",
        session_id=session_id,
        agent_name=agent_name,
        steps=final_steps,
        start_time_ns=start_ns,
        end_time_ns=end_ns,
        status="ok" if combined_output.strip() else "error",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _original_tool_names_from_trajectory(trajectory: Trajectory) -> set[str]:
    """Collect tool names that were actually executed in the original trajectory."""
    names: set[str] = set()
    for step in trajectory.steps:
        for tc in step.tool_calls:
            if tc.error is None:
                names.add(tc.name)
    return names


async def replay_trajectory(
    trajectory: Trajectory,
    *,
    agent_builder: Callable[[], BuiltAgent | Awaitable[BuiltAgent]],
    prompts: list[str] | None = None,
    prompt: str | None = None,
    config: ReplayConfig | None = None,
    initial_state: dict[str, Any] | None = None,
    app_name: str = "optimization_replay",
    user_id: str = "optimizer",
) -> ReplayResult:
    """Replay a trajectory by running a fresh agent session.

    Supports single-turn and multi-turn replay with tool approval handling
    and an optional user simulator for contextual follow-up messages.

    Parameters
    ----------
    trajectory:
        The source trajectory to replay.
    agent_builder:
        A callable that returns a ``BuiltAgent`` (sync or async).
    prompts:
        Override the list of prompts to replay.  If ``None``, all user
        prompts are extracted from the trajectory.
    prompt:
        Single-prompt shorthand.  Overrides ``prompts`` if set.
    config:
        Replay configuration (tool approval policy, user simulator).
        Defaults to ``ReplayConfig()`` (auto-approve all tools).
    initial_state:
        Optional initial session state for the replay.
    app_name:
        ADK app name for the replay session.
    user_id:
        ADK user ID for the replay session.

    Returns
    -------
    ReplayResult
        The replay result with per-turn outputs, event summaries,
        and a reconstructed Trajectory.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    resolved_config = config or ReplayConfig()

    # Resolve prompts.
    if prompt is not None:
        resolved_prompts = [prompt]
    elif prompts is not None:
        resolved_prompts = list(prompts)
    else:
        resolved_prompts = extract_all_user_prompts(trajectory)
        if not resolved_prompts:
            raise ValueError(
                f"Could not extract any user prompts from trajectory {trajectory.trace_id}."
            )

    original_tool_names = _original_tool_names_from_trajectory(trajectory)

    # Build the agent.
    result = agent_builder()
    built: BuiltAgent
    if asyncio.iscoroutine(result) or asyncio.isfuture(result):
        built = await result  # type: ignore[misc]
    else:
        built = result  # type: ignore[assignment]

    cleanup = built.cleanup
    try:
        runner = InMemoryRunner(agent=built.agent, app_name=app_name)

        state: dict[str, Any] = {"files": {}}
        if initial_state:
            state.update(initial_state)

        session = await runner.session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            state=state,
        )

        agent_name = built.agent.name or trajectory.agent_name or ""
        all_text_chunks: list[str] = []
        all_events: list[dict[str, Any]] = []
        all_steps: list[AgentStep] = []
        per_turn_outputs: list[str] = []
        sent_prompts: list[str] = []

        start_ns = time.time_ns()

        # Always send the first prompt (the task).
        first_prompt = resolved_prompts[0]
        remaining_prompts = list(resolved_prompts[1:])

        user_message = types.Content(
            role="user",
            parts=[types.Part(text=first_prompt)],
        )

        turn_texts, turn_events, turn_steps = await _collect_turn_events(
            runner,
            session_id=session.id,
            user_id=user_id,
            message=user_message,
            agent_name=agent_name,
            config=resolved_config,
            original_tool_names=original_tool_names,
        )

        turn_output = "".join(turn_texts)
        per_turn_outputs.append(turn_output)
        all_text_chunks.extend(turn_texts)
        all_events.extend(turn_events)
        all_steps.extend(turn_steps)
        sent_prompts.append(first_prompt)

        # Handle follow-up turns.
        turn_index = 0
        while remaining_prompts or resolved_config.user_simulator:
            turn_index += 1
            agent_last_output = turn_output

            # Determine the next user message.
            if resolved_config.user_simulator is not None:
                # Use the simulator to generate a contextual response.
                sim_result = resolved_config.user_simulator(
                    first_prompt,
                    list(sent_prompts),
                    agent_last_output,
                )
                if asyncio.iscoroutine(sim_result):
                    next_prompt = await sim_result  # type: ignore[misc]
                else:
                    next_prompt = sim_result  # type: ignore[assignment]

                # Simulator returning empty string signals end of conversation.
                if not next_prompt or not next_prompt.strip():
                    break
            elif remaining_prompts:
                next_prompt = remaining_prompts.pop(0)
            else:
                break

            user_message = types.Content(
                role="user",
                parts=[types.Part(text=next_prompt)],
            )

            turn_texts, turn_events, turn_steps = await _collect_turn_events(
                runner,
                session_id=session.id,
                user_id=user_id,
                message=user_message,
                agent_name=agent_name,
                config=resolved_config,
                original_tool_names=original_tool_names,
            )

            turn_output = "".join(turn_texts)
            per_turn_outputs.append(turn_output)
            all_text_chunks.extend(turn_texts)
            all_events.extend(turn_events)
            all_steps.extend(turn_steps)
            sent_prompts.append(next_prompt)

        end_ns = time.time_ns()
        output_text = "".join(all_text_chunks)

        replay_traj = _build_replay_trajectory(
            source_trace_id=trajectory.trace_id,
            session_id=session.id,
            agent_name=agent_name,
            prompts=sent_prompts,
            per_turn_outputs=per_turn_outputs,
            all_steps=all_steps,
            start_ns=start_ns,
            end_ns=end_ns,
        )

        return ReplayResult(
            source_trace_id=trajectory.trace_id,
            replay_session_id=session.id,
            prompts=sent_prompts,
            output_text=output_text,
            events=all_events,
            replay_trajectory=replay_traj,
            per_turn_outputs=per_turn_outputs,
        )

    finally:
        if cleanup is not None:
            await cleanup()
