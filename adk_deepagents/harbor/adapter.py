"""HarborAdapter — BaseAgent implementation for Google ADK deepagents.

The fixed boundary: Harbor integration, ATIF trajectory serialization,
and ADK runner wiring. Projects import this and never modify it.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

# ---------------------------------------------------------------------------
# ATIF serialization
# ---------------------------------------------------------------------------


def to_atif(events: list, model: str, agent_name: str, duration_ms: int = 0) -> dict:
    """Convert ADK runner events to an ATIF trajectory dict."""
    steps: list[dict] = []
    step_id = 0
    now = datetime.now(UTC).isoformat()
    total_input_tokens = 0
    total_output_tokens = 0
    pending_calls: dict[str, dict] = {}

    def _step(source: str, message: str, **extra: Any) -> dict:
        nonlocal step_id
        step_id += 1
        s = {"step_id": step_id, "timestamp": now, "source": source, "message": message}
        s.update({k: v for k, v in extra.items() if v is not None})
        return s

    for event in events:
        if not event.content or not event.content.parts:
            continue

        for part in event.content.parts:
            if getattr(part, "text", None) and event.author != "user":
                steps.append(_step("agent", part.text, model_name=model))

            elif getattr(part, "function_call", None):
                fc = part.function_call
                call_id = getattr(fc, "id", None) or fc.name
                pending_calls[call_id] = {
                    "name": fc.name,
                    "args": dict(fc.args) if fc.args else {},
                    "call_id": call_id,
                }

            elif getattr(part, "function_response", None):
                fr = part.function_response
                call_id = getattr(fr, "id", None) or fr.name
                call_info = pending_calls.pop(
                    call_id, {"name": fr.name, "args": {}, "call_id": call_id}
                )
                response = fr.response or {}
                output = (
                    str(response.get("output", response))
                    if isinstance(response, dict)
                    else str(response)
                )
                steps.append(
                    _step(
                        "agent",
                        f"Tool: {call_info['name']}",
                        tool_calls=[
                            {
                                "tool_call_id": call_id,
                                "function_name": call_info["name"],
                                "arguments": call_info["args"],
                            }
                        ],
                        observation={"results": [{"source_call_id": call_id, "content": output}]},
                    )
                )

        usage = getattr(event, "usage_metadata", None)
        if usage:
            total_input_tokens += getattr(usage, "prompt_token_count", 0) or 0
            total_output_tokens += getattr(usage, "candidates_token_count", 0) or 0

    if not steps:
        steps.append(_step("user", "(empty)"))

    return {
        "schema_version": "ATIF-v1.6",
        "session_id": "adk-session",
        "agent": {"name": agent_name, "version": "0.1.0", "model_name": model},
        "steps": steps,
        "final_metrics": {
            "total_prompt_tokens": total_input_tokens,
            "total_completion_tokens": total_output_tokens,
            "total_cached_tokens": 0,
            "total_cost_usd": None,
            "total_steps": len(steps),
            "extra": {"duration_ms": duration_ms},
        },
    }


# ---------------------------------------------------------------------------
# ADK runner
# ---------------------------------------------------------------------------


async def _run_agent(
    agent: LlmAgent,
    instruction: str,
    app_name: str,
) -> tuple[list, str]:
    """Run an ADK LlmAgent on an instruction. Returns (events, session_id)."""
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)
    session = await session_service.create_session(app_name=app_name, user_id="harbor")

    events = []
    async for event in runner.run_async(
        user_id="harbor",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=instruction)]),
    ):
        events.append(event)

    return events, session.id


# ---------------------------------------------------------------------------
# Harbor adapter
# ---------------------------------------------------------------------------

BuildAgentFn = Callable[[BaseEnvironment], LlmAgent | Coroutine[Any, Any, LlmAgent]]


class HarborAdapter(BaseAgent):
    """Harbor BaseAgent adapter for Google ADK deepagents.

    The agent runs host-side; all tool calls proxy into the Harbor container
    via ``environment.exec()``.

    Usage — factory pattern (most common):

        AutoAgent = HarborAdapter.create(create_agent, agent_name="my-agent", model=MODEL)

    Usage — subclass pattern:

        class AutoAgent(HarborAdapter):
            agent_name = "my-agent"
            model = "gemini-2.0-flash"

            def build_agent(self, environment):
                return create_deep_agent(...)
    """

    SUPPORTS_ATIF = True

    agent_name: str = "adk-autoagent"
    model: str = "gemini-2.0-flash"

    def build_agent(self, environment: BaseEnvironment) -> LlmAgent | Coroutine[Any, Any, LlmAgent]:
        raise NotImplementedError

    @staticmethod
    def name() -> str:
        return "adk-autoagent"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        await environment.exec(command="mkdir -p /task")
        instr_file = self.logs_dir / "instruction.md"
        instr_file.write_text(instruction)
        await environment.upload_file(source_path=instr_file, target_path="/task/instruction.md")

        t0 = time.time()

        # Support both sync and async build_agent implementations
        result = self.build_agent(environment)
        agent = await result if asyncio.iscoroutine(result) else result

        events, session_id = await _run_agent(agent, instruction, app_name=self.agent_name)
        duration_ms = int((time.time() - t0) * 1000)

        atif = to_atif(
            events, model=self.model, agent_name=self.agent_name, duration_ms=duration_ms
        )
        atif["session_id"] = session_id

        traj_path = self.logs_dir / "trajectory.json"
        traj_path.write_text(json.dumps(atif, indent=2))

        try:
            fm = atif["final_metrics"]
            context.n_input_tokens = fm["total_prompt_tokens"]
            context.n_output_tokens = fm["total_completion_tokens"]
        except Exception:
            pass

        fm = atif["final_metrics"]
        print(
            f"steps={fm['total_steps']} duration_ms={duration_ms} "
            f"input={fm['total_prompt_tokens']} output={fm['total_completion_tokens']}"
        )

    @classmethod
    def create(
        cls,
        build_agent_fn: BuildAgentFn,
        agent_name: str = "adk-autoagent",
        model: str = "gemini-2.0-flash",
    ) -> type[HarborAdapter]:
        """Factory: create a concrete HarborAdapter class from a build_agent function.

        The returned class is suitable for use as ``--agent-import-path``.

            AutoAgent = HarborAdapter.create(create_agent, agent_name="my-agent", model=MODEL)
        """
        return type(
            "AutoAgent",
            (cls,),
            {
                "agent_name": agent_name,
                "model": model,
                "build_agent": lambda self, env: build_agent_fn(env),
                "name": staticmethod(lambda: agent_name),
            },
        )
