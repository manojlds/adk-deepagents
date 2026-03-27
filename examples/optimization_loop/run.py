"""Self-improving agent example — optimization loop.

Demonstrates the full optimization cycle:

1. Define an agent with an instruction and skills
2. Run it on test prompts to produce "seed" trajectories
3. Feed trajectories into the optimization loop
4. The loop replays each prompt, evaluates with an LLM judge,
   then suggests and applies instruction improvements
5. Iterate until the agent's scores converge

Usage:
    # Requires GOOGLE_API_KEY, OPENAI_API_KEY, or equivalent
    uv run python examples/optimization_loop/run.py

    # Use a specific model:
    ADK_DEEPAGENTS_MODEL=openai/gpt-4o-mini uv run python examples/optimization_loop/run.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model to use for the agent, evaluator, and reflector.
# Override with ADK_DEEPAGENTS_MODEL or LITELLM_MODEL env vars.
MODEL = os.environ.get(
    "ADK_DEEPAGENTS_MODEL",
    os.environ.get("LITELLM_MODEL", "gemini-2.5-flash"),
)

# The test prompts to optimize over.
# Single-turn prompts: the agent handles these in one shot.
TEST_PROMPTS = [
    "Write a short blog post about why Python is great for beginners.",
    "Create a tweet thread (3 tweets) about the benefits of open source software.",
    "Write a LinkedIn post announcing a new AI-powered developer tool.",
]

# Multi-turn prompts: the first message starts a conversation, and the
# user simulator generates follow-up messages based on what the agent says.
MULTI_TURN_PROMPTS = [
    "I need a blog post about Kubernetes for beginners. Let's start with an outline.",
]

# Initial agent instruction (intentionally minimal — room for improvement).
INITIAL_INSTRUCTION = (
    "You are a content creation assistant. You help users create "
    "blog posts, social media content, and other written materials."
)

# Path to skills (relative to this example dir).
EXAMPLE_DIR = Path(__file__).parent
SKILLS_DIR = str(EXAMPLE_DIR / "skills")

# Number of optimization iterations.
MAX_ITERATIONS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


async def seed_trajectories(
    *,
    agent_kwargs: dict,
    prompts: list[str],
    model,
) -> list:
    """Run the agent on test prompts and capture trajectories.

    Returns a list of Trajectory objects built from the replay events.
    """
    import time

    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from adk_deepagents import create_deep_agent
    from adk_deepagents.optimization.replay import _build_replay_trajectory
    from adk_deepagents.optimization.trajectory import AgentStep, ToolCall, Trajectory

    trajectories: list[Trajectory] = []

    for i, prompt in enumerate(prompts):
        print(f"\n  Seed run {i + 1}/{len(prompts)}: {prompt[:60]}...")

        agent = create_deep_agent(**{**agent_kwargs, "model": model})
        runner = InMemoryRunner(agent=agent, app_name="seed_run")
        session = await runner.session_service.create_session(
            app_name="seed_run",
            user_id="optimizer",
            state={"files": {}},
        )

        user_message = types.Content(role="user", parts=[types.Part(text=prompt)])

        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        tool_responses: dict[str, object] = {}
        start_ns = time.time_ns()

        async for event in runner.run_async(
            session_id=session.id,
            user_id="optimizer",
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
                fr = getattr(part, "function_response", None)
                if fr and getattr(fr, "name", None):
                    resp = None
                    if hasattr(fr, "response") and fr.response:
                        resp = dict(fr.response) if hasattr(fr.response, "items") else fr.response
                    tool_responses[fr.name] = resp

        end_ns = time.time_ns()
        output = "".join(text_chunks)

        # Patch tool responses into tool calls.
        for tc in tool_calls:
            if tc.name in tool_responses:
                tc.response = tool_responses[tc.name]

        traj = _build_replay_trajectory(
            source_trace_id=f"seed-{i}",
            session_id=session.id,
            agent_name=agent.name,
            prompts=[prompt],
            per_turn_outputs=[output],
            all_steps=[
                AgentStep(
                    agent_name=agent.name,
                    tool_calls=tool_calls,
                ),
            ],
            start_ns=start_ns,
            end_ns=end_ns,
        )
        # Mark seed trajectories as golden baseline.
        traj.is_golden = True
        trajectories.append(traj)

        preview = output[:120].replace("\n", " ")
        print(f"    ✓ {len(output)} chars, {len(tool_calls)} tool call(s)")
        print(f"    Preview: {preview}...")

    return trajectories


def _build_user_simulator(model):
    """Build an LLM-backed user simulator for multi-turn replay.

    The simulator plays the role of a user reviewing content drafts.
    It generates contextual follow-up messages based on what the agent
    actually produced (not canned responses), and returns "" to end
    the conversation after 2 follow-ups.
    """

    async def simulate(
        original_task: str,
        previous_messages: list[str],
        agent_output: str,
    ) -> str:
        # End after 2 follow-up messages (3 turns total).
        turn_number = len(previous_messages)
        if turn_number >= 3:
            return ""

        from google.adk.agents import LlmAgent
        from google.adk.runners import InMemoryRunner
        from google.genai import types

        agent = LlmAgent(
            name="user_simulator",
            model=model,
            instruction=(
                "You are simulating a user who is collaborating with a content "
                "creation assistant. You started with this request:\n\n"
                f'  "{original_task}"\n\n'
                "The assistant just responded. Give a short, natural follow-up "
                "message. Examples:\n"
                '- Ask for revisions ("Make the intro more engaging")\n'
                '- Request additions ("Add a section about monitoring")\n'
                '- Give approval with feedback ("Looks good, now write the full post")\n'
                "- Ask clarifying questions\n\n"
                "Keep it to 1-2 sentences. Be specific about what you want changed."
            ),
        )

        runner = InMemoryRunner(agent=agent, app_name="user_sim")
        session = await runner.session_service.create_session(
            app_name="user_sim", user_id="simulator"
        )

        prompt = (
            f"The assistant's latest response:\n\n{agent_output[:500]}\n\n"
            f"This is turn {turn_number + 1}. Generate your follow-up message."
        )

        texts: list[str] = []
        async for event in runner.run_async(
            session_id=session.id,
            user_id="simulator",
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    text = getattr(part, "text", None)
                    if isinstance(text, str) and text:
                        texts.append(text)

        return "".join(texts).strip()

    return simulate


async def run_example():
    """Run the full optimization example."""
    from adk_deepagents import create_deep_agent
    from adk_deepagents.optimization import (
        BuiltAgent,
        OptimizationCandidate,
        ReplayConfig,
        TrajectoryStore,
        run_optimization_loop,
    )

    model = _resolve_model()
    print(f"Model: {MODEL}")
    print(f"Skills: {SKILLS_DIR}")
    print(f"Max iterations: {MAX_ITERATIONS}")
    print(f"Test prompts: {len(TEST_PROMPTS)}")

    # Check if skills dir exists; if not, skip skills.
    skills_dirs = [SKILLS_DIR] if Path(SKILLS_DIR).is_dir() else []
    if not skills_dirs:
        print("(No skills directory found — running without skills)")

    # Base agent configuration.
    base_kwargs: dict = {
        "name": "content_agent",
        "instruction": INITIAL_INSTRUCTION,
        "execution": "local",
    }
    if skills_dirs:
        base_kwargs["skills"] = skills_dirs

    # -----------------------------------------------------------------------
    # Step 1: Seed runs — produce baseline trajectories
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Seed runs (producing baseline trajectories)")
    print("=" * 60)

    # Single-turn seeds.
    seed_trajs = await seed_trajectories(
        agent_kwargs=base_kwargs,
        prompts=TEST_PROMPTS,
        model=model,
    )

    # Multi-turn seeds (run with user simulator to produce a conversation).
    print("\n  Multi-turn seed runs (with user simulator):")
    for i, mt_prompt in enumerate(MULTI_TURN_PROMPTS):
        print(f"\n  Multi-turn {i + 1}/{len(MULTI_TURN_PROMPTS)}: {mt_prompt[:60]}...")

        from adk_deepagents.optimization.replay import (
            BuiltAgent as BA,
        )
        from adk_deepagents.optimization.replay import (
            ReplayConfig as RC,
        )
        from adk_deepagents.optimization.replay import (
            replay_trajectory,
        )
        from adk_deepagents.optimization.trajectory import AgentStep as AS
        from adk_deepagents.optimization.trajectory import ModelCall as MC
        from adk_deepagents.optimization.trajectory import Trajectory as Traj

        seed_traj = Traj(
            trace_id=f"multi-turn-seed-{i}",
            agent_name=base_kwargs["name"],
            steps=[
                AS(
                    agent_name=base_kwargs["name"],
                    model_call=MC(
                        model="",
                        input_tokens=0,
                        output_tokens=0,
                        duration_ms=0,
                        request={"contents": [{"role": "user", "parts": [{"text": mt_prompt}]}]},
                    ),
                )
            ],
            is_golden=True,
        )

        user_sim = _build_user_simulator(model)
        mt_config = RC(
            tool_approval="auto_approve",
            user_simulator=user_sim,
        )

        def _mt_builder() -> BA:
            agent = create_deep_agent(**{**base_kwargs, "model": model})
            return BA(agent=agent)

        mt_result = await replay_trajectory(
            seed_traj,
            agent_builder=_mt_builder,
            config=mt_config,
        )

        if mt_result.replay_trajectory is not None:
            mt_result.replay_trajectory.is_golden = True
            seed_trajs.append(mt_result.replay_trajectory)
            print(f"    ✓ {len(mt_result.per_turn_outputs)} turns, {len(mt_result.events)} events")
            for t, turn_out in enumerate(mt_result.per_turn_outputs):
                preview = turn_out[:80].replace("\n", " ")
                print(f"    Turn {t + 1}: {preview}...")

    print(f"\nProduced {len(seed_trajs)} seed trajectories total.")

    # -----------------------------------------------------------------------
    # Step 2: Persist seed trajectories
    # -----------------------------------------------------------------------
    tmp_dir = tempfile.mkdtemp(prefix="optim_example_")
    store_dir = Path(tmp_dir) / "trajectories"
    store = TrajectoryStore(store_dir)

    for traj in seed_trajs:
        store.save(traj)
    print(f"Stored trajectories in: {store_dir}")

    # -----------------------------------------------------------------------
    # Step 3: Run the optimization loop
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Optimization loop")
    print("=" * 60)

    base_candidate = OptimizationCandidate(agent_kwargs=base_kwargs)

    def on_iteration(iteration_result):
        """Print progress after each iteration."""
        it = iteration_result
        print(f"\n--- Iteration {it.iteration} complete ---")
        if it.average_score is not None:
            print(f"  Average score: {it.average_score:.3f}")
        if it.average_delta is not None:
            print(f"  Average delta: {it.average_delta:+.3f}")
        print(f"  Regressions: {it.regressions}")
        print(f"  Suggestions: {len(it.suggestions)}")
        for s in it.suggestions:
            tag = " [auto-apply]" if s.auto_applicable else " [manual]"
            print(f"    • {s.kind}{tag}: {s.rationale[:80]}")

    def agent_builder_factory(candidate: OptimizationCandidate) -> BuiltAgent:
        kwargs = {**candidate.agent_kwargs, "model": model}
        agent = create_deep_agent(**kwargs)
        return BuiltAgent(agent=agent)

    # ReplayConfig with user simulator for multi-turn trajectories.
    replay_cfg = ReplayConfig(
        tool_approval="auto_approve",
        user_simulator=_build_user_simulator(model),
    )

    result = await run_optimization_loop(
        trajectories=seed_trajs,
        base_candidate=base_candidate,
        agent_builder_factory=agent_builder_factory,
        evaluator_model=model,
        replay_config=replay_cfg,
        store=store,
        max_iterations=MAX_ITERATIONS,
        apply_mode="prompt_and_skills",
        on_iteration=on_iteration,
    )

    # -----------------------------------------------------------------------
    # Step 4: Report results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Stopped: {result.stopped_reason}")
    print(f"Iterations: {len(result.iterations)}")

    # Show score progression.
    print("\nScore progression:")
    for it in result.iterations:
        score = f"{it.average_score:.3f}" if it.average_score is not None else "N/A"
        delta = f"{it.average_delta:+.3f}" if it.average_delta is not None else "N/A"
        print(f"  Iteration {it.iteration}: avg_score={score}, avg_delta={delta}")

    # Show the optimized instruction.
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

    # Show all suggestions across iterations.
    all_suggestions = [s for it in result.iterations for s in it.suggestions]
    if all_suggestions:
        print(f"\nAll suggestions ({len(all_suggestions)}):")
        for s in all_suggestions:
            tag = "✓ applied" if s.auto_applicable else "→ manual"
            print(f"  [{tag}] {s.kind}: {s.target}")
            print(f"    Rationale: {s.rationale[:120]}")
            if s.kind == "tool_definition_note":
                print(f"    Proposal: {s.proposal[:120]}")

    print(f"\nTrajectory store: {store_dir}")
    print(f"Stored {len(store.list_ids())} trajectories total.")


def main():
    """Entrypoint."""
    print("=" * 60)
    print("Self-Improving Agent — Optimization Loop Example")
    print("=" * 60)
    asyncio.run(run_example())


if __name__ == "__main__":
    main()
