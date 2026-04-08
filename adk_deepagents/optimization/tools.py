"""Experience query tools for meta-agent introspection.

Exposes ``TrajectoryStore``, ``ScoreHistory``, and ``LearningsStore``
as ADK-compatible function tools so the reflector or a meta-agent can
query its own optimization history.
"""

from __future__ import annotations

from collections.abc import Callable

from adk_deepagents.optimization.history import ScoreHistory
from adk_deepagents.optimization.learnings import LearningsStore
from adk_deepagents.optimization.store import TrajectoryStore


def create_experience_tools(
    store: TrajectoryStore,
    history: ScoreHistory,
    learnings: LearningsStore | None = None,
) -> list[Callable]:
    """Create tools for querying the optimization experience store.

    Parameters
    ----------
    store:
        Trajectory store containing per-run execution traces.
    history:
        Score history tracking per-iteration results.
    learnings:
        Optional learnings store for cross-session memory.

    Returns
    -------
    list[Callable]
        Tools suitable for adding to an ADK agent's tool list.
    """

    def list_trajectories(
        *,
        sort_by: str = "score",
        limit: int = 20,
        status: str | None = None,
    ) -> str:
        """List evaluated trajectories, ranked by score.

        Args:
            sort_by: Sort key — "score" (default) or "time".
            limit: Maximum number of results (default 20).
            status: Filter by status — "ok", "error", or "unset".
        """
        trajectories = store.list_trajectories(status=status)

        if sort_by == "score":
            trajectories.sort(key=lambda t: t.score if t.score is not None else -1, reverse=True)
        else:
            trajectories.sort(key=lambda t: t.start_time_ns, reverse=True)

        trajectories = trajectories[:limit]

        if not trajectories:
            return "No trajectories found."

        lines: list[str] = []
        lines.append(f"Trajectories ({len(trajectories)} shown):")
        for t in trajectories:
            score_str = f"{t.score:.3f}" if t.score is not None else "N/A"
            golden = " ★" if t.is_golden else ""
            steps = len(t.steps)
            tokens = t.total_input_tokens + t.total_output_tokens
            lines.append(
                f"  {t.trace_id[:16]}  score={score_str}{golden}  "
                f"status={t.status}  steps={steps}  tokens={tokens}"
            )

        return "\n".join(lines)

    def show_failures(*, trace_id: str) -> str:
        """Show failed tool calls and issues for a specific trajectory.

        Args:
            trace_id: The trajectory trace ID to inspect.
        """
        traj = store.load(trace_id)
        if traj is None:
            return f"Trajectory {trace_id} not found."

        lines: list[str] = []
        lines.append(f"Trajectory: {traj.trace_id}")
        lines.append(f"Status: {traj.status}")
        lines.append(f"Score: {traj.score if traj.score is not None else 'N/A'}")
        lines.append(f"Steps: {len(traj.steps)}")
        lines.append("")

        # Tool errors
        errors: list[str] = []
        for i, step in enumerate(traj.steps, 1):
            for tc in step.tool_calls:
                if tc.error:
                    args_preview = str(tc.args)[:100]
                    errors.append(f"  Step {i}: {tc.name}({args_preview}) → ERROR: {tc.error}")

        if errors:
            lines.append("Tool errors:")
            lines.extend(errors)
        else:
            lines.append("No tool errors.")

        # Feedback
        if traj.feedback:
            lines.append("")
            lines.append("Feedback:")
            for fb in traj.feedback:
                rating = f"{fb.rating:.3f}" if fb.rating is not None else "N/A"
                lines.append(f"  [{fb.source}] rating={rating}: {fb.comment[:120]}")
                issues = fb.metadata.get("issues", [])
                for issue in issues:
                    lines.append(f"    - {issue}")

        return "\n".join(lines)

    def diff_trajectories(*, trace_id_a: str, trace_id_b: str) -> str:
        """Compare two trajectories — show score changes and different tool usage.

        Args:
            trace_id_a: First trajectory trace ID.
            trace_id_b: Second trajectory trace ID.
        """
        traj_a = store.load(trace_id_a)
        traj_b = store.load(trace_id_b)

        if traj_a is None:
            return f"Trajectory {trace_id_a} not found."
        if traj_b is None:
            return f"Trajectory {trace_id_b} not found."

        lines: list[str] = []
        lines.append(f"Comparing {trace_id_a[:16]} vs {trace_id_b[:16]}:")

        # Scores
        score_a = traj_a.score if traj_a.score is not None else 0.0
        score_b = traj_b.score if traj_b.score is not None else 0.0
        delta = score_b - score_a
        lines.append(f"  Score: {score_a:.3f} → {score_b:.3f} (Δ {delta:+.3f})")
        lines.append(f"  Status: {traj_a.status} → {traj_b.status}")
        lines.append(f"  Steps: {len(traj_a.steps)} → {len(traj_b.steps)}")

        # Token comparison
        tokens_a = traj_a.total_input_tokens + traj_a.total_output_tokens
        tokens_b = traj_b.total_input_tokens + traj_b.total_output_tokens
        lines.append(f"  Tokens: {tokens_a} → {tokens_b}")

        # Tool usage comparison
        tools_a: dict[str, int] = {}
        tools_b: dict[str, int] = {}
        for step in traj_a.steps:
            for tc in step.tool_calls:
                tools_a[tc.name] = tools_a.get(tc.name, 0) + 1
        for step in traj_b.steps:
            for tc in step.tool_calls:
                tools_b[tc.name] = tools_b.get(tc.name, 0) + 1

        all_tools = sorted(set(tools_a.keys()) | set(tools_b.keys()))
        if all_tools:
            lines.append("  Tool usage:")
            for tool in all_tools:
                count_a = tools_a.get(tool, 0)
                count_b = tools_b.get(tool, 0)
                if count_a != count_b:
                    lines.append(f"    {tool}: {count_a} → {count_b}")

        return "\n".join(lines)

    def show_score_history(*, last_n: int = 20) -> str:
        """Show the optimization score progression.

        Args:
            last_n: Number of recent entries to show (default 20).
        """
        return history.summary(last_n=last_n)

    tools: list[Callable] = [
        list_trajectories,
        show_failures,
        diff_trajectories,
        show_score_history,
    ]

    if learnings is not None:
        _learnings = learnings

        def show_learnings(
            *,
            category: str | None = None,
            last_n: int = 10,
        ) -> str:
            """Show recent learnings from the optimization process.

            Args:
                category: Filter by category — "confirmed_pattern",
                    "successful_change", "failed_attempt", or "open_question".
                last_n: Maximum entries to show (default 10).
            """
            if category is not None:
                entries = _learnings.by_category(category)[-last_n:]
            else:
                entries = _learnings.recent(last_n)

            if not entries:
                return "No learnings recorded yet."

            lines: list[str] = []
            lines.append(f"Learnings ({len(entries)} shown):")
            for e in entries:
                kind = f" [{e.suggestion_kind}]" if e.suggestion_kind else ""
                delta = ""
                if e.score_before is not None and e.score_after is not None:
                    delta = f" ({e.score_before:.3f} → {e.score_after:.3f})"
                lines.append(f"  [{e.category}] iter {e.iteration}{kind}: {e.summary}{delta}")

            return "\n".join(lines)

        tools.append(show_learnings)

    return tools
