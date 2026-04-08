"""Learnings store — structured cross-session memory for optimization.

Records what was tried, what worked, what failed, and confirmed patterns.
The reflector reads learnings to avoid repeating failed attempts and to
build on successful strategies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class LearningEntry:
    """A single learning from the optimization process."""

    iteration: int
    category: Literal[
        "confirmed_pattern",
        "successful_change",
        "failed_attempt",
        "open_question",
    ]
    summary: str
    evidence_trace_ids: list[str] = field(default_factory=list)
    suggestion_kind: str | None = None
    score_before: float | None = None
    score_after: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LearningsStore:
    """JSONL-backed learnings store for cross-session memory.

    Each line is a JSON-serialized ``LearningEntry``.  Supports filtering
    by category and rendering as prompt context for the reflector.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[LearningEntry] = self._load()

    def _load(self) -> list[LearningEntry]:
        if not self._path.exists():
            return []
        entries: list[LearningEntry] = []
        for line_num, line in enumerate(
            self._path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entries.append(
                    LearningEntry(
                        **{k: v for k, v in data.items() if k in LearningEntry.__dataclass_fields__}
                    )
                )
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Skipping malformed learning at line %d: %s", line_num, exc)
        return entries

    def append(self, entry: LearningEntry) -> None:
        """Append an entry and flush to disk."""
        self._entries.append(entry)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False))
            f.write("\n")

    def recent(self, n: int = 20) -> list[LearningEntry]:
        """Return the N most recent entries."""
        return self._entries[-n:]

    def by_category(
        self,
        category: str,
    ) -> list[LearningEntry]:
        """Return all entries matching a category."""
        return [e for e in self._entries if e.category == category]

    def failed_attempts(self) -> list[LearningEntry]:
        """Return all failed attempt entries."""
        return self.by_category("failed_attempt")

    def successful_changes(self) -> list[LearningEntry]:
        """Return all successful change entries."""
        return self.by_category("successful_change")

    def to_prompt_context(self, *, max_entries: int = 10) -> str:
        """Render recent learnings as text for the reflector prompt.

        Groups by category and shows the most recent entries, prioritizing
        failed attempts (to avoid repeating them) and successful changes
        (to build on them).
        """
        if not self._entries:
            return ""

        lines: list[str] = []
        lines.append("## Optimization Learnings (from previous iterations)")
        lines.append("")

        # Prioritize: failed attempts first, then successful, then patterns
        failed = self.by_category("failed_attempt")[-max_entries:]
        successful = self.by_category("successful_change")[-max_entries:]
        patterns = self.by_category("confirmed_pattern")[-max_entries:]
        questions = self.by_category("open_question")[-max_entries:]

        if failed:
            lines.append("### Failed Attempts (DO NOT repeat these)")
            for entry in failed:
                kind = f" [{entry.suggestion_kind}]" if entry.suggestion_kind else ""
                delta = ""
                if entry.score_before is not None and entry.score_after is not None:
                    delta = f" (score: {entry.score_before:.3f} → {entry.score_after:.3f})"
                lines.append(f"- Iter {entry.iteration}{kind}: {entry.summary}{delta}")
            lines.append("")

        if successful:
            lines.append("### Successful Changes (build on these)")
            for entry in successful:
                kind = f" [{entry.suggestion_kind}]" if entry.suggestion_kind else ""
                delta = ""
                if entry.score_before is not None and entry.score_after is not None:
                    delta = f" (score: {entry.score_before:.3f} → {entry.score_after:.3f})"
                lines.append(f"- Iter {entry.iteration}{kind}: {entry.summary}{delta}")
            lines.append("")

        if patterns:
            lines.append("### Confirmed Patterns")
            for entry in patterns:
                lines.append(f"- {entry.summary}")
            lines.append("")

        if questions:
            lines.append("### Open Questions")
            for entry in questions:
                lines.append(f"- {entry.summary}")
            lines.append("")

        return "\n".join(lines)

    def entries(self) -> list[LearningEntry]:
        """Return all entries in order."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
