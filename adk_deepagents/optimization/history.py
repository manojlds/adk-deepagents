"""Score history — append-only audit trail for optimization runs.

Tracks per-iteration scores, acceptance decisions, and cost. Backed by
a JSONL file for easy inspection and append-only durability.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HistoryEntry:
    """One row in the optimization history."""

    iteration: int
    val_score: float
    pass_rate: float
    cost_usd: float = 0.0
    accepted: bool = True
    description: str = ""
    timestamp: str = ""
    candidate_kwargs: dict[str, Any] | None = None
    train_scores: dict[str, float] | None = None
    test_scores: dict[str, float] | None = None


class ScoreHistory:
    """Append-only JSONL-backed score history.

    Each line is a JSON-serialized ``HistoryEntry``.  Supports querying
    the historical best score (for monotonic gating) and rendering a
    sparkline for quick visual progress tracking.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[HistoryEntry] = self._load()

    def _load(self) -> list[HistoryEntry]:
        if not self._path.exists():
            return []
        entries: list[HistoryEntry] = []
        for line_num, line in enumerate(
            self._path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entries.append(
                    HistoryEntry(
                        **{k: v for k, v in data.items() if k in HistoryEntry.__dataclass_fields__}
                    )
                )
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Skipping malformed history line %d: %s", line_num, exc)
        return entries

    def append(self, entry: HistoryEntry) -> None:
        """Append an entry and flush to disk."""
        self._entries.append(entry)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False))
            f.write("\n")

    def best_val_score(self) -> float | None:
        """Return the best val_score among accepted entries, or None."""
        accepted = [e.val_score for e in self._entries if e.accepted]
        return max(accepted) if accepted else None

    def best_pass_rate(self) -> float | None:
        """Return the best pass_rate among accepted entries, or None."""
        accepted = [e.pass_rate for e in self._entries if e.accepted]
        return max(accepted) if accepted else None

    def latest(self) -> HistoryEntry | None:
        """Return the most recent entry, or None."""
        return self._entries[-1] if self._entries else None

    def entries(self) -> list[HistoryEntry]:
        """Return all entries in order."""
        return list(self._entries)

    def accepted_entries(self) -> list[HistoryEntry]:
        """Return only accepted entries."""
        return [e for e in self._entries if e.accepted]

    def sparkline(self) -> str:
        """Render a sparkline of val_scores for quick visual progress.

        Uses Unicode block characters. Only shows accepted entries.
        """
        scores = [e.val_score for e in self._entries if e.accepted]
        if not scores:
            return "(no data)"

        blocks = " ▁▂▃▄▅▆▇█"
        lo = min(scores)
        hi = max(scores)
        span = hi - lo if hi > lo else 1.0

        chars: list[str] = []
        for s in scores:
            idx = int((s - lo) / span * (len(blocks) - 1))
            idx = max(0, min(len(blocks) - 1, idx))
            chars.append(blocks[idx])
        return "".join(chars)

    def summary(self, last_n: int = 10) -> str:
        """Return a human-readable summary of recent history."""
        recent = self._entries[-last_n:]
        if not recent:
            return "No optimization history."

        lines: list[str] = []
        lines.append(f"Score history ({len(self._entries)} total, showing last {len(recent)}):")
        lines.append(f"  Sparkline: {self.sparkline()}")
        best = self.best_val_score()
        if best is not None:
            lines.append(f"  Best val_score: {best:.3f}")

        for e in recent:
            status = "✓" if e.accepted else "✗"
            desc = f" — {e.description}" if e.description else ""
            lines.append(
                f"  [{status}] iter {e.iteration}: "
                f"val={e.val_score:.3f} pass={e.pass_rate:.1%} "
                f"cost=${e.cost_usd:.4f}{desc}"
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)
