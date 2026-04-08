"""Tests for optimization/history.py — ScoreHistory."""

from __future__ import annotations

from adk_deepagents.optimization.history import HistoryEntry, ScoreHistory

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestScoreHistoryInit:
    def test_empty_file(self, tmp_path):
        h = ScoreHistory(tmp_path / "history.jsonl")
        assert len(h) == 0
        assert h.entries() == []

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "history.jsonl"
        ScoreHistory(path)
        assert path.parent.exists()


# ---------------------------------------------------------------------------
# Append and persistence
# ---------------------------------------------------------------------------


class TestAppendPersistence:
    def test_append_and_reload(self, tmp_path):
        path = tmp_path / "history.jsonl"
        h1 = ScoreHistory(path)
        h1.append(HistoryEntry(iteration=1, val_score=0.7, pass_rate=0.5))
        h1.append(HistoryEntry(iteration=2, val_score=0.8, pass_rate=0.6))
        assert len(h1) == 2

        h2 = ScoreHistory(path)
        assert len(h2) == 2
        assert h2.entries()[0].iteration == 1
        assert h2.entries()[1].val_score == 0.8


# ---------------------------------------------------------------------------
# best_val_score
# ---------------------------------------------------------------------------


class TestBestValScore:
    def test_no_entries(self, tmp_path):
        h = ScoreHistory(tmp_path / "h.jsonl")
        assert h.best_val_score() is None

    def test_single_entry(self, tmp_path):
        h = ScoreHistory(tmp_path / "h.jsonl")
        h.append(HistoryEntry(iteration=1, val_score=0.6, pass_rate=0.5))
        assert h.best_val_score() == 0.6

    def test_multiple_only_accepted(self, tmp_path):
        h = ScoreHistory(tmp_path / "h.jsonl")
        h.append(HistoryEntry(iteration=1, val_score=0.5, pass_rate=0.3, accepted=True))
        h.append(HistoryEntry(iteration=2, val_score=0.9, pass_rate=0.8, accepted=False))
        h.append(HistoryEntry(iteration=3, val_score=0.7, pass_rate=0.6, accepted=True))
        assert h.best_val_score() == 0.7


# ---------------------------------------------------------------------------
# best_pass_rate
# ---------------------------------------------------------------------------


class TestBestPassRate:
    def test_no_entries(self, tmp_path):
        h = ScoreHistory(tmp_path / "h.jsonl")
        assert h.best_pass_rate() is None

    def test_returns_best_accepted(self, tmp_path):
        h = ScoreHistory(tmp_path / "h.jsonl")
        h.append(HistoryEntry(iteration=1, val_score=0.5, pass_rate=0.4, accepted=True))
        h.append(HistoryEntry(iteration=2, val_score=0.6, pass_rate=0.9, accepted=False))
        h.append(HistoryEntry(iteration=3, val_score=0.7, pass_rate=0.6, accepted=True))
        assert h.best_pass_rate() == 0.6


# ---------------------------------------------------------------------------
# sparkline
# ---------------------------------------------------------------------------


class TestSparkline:
    def test_no_data(self, tmp_path):
        h = ScoreHistory(tmp_path / "h.jsonl")
        assert h.sparkline() == "(no data)"

    def test_non_empty(self, tmp_path):
        h = ScoreHistory(tmp_path / "h.jsonl")
        for i in range(5):
            h.append(HistoryEntry(iteration=i, val_score=i * 0.2, pass_rate=0.5))
        line = h.sparkline()
        assert len(line) > 0
        assert any(ch in line for ch in "▁▂▃▄▅▆▇█")


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_empty(self, tmp_path):
        h = ScoreHistory(tmp_path / "h.jsonl")
        assert "No optimization history" in h.summary()

    def test_non_empty(self, tmp_path):
        h = ScoreHistory(tmp_path / "h.jsonl")
        h.append(HistoryEntry(iteration=1, val_score=0.5, pass_rate=0.4))
        s = h.summary()
        assert "Score history" in s
        assert "iter 1" in s


# ---------------------------------------------------------------------------
# latest
# ---------------------------------------------------------------------------


class TestLatest:
    def test_none_when_empty(self, tmp_path):
        h = ScoreHistory(tmp_path / "h.jsonl")
        assert h.latest() is None

    def test_returns_last(self, tmp_path):
        h = ScoreHistory(tmp_path / "h.jsonl")
        h.append(HistoryEntry(iteration=1, val_score=0.5, pass_rate=0.3))
        h.append(HistoryEntry(iteration=2, val_score=0.8, pass_rate=0.7))
        latest = h.latest()
        assert latest is not None
        assert latest.iteration == 2


# ---------------------------------------------------------------------------
# Corrupt JSONL
# ---------------------------------------------------------------------------


class TestCorruptJsonl:
    def test_skips_bad_lines(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(
            '{"iteration":1,"val_score":0.5,"pass_rate":0.3}\n'
            "not json at all\n"
            '{"iteration":2,"val_score":0.8,"pass_rate":0.6}\n',
            encoding="utf-8",
        )
        h = ScoreHistory(path)
        assert len(h) == 2
        assert h.entries()[0].iteration == 1
        assert h.entries()[1].iteration == 2
