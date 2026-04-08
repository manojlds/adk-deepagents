"""Tests for optimization/learnings.py — LearningsStore."""

from __future__ import annotations

from adk_deepagents.optimization.learnings import LearningEntry, LearningsStore

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestLearningsStoreInit:
    def test_empty_file(self, tmp_path):
        store = LearningsStore(tmp_path / "learnings.jsonl")
        assert len(store) == 0
        assert store.entries() == []

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "learnings.jsonl"
        LearningsStore(path)
        assert path.parent.exists()


# ---------------------------------------------------------------------------
# Append and persistence
# ---------------------------------------------------------------------------


class TestAppendPersistence:
    def test_append_and_reload(self, tmp_path):
        path = tmp_path / "learnings.jsonl"
        s1 = LearningsStore(path)
        s1.append(LearningEntry(iteration=1, category="failed_attempt", summary="bad idea"))
        s1.append(LearningEntry(iteration=2, category="successful_change", summary="good idea"))
        assert len(s1) == 2

        s2 = LearningsStore(path)
        assert len(s2) == 2
        assert s2.entries()[0].summary == "bad idea"
        assert s2.entries()[1].category == "successful_change"


# ---------------------------------------------------------------------------
# recent
# ---------------------------------------------------------------------------


class TestRecent:
    def test_with_limit(self, tmp_path):
        store = LearningsStore(tmp_path / "l.jsonl")
        for i in range(10):
            store.append(LearningEntry(iteration=i, category="confirmed_pattern", summary=f"p{i}"))
        recent = store.recent(3)
        assert len(recent) == 3
        assert recent[0].summary == "p7"
        assert recent[2].summary == "p9"

    def test_fewer_than_limit(self, tmp_path):
        store = LearningsStore(tmp_path / "l.jsonl")
        store.append(LearningEntry(iteration=1, category="open_question", summary="q1"))
        recent = store.recent(5)
        assert len(recent) == 1


# ---------------------------------------------------------------------------
# by_category
# ---------------------------------------------------------------------------


class TestByCategory:
    def test_filters_correctly(self, tmp_path):
        store = LearningsStore(tmp_path / "l.jsonl")
        store.append(LearningEntry(iteration=1, category="failed_attempt", summary="f1"))
        store.append(LearningEntry(iteration=2, category="successful_change", summary="s1"))
        store.append(LearningEntry(iteration=3, category="failed_attempt", summary="f2"))

        failed = store.by_category("failed_attempt")
        assert len(failed) == 2
        assert all(e.category == "failed_attempt" for e in failed)

    def test_no_match(self, tmp_path):
        store = LearningsStore(tmp_path / "l.jsonl")
        store.append(LearningEntry(iteration=1, category="failed_attempt", summary="f1"))
        assert store.by_category("open_question") == []


# ---------------------------------------------------------------------------
# Shortcuts
# ---------------------------------------------------------------------------


class TestShortcuts:
    def test_failed_attempts(self, tmp_path):
        store = LearningsStore(tmp_path / "l.jsonl")
        store.append(LearningEntry(iteration=1, category="failed_attempt", summary="f"))
        store.append(LearningEntry(iteration=2, category="successful_change", summary="s"))
        assert len(store.failed_attempts()) == 1
        assert store.failed_attempts()[0].summary == "f"

    def test_successful_changes(self, tmp_path):
        store = LearningsStore(tmp_path / "l.jsonl")
        store.append(LearningEntry(iteration=1, category="failed_attempt", summary="f"))
        store.append(LearningEntry(iteration=2, category="successful_change", summary="s"))
        assert len(store.successful_changes()) == 1
        assert store.successful_changes()[0].summary == "s"


# ---------------------------------------------------------------------------
# to_prompt_context
# ---------------------------------------------------------------------------


class TestToPromptContext:
    def test_empty_store(self, tmp_path):
        store = LearningsStore(tmp_path / "l.jsonl")
        assert store.to_prompt_context() == ""

    def test_with_entries(self, tmp_path):
        store = LearningsStore(tmp_path / "l.jsonl")
        store.append(
            LearningEntry(
                iteration=1,
                category="failed_attempt",
                summary="tried X, broke Y",
                score_before=0.5,
                score_after=0.3,
            )
        )
        store.append(
            LearningEntry(
                iteration=2,
                category="successful_change",
                summary="improved Z",
                suggestion_kind="prompt_edit",
            )
        )
        store.append(LearningEntry(iteration=3, category="confirmed_pattern", summary="always A"))
        ctx = store.to_prompt_context()
        assert "Optimization Learnings" in ctx
        assert "Failed Attempts" in ctx
        assert "tried X, broke Y" in ctx
        assert "0.500 → 0.300" in ctx
        assert "Successful Changes" in ctx
        assert "[prompt_edit]" in ctx
        assert "Confirmed Patterns" in ctx
        assert "always A" in ctx


# ---------------------------------------------------------------------------
# Corrupt JSONL
# ---------------------------------------------------------------------------


class TestCorruptJsonl:
    def test_skips_bad_lines(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(
            '{"iteration":1,"category":"failed_attempt","summary":"ok"}\n'
            "garbage line\n"
            '{"iteration":2,"category":"open_question","summary":"hmm"}\n',
            encoding="utf-8",
        )
        store = LearningsStore(path)
        assert len(store) == 2
        assert store.entries()[0].iteration == 1
        assert store.entries()[1].category == "open_question"
