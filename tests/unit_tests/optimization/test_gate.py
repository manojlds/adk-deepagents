"""Tests for optimization/gate.py — RegressionSuite and gate types."""

from __future__ import annotations

from adk_deepagents.optimization.gate import (
    GateConfig,
    GateResult,
    GateStepResult,
    RegressionSuite,
    RegressionTask,
)

# ---------------------------------------------------------------------------
# RegressionSuite.load
# ---------------------------------------------------------------------------


class TestRegressionSuiteLoad:
    def test_nonexistent_file(self, tmp_path):
        suite = RegressionSuite.load(tmp_path / "missing.json")
        assert len(suite) == 0
        assert suite.task_ids() == []

    def test_corrupt_file(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json", encoding="utf-8")
        suite = RegressionSuite.load(path)
        assert len(suite) == 0


# ---------------------------------------------------------------------------
# Save and re-load round-trip
# ---------------------------------------------------------------------------


class TestRegressionSuiteSaveLoad:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "suite.json"
        suite1 = RegressionSuite(
            path=path,
            tasks={
                "t1": RegressionTask(
                    task_id="t1",
                    added_at_iteration=1,
                    last_reward=0.9,
                    last_passed=True,
                ),
                "t2": RegressionTask(
                    task_id="t2",
                    added_at_iteration=2,
                    last_reward=0.0,
                    last_passed=False,
                ),
            },
        )
        suite1.save()

        suite2 = RegressionSuite.load(path)
        assert len(suite2) == 2
        assert suite2.tasks["t1"].last_reward == 0.9
        assert suite2.tasks["t1"].last_passed is True
        assert suite2.tasks["t2"].added_at_iteration == 2
        assert suite2.tasks["t2"].last_passed is False


# ---------------------------------------------------------------------------
# promote
# ---------------------------------------------------------------------------


class TestRegressionSuitePromote:
    def test_adds_new_tasks(self, tmp_path):
        path = tmp_path / "suite.json"
        suite = RegressionSuite(path=path)
        count = suite.promote(["t1", "t2"], iteration=5)
        assert count == 2
        assert len(suite) == 2
        assert suite.tasks["t1"].added_at_iteration == 5
        assert suite.tasks["t1"].last_passed is True

    def test_no_duplicates(self, tmp_path):
        path = tmp_path / "suite.json"
        suite = RegressionSuite(
            path=path,
            tasks={"t1": RegressionTask(task_id="t1", added_at_iteration=1)},
        )
        count = suite.promote(["t1", "t2"], iteration=3)
        assert count == 1
        assert len(suite) == 2
        # t1 should keep its original iteration
        assert suite.tasks["t1"].added_at_iteration == 1

    def test_promote_with_rewards(self, tmp_path):
        path = tmp_path / "suite.json"
        suite = RegressionSuite(path=path)
        count = suite.promote(
            ["t1"],
            iteration=1,
            rewards={"t1": 0.85},
        )
        assert count == 1
        assert suite.tasks["t1"].last_reward == 0.85

    def test_promote_saves_to_disk(self, tmp_path):
        path = tmp_path / "suite.json"
        suite = RegressionSuite(path=path)
        suite.promote(["t1"], iteration=1)
        assert path.exists()

        reloaded = RegressionSuite.load(path)
        assert "t1" in reloaded.tasks

    def test_promote_empty_returns_zero(self, tmp_path):
        path = tmp_path / "suite.json"
        suite = RegressionSuite(path=path)
        count = suite.promote([])
        assert count == 0


# ---------------------------------------------------------------------------
# update_results
# ---------------------------------------------------------------------------


class TestRegressionSuiteUpdateResults:
    def test_updates_existing(self, tmp_path):
        path = tmp_path / "suite.json"
        suite = RegressionSuite(
            path=path,
            tasks={
                "t1": RegressionTask(task_id="t1", last_reward=1.0, last_passed=True),
                "t2": RegressionTask(task_id="t2", last_reward=0.5, last_passed=True),
            },
        )
        suite.update_results({"t1": 0.0, "t2": 0.8})
        assert suite.tasks["t1"].last_reward == 0.0
        assert suite.tasks["t1"].last_passed is False
        assert suite.tasks["t2"].last_reward == 0.8
        assert suite.tasks["t2"].last_passed is True

    def test_ignores_unknown_ids(self, tmp_path):
        path = tmp_path / "suite.json"
        suite = RegressionSuite(path=path)
        suite.update_results({"unknown": 1.0})
        assert len(suite) == 0


# ---------------------------------------------------------------------------
# GateConfig defaults
# ---------------------------------------------------------------------------


class TestGateConfig:
    def test_defaults(self):
        cfg = GateConfig()
        assert cfg.regression_threshold == 0.8
        assert cfg.require_improvement is True
        assert cfg.auto_promote is True
        assert cfg.skip_if_empty_suite is True

    def test_custom(self):
        cfg = GateConfig(regression_threshold=0.9, auto_promote=False)
        assert cfg.regression_threshold == 0.9
        assert cfg.auto_promote is False


# ---------------------------------------------------------------------------
# GateResult
# ---------------------------------------------------------------------------


class TestGateResult:
    def test_creation(self):
        result = GateResult(
            passed=True,
            step_results=[
                GateStepResult(step=1, name="regression_suite", passed=True),
            ],
            promoted_tasks=["t1"],
        )
        assert result.passed is True
        assert len(result.step_results) == 1
        assert result.promoted_tasks == ["t1"]
        assert result.test_result is None

    def test_failed(self):
        result = GateResult(passed=False)
        assert result.passed is False
        assert result.step_results == []
        assert result.promoted_tasks == []
