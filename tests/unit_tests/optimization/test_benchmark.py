"""Tests for optimization/benchmark.py dataclasses and helpers."""

from __future__ import annotations

from adk_deepagents.optimization.benchmark import (
    BenchmarkResult,
    LocalBenchmarkRunner,
    TaskResult,
    TaskSpec,
    _score_json_match,
    _values_match,
)

# ---------------------------------------------------------------------------
# TaskSpec
# ---------------------------------------------------------------------------


class TestTaskSpec:
    def test_defaults(self):
        spec = TaskSpec(task_id="t1", instruction="Do something")
        assert spec.task_id == "t1"
        assert spec.instruction == "Do something"
        assert spec.workspace_files == {}
        assert spec.expected_output is None
        assert spec.verify_command is None
        assert spec.timeout_seconds == 120.0
        assert spec.metadata == {}

    def test_with_fields(self):
        spec = TaskSpec(
            task_id="t2",
            instruction="Write code",
            workspace_files={"/main.py": "print('hi')"},
            expected_output={"result": 42},
            timeout_seconds=60.0,
            metadata={"difficulty": "easy"},
        )
        assert spec.workspace_files == {"/main.py": "print('hi')"}
        assert spec.expected_output == {"result": 42}
        assert spec.timeout_seconds == 60.0
        assert spec.metadata["difficulty"] == "easy"


# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------


class TestTaskResult:
    def test_defaults(self):
        r = TaskResult(task_id="t1", reward=0.8, passed=True)
        assert r.task_id == "t1"
        assert r.reward == 0.8
        assert r.passed is True
        assert r.trajectory is None
        assert r.cost_usd == 0.0
        assert r.duration_ms == 0.0
        assert r.output is None
        assert r.error is None

    def test_with_error(self):
        r = TaskResult(task_id="t1", reward=0.0, passed=False, error="timeout")
        assert r.error == "timeout"
        assert r.passed is False


# ---------------------------------------------------------------------------
# BenchmarkResult.from_task_results
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    def test_from_empty(self):
        br = BenchmarkResult.from_task_results({})
        assert br.pass_rate == 0.0
        assert br.mean_reward == 0.0
        assert br.total_cost_usd == 0.0
        assert br.task_results == {}

    def test_from_mixed_results(self):
        results = {
            "t1": TaskResult(task_id="t1", reward=1.0, passed=True, cost_usd=0.01),
            "t2": TaskResult(task_id="t2", reward=0.0, passed=False, cost_usd=0.02),
            "t3": TaskResult(task_id="t3", reward=0.5, passed=True, cost_usd=0.03),
        }
        br = BenchmarkResult.from_task_results(results, split="val", duration_ms=500.0)
        assert len(br.task_results) == 3
        assert br.split == "val"
        assert br.duration_ms == 500.0
        assert br.pass_rate == 2 / 3
        assert abs(br.mean_reward - 0.5) < 1e-9
        assert abs(br.total_cost_usd - 0.06) < 1e-9

    def test_from_all_passing(self):
        results = {
            "t1": TaskResult(task_id="t1", reward=1.0, passed=True),
            "t2": TaskResult(task_id="t2", reward=0.9, passed=True),
        }
        br = BenchmarkResult.from_task_results(results)
        assert br.pass_rate == 1.0
        assert abs(br.mean_reward - 0.95) < 1e-9


# ---------------------------------------------------------------------------
# _values_match
# ---------------------------------------------------------------------------


class TestValuesMatch:
    def test_numeric_exact(self):
        assert _values_match(42, 42) is True

    def test_numeric_tolerance(self):
        assert _values_match(1.0, 1.005) is True
        assert _values_match(1.0, 1.02) is False

    def test_string_case_insensitive(self):
        assert _values_match("Hello", "hello") is True
        assert _values_match("  ABC  ", "abc") is True
        assert _values_match("abc", "xyz") is False

    def test_list_match(self):
        assert _values_match([1, 2, 3], [1, 2, 3]) is True
        assert _values_match([1, 2], [1, 2, 3]) is False
        assert _values_match(["a", "B"], ["A", "b"]) is True

    def test_dict_match(self):
        assert _values_match({"a": 1}, {"a": 1}) is True
        assert _values_match({"a": 1}, {"a": 2}) is False
        assert _values_match({"a": 1}, {"b": 1}) is False

    def test_mixed_types(self):
        assert _values_match("hello", 42) is False

    def test_nested(self):
        expected = {"items": [1, 2], "name": "Test"}
        actual = {"items": [1, 2], "name": "test"}
        assert _values_match(expected, actual) is True


# ---------------------------------------------------------------------------
# _score_json_match
# ---------------------------------------------------------------------------


class TestScoreJsonMatch:
    def test_full_match(self):
        expected = {"a": 1, "b": "hello"}
        actual = {"a": 1, "b": "HELLO"}
        score, detail = _score_json_match(expected, actual)
        assert score == 1.0

    def test_partial_match(self):
        expected = {"a": 1, "b": "hello"}
        actual = {"a": 1, "b": 999}
        score, detail = _score_json_match(expected, actual)
        # 1 exact + 1 partial*0.3 out of 2
        assert abs(score - (1 + 0.3) / 2) < 1e-9
        assert "WRONG" in detail

    def test_missing_fields(self):
        expected = {"a": 1, "b": 2}
        actual = {"a": 1}
        score, detail = _score_json_match(expected, actual)
        assert score == 0.5
        assert "MISSING" in detail

    def test_empty_expected(self):
        score, detail = _score_json_match({}, {"a": 1})
        assert score == 1.0


# ---------------------------------------------------------------------------
# LocalBenchmarkRunner.list_task_ids
# ---------------------------------------------------------------------------


class TestLocalBenchmarkRunnerListTaskIds:
    def test_returns_correct_ids(self):
        tasks = [
            TaskSpec(task_id="alpha", instruction="do alpha"),
            TaskSpec(task_id="beta", instruction="do beta"),
            TaskSpec(task_id="gamma", instruction="do gamma"),
        ]
        runner = LocalBenchmarkRunner(tasks)
        assert runner.list_task_ids() == ["alpha", "beta", "gamma"]

    def test_empty_tasks(self):
        runner = LocalBenchmarkRunner([])
        assert runner.list_task_ids() == []
