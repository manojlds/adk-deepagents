"""Deterministic JSON field verifier for data extraction tasks.

Scores agent output against expected JSON with field-level matching:
- Each field present and correct: full credit
- Each field present but wrong: partial credit (0.3)
- Each field missing: no credit
- Numeric tolerance: 0.01
- String comparison: case-insensitive, whitespace-normalized
- Array comparison: element-by-element with order tolerance
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def verify_json_output(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> tuple[float, str]:
    """Compare expected and actual JSON, return (score, details).

    Score is 0.0-1.0 based on field-level matching.
    """
    if not expected:
        return 1.0 if not actual else 0.0, "Empty expected"

    total = 0
    matched = 0
    partial = 0
    details: list[str] = []

    for key, exp_val in expected.items():
        total += 1
        act_val = actual.get(key)

        if act_val is None:
            details.append(f"MISSING: {key}")
            continue

        match_result = _deep_match(exp_val, act_val, path=key)
        if match_result == 1.0:
            matched += 1
        elif match_result > 0:
            partial += 1
            details.append(f"PARTIAL ({match_result:.0%}): {key}")
        else:
            details.append(f"WRONG: {key} (expected {_preview(exp_val)}, got {_preview(act_val)})")

    # Check for unexpected extra keys
    extra = set(actual.keys()) - set(expected.keys())
    if extra:
        details.append(f"EXTRA keys: {', '.join(sorted(extra))}")

    if total == 0:
        return 1.0, "No fields to check"

    score = (matched + partial * 0.3) / total
    summary = f"{matched}/{total} exact, {partial} partial, {total - matched - partial} missing"
    if details:
        summary += "\n" + "\n".join(details)
    return round(score, 4), summary


def _deep_match(expected: Any, actual: Any, path: str = "") -> float:
    """Recursively match two values, returning a score 0.0-1.0."""
    if expected is None:
        return 1.0 if actual is None else 0.0

    if actual is None:
        return 0.0

    # Numeric
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return 1.0 if abs(expected - actual) < 0.01 else 0.0

    # String
    if isinstance(expected, str) and isinstance(actual, str):
        return 1.0 if _normalize_str(expected) == _normalize_str(actual) else 0.0

    # List
    if isinstance(expected, list) and isinstance(actual, list):
        return _match_lists(expected, actual, path)

    # Dict
    if isinstance(expected, dict) and isinstance(actual, dict):
        return _match_dicts(expected, actual, path)

    # Bool
    if isinstance(expected, bool) and isinstance(actual, bool):
        return 1.0 if expected == actual else 0.0

    return 1.0 if expected == actual else 0.0


def _match_lists(expected: list, actual: list, path: str) -> float:
    """Match two lists element-by-element. Tries both ordered and best-match."""
    if not expected:
        return 1.0 if not actual else 0.5

    if not actual:
        return 0.0

    # Try ordered match first
    ordered_score = _ordered_list_match(expected, actual, path)

    # If ordered match is good enough, use it
    if ordered_score >= 0.8:
        return ordered_score

    # Try best-match (unordered) for lists of dicts
    if expected and isinstance(expected[0], dict) and actual and isinstance(actual[0], dict):
        unordered_score = _unordered_list_match(expected, actual, path)
        return max(ordered_score, unordered_score)

    return ordered_score


def _ordered_list_match(expected: list, actual: list, path: str) -> float:
    """Score ordered element-by-element list match."""
    max_len = max(len(expected), len(actual))
    if max_len == 0:
        return 1.0

    total_score = 0.0
    for i in range(max_len):
        if i < len(expected) and i < len(actual):
            total_score += _deep_match(expected[i], actual[i], f"{path}[{i}]")
    return total_score / max_len


def _unordered_list_match(expected: list, actual: list, path: str) -> float:
    """Best-match scoring for unordered lists of dicts."""
    if not expected:
        return 1.0 if not actual else 0.5

    used: set[int] = set()
    total_score = 0.0

    for exp_item in expected:
        best_score = 0.0
        best_idx = -1
        for j, act_item in enumerate(actual):
            if j in used:
                continue
            s = _deep_match(exp_item, act_item, path)
            if s > best_score:
                best_score = s
                best_idx = j
        if best_idx >= 0:
            used.add(best_idx)
        total_score += best_score

    max_len = max(len(expected), len(actual))
    return total_score / max_len


def _match_dicts(expected: dict, actual: dict, path: str) -> float:
    """Score field-level dict match."""
    if not expected:
        return 1.0 if not actual else 0.5

    total = len(expected)
    score = 0.0

    for key, exp_val in expected.items():
        act_val = actual.get(key)
        if act_val is None and exp_val is not None:
            continue
        score += _deep_match(exp_val, act_val, f"{path}.{key}")

    return score / total if total > 0 else 1.0


def _normalize_str(s: str) -> str:
    """Normalize a string for comparison."""
    return " ".join(s.strip().lower().split())


def _preview(value: Any, max_len: int = 60) -> str:
    """Short preview of a value."""
    s = str(value)
    return s[:max_len] + "..." if len(s) > max_len else s


def verify_from_files(
    expected_path: str | Path,
    actual_path: str | Path,
) -> tuple[float, str]:
    """Verify by comparing two JSON files."""
    try:
        expected = json.loads(Path(expected_path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return 0.0, f"Failed to read expected: {exc}"

    try:
        actual = json.loads(Path(actual_path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return 0.0, f"Failed to read actual: {exc}"

    return verify_json_output(expected, actual)
