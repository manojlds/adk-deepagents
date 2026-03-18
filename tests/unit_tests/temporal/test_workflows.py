"""Tests for Temporal workflow configuration hooks."""

from __future__ import annotations

import pytest

pytest.importorskip("temporalio")


def test_workflow_class_name():
    from adk_deepagents.temporal.workflows import DynamicTaskWorkflow

    assert DynamicTaskWorkflow.__name__ == "DynamicTaskWorkflow"


def test_configure_workflow_updates_module_state():
    from adk_deepagents.temporal import workflows
    from adk_deepagents.temporal.workflows import configure_workflow

    configure_workflow(
        activity_timeout_seconds=250.0, retry_max_attempts=3, idle_timeout_seconds=300.0
    )
    assert workflows._activity_timeout_seconds == 250.0
    assert workflows._retry_max_attempts == 3
    assert workflows._idle_timeout_seconds == 300.0

    configure_workflow()
    assert workflows._activity_timeout_seconds == 120.0
    assert workflows._retry_max_attempts == 1
    assert workflows._idle_timeout_seconds == 600.0
