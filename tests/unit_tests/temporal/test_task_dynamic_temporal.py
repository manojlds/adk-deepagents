"""Tests for Temporal path in dynamic task tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from adk_deepagents.tools.task_dynamic import _run_dynamic_task_temporal
from adk_deepagents.types import DynamicTaskConfig, TemporalTaskConfig

pytest.importorskip("temporalio")


async def test_run_dynamic_task_temporal_dispatches_to_client():
    expected = {
        "result": "temporal result",
        "function_calls": ["read_file"],
        "files": {},
        "todos": [],
        "timed_out": False,
        "error": None,
    }

    with patch(
        "adk_deepagents.temporal.client.run_task_via_temporal",
        new_callable=AsyncMock,
        return_value=expected,
    ) as dispatch_mock:
        result = await _run_dynamic_task_temporal(
            snapshot_data={
                "subagent_type": "general_purpose",
                "prompt": "do work",
                "depth": 1,
                "files": {},
                "todos": [],
                "history": [],
                "model_override": None,
                "subagent_spec": {
                    "name": "runtime_specialist",
                    "description": "Runtime specialist",
                },
                "timeout_seconds": 30.0,
            },
            logical_parent_id="parent",
            task_id="task_1",
            task_config=DynamicTaskConfig(temporal=TemporalTaskConfig()),
        )

    assert result == expected
    dispatch_mock.assert_awaited_once()

    call_kwargs = dispatch_mock.call_args.kwargs
    assert call_kwargs["snapshot"].subagent_type == "general_purpose"
    assert call_kwargs["snapshot"].prompt == "do work"
    assert call_kwargs["snapshot"].subagent_spec == {
        "name": "runtime_specialist",
        "description": "Runtime specialist",
    }
    assert call_kwargs["logical_parent_id"] == "parent"
    assert call_kwargs["task_id"] == "task_1"
