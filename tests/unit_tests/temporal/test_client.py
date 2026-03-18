"""Tests for Temporal client dispatch logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adk_deepagents.temporal.activities import TaskResult, TaskSnapshot
from adk_deepagents.temporal.client import (
    _SPEC_HASH_CACHE,
    _remember_spec_hash,
    _subagent_spec_hash,
    _workflow_id,
    run_task_via_temporal,
)
from adk_deepagents.types import DynamicTaskConfig, TemporalTaskConfig

pytest.importorskip("temporalio")


@pytest.fixture(autouse=True)
def _clear_spec_hash_cache() -> None:
    _SPEC_HASH_CACHE.clear()


def test_workflow_id_uses_prefix_parent_and_task():
    config = TemporalTaskConfig(workflow_id_prefix="prefix")
    assert _workflow_id(config, "parent", "task_1") == "prefix:parent:task_1"


def test_subagent_spec_hash_is_stable_across_key_order():
    spec_a = {
        "name": "runtime_specialist",
        "description": "Runtime specialist",
        "tool_names": ["read_file", "write_file"],
    }
    spec_b = {
        "tool_names": ["read_file", "write_file"],
        "description": "Runtime specialist",
        "name": "runtime_specialist",
    }

    assert _subagent_spec_hash(spec_a) == _subagent_spec_hash(spec_b)


def test_spec_hash_cache_uses_lru_style_eviction():
    with patch("adk_deepagents.temporal.client._SPEC_HASH_CACHE_MAX_ENTRIES", 2):
        _remember_spec_hash("wf-1", "a")
        _remember_spec_hash("wf-2", "b")
        _remember_spec_hash("wf-1", "a2")  # refresh recency for wf-1
        _remember_spec_hash("wf-3", "c")

    assert list(_SPEC_HASH_CACHE.keys()) == ["wf-1", "wf-3"]
    assert _SPEC_HASH_CACHE["wf-1"] == "a2"
    assert _SPEC_HASH_CACHE["wf-3"] == "c"


async def test_run_task_via_temporal_without_config_returns_error():
    result = await run_task_via_temporal(
        snapshot=TaskSnapshot(subagent_type="general_purpose", prompt="hi"),
        logical_parent_id="parent",
        task_id="task_1",
        task_config=DynamicTaskConfig(temporal=None),
    )
    assert result["error"] == "Temporal config is None"


async def test_run_task_via_temporal_handles_connection_failure():
    with patch(
        "adk_deepagents.temporal.client._get_or_create_client",
        new_callable=AsyncMock,
        side_effect=ConnectionError("refused"),
    ):
        result = await run_task_via_temporal(
            snapshot=TaskSnapshot(subagent_type="general_purpose", prompt="hi"),
            logical_parent_id="parent",
            task_id="task_1",
            task_config=DynamicTaskConfig(temporal=TemporalTaskConfig()),
        )

    assert "connection failed" in result["error"].lower()


async def test_run_task_via_temporal_starts_new_workflow_when_not_found():
    expected = TaskResult(result="done", function_calls=["write_file"]).to_dict()

    mock_handle = AsyncMock()
    mock_handle.execute_update = AsyncMock(return_value=expected)

    mock_client = AsyncMock()
    missing_exc = RuntimeError("missing")
    existing_handle = AsyncMock()
    existing_handle.describe = AsyncMock(side_effect=missing_exc)

    mock_client.get_workflow_handle = MagicMock(return_value=existing_handle)
    mock_client.start_workflow = AsyncMock(return_value=mock_handle)

    with (
        patch(
            "adk_deepagents.temporal.client._get_or_create_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ),
        patch("adk_deepagents.temporal.client._is_not_found_error", return_value=True),
    ):
        result = await run_task_via_temporal(
            snapshot=TaskSnapshot(
                subagent_type="general_purpose",
                prompt="work",
                subagent_spec={
                    "name": "runtime_specialist",
                    "description": "Runtime specialist",
                    "tool_names": ["read_file"],
                },
            ),
            logical_parent_id="parent",
            task_id="task_1",
            task_config=DynamicTaskConfig(temporal=TemporalTaskConfig()),
        )

    assert result["result"] == "done"
    mock_client.start_workflow.assert_awaited_once()
    mock_handle.execute_update.assert_awaited_once()
    update_args = mock_handle.execute_update.await_args.kwargs["arg"]
    assert update_args["subagent_spec"]["name"] == "runtime_specialist"


async def test_run_task_via_temporal_omits_subagent_spec_when_hash_unchanged():
    config = TemporalTaskConfig()
    expected = TaskResult(result="done").to_dict()

    running_desc = MagicMock()
    from temporalio.client import WorkflowExecutionStatus

    running_desc.status = WorkflowExecutionStatus.RUNNING

    existing_handle = AsyncMock()
    existing_handle.describe = AsyncMock(return_value=running_desc)
    existing_handle.execute_update = AsyncMock(return_value=expected)

    mock_client = AsyncMock()
    mock_client.get_workflow_handle = MagicMock(return_value=existing_handle)

    spec = {
        "name": "runtime_specialist",
        "description": "Runtime specialist",
        "tool_names": ["read_file"],
    }
    wf_id = _workflow_id(config, "parent", "task_1")
    _SPEC_HASH_CACHE[wf_id] = _subagent_spec_hash(spec)

    with patch(
        "adk_deepagents.temporal.client._get_or_create_client",
        new_callable=AsyncMock,
        return_value=mock_client,
    ):
        result = await run_task_via_temporal(
            snapshot=TaskSnapshot(
                subagent_type="general_purpose",
                prompt="work",
                subagent_spec=spec,
            ),
            logical_parent_id="parent",
            task_id="task_1",
            task_config=DynamicTaskConfig(temporal=config),
        )

    assert result["result"] == "done"
    update_args = existing_handle.execute_update.await_args.kwargs["arg"]
    assert "subagent_spec_hash" in update_args
    assert "subagent_spec" not in update_args


async def test_run_task_via_temporal_returns_error_on_update_failure():
    mock_handle = AsyncMock()
    mock_handle.execute_update = AsyncMock(side_effect=RuntimeError("update failed"))

    running_desc = MagicMock()
    from temporalio.client import WorkflowExecutionStatus

    running_desc.status = WorkflowExecutionStatus.RUNNING

    existing_handle = AsyncMock()
    existing_handle.describe = AsyncMock(return_value=running_desc)
    existing_handle.execute_update = mock_handle.execute_update

    mock_client = AsyncMock()
    mock_client.get_workflow_handle = MagicMock(return_value=existing_handle)

    with patch(
        "adk_deepagents.temporal.client._get_or_create_client",
        new_callable=AsyncMock,
        return_value=mock_client,
    ):
        result = await run_task_via_temporal(
            snapshot=TaskSnapshot(subagent_type="general_purpose", prompt="work"),
            logical_parent_id="parent",
            task_id="task_1",
            task_config=DynamicTaskConfig(temporal=TemporalTaskConfig()),
        )

    assert "execution failed" in result["error"].lower()
