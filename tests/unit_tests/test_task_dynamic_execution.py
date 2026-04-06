"""Unit tests for dynamic task execution helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from adk_deepagents.tools.task_dynamic_execution import (
    _new_structured_result_state,
    _parse_dynamic_task_result_payload,
    _run_dynamic_task_a2a,
)
from adk_deepagents.types import A2ATaskConfig, DynamicTaskConfig


class _FakePart:
    def __init__(self, *, text: str | None = None, data: object | None = None) -> None:
        self.text = text
        self.data = data


class _FakeArtifact:
    def __init__(self, parts: list[object] | None = None) -> None:
        self.parts = parts or []


class _FakeMessage:
    def __init__(self, parts: list[object] | None = None) -> None:
        self.parts = parts or []


class _FakeStatus:
    def __init__(self, message: object | None = None) -> None:
        self.message = message


class _FakeTask:
    def __init__(
        self,
        *,
        artifacts: list[object] | None = None,
        status: object | None = None,
    ) -> None:
        self.artifacts = artifacts or []
        self.status = status


class _FakeClient:
    def __init__(self, events: list[object]) -> None:
        self._events = events
        self.closed = False

    async def send_message(self, _request):
        for item in self._events:
            yield item

    async def close(self) -> None:
        self.closed = True


class _FakeClientFactory:
    events: list[object] = []
    last_client: _FakeClient | None = None
    last_connect_kwargs: dict[str, object] = {}

    @classmethod
    async def connect(cls, url: str, *, client_config):
        cls.last_connect_kwargs = {
            "url": url,
            "streaming": getattr(client_config, "streaming", None),
        }
        client = _FakeClient(cls.events)
        cls.last_client = client
        return client


class _FakeClientConfig:
    def __init__(self, *, streaming: bool) -> None:
        self.streaming = streaming


class _FakeRole:
    ROLE_USER = "ROLE_USER"


class _FakeMessageRequest:
    def __init__(
        self,
        *,
        role,
        parts,
        message_id: str,
        context_id: str,
    ) -> None:
        self.role = role
        self.parts = parts
        self.message_id = message_id
        self.context_id = context_id


def _install_fake_a2a_modules(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "a2a", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "a2a.client", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "a2a.client.client",
        SimpleNamespace(ClientConfig=_FakeClientConfig),
    )
    monkeypatch.setitem(
        sys.modules,
        "a2a.client.client_factory",
        SimpleNamespace(ClientFactory=_FakeClientFactory),
    )
    monkeypatch.setitem(sys.modules, "a2a.types", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "a2a.types.a2a_pb2",
        SimpleNamespace(
            Message=_FakeMessageRequest,
            Part=_FakePart,
            Role=_FakeRole,
        ),
    )


def test_parse_dynamic_task_result_payload_accepts_schema_envelope():
    payload = {
        "schema": "adk_deepagents.dynamic_task_result.v1",
        "payload": {
            "result": "done",
            "function_calls": ["read_file", "read_file", 123],
            "files": {"/notes.txt": {"content": "ok"}},
            "todos": [{"content": "ship", "status": "pending", "priority": "high"}],
            "error": "",
        },
    }

    parsed = _parse_dynamic_task_result_payload(payload)

    assert parsed == {
        "result": "done",
        "function_calls": ["read_file"],
        "files": {"/notes.txt": {"content": "ok"}},
        "todos": [{"content": "ship", "status": "pending", "priority": "high"}],
        "error": "",
    }


def test_parse_dynamic_task_result_payload_rejects_unrelated_json():
    assert _parse_dynamic_task_result_payload({"foo": "bar"}) is None
    assert _parse_dynamic_task_result_payload('{"foo": "bar"}') is None


@pytest.mark.asyncio
async def test_run_dynamic_task_a2a_prefers_structured_artifact_payload(monkeypatch):
    _install_fake_a2a_modules(monkeypatch)

    structured_payload = {
        "schema": "adk_deepagents.dynamic_task_result.v1",
        "payload": {
            "result": "Structured result",
            "function_calls": ["grep", "read_file", "grep"],
            "files": {"/report.md": {"content": "Done"}},
            "todos": [{"content": "Verify", "status": "completed", "priority": "medium"}],
            "error": "",
        },
    }
    task = _FakeTask(
        artifacts=[
            _FakeArtifact(parts=[_FakePart(data=structured_payload)]),
        ]
    )
    _FakeClientFactory.events = [(task, None)]

    result = await _run_dynamic_task_a2a(
        prompt="delegate",
        task_id="task_1",
        subagent_type="general_purpose",
        task_config=DynamicTaskConfig(a2a=A2ATaskConfig(agent_url="http://example.local")),
    )

    assert result == {
        "result": "Structured result",
        "function_calls": ["grep", "read_file"],
        "files": {"/report.md": {"content": "Done"}},
        "todos": [{"content": "Verify", "status": "completed", "priority": "medium"}],
        "timed_out": False,
        "error": None,
    }
    assert _FakeClientFactory.last_connect_kwargs == {
        "url": "http://example.local",
        "streaming": True,
    }
    assert _FakeClientFactory.last_client is not None
    assert _FakeClientFactory.last_client.closed is True


@pytest.mark.asyncio
async def test_run_dynamic_task_a2a_merges_function_calls_and_latest_state(monkeypatch):
    _install_fake_a2a_modules(monkeypatch)

    task_1 = _FakeTask(
        artifacts=[
            _FakeArtifact(
                parts=[
                    _FakePart(
                        data={
                            "schema": "adk_deepagents.dynamic_task_result.v1",
                            "payload": {
                                "function_calls": ["glob"],
                                "files": {"/tmp/a.txt": {"content": "A"}},
                                "todos": [
                                    {
                                        "content": "step 1",
                                        "status": "pending",
                                        "priority": "low",
                                    }
                                ],
                            },
                        }
                    )
                ]
            )
        ]
    )
    task_2 = _FakeTask(
        artifacts=[
            _FakeArtifact(
                parts=[
                    _FakePart(
                        data={
                            "schema": "adk_deepagents.dynamic_task_result.v1",
                            "payload": {
                                "result": "final",
                                "function_calls": ["read_file", "glob"],
                                "files": {"/tmp/b.txt": {"content": "B"}},
                                "todos": [
                                    {
                                        "content": "step 2",
                                        "status": "completed",
                                        "priority": "high",
                                    }
                                ],
                            },
                        }
                    )
                ]
            )
        ]
    )
    _FakeClientFactory.events = [(task_1, None), (task_2, None)]

    result = await _run_dynamic_task_a2a(
        prompt="delegate",
        task_id="task_2",
        subagent_type="general_purpose",
        task_config=DynamicTaskConfig(a2a=A2ATaskConfig()),
    )

    assert result["result"] == "final"
    assert result["function_calls"] == ["glob", "read_file"]
    assert result["files"] == {"/tmp/b.txt": {"content": "B"}}
    assert result["todos"] == [{"content": "step 2", "status": "completed", "priority": "high"}]
    assert result["error"] is None
    assert result["timed_out"] is False


@pytest.mark.asyncio
async def test_run_dynamic_task_a2a_falls_back_to_plain_text(monkeypatch):
    _install_fake_a2a_modules(monkeypatch)

    task = _FakeTask(
        artifacts=[
            _FakeArtifact(parts=[_FakePart(text="Hello")]),
            _FakeArtifact(parts=[_FakePart(text="Hello")]),
        ],
    )
    _FakeClientFactory.events = [(task, None), _FakeMessage(parts=[_FakePart(text="World")])]

    result = await _run_dynamic_task_a2a(
        prompt="delegate",
        task_id="task_3",
        subagent_type="general_purpose",
        task_config=DynamicTaskConfig(a2a=A2ATaskConfig()),
    )

    assert result == {
        "result": "Hello\nWorld",
        "function_calls": [],
        "files": {},
        "todos": [],
        "timed_out": False,
        "error": None,
    }


@pytest.mark.asyncio
async def test_run_dynamic_task_a2a_uses_structured_error(monkeypatch):
    _install_fake_a2a_modules(monkeypatch)

    task = _FakeTask(
        artifacts=[
            _FakeArtifact(
                parts=[
                    _FakePart(
                        data={
                            "schema": "adk_deepagents.dynamic_task_result.v1",
                            "payload": {
                                "result": "partial",
                                "error": "remote task failed",
                            },
                        }
                    )
                ]
            )
        ]
    )
    _FakeClientFactory.events = [(task, None)]

    result = await _run_dynamic_task_a2a(
        prompt="delegate",
        task_id="task_4",
        subagent_type="general_purpose",
        task_config=DynamicTaskConfig(a2a=A2ATaskConfig()),
    )

    assert result["result"] == "partial"
    assert result["error"] == "remote task failed"


def test_new_structured_result_state_defaults():
    state = _new_structured_result_state()
    assert state == {
        "has_result": False,
        "result": "",
        "has_function_calls": False,
        "function_calls": [],
        "has_files": False,
        "files": {},
        "has_todos": False,
        "todos": [],
        "has_error": False,
        "error": "",
    }
