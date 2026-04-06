"""Shared fixtures and helpers for integration tests."""

from __future__ import annotations

import os
import uuid
from typing import Any

import litellm
from dotenv import load_dotenv
from google.genai import types

from adk_deepagents import to_a2a_app
from adk_deepagents.backends.state import StateBackend

# Load local .env for developer-friendly LLM test runs.
load_dotenv()

# Use httpx transport to avoid aiohttp cleanup warnings on Python >=3.12.7.
litellm.disable_aiohttp_transport = True

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_litellm_model():
    """Create a LiteLlm model using available environment variables.

    Reads model from ADK_DEEPAGENTS_MODEL first, then LITELLM_MODEL for
    backward compatibility. API credentials come from OPENAI_API_KEY (or
    OPENCODE_API_KEY) and OPENAI_API_BASE.
    """
    from google.adk.models.lite_llm import LiteLlm

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENCODE_API_KEY", "")
    api_base = os.environ.get("OPENAI_API_BASE", "https://opencode.ai/zen/v1")
    model = os.environ.get("ADK_DEEPAGENTS_MODEL") or os.environ.get(
        "LITELLM_MODEL", "openai/gpt-4o-mini"
    )

    return LiteLlm(
        model=model,
        api_key=api_key,
        api_base=api_base,
    )


def backend_factory(state: dict[str, Any]) -> StateBackend:
    """Default backend factory for integration tests."""
    return StateBackend(state)


class _A2AIntegrationSession:
    """In-process session shim for A2A-backed integration tests."""

    def __init__(self, session_id: str) -> None:
        self.id = session_id


class _A2AIntegrationSessionService:
    """Session service shim that maps helper user IDs to A2A session user IDs."""

    def __init__(self, *, inner: Any, app_name: str, runner_user_id: str) -> None:
        self._inner = inner
        self._app_name = app_name
        self._runner_user_id = runner_user_id

    def _mapped_user_id(self, user_id: str) -> str:
        if user_id == "test_user":
            return self._runner_user_id
        return user_id

    async def get_session(self, *, app_name: str, user_id: str, session_id: str) -> Any:
        return await self._inner.get_session(
            app_name=app_name or self._app_name,
            user_id=self._mapped_user_id(user_id),
            session_id=session_id,
        )

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Any:
        return await self._inner.create_session(
            app_name=app_name or self._app_name,
            user_id=self._mapped_user_id(user_id),
            state=state,
            session_id=session_id,
        )


class _A2AIntegrationRunner:
    """A lightweight runner-like wrapper backed by an in-process A2A app."""

    def __init__(
        self,
        *,
        app: Any,
        user_id: str,
        context_id: str,
        adk_runner: Any,
        runner_user_id: str,
    ) -> None:
        self._app = app
        self._user_id = user_id
        self._context_id = context_id
        self._adk_runner = adk_runner
        self._runner_user_id = runner_user_id
        self._started = False
        self._messages: list[types.Content] = []
        self._state: dict[str, Any] = {}
        self._session_service = _A2AIntegrationSessionService(
            inner=adk_runner.session_service,
            app_name=adk_runner.app_name,
            runner_user_id=runner_user_id,
        )

    @property
    def context_id(self) -> str:
        return self._context_id

    @property
    def messages(self) -> list[types.Content]:
        return list(self._messages)

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._state)

    @property
    def session_service(self) -> Any:
        return self._session_service

    @staticmethod
    def _append_text_if_new(texts: list[str], value: Any) -> None:
        if not isinstance(value, str) or not value:
            return
        if not texts or texts[-1] != value:
            texts.append(value)

    def _consume_part(
        self,
        part: Any,
        *,
        texts: list[str],
        function_calls: list[str],
        function_responses: list[str],
        task_payloads: list[dict[str, Any]],
    ) -> None:
        root = part.get("root") if isinstance(part, dict) else getattr(part, "root", None)
        candidate = root if root is not None else part

        text = (
            candidate.get("text")
            if isinstance(candidate, dict)
            else getattr(candidate, "text", None)
        )
        self._append_text_if_new(texts, text)

        data = (
            candidate.get("data")
            if isinstance(candidate, dict)
            else getattr(candidate, "data", None)
        )
        if not isinstance(data, dict):
            return

        metadata = (
            candidate.get("metadata")
            if isinstance(candidate, dict)
            else getattr(candidate, "metadata", None)
        )
        adk_type = metadata.get("adk_type") if isinstance(metadata, dict) else None

        name = data.get("name")
        if not isinstance(name, str) or not name:
            return

        is_call = adk_type == "function_call" or "args" in data
        is_response = adk_type == "function_response" or "response" in data

        if is_call:
            function_calls.append(name)

        if is_response:
            function_responses.append(name)
            response_payload = data.get("response")
            if name == "task" and isinstance(response_payload, dict):
                task_payloads.append(response_payload)

    def _consume_message(
        self,
        message: Any,
        *,
        texts: list[str],
        function_calls: list[str],
        function_responses: list[str],
        task_payloads: list[dict[str, Any]],
    ) -> None:
        if message is None:
            return

        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if isinstance(role, str) and role.lower() == "user":
            return

        parts = (
            message.get("parts") if isinstance(message, dict) else getattr(message, "parts", None)
        )
        if isinstance(parts, list):
            for part in parts:
                self._consume_part(
                    part,
                    texts=texts,
                    function_calls=function_calls,
                    function_responses=function_responses,
                    task_payloads=task_payloads,
                )

    def _consume_task(
        self,
        task_obj: Any,
        *,
        texts: list[str],
        function_calls: list[str],
        function_responses: list[str],
        task_payloads: list[dict[str, Any]],
    ) -> None:
        if task_obj is None:
            return

        history = (
            task_obj.get("history")
            if isinstance(task_obj, dict)
            else getattr(task_obj, "history", None)
        )
        if isinstance(history, list):
            for message in history:
                self._consume_message(
                    message,
                    texts=texts,
                    function_calls=function_calls,
                    function_responses=function_responses,
                    task_payloads=task_payloads,
                )

        artifacts = (
            task_obj.get("artifacts")
            if isinstance(task_obj, dict)
            else getattr(task_obj, "artifacts", None)
        )
        if isinstance(artifacts, list):
            for artifact in artifacts:
                artifact_parts = (
                    artifact.get("parts")
                    if isinstance(artifact, dict)
                    else getattr(artifact, "parts", None)
                )
                if not isinstance(artifact_parts, list):
                    continue
                for part in artifact_parts:
                    self._consume_part(
                        part,
                        texts=texts,
                        function_calls=function_calls,
                        function_responses=function_responses,
                        task_payloads=task_payloads,
                    )

        status = (
            task_obj.get("status")
            if isinstance(task_obj, dict)
            else getattr(task_obj, "status", None)
        )
        status_message = (
            status.get("message") if isinstance(status, dict) else getattr(status, "message", None)
        )
        self._consume_message(
            status_message,
            texts=texts,
            function_calls=function_calls,
            function_responses=function_responses,
            task_payloads=task_payloads,
        )

    async def run_turn(
        self,
        prompt: str,
    ) -> tuple[list[str], list[str], list[str], list[dict[str, Any]]]:
        import httpx
        from a2a.client.client import ClientConfig
        from a2a.client.client_factory import ClientFactory
        from a2a.client.middleware import ClientCallContext, ClientCallInterceptor
        from a2a.types import Message, Part, Role, TextPart
        from google.adk.a2a.converters.utils import _get_adk_metadata_key

        request_message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=prompt))],
            message_id=str(uuid.uuid4()),
            context_id=self._context_id,
        )

        if not self._started:
            await self._app.router.startup()
            self._started = True

        class _A2AStateDeltaInterceptor(ClientCallInterceptor):
            async def intercept(
                self,
                method_name: str,
                request_payload: dict[str, Any],
                http_kwargs: dict[str, Any],
                agent_card: Any,
                context: ClientCallContext | None,
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                del method_name, agent_card

                if context is None:
                    return request_payload, http_kwargs

                state_delta = context.state.get("state_delta")
                if not isinstance(state_delta, dict) or not state_delta:
                    return request_payload, http_kwargs

                params = request_payload.get("params")
                if not isinstance(params, dict):
                    return request_payload, http_kwargs

                metadata = params.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                    params["metadata"] = metadata

                metadata[_get_adk_metadata_key("state_delta")] = state_delta
                return request_payload, http_kwargs

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self._app),
            base_url="http://a2a.local",
            timeout=120.0,
        ) as http_client:
            client = await ClientFactory.connect(
                "http://a2a.local",
                client_config=ClientConfig(streaming=True, httpx_client=http_client),
                interceptors=[_A2AStateDeltaInterceptor()],
            )

            texts: list[str] = []
            function_calls: list[str] = []
            function_responses: list[str] = []
            task_payloads: list[dict[str, Any]] = []

            async for event in client.send_message(request_message):
                if isinstance(event, tuple):
                    task_obj, _ = event
                    self._consume_task(
                        task_obj,
                        texts=texts,
                        function_calls=function_calls,
                        function_responses=function_responses,
                        task_payloads=task_payloads,
                    )
                    continue

                self._consume_message(
                    event,
                    texts=texts,
                    function_calls=function_calls,
                    function_responses=function_responses,
                    task_payloads=task_payloads,
                )

            close = getattr(client, "close", None)
            if callable(close):
                await close()

        updated_session = await self._adk_runner.session_service.get_session(
            app_name=self._adk_runner.app_name,
            user_id=self._runner_user_id,
            session_id=self._context_id,
        )
        if updated_session is not None and isinstance(updated_session.state, dict):
            self._state = dict(updated_session.state)

        self._messages.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
        if texts:
            self._messages.append(
                types.Content(role="model", parts=[types.Part(text="\n".join(texts))])
            )
        return texts, function_calls, function_responses, task_payloads


def _llm_a2a_mode_enabled() -> bool:
    mode = os.environ.get("ADK_DEEPAGENTS_LLM_TRANSPORT", "inmemory").strip().lower()
    return mode == "a2a"


def _initial_state_for_run(state: dict[str, Any] | None = None) -> dict[str, Any]:
    initial_state: dict[str, Any] = {
        "files": {},
        "_backend_factory": backend_factory,
    }
    if state:
        initial_state.update(state)
    return initial_state


async def _build_a2a_runner(
    *,
    agent: Any,
    initial_state: dict[str, Any],
) -> tuple[_A2AIntegrationRunner, _A2AIntegrationSession]:
    from google.adk.runners import InMemoryRunner

    runner_impl = InMemoryRunner(agent=agent, app_name="integration_test")
    session_impl = await runner_impl.session_service.create_session(
        app_name="integration_test",
        user_id="test_user",
        state=initial_state,
    )
    runner_user_id = f"A2A_USER_{session_impl.id}"

    a2a_app = to_a2a_app(
        agent,
        host="a2a.local",
        port=80,
        protocol="http",
        runner=runner_impl,
    )

    runner = _A2AIntegrationRunner(
        app=a2a_app,
        user_id="test_user",
        context_id=session_impl.id,
        adk_runner=runner_impl,
        runner_user_id=runner_user_id,
    )
    return runner, _A2AIntegrationSession(session_impl.id)


async def run_agent(agent, prompt: str, *, state: dict[str, Any] | None = None):
    """Run *agent* with a single user prompt and return (texts, runner, session).

    Returns all text responses, the runner instance, and the session object
    so callers can send follow-up messages on the same session.
    """
    if _llm_a2a_mode_enabled():
        initial_state = _initial_state_for_run(state)
        runner, session = await _build_a2a_runner(agent=agent, initial_state=initial_state)
        texts, _calls, _responses, _task_payloads = await runner.run_turn(prompt)
        return texts, runner, session

    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=agent, app_name="integration_test")
    initial_state = _initial_state_for_run(state)

    session = await runner.session_service.create_session(
        app_name="integration_test",
        user_id="test_user",
        state=initial_state,
    )

    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    texts: list[str] = []

    async for event in runner.run_async(
        session_id=session.id,
        user_id="test_user",
        new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    texts.append(part.text)

    return texts, runner, session


async def run_agent_with_events(
    agent,
    prompt: str,
    *,
    state: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], list[str], Any, Any]:
    """Run *agent* and return text output plus tool call/response names.

    Returns ``(texts, function_calls, function_responses, runner, session)``.
    """
    if _llm_a2a_mode_enabled():
        initial_state = _initial_state_for_run(state)
        runner, session = await _build_a2a_runner(agent=agent, initial_state=initial_state)
        texts, function_calls, function_responses, _task_payloads = await runner.run_turn(prompt)
        return texts, function_calls, function_responses, runner, session

    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=agent, app_name="integration_test")
    initial_state = _initial_state_for_run(state)

    session = await runner.session_service.create_session(
        app_name="integration_test",
        user_id="test_user",
        state=initial_state,
    )

    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    texts: list[str] = []
    function_calls: list[str] = []
    function_responses: list[str] = []

    async for event in runner.run_async(
        session_id=session.id,
        user_id="test_user",
        new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    texts.append(part.text)
                if hasattr(part, "function_call") and part.function_call:
                    name = part.function_call.name
                    if isinstance(name, str) and name:
                        function_calls.append(name)
                if hasattr(part, "function_response") and part.function_response:
                    name = part.function_response.name
                    if isinstance(name, str) and name:
                        function_responses.append(name)

    return texts, function_calls, function_responses, runner, session


async def run_agent_with_task_payloads(
    agent,
    prompt: str,
    *,
    state: dict[str, Any] | None = None,
) -> tuple[list[str], list[dict[str, Any]], Any, Any]:
    """Run *agent* and collect dynamic ``task`` tool response payloads.

    Returns ``(texts, task_payloads, runner, session)``.
    """
    if _llm_a2a_mode_enabled():
        initial_state = _initial_state_for_run(state)
        runner, session = await _build_a2a_runner(agent=agent, initial_state=initial_state)
        texts, _function_calls, _function_responses, task_payloads = await runner.run_turn(prompt)
        return texts, task_payloads, runner, session

    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=agent, app_name="integration_test")
    initial_state = _initial_state_for_run(state)

    session = await runner.session_service.create_session(
        app_name="integration_test",
        user_id="test_user",
        state=initial_state,
    )

    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    texts: list[str] = []
    task_payloads: list[dict[str, Any]] = []

    async for event in runner.run_async(
        session_id=session.id,
        user_id="test_user",
        new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    texts.append(part.text)

                function_response = getattr(part, "function_response", None)
                if function_response is None:
                    continue

                if getattr(function_response, "name", None) != "task":
                    continue

                payload = getattr(function_response, "response", None)
                if isinstance(payload, dict):
                    task_payloads.append(payload)

    return texts, task_payloads, runner, session


async def get_file_content(runner, session) -> dict[str, str]:
    """Return a dict of {path: content_str} from the session's file state."""
    if _llm_a2a_mode_enabled() and isinstance(runner, _A2AIntegrationRunner):
        files = runner.state.get("files", {})
        result: dict[str, str] = {}
        for path, file_data in files.items():
            if isinstance(file_data, dict) and "content" in file_data:
                content_value = file_data["content"]
                if isinstance(content_value, list):
                    result[path] = "\n".join(str(line) for line in content_value)
                elif isinstance(content_value, str):
                    result[path] = content_value
            elif isinstance(file_data, str):
                result[path] = file_data
        return result

    updated = await runner.session_service.get_session(
        app_name="integration_test",
        user_id="test_user",
        session_id=session.id,
    )
    files = updated.state.get("files", {})
    result: dict[str, str] = {}
    for path, file_data in files.items():
        if isinstance(file_data, dict) and "content" in file_data:
            result[path] = "\n".join(file_data["content"])
        elif isinstance(file_data, str):
            result[path] = file_data
    return result


async def send_followup(runner, session, prompt: str) -> list[str]:
    """Send a follow-up message on an existing session and return text responses."""
    if _llm_a2a_mode_enabled() and isinstance(runner, _A2AIntegrationRunner):
        texts, _calls, _responses, _task_payloads = await runner.run_turn(prompt)
        return texts

    from google.genai import types

    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    texts: list[str] = []

    async for event in runner.run_async(
        session_id=session.id,
        user_id="test_user",
        new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    texts.append(part.text)

    return texts


async def send_followup_with_events(
    runner,
    session,
    prompt: str,
) -> tuple[list[str], list[str], list[str]]:
    """Send a follow-up and return text output plus tool call/response names."""
    if _llm_a2a_mode_enabled() and isinstance(runner, _A2AIntegrationRunner):
        texts, function_calls, function_responses, _task_payloads = await runner.run_turn(prompt)
        return texts, function_calls, function_responses

    from google.genai import types

    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    texts: list[str] = []
    function_calls: list[str] = []
    function_responses: list[str] = []

    async for event in runner.run_async(
        session_id=session.id,
        user_id="test_user",
        new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    texts.append(part.text)
                if hasattr(part, "function_call") and part.function_call:
                    name = part.function_call.name
                    if isinstance(name, str) and name:
                        function_calls.append(name)
                if hasattr(part, "function_response") and part.function_response:
                    name = part.function_response.name
                    if isinstance(name, str) and name:
                        function_responses.append(name)

    return texts, function_calls, function_responses
