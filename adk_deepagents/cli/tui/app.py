"""Main Textual application for the adk-deepagents TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header

from adk_deepagents.cli.tui.agent_service import AgentService, UiUpdate
from adk_deepagents.cli.tui.widgets import ApprovalBox, MessageDisplay, PromptInput
from adk_deepagents.types import DynamicTaskConfig


@dataclass
class TuiConfig:
    """All configuration needed to start the TUI."""

    agent_name: str
    user_id: str
    session_id: str
    db_path: Path
    model: str | None = None
    first_prompt: str | None = None
    auto_approve: bool = False
    dynamic_task_config: DynamicTaskConfig | None = None
    memory_sources: list[str] = field(default_factory=list)
    memory_source_paths: dict[str, Path] = field(default_factory=dict)
    skills_dirs: list[str] = field(default_factory=list)


class DeepAgentTui(App[None]):
    """Full-screen TUI for interacting with adk-deepagents."""

    TITLE = "adk-deepagents"
    SUB_TITLE = "agent"

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }

    #messages {
        height: 1fr;
    }

    #composer {
        height: auto;
        min-height: 3;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, config: TuiConfig) -> None:
        super().__init__()
        self._config = config
        self._service = AgentService(
            agent_name=config.agent_name,
            user_id=config.user_id,
            model=config.model,
            db_path=config.db_path,
            auto_approve=config.auto_approve,
            session_id=config.session_id,
            dynamic_task_config=config.dynamic_task_config,
            memory_sources=config.memory_sources,
            memory_source_paths=config.memory_source_paths,
            skills_dirs=config.skills_dirs,
        )
        self._waiting_for_approval = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield MessageDisplay(id="messages")
        yield PromptInput(id="composer")
        yield Footer()

    async def on_mount(self) -> None:
        self._service.initialize()
        self.run_worker(self._pump_updates(), exclusive=False)

        messages = self.query_one(MessageDisplay)
        messages.add_system_message(f"Thread {self._config.session_id} — type /help for commands")

        if self._config.first_prompt:
            await self._service.handle_input(self._config.first_prompt)

    async def _pump_updates(self) -> None:
        """Consume UI updates from the agent service forever."""
        while True:
            update = await self._service.updates.get()
            self._apply_update(update)

    def _apply_update(self, update: UiUpdate) -> None:
        messages = self.query_one(MessageDisplay)
        composer = self.query_one(PromptInput)

        if update.kind == "user_message":
            messages.end_assistant_message()
            messages.add_user_message(update.text or "")

        elif update.kind == "assistant_delta":
            messages.append_assistant_delta(update.text or "")

        elif update.kind == "thought_delta":
            messages.append_thought_delta(update.text or "")

        elif update.kind == "tool_call":
            messages.end_assistant_message()
            messages.add_tool_call(
                update.tool_name or "unknown_tool",
                detail=update.tool_detail,
            )

        elif update.kind == "tool_result":
            messages.end_assistant_message()
            messages.add_tool_result(
                update.tool_name or "unknown_tool",
                detail=update.tool_detail,
            )

        elif update.kind == "system":
            messages.end_assistant_message()
            messages.add_system_message(update.text or "")

        elif update.kind == "error":
            messages.end_assistant_message()
            messages.add_system_message(update.text or "", error=True)

        elif update.kind == "approval_request":
            messages.end_assistant_message()
            messages.add_approval_prompt(
                tool_name=update.approval_tool_name or "unknown_tool",
                hint=update.approval_hint,
                args_preview=update.approval_args_preview or "{}",
                request_id=update.request_id or "",
            )
            composer.disable_input()
            self._waiting_for_approval = True

        elif update.kind == "turn_started":
            composer.disable_input()

        elif update.kind == "turn_finished":
            messages.end_assistant_message()
            if not self._waiting_for_approval:
                composer.enable_input()

        elif update.kind == "clear_transcript":
            messages.clear_transcript()

        elif update.kind == "exit":
            self.exit()

    async def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        await self._service.handle_input(event.value)

    def on_approval_box_resolved(self, event: ApprovalBox.Resolved) -> None:
        self._service.resolve_approval(event.approved, event.always)
        self._waiting_for_approval = False
        self.query_one(PromptInput).enable_input()
