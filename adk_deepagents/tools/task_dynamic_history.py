"""History normalization and prompt shaping for dynamic tasks."""

from __future__ import annotations

from typing import Any

from adk_deepagents.types import DynamicTaskConfig

_TASK_HISTORY_MAX_ENTRIES = 12
_TASK_HISTORY_MAX_PROMPT_CHARS = 1200
_TASK_HISTORY_MAX_RESULT_CHARS = 2400


def _truncate_history_text(value: Any, *, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _normalized_task_history(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue

        prompt = _truncate_history_text(
            item.get("prompt"), max_chars=_TASK_HISTORY_MAX_PROMPT_CHARS
        )
        result = _truncate_history_text(
            item.get("result"), max_chars=_TASK_HISTORY_MAX_RESULT_CHARS
        )
        if not prompt and not result:
            continue

        normalized.append({"prompt": prompt, "result": result})

    return normalized[-_TASK_HISTORY_MAX_ENTRIES:]


def _append_task_history_entry(*, task_state: dict[str, Any], prompt: str, result: str) -> None:
    history = _normalized_task_history(task_state.get("history"))
    history.append(
        {
            "prompt": _truncate_history_text(prompt, max_chars=_TASK_HISTORY_MAX_PROMPT_CHARS),
            "result": _truncate_history_text(result, max_chars=_TASK_HISTORY_MAX_RESULT_CHARS),
        }
    )
    task_state["history"] = history[-_TASK_HISTORY_MAX_ENTRIES:]


def _build_resume_prompt(*, history: list[dict[str, str]], prompt: str) -> str:
    if not history:
        return prompt

    lines = [
        "Continue this delegated task using the prior context below.",
        "",
        "Previous delegated turns:",
    ]

    for index, item in enumerate(history, start=1):
        previous_prompt = item.get("prompt", "")
        previous_result = item.get("result", "")
        if previous_prompt:
            lines.append(f"{index}. User instruction: {previous_prompt}")
        if previous_result:
            lines.append(f"{index}. Your previous response: {previous_result}")

    lines.extend(
        [
            "",
            "New instruction:",
            prompt,
        ]
    )
    return "\n".join(lines)


def _dynamic_task_tool_doc(config: DynamicTaskConfig) -> str:
    """Build dynamic task tool docs with live concurrency limits."""
    return (
        "Run a task in a dynamic sub-agent.\n\n"
        "Dynamic concurrency limits:\n"
        f"- max_parallel={config.max_parallel}\n"
        f"- concurrency_policy={config.concurrency_policy}\n"
        f"- queue_timeout_seconds={config.queue_timeout_seconds}\n\n"
        "When delegating many tasks:\n"
        f"- Launch in waves of <= {config.max_parallel} concurrent task calls\n"
        "- Wait for one wave to complete before starting the next\n"
        "- Use a stable task_id when you want continuity; "
        "first use creates it, later uses resume it"
    )
