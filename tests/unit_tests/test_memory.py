"""Tests for memory module."""

from adk_deepagents.backends.state import StateBackend
from adk_deepagents.backends.utils import create_file_data
from adk_deepagents.memory import format_memory, load_memory


def test_load_memory_from_state_backend():
    state = {
        "files": {
            "/AGENTS.md": create_file_data("# Memory\nI am a coding agent."),
        }
    }
    backend = StateBackend(state)
    contents = load_memory(backend, ["/AGENTS.md"])
    assert "/AGENTS.md" in contents
    assert "coding agent" in contents["/AGENTS.md"]


def test_load_memory_missing_file():
    state = {"files": {}}
    backend = StateBackend(state)
    contents = load_memory(backend, ["/AGENTS.md"])
    assert contents == {}


def test_format_memory_with_content():
    contents = {"/AGENTS.md": "I am helpful."}
    result = format_memory(contents, ["/AGENTS.md"])
    assert "I am helpful." in result
    assert "agent_memory" in result


def test_format_memory_empty():
    result = format_memory({}, ["/AGENTS.md"])
    assert "No memory loaded" in result
