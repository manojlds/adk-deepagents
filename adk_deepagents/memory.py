"""Memory module — AGENTS.md loading and formatting.

Ported from deepagents.middleware.memory.
Loads AGENTS.md files from the backend and formats them for system prompt injection.
"""

from __future__ import annotations

from adk_deepagents.backends.protocol import Backend
from adk_deepagents.prompts import MEMORY_SYSTEM_PROMPT


def load_memory(backend: Backend, sources: list[str]) -> dict[str, str]:
    """Load memory files from the backend.

    Parameters
    ----------
    backend:
        The backend to read files from.
    sources:
        List of file paths to load (e.g., ``["./AGENTS.md"]``).

    Returns
    -------
    dict[str, str]
        Mapping of path → content for successfully loaded files.
    """
    contents: dict[str, str] = {}
    results = backend.download_files(sources)
    for resp in results:
        if resp.content is not None:
            try:
                contents[resp.path] = resp.content.decode("utf-8")
            except (UnicodeDecodeError, AttributeError):
                if isinstance(resp.content, str):
                    contents[resp.path] = resp.content
    return contents


def format_memory(contents: dict[str, str], sources: list[str]) -> str:
    """Format loaded memory for system prompt injection.

    Parameters
    ----------
    contents:
        Mapping of path → content from ``load_memory``.
    sources:
        Original source paths (for ordering).

    Returns
    -------
    str
        Formatted memory prompt ready for injection.
    """
    if not contents:
        return MEMORY_SYSTEM_PROMPT.format(agent_memory="(No memory loaded)")

    sections: list[str] = []
    for path in sources:
        content = contents.get(path)
        if content:
            sections.append(f"### {path}\n{content}")

    agent_memory = "\n\n".join(sections) if sections else "(No memory loaded)"
    return MEMORY_SYSTEM_PROMPT.format(agent_memory=agent_memory)
