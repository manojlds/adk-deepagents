"""Filesystem tools for the deep agent.

Each tool resolves the backend from ``tool_context.state["_backend"]`` and
delegates to the corresponding backend method. State updates (for
``StateBackend``) are applied via ``tool_context.state``.

Ported from deepagents.middleware.filesystem tool definitions.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools import ToolContext

from adk_deepagents.backends.protocol import Backend, FileData
from adk_deepagents.backends.utils import (
    format_grep_matches,
    truncate_if_too_long,
    validate_path,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_backend(tool_context: ToolContext) -> Backend:
    """Resolve the backend from tool context state."""
    backend = tool_context.state.get("_backend")
    if backend is None:
        factory = tool_context.state.get("_backend_factory")
        if factory is not None:
            backend = factory(tool_context.state)
            tool_context.state["_backend"] = backend
    if backend is None:
        raise RuntimeError(
            "No backend configured. Set state['_backend'] or state['_backend_factory']."
        )
    return backend


def _apply_files_update(
    tool_context: ToolContext,
    files_update: dict[str, FileData] | None,
) -> None:
    """Merge files_update into session state (for StateBackend)."""
    if files_update is None:
        return
    files = tool_context.state.get("files", {})
    files.update(files_update)
    tool_context.state["files"] = files


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def ls(path: str, tool_context: ToolContext) -> dict:
    """List files and directories at the given path.

    Returns name, type (file/dir), size, and modification time for each entry.

    Args:
        path: Absolute path to list (must start with /).
    """
    try:
        validated = validate_path(path)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    backend = _get_backend(tool_context)
    entries = backend.ls_info(validated)
    return {"status": "success", "entries": entries}


def read_file(
    file_path: str,
    tool_context: ToolContext,
    offset: int = 0,
    limit: int = 100,
) -> dict:
    """Read a file from the filesystem with optional pagination.

    Returns content with line numbers. For large files, use offset and limit
    to paginate through the content.

    Args:
        file_path: Absolute path to the file (must start with /).
        offset: Line number to start reading from (0-based). Defaults to 0.
        limit: Maximum number of lines to return. Defaults to 100.
    """
    try:
        validated = validate_path(file_path)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    backend = _get_backend(tool_context)
    content = backend.read(validated, offset=offset, limit=limit)

    if content.startswith("Error:"):
        return {"status": "error", "message": content}

    return {"status": "success", "content": content}


def write_file(file_path: str, content: str, tool_context: ToolContext) -> dict:
    """Create a new file with the given content.

    Cannot overwrite existing files. Use edit_file for modifications.

    Args:
        file_path: Absolute path for the new file (must start with /).
        content: The content to write to the file.
    """
    try:
        validated = validate_path(file_path)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    backend = _get_backend(tool_context)
    result = backend.write(validated, content)

    if result.error:
        if result.error == "invalid_path":
            return {
                "status": "error",
                "message": f"File already exists: {validated}. Use edit_file to modify.",
            }
        return {"status": "error", "message": result.error}

    _apply_files_update(tool_context, result.files_update)
    return {"status": "success", "path": result.path}


def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    tool_context: ToolContext,
    replace_all: bool = False,
) -> dict:
    """Edit a file by replacing a specific string with a new string.

    The old_string must uniquely identify the text to replace, unless
    replace_all is set to True.

    Args:
        file_path: Absolute path to the file (must start with /).
        old_string: The exact text to find and replace.
        new_string: The replacement text.
        replace_all: If True, replace all occurrences. Defaults to False.
    """
    try:
        validated = validate_path(file_path)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    backend = _get_backend(tool_context)
    result = backend.edit(validated, old_string, new_string, replace_all)

    if result.error:
        return {"status": "error", "message": str(result.error)}

    _apply_files_update(tool_context, result.files_update)
    return {
        "status": "success",
        "path": result.path,
        "occurrences": result.occurrences,
    }


def glob(pattern: str, tool_context: ToolContext, path: str = "/") -> dict:
    """Find files matching a glob pattern.

    Supports ** for recursive matching and {} for alternatives.

    Args:
        pattern: Glob pattern to match (e.g., "**/*.py", "src/{a,b}/*.ts").
        path: Base directory to search from. Defaults to "/".
    """
    try:
        validated_path = validate_path(path)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    backend = _get_backend(tool_context)
    entries = backend.glob_info(pattern, validated_path)
    return {"status": "success", "entries": entries}


def grep(
    pattern: str,
    tool_context: ToolContext,
    path: str | None = None,
    glob: str | None = None,
    output_mode: str = "files_with_matches",
) -> dict:
    """Search for a text pattern within files.

    Searches for literal text matches across files. Can filter by path
    and/or glob pattern.

    Args:
        pattern: The text pattern to search for (literal match).
        path: Optional directory to limit the search to.
        glob: Optional glob pattern to filter files (e.g., "*.py").
        output_mode: One of "files_with_matches" (default), "content", or "count".
    """
    validated_path = None
    if path:
        try:
            validated_path = validate_path(path)
        except ValueError as e:
            return {"status": "error", "message": str(e)}

    backend = _get_backend(tool_context)
    raw_result = backend.grep_raw(pattern, validated_path, glob)

    if isinstance(raw_result, str):
        # Already formatted (e.g., from ripgrep backend)
        return {"status": "success", "result": truncate_if_too_long(raw_result)}

    formatted = format_grep_matches(raw_result, output_mode)
    return {"status": "success", "result": truncate_if_too_long(formatted)}
