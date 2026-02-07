"""Shared backend utilities.

Ported from deepagents.backends.utils with minor adaptations.
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime

from adk_deepagents.backends.protocol import FileData, GrepMatch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_LINE_LENGTH = 5000
LINE_NUMBER_WIDTH = 6
TOOL_RESULT_TOKEN_LIMIT = 20000
NUM_CHARS_PER_TOKEN = 4
EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def normalize_path(path: str) -> str:
    """Return a canonical ``/``-prefixed path with no trailing slash."""
    path = path.replace("\\", "/")
    path = os.path.normpath(path)
    path = path.replace("\\", "/")
    if not path.startswith("/"):
        path = "/" + path
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return path


def validate_path(
    path: str,
    *,
    allowed_prefixes: list[str] | None = None,
) -> str:
    """Validate and normalize a file path.

    Raises ``ValueError`` on traversal attempts or invalid paths.
    """
    if ".." in path.split("/"):
        raise ValueError(f"Path traversal not allowed: {path}")
    if "~" in path:
        raise ValueError(f"Home directory expansion not allowed: {path}")
    # Reject Windows absolute paths
    if len(path) >= 2 and path[1] == ":" and path[0].isalpha():
        raise ValueError(f"Windows absolute paths not allowed: {path}")

    normalized = normalize_path(path)

    if allowed_prefixes and not any(normalized.startswith(p) for p in allowed_prefixes):
        raise ValueError(f"Path {normalized} not within allowed prefixes: {allowed_prefixes}")

    return normalized


# ---------------------------------------------------------------------------
# File data helpers
# ---------------------------------------------------------------------------


def create_file_data(content: str, created_at: str | None = None) -> FileData:
    """Create a ``FileData`` dict from string content."""
    now = datetime.now(UTC).isoformat()
    lines = content.split("\n") if content else []
    return FileData(
        content=lines,
        created_at=created_at or now,
        modified_at=now,
    )


def update_file_data(file_data: FileData, content: str) -> FileData:
    """Return a new ``FileData`` with updated content, preserving ``created_at``."""
    lines = content.split("\n") if content else []
    return FileData(
        content=lines,
        created_at=file_data.get("created_at", datetime.now(UTC).isoformat()),
        modified_at=datetime.now(UTC).isoformat(),
    )


def file_data_to_string(file_data: FileData) -> str:
    """Join file data lines back into a single string."""
    return "\n".join(file_data.get("content", []))


# ---------------------------------------------------------------------------
# Content formatting
# ---------------------------------------------------------------------------


def format_content_with_line_numbers(content: str, start_line: int = 1) -> str:
    """Format content with ``cat -n`` style line numbers.

    Long lines (> MAX_LINE_LENGTH) are chunked with continuation markers
    (e.g. ``5.1``, ``5.2``).
    """
    lines = content.split("\n")
    result_parts: list[str] = []
    for i, line in enumerate(lines):
        line_num = start_line + i
        if len(line) <= MAX_LINE_LENGTH:
            result_parts.append(f"{line_num:>{LINE_NUMBER_WIDTH}}\t{line}")
        else:
            # Chunk long lines
            chunks = [line[j : j + MAX_LINE_LENGTH] for j in range(0, len(line), MAX_LINE_LENGTH)]
            for ci, chunk in enumerate(chunks):
                if ci == 0:
                    result_parts.append(f"{line_num:>{LINE_NUMBER_WIDTH}}\t{chunk}")
                else:
                    label = f"{line_num}.{ci}"
                    result_parts.append(f"{label:>{LINE_NUMBER_WIDTH}}\t{chunk}")
    return "\n".join(result_parts)


def format_read_response(
    file_data: FileData,
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Format file data for a read response with pagination."""
    lines = file_data.get("content", [])
    total_lines = len(lines)

    if total_lines == 0:
        return EMPTY_CONTENT_WARNING

    selected = lines[offset : offset + limit]
    if not selected:
        return f"No content at offset {offset} (file has {total_lines} lines)"

    content = "\n".join(selected)
    formatted = format_content_with_line_numbers(content, start_line=offset + 1)

    if offset + limit < total_lines:
        remaining = total_lines - (offset + limit)
        formatted += (
            f"\n\n... ({remaining} more lines. Use offset={offset + limit} to continue reading)"
        )

    return formatted


# ---------------------------------------------------------------------------
# String replacement
# ---------------------------------------------------------------------------


def perform_string_replacement(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> tuple[str, int] | str:
    """Perform string replacement.

    Returns ``(new_content, count)`` on success or an error string on failure.
    """
    if old_string == new_string:
        return "old_string and new_string are identical"

    count = content.count(old_string)
    if count == 0:
        return "old_string not found in file content"

    if not replace_all and count > 1:
        return (
            f"old_string appears {count} times. "
            "Provide more context to make it unique, or set replace_all=True."
        )

    if replace_all:
        new_content = content.replace(old_string, new_string)
    else:
        new_content = content.replace(old_string, new_string, 1)

    return (new_content, count)


# ---------------------------------------------------------------------------
# Grep / glob helpers
# ---------------------------------------------------------------------------


def grep_matches_from_files(
    files: dict[str, FileData],
    pattern: str,
    path: str | None = None,
    glob_pattern: str | None = None,
) -> list[GrepMatch]:
    """Search for literal *pattern* in files, returning structured matches."""
    matches: list[GrepMatch] = []
    filtered = filter_files_by_path(files, path) if path else files

    if glob_pattern:
        filtered = glob_search_files(filtered, glob_pattern, path or "/")

    for file_path, file_data in sorted(filtered.items()):
        lines = file_data.get("content", [])
        for line_num, line in enumerate(lines, start=1):
            if pattern in line:
                matches.append(GrepMatch(path=file_path, line=line_num, text=line))

    return matches


def format_grep_matches(
    matches: list[GrepMatch],
    output_mode: str = "files_with_matches",
) -> str:
    """Format grep matches according to the output mode."""
    if not matches:
        return "No matches found."

    if output_mode == "files_with_matches":
        seen: set[str] = set()
        paths: list[str] = []
        for m in matches:
            if m["path"] not in seen:
                seen.add(m["path"])
                paths.append(m["path"])
        return "\n".join(paths)

    if output_mode == "count":
        from collections import Counter

        counts: dict[str, int] = Counter(m["path"] for m in matches)
        return "\n".join(f"{p}: {c}" for p, c in sorted(counts.items()))

    # Default: content mode
    lines: list[str] = []
    for m in matches:
        lines.append(f"{m['path']}:{m['line']}:{m['text']}")
    return "\n".join(lines)


def filter_files_by_path(
    files: dict[str, FileData],
    path: str,
) -> dict[str, FileData]:
    """Filter files dict to entries at or under *path*."""
    normalized = normalize_path(path)
    if normalized == "/":
        return dict(files)  # Root matches everything
    result: dict[str, FileData] = {}
    for fp, fd in files.items():
        norm_fp = normalize_path(fp)
        if norm_fp == normalized or norm_fp.startswith(normalized + "/"):
            result[fp] = fd
    return result


def glob_search_files(
    files: dict[str, FileData],
    pattern: str,
    path: str = "/",
) -> dict[str, FileData]:
    """Filter files using a glob pattern.

    Uses ``wcmatch`` for glob matching with brace expansion and globstar.
    """
    try:
        from wcmatch import glob as wc_glob
    except ImportError:
        # Fallback to fnmatch
        import fnmatch

        result: dict[str, FileData] = {}
        for fp, fd in files.items():
            if fnmatch.fnmatch(fp, pattern):
                result[fp] = fd
        return result

    normalized_path = normalize_path(path)
    flags = wc_glob.BRACE | wc_glob.GLOBSTAR

    result = {}
    for fp, fd in files.items():
        norm_fp = normalize_path(fp)
        # Check file is under the search path
        if normalized_path == "/":
            # Root path — all files match
            pass
        elif not (norm_fp == normalized_path or norm_fp.startswith(normalized_path + "/")):
            continue
        # Match the relative path against the pattern
        if normalized_path == "/":
            rel = norm_fp.lstrip("/")
        else:
            rel = norm_fp[len(normalized_path) :].lstrip("/")
        if wc_glob.globmatch(rel, pattern, flags=flags):
            result[fp] = fd
    return result


# ---------------------------------------------------------------------------
# Truncation / eviction helpers
# ---------------------------------------------------------------------------


def truncate_if_too_long(
    result: str,
    token_limit: int = TOOL_RESULT_TOKEN_LIMIT,
) -> str:
    """Truncate a string if it exceeds the rough token limit."""
    char_limit = token_limit * NUM_CHARS_PER_TOKEN
    if len(result) <= char_limit:
        return result
    half = char_limit // 2
    return (
        result[:half]
        + f"\n\n... (truncated {len(result) - char_limit} characters) ...\n\n"
        + result[-half:]
    )


def create_content_preview(content: str, max_lines: int = 20) -> str:
    """Create a head + tail preview of large content."""
    lines = content.split("\n")
    if len(lines) <= max_lines:
        return content
    head = lines[: max_lines // 2]
    tail = lines[-(max_lines // 2) :]
    omitted = len(lines) - max_lines
    return "\n".join(head) + f"\n\n... ({omitted} lines omitted) ...\n\n" + "\n".join(tail)


def sanitize_tool_call_id(tool_call_id: str) -> str:
    """Sanitize a tool call ID for use as a file path component."""
    return re.sub(r"[./\\]", "_", tool_call_id)
