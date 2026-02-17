"""Filesystem backend — reads/writes to the local filesystem.

Ported from deepagents.backends.filesystem with adaptations for ADK.
Files are persisted directly to disk, so ``WriteResult.files_update`` and
``EditResult.files_update`` are always ``None`` (no state merging needed).
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from adk_deepagents.backends.protocol import (
    Backend,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)
from adk_deepagents.backends.utils import (
    EMPTY_CONTENT_WARNING,
    format_content_with_line_numbers,
    perform_string_replacement,
)


class FilesystemBackend(Backend):
    """Backend that reads/writes to the local filesystem.

    Parameters
    ----------
    root_dir:
        Root directory for file operations. Defaults to the current working
        directory. In virtual mode, all paths are resolved relative to this.
    virtual_mode:
        If ``True``, enforce that all paths stay within *root_dir*. Prevents
        path traversal and access outside the root. Defaults to ``False``.
    max_file_size_mb:
        Maximum file size in megabytes for read operations. Defaults to 10.
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        *,
        virtual_mode: bool = False,
        max_file_size_mb: float = 10,
    ) -> None:
        self._root = Path(root_dir).resolve() if root_dir else Path.cwd()
        self._virtual_mode = virtual_mode
        self._max_file_size = int(max_file_size_mb * 1024 * 1024)

    # -- path resolution ----------------------------------------------------

    def _resolve_path(self, key: str) -> Path:
        """Resolve a virtual path to a real filesystem path.

        In virtual mode, paths are relative to root and cannot escape it.
        In non-virtual mode, paths are used as-is.
        """
        if self._virtual_mode:
            # Strip leading / and resolve relative to root
            rel = key.lstrip("/")
            if not rel:
                return self._root
            resolved = (self._root / rel).resolve()
            # Prevent escape from root directory
            try:
                resolved.relative_to(self._root)
            except ValueError as err:
                raise ValueError(f"Path escapes root directory: {key}") from err
            return resolved
        else:
            # Non-virtual: treat as absolute path, or relative to root
            p = Path(key)
            if p.is_absolute():
                return p
            return (self._root / key).resolve()

    # -- Backend implementation ---------------------------------------------

    def ls_info(self, path: str) -> list[FileInfo]:
        try:
            resolved = self._resolve_path(path)
        except ValueError:
            return []

        if not resolved.exists():
            return []

        if resolved.is_file():
            stat = resolved.stat()
            return [
                FileInfo(
                    path=path,
                    is_dir=False,
                    size=stat.st_size,
                    modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                )
            ]

        if not resolved.is_dir():
            return []

        entries: list[FileInfo] = []
        try:
            for child in sorted(resolved.iterdir()):
                child_path = f"{path.rstrip('/')}/{child.name}" if path != "/" else f"/{child.name}"
                try:
                    stat = child.stat()
                    entries.append(
                        FileInfo(
                            path=child_path,
                            is_dir=child.is_dir(),
                            size=stat.st_size if child.is_file() else 0,
                            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                        )
                    )
                except OSError:
                    # Skip files we can't stat (broken symlinks, etc.)
                    continue
        except PermissionError:
            return []

        return entries

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        try:
            resolved = self._resolve_path(file_path)
        except ValueError as e:
            return f"Error: {e}"

        if not resolved.exists():
            return f"Error: file not found: {file_path}"
        if resolved.is_dir():
            return f"Error: is a directory: {file_path}"

        # Check file size
        file_size = resolved.stat().st_size
        if file_size > self._max_file_size:
            return (
                f"Error: file too large ({file_size} bytes, "
                f"max {self._max_file_size} bytes): {file_path}"
            )

        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
        except PermissionError:
            return f"Error: permission denied: {file_path}"
        except OSError as e:
            return f"Error: {e}"

        if not content:
            return EMPTY_CONTENT_WARNING

        lines = content.split("\n")
        total_lines = len(lines)
        selected = lines[offset : offset + limit]

        if not selected:
            return f"No content at offset {offset} (file has {total_lines} lines)"

        formatted = format_content_with_line_numbers("\n".join(selected), start_line=offset + 1)

        if offset + limit < total_lines:
            remaining = total_lines - (offset + limit)
            formatted += (
                f"\n\n... ({remaining} more lines. Use offset={offset + limit} to continue reading)"
            )

        return formatted

    def write(self, file_path: str, content: str) -> WriteResult:
        try:
            resolved = self._resolve_path(file_path)
        except ValueError:
            return WriteResult(error="invalid_path", path=file_path)

        if resolved.exists():
            return WriteResult(error="already_exists", path=file_path)

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
        except PermissionError:
            return WriteResult(error="permission_denied", path=file_path)
        except OSError:
            return WriteResult(error="invalid_path", path=file_path)

        # files_update is None — file is persisted directly to disk
        return WriteResult(path=file_path, files_update=None)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        try:
            resolved = self._resolve_path(file_path)
        except ValueError:
            return EditResult(error="invalid_path", path=file_path)

        if not resolved.exists():
            return EditResult(error="file_not_found", path=file_path)
        if resolved.is_dir():
            return EditResult(error="is_directory", path=file_path)

        try:
            current_content = resolved.read_text(encoding="utf-8")
        except PermissionError:
            return EditResult(error="permission_denied", path=file_path)
        except OSError as e:
            return EditResult(error=str(e), path=file_path)

        result = perform_string_replacement(current_content, old_string, new_string, replace_all)

        if isinstance(result, str):
            return EditResult(error=result, path=file_path)

        new_content, count = result

        try:
            resolved.write_text(new_content, encoding="utf-8")
        except PermissionError:
            return EditResult(error="permission_denied", path=file_path)
        except OSError as e:
            return EditResult(error=str(e), path=file_path)

        # files_update is None — file is persisted directly to disk
        return EditResult(path=file_path, files_update=None, occurrences=count)

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        search_path = self._resolve_path(path) if path else self._root

        # Try ripgrep first for performance
        rg_result = self._grep_with_ripgrep(pattern, search_path, glob)
        if rg_result is not None:
            return rg_result

        # Fallback to Python-based search
        return self._grep_python(pattern, search_path, glob)

    def _grep_with_ripgrep(
        self,
        pattern: str,
        search_path: Path,
        glob_pattern: str | None,
    ) -> list[GrepMatch] | None:
        """Try to grep using ripgrep. Returns None if rg is not available."""
        cmd = ["rg", "--json", "-F", pattern, str(search_path)]
        if glob_pattern:
            cmd.extend(["--glob", glob_pattern])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

        if result.returncode not in (0, 1):
            return None

        import json

        matches: list[GrepMatch] = []
        for line in result.stdout.splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") != "match":
                continue
            match_data = data["data"]
            file_path = match_data["path"]["text"]
            # Make path relative to root in virtual mode
            if self._virtual_mode:
                try:
                    rel = str(Path(file_path).relative_to(self._root))
                    file_path = "/" + rel
                except ValueError:
                    pass
            line_number = match_data["line_number"]
            text = match_data["lines"]["text"].rstrip("\n")
            matches.append(GrepMatch(path=file_path, line=line_number, text=text))
        return matches

    def _grep_python(
        self,
        pattern: str,
        search_path: Path,
        glob_pattern: str | None,
    ) -> list[GrepMatch]:
        """Fallback Python-based grep."""
        matches: list[GrepMatch] = []

        if search_path.is_file():
            files = [search_path]
        elif search_path.is_dir():
            if glob_pattern:
                files = sorted(search_path.rglob(glob_pattern))
            else:
                files = sorted(search_path.rglob("*"))
        else:
            return matches

        for file_path in files:
            if not file_path.is_file():
                continue
            if file_path.stat().st_size > self._max_file_size:
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except (PermissionError, OSError):
                continue

            for line_num, line in enumerate(content.split("\n"), start=1):
                if pattern in line:
                    # Build path string
                    if self._virtual_mode:
                        try:
                            rel = str(file_path.relative_to(self._root))
                            display_path = "/" + rel
                        except ValueError:
                            display_path = str(file_path)
                    else:
                        display_path = str(file_path)
                    matches.append(GrepMatch(path=display_path, line=line_num, text=line))

        return matches

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        try:
            search_path = self._resolve_path(path)
        except ValueError:
            return []

        if not search_path.is_dir():
            return []

        entries: list[FileInfo] = []
        for match in sorted(search_path.rglob(pattern)):
            if not match.is_file():
                continue
            try:
                stat = match.stat()
            except OSError:
                continue

            if self._virtual_mode:
                try:
                    rel = str(match.relative_to(self._root))
                    display_path = "/" + rel
                except ValueError:
                    display_path = str(match)
            else:
                display_path = str(match)

            entries.append(
                FileInfo(
                    path=display_path,
                    is_dir=False,
                    size=stat.st_size,
                    modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                )
            )

        return entries

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        results: list[FileUploadResponse] = []
        for name, content in files:
            try:
                resolved = self._resolve_path(name)
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_bytes(content)
                results.append(FileUploadResponse(path=name))
            except (ValueError, PermissionError, OSError):
                results.append(FileUploadResponse(path=name, error="permission_denied"))
        return results

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        results: list[FileDownloadResponse] = []
        for p in paths:
            try:
                resolved = self._resolve_path(p)
            except ValueError:
                results.append(FileDownloadResponse(path=p, error="invalid_path"))
                continue

            if not resolved.exists():
                results.append(FileDownloadResponse(path=p, error="file_not_found"))
                continue
            if resolved.is_dir():
                results.append(FileDownloadResponse(path=p, error="is_directory"))
                continue

            try:
                content = resolved.read_bytes()
                results.append(FileDownloadResponse(path=p, content=content))
            except PermissionError:
                results.append(FileDownloadResponse(path=p, error="permission_denied"))
            except OSError:
                results.append(FileDownloadResponse(path=p, error="file_not_found"))

        return results
