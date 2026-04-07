"""Filesystem backend with explicit memory source path mappings."""

from __future__ import annotations

import fnmatch
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from adk_deepagents.backends.filesystem import FilesystemBackend
from adk_deepagents.backends.protocol import FileDownloadResponse, FileInfo, GrepMatch

_DEFAULT_EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)

_DEFAULT_EXCLUDE_PATTERNS = (
    "**/*.pyc",
    "**/*.pyo",
)


class MemoryMappedFilesystemBackend(FilesystemBackend):
    """Workspace filesystem backend with explicit memory-source path mappings.

    This backend keeps normal file operations sandboxed to ``root_dir`` while
    allowing reads from explicit source keys mapped to paths outside the root.
    """

    def __init__(
        self,
        *,
        root_dir: Path,
        memory_source_paths: Mapping[str, Path] | None = None,
        respect_gitignore: bool = True,
        exclude_patterns: Sequence[str] | None = None,
    ) -> None:
        super().__init__(root_dir=root_dir, virtual_mode=True)
        self._memory_source_paths = {
            source: path.expanduser().resolve()
            for source, path in (memory_source_paths or {}).items()
        }
        self._respect_gitignore = respect_gitignore
        self._exclude_patterns = tuple(exclude_patterns or ())
        self._is_git_repo: bool | None = None

    def ls_info(self, path: str) -> list[FileInfo]:
        entries = super().ls_info(path)
        return self._filter_file_info_entries(entries)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        entries = super().glob_info(pattern, path)
        return self._filter_file_info_entries(entries)

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch]:
        return self._filter_grep_matches(super().grep_raw(pattern, path=path, glob=glob))

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        results: list[FileDownloadResponse] = []
        for source in paths:
            mapped_path = self._memory_source_paths.get(source)
            if mapped_path is None:
                results.extend(super().download_files([source]))
                continue

            results.append(self._download_mapped_source(source=source, path=mapped_path))

        return results

    def _filter_file_info_entries(self, entries: list[FileInfo]) -> list[FileInfo]:
        hidden = self._hidden_original_paths([entry.get("path") for entry in entries])
        return [entry for entry in entries if entry.get("path") not in hidden]

    def _filter_grep_matches(self, matches: list[GrepMatch]) -> list[GrepMatch]:
        hidden = self._hidden_original_paths([match.get("path") for match in matches])
        return [match for match in matches if match.get("path") not in hidden]

    def _hidden_original_paths(self, raw_paths: Sequence[object]) -> set[str]:
        rel_by_original: dict[str, str] = {}
        for raw in raw_paths:
            if not isinstance(raw, str):
                continue
            rel = self._to_workspace_relative_path(raw)
            if rel is None:
                continue
            rel_by_original[raw] = rel

        if not rel_by_original:
            return set()

        hidden_relative = {
            rel for rel in set(rel_by_original.values()) if self._is_default_hidden(rel)
        }

        remaining = [rel for rel in set(rel_by_original.values()) if rel not in hidden_relative]
        hidden_relative.update(self._git_ignored_paths(remaining))

        return {original for original, rel in rel_by_original.items() if rel in hidden_relative}

    def _to_workspace_relative_path(self, path: str) -> str | None:
        normalized = path.strip()
        if not normalized:
            return None

        if normalized.startswith("/"):
            return PurePosixPath(normalized.lstrip("/")).as_posix()

        parsed = Path(normalized)
        if parsed.is_absolute():
            try:
                return parsed.resolve().relative_to(self._root).as_posix()
            except ValueError:
                return None

        return PurePosixPath(normalized).as_posix()

    def _is_default_hidden(self, relative_path: str) -> bool:
        if not relative_path:
            return False

        parts = PurePosixPath(relative_path).parts
        if any(part in _DEFAULT_EXCLUDED_DIR_NAMES for part in parts):
            return True

        if any(fnmatch.fnmatch(relative_path, pattern) for pattern in _DEFAULT_EXCLUDE_PATTERNS):
            return True

        return any(fnmatch.fnmatch(relative_path, pattern) for pattern in self._exclude_patterns)

    def _git_ignored_paths(self, relative_paths: Sequence[str]) -> set[str]:
        if not self._respect_gitignore:
            return set()
        if not relative_paths:
            return set()
        if not self._is_within_git_repo():
            return set()

        payload = "\n".join(relative_paths) + "\n"
        try:
            result = subprocess.run(
                ["git", "-C", str(self._root), "check-ignore", "--stdin"],
                input=payload,
                text=True,
                capture_output=True,
                check=False,
                timeout=5,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return set()

        if result.returncode not in {0, 1}:
            return set()

        ignored: set[str] = set()
        for line in result.stdout.splitlines():
            normalized = line.strip().lstrip("./").rstrip("/")
            if normalized:
                ignored.add(PurePosixPath(normalized).as_posix())
        return ignored

    def _is_within_git_repo(self) -> bool:
        if self._is_git_repo is not None:
            return self._is_git_repo

        try:
            result = subprocess.run(
                ["git", "-C", str(self._root), "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            self._is_git_repo = False
            return False

        self._is_git_repo = result.returncode == 0 and result.stdout.strip() == "true"
        return self._is_git_repo

    @staticmethod
    def _download_mapped_source(*, source: str, path: Path) -> FileDownloadResponse:
        if not path.exists():
            return FileDownloadResponse(path=source, error="file_not_found")
        if path.is_dir():
            return FileDownloadResponse(path=source, error="is_directory")

        try:
            content = path.read_bytes()
        except PermissionError:
            return FileDownloadResponse(path=source, error="permission_denied")
        except OSError:
            return FileDownloadResponse(path=source, error="file_not_found")

        return FileDownloadResponse(path=source, content=content)
