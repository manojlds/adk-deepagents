"""State backend — stores files in a dict (maps to ADK session.state).

Ported from deepagents.backends.state with adaptations for ADK session state.
"""

from __future__ import annotations

from typing import Any

from adk_deepagents.backends.protocol import (
    Backend,
    EditResult,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)
from adk_deepagents.backends.utils import (
    create_file_data,
    file_data_to_string,
    format_read_response,
    glob_search_files,
    grep_matches_from_files,
    normalize_path,
    perform_string_replacement,
    update_file_data,
)


class StateBackend(Backend):
    """Backend that stores files in a dict (session state).

    In deepagents this uses ``runtime.state["files"]``. In ADK, the state
    dict is provided directly (from ``tool_context.state`` or a factory).

    Parameters
    ----------
    state:
        The session state dict. Files are stored under ``state["files"]``.
    """

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    # -- internal helpers ---------------------------------------------------

    @property
    def _files(self) -> dict[str, FileData]:
        return self._state.get("files", {})

    def _set_files(self, files: dict[str, FileData]) -> None:
        self._state["files"] = files

    # -- Backend implementation ---------------------------------------------

    def ls_info(self, path: str) -> list[FileInfo]:
        normalized = normalize_path(path)
        files = self._files
        entries: dict[str, FileInfo] = {}

        for fp in files:
            norm_fp = normalize_path(fp)
            # Exact file match
            if norm_fp == normalized:
                fd = files[fp]
                content = fd.get("content", [])
                size = sum(len(line) for line in content) + max(0, len(content) - 1)
                return [
                    FileInfo(
                        path=norm_fp,
                        is_dir=False,
                        size=size,
                        modified_at=fd.get("modified_at", ""),
                    )
                ]

            # Directory listing
            prefix = normalized if normalized.endswith("/") else normalized + "/"
            if not norm_fp.startswith(prefix):
                continue

            remainder = norm_fp[len(prefix) :]
            if "/" in remainder:
                # Subdirectory
                dir_name = remainder.split("/")[0]
                dir_path = prefix + dir_name
                if dir_path not in entries:
                    entries[dir_path] = FileInfo(path=dir_path, is_dir=True)
            else:
                # Direct child file
                fd = files[fp]
                content = fd.get("content", [])
                size = sum(len(line) for line in content) + max(0, len(content) - 1)
                entries[norm_fp] = FileInfo(
                    path=norm_fp,
                    is_dir=False,
                    size=size,
                    modified_at=fd.get("modified_at", ""),
                )

        return sorted(entries.values(), key=lambda e: e["path"])

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        normalized = normalize_path(file_path)
        files = self._files
        file_data = files.get(normalized)
        if file_data is None:
            return f"Error: file not found: {normalized}"
        return format_read_response(file_data, offset, limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        normalized = normalize_path(file_path)
        files = self._files

        if normalized in files:
            return WriteResult(
                error="invalid_path",
                path=normalized,
            )

        file_data = create_file_data(content)
        files_update = {normalized: file_data}
        return WriteResult(
            path=normalized,
            files_update=files_update,
        )

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        normalized = normalize_path(file_path)
        files = self._files
        file_data = files.get(normalized)

        if file_data is None:
            return EditResult(error="file_not_found", path=normalized)

        current_content = file_data_to_string(file_data)
        result = perform_string_replacement(current_content, old_string, new_string, replace_all)

        if isinstance(result, str):
            # Error message
            return EditResult(error=result, path=normalized)

        new_content, count = result
        new_file_data = update_file_data(file_data, new_content)
        files_update = {normalized: new_file_data}
        return EditResult(
            path=normalized,
            files_update=files_update,
            occurrences=count,
        )

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        matches = grep_matches_from_files(self._files, pattern, path, glob)
        return matches

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        matched = glob_search_files(self._files, pattern, path)
        result: list[FileInfo] = []
        for fp, fd in sorted(matched.items()):
            content = fd.get("content", [])
            size = sum(len(line) for line in content) + max(0, len(content) - 1)
            result.append(
                FileInfo(
                    path=normalize_path(fp),
                    is_dir=False,
                    size=size,
                    modified_at=fd.get("modified_at", ""),
                )
            )
        return result

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        raise NotImplementedError("StateBackend does not support upload_files")

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        files = self._files
        results: list[FileDownloadResponse] = []
        for p in paths:
            normalized = normalize_path(p)
            file_data = files.get(normalized)
            if file_data is None:
                results.append(FileDownloadResponse(path=normalized, error="file_not_found"))
            else:
                content_str = file_data_to_string(file_data)
                results.append(
                    FileDownloadResponse(path=normalized, content=content_str.encode("utf-8"))
                )
        return results
