"""Store backend — cross-thread persistence via a shared dict-based store.

Provides namespace-scoped file storage that persists across different
conversation sessions, enabling agents to share data across threads.

Unlike StateBackend (which is tied to a single session's state dict),
StoreBackend uses a shared mutable store that can be passed to multiple
sessions.
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


class StoreBackend(Backend):
    """Backend that persists files in a shared dict-based store.

    Files are scoped by a ``namespace`` prefix so that different agents
    or workflows can operate in isolated namespaces while sharing the
    same underlying store.

    Parameters
    ----------
    store:
        A shared mutable dict that persists across sessions.  The same
        ``store`` instance must be passed to all sessions that need to
        share data.  Files are stored under ``store["files"]``.
    namespace:
        Optional namespace prefix for path isolation.  When set, all
        paths are transparently prefixed with ``/<namespace>``.  If
        ``None``, paths are stored as-is.

    Example
    -------
    >>> shared = {}
    >>> backend_a = StoreBackend(shared, namespace="project1")
    >>> backend_b = StoreBackend(shared, namespace="project1")
    >>> # Write in session A
    >>> result = backend_a.write("/readme.md", "# Hello")
    >>> shared["files"].update(result.files_update)
    >>> # Read in session B — same store, same namespace
    >>> content = backend_b.read("/readme.md")
    """

    def __init__(
        self,
        store: dict[str, Any],
        namespace: str | None = None,
    ) -> None:
        self._store = store
        self._namespace = namespace
        if "files" not in self._store:
            self._store["files"] = {}

    # -- internal helpers ---------------------------------------------------

    @property
    def _files(self) -> dict[str, FileData]:
        return self._store.get("files", {})

    def _set_files(self, files: dict[str, FileData]) -> None:
        self._store["files"] = files

    def _ns_path(self, path: str) -> str:
        """Prefix *path* with the namespace (if set)."""
        normalized = normalize_path(path)
        if self._namespace:
            ns_prefix = normalize_path(self._namespace)
            if not normalized.startswith(ns_prefix + "/") and normalized != ns_prefix:
                return ns_prefix + normalized
        return normalized

    def _strip_ns(self, path: str) -> str:
        """Remove the namespace prefix from *path* (if set)."""
        if self._namespace:
            ns_prefix = normalize_path(self._namespace)
            if path.startswith(ns_prefix + "/"):
                return path[len(ns_prefix) :]
            if path == ns_prefix:
                return "/"
        return path

    def _ns_files(self) -> dict[str, FileData]:
        """Return only files within the current namespace."""
        files = self._files
        if not self._namespace:
            return dict(files)
        ns_prefix = normalize_path(self._namespace)
        return {
            fp: fd for fp, fd in files.items() if fp.startswith(ns_prefix + "/") or fp == ns_prefix
        }

    # -- Backend implementation ---------------------------------------------

    def ls_info(self, path: str) -> list[FileInfo]:
        ns_path = self._ns_path(path)
        files = self._files
        entries: dict[str, FileInfo] = {}

        for fp in files:
            norm_fp = normalize_path(fp)
            # Exact file match
            if norm_fp == ns_path:
                fd = files[fp]
                content = fd.get("content", [])
                size = sum(len(line) for line in content) + max(0, len(content) - 1)
                return [
                    FileInfo(
                        path=self._strip_ns(norm_fp),
                        is_dir=False,
                        size=size,
                        modified_at=fd.get("modified_at", ""),
                    )
                ]

            # Directory listing
            prefix = ns_path if ns_path.endswith("/") else ns_path + "/"
            if not norm_fp.startswith(prefix):
                continue

            remainder = norm_fp[len(prefix) :]
            if "/" in remainder:
                # Subdirectory
                dir_name = remainder.split("/")[0]
                dir_path = prefix + dir_name
                ext_path = self._strip_ns(dir_path)
                if ext_path not in entries:
                    entries[ext_path] = FileInfo(path=ext_path, is_dir=True)
            else:
                # Direct child file
                fd = files[fp]
                content = fd.get("content", [])
                size = sum(len(line) for line in content) + max(0, len(content) - 1)
                ext_path = self._strip_ns(norm_fp)
                entries[ext_path] = FileInfo(
                    path=ext_path,
                    is_dir=False,
                    size=size,
                    modified_at=fd.get("modified_at", ""),
                )

        return sorted(entries.values(), key=lambda e: e["path"])

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        ns_path = self._ns_path(file_path)
        files = self._files
        file_data = files.get(ns_path)
        if file_data is None:
            return f"Error: file not found: {normalize_path(file_path)}"
        return format_read_response(file_data, offset, limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        ns_path = self._ns_path(file_path)
        files = self._files

        if ns_path in files:
            return WriteResult(
                error="already_exists",
                path=normalize_path(file_path),
            )

        file_data = create_file_data(content)
        files_update = {ns_path: file_data}
        return WriteResult(
            path=normalize_path(file_path),
            files_update=files_update,
        )

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        ns_path = self._ns_path(file_path)
        files = self._files
        file_data = files.get(ns_path)

        if file_data is None:
            return EditResult(error="file_not_found", path=normalize_path(file_path))

        current_content = file_data_to_string(file_data)
        result = perform_string_replacement(current_content, old_string, new_string, replace_all)

        if isinstance(result, str):
            return EditResult(error=result, path=normalize_path(file_path))

        new_content, count = result
        new_file_data = update_file_data(file_data, new_content)
        files_update = {ns_path: new_file_data}
        return EditResult(
            path=normalize_path(file_path),
            files_update=files_update,
            occurrences=count,
        )

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        ns_files = self._ns_files()
        # Remap to external paths for grep
        external_files: dict[str, FileData] = {
            self._strip_ns(fp): fd for fp, fd in ns_files.items()
        }
        matches = grep_matches_from_files(external_files, pattern, path, glob)
        return matches

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        ns_files = self._ns_files()
        # Remap to external paths for glob search
        external_files: dict[str, FileData] = {
            self._strip_ns(fp): fd for fp, fd in ns_files.items()
        }
        matched = glob_search_files(external_files, pattern, path)
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
        responses: list[FileUploadResponse] = []
        for file_name, content in files:
            ns_path = self._ns_path(file_name)
            if ns_path in self._files:
                responses.append(
                    FileUploadResponse(path=normalize_path(file_name), error="already_exists")
                )
            else:
                file_data = create_file_data(content.decode("utf-8", errors="replace"))
                self._files[ns_path] = file_data
                responses.append(FileUploadResponse(path=normalize_path(file_name)))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        files = self._files
        results: list[FileDownloadResponse] = []
        for p in paths:
            ns_path = self._ns_path(p)
            file_data = files.get(ns_path)
            if file_data is None:
                results.append(FileDownloadResponse(path=normalize_path(p), error="file_not_found"))
            else:
                content_str = file_data_to_string(file_data)
                results.append(
                    FileDownloadResponse(
                        path=normalize_path(p), content=content_str.encode("utf-8")
                    )
                )
        return results

    # -- async overrides (direct, no asyncio.to_thread needed) ---------------

    async def als_info(self, path: str) -> list[FileInfo]:
        """Async :meth:`ls_info` — direct call (in-memory, no I/O)."""
        return self.ls_info(path)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Async :meth:`read` — direct call (in-memory, no I/O)."""
        return self.read(file_path, offset, limit)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Async :meth:`write` — direct call (in-memory, no I/O)."""
        return self.write(file_path, content)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Async :meth:`edit` — direct call (in-memory, no I/O)."""
        return self.edit(file_path, old_string, new_string, replace_all)

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Async :meth:`grep_raw` — direct call (in-memory, no I/O)."""
        return self.grep_raw(pattern, path, glob)

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Async :meth:`glob_info` — direct call (in-memory, no I/O)."""
        return self.glob_info(pattern, path)
