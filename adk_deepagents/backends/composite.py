"""Composite backend — path-based routing to multiple backends.

Routes file operations to different backends based on path prefixes.
For example, ``/workspace/*`` can go to a ``FilesystemBackend`` while
everything else goes to a ``StateBackend``.

Ported from deepagents.backends.composite.
"""

from __future__ import annotations

from adk_deepagents.backends.protocol import (
    Backend,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)
from adk_deepagents.backends.utils import normalize_path


class CompositeBackend(Backend):
    """Backend that routes operations by path prefix to child backends.

    Parameters
    ----------
    default:
        The fallback backend used when no route matches.
    routes:
        Mapping of path prefix → backend. Prefixes should be normalized
        (e.g., ``"/workspace"``). Longer prefixes take priority.

    Example
    -------
    >>> from adk_deepagents.backends import StateBackend, FilesystemBackend
    >>> composite = CompositeBackend(
    ...     default=StateBackend(state),
    ...     routes={"/workspace": FilesystemBackend(root_dir="./workspace")},
    ... )
    """

    def __init__(
        self,
        default: Backend,
        routes: dict[str, Backend] | None = None,
    ) -> None:
        self._default = default
        # Normalize and sort routes by prefix length (longest first for specificity)
        self._routes: list[tuple[str, Backend]] = []
        if routes:
            for prefix, backend in routes.items():
                normalized = normalize_path(prefix)
                self._routes.append((normalized, backend))
            self._routes.sort(key=lambda r: len(r[0]), reverse=True)

    @property
    def default(self) -> Backend:
        """The fallback backend."""
        return self._default

    @property
    def routes(self) -> list[tuple[str, Backend]]:
        """Sorted list of ``(prefix, backend)`` pairs."""
        return list(self._routes)

    def _resolve(self, path: str) -> Backend:
        """Find the backend for a given path."""
        normalized = normalize_path(path)
        for prefix, backend in self._routes:
            if normalized == prefix or normalized.startswith(prefix + "/"):
                return backend
        return self._default

    def _resolve_all(self, path: str | None = None) -> list[Backend]:
        """Get all backends that could serve paths under *path*.

        Used for operations like grep/glob that may span multiple backends.
        """
        if path is None or normalize_path(path) == "/":
            # Search all backends
            backends = [self._default]
            backends.extend(b for _, b in self._routes)
            return backends
        return [self._resolve(path)]

    # ----- ls_info -----

    def ls_info(self, path: str) -> list[FileInfo]:
        """List files at *path* using the appropriate backend."""
        return self._resolve(path).ls_info(path)

    # ----- read -----

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read a file from the appropriate backend."""
        return self._resolve(file_path).read(file_path, offset, limit)

    # ----- write -----

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write a file to the appropriate backend."""
        return self._resolve(file_path).write(file_path, content)

    # ----- edit -----

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file in the appropriate backend."""
        return self._resolve(file_path).edit(file_path, old_string, new_string, replace_all)

    # ----- grep_raw -----

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Search across all relevant backends and merge results."""
        backends = self._resolve_all(path)

        all_matches: list[GrepMatch] = []
        for backend in backends:
            result = backend.grep_raw(pattern, path, glob)
            if isinstance(result, list):
                all_matches.extend(result)
            elif isinstance(result, str) and result != "No matches found." and not all_matches:
                # Backend returned a formatted string with no prior list matches
                return result

        return all_matches

    # ----- glob_info -----

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Search across all relevant backends and merge results."""
        backends = self._resolve_all(path)
        all_results: list[FileInfo] = []
        seen_paths: set[str] = set()
        for backend in backends:
            for info in backend.glob_info(pattern, path):
                if info["path"] not in seen_paths:
                    seen_paths.add(info["path"])
                    all_results.append(info)
        return all_results

    # ----- upload_files -----

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files to the appropriate backends."""
        responses: list[FileUploadResponse] = []
        for file_name, content in files:
            backend = self._resolve(file_name)
            result = backend.upload_files([(file_name, content)])
            responses.extend(result)
        return responses

    # ----- download_files -----

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the appropriate backends."""
        responses: list[FileDownloadResponse] = []
        for path in paths:
            backend = self._resolve(path)
            result = backend.download_files([path])
            responses.extend(result)
        return responses

    # ----- async delegation -----

    async def als_info(self, path: str) -> list[FileInfo]:
        """Async :meth:`ls_info` — delegates to resolved backend."""
        return await self._resolve(path).als_info(path)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Async :meth:`read` — delegates to resolved backend."""
        return await self._resolve(file_path).aread(file_path, offset, limit)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Async :meth:`write` — delegates to resolved backend."""
        return await self._resolve(file_path).awrite(file_path, content)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Async :meth:`edit` — delegates to resolved backend."""
        return await self._resolve(file_path).aedit(file_path, old_string, new_string, replace_all)

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Async :meth:`grep_raw` — delegates to resolved backends and merges."""
        backends = self._resolve_all(path)

        all_matches: list[GrepMatch] = []
        for backend in backends:
            result = await backend.agrep_raw(pattern, path, glob)
            if isinstance(result, list):
                all_matches.extend(result)
            elif isinstance(result, str) and result != "No matches found." and not all_matches:
                return result

        return all_matches

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Async :meth:`glob_info` — delegates to resolved backends and merges."""
        backends = self._resolve_all(path)
        all_results: list[FileInfo] = []
        seen_paths: set[str] = set()
        for backend in backends:
            for info in await backend.aglob_info(pattern, path):
                if info["path"] not in seen_paths:
                    seen_paths.add(info["path"])
                    all_results.append(info)
        return all_results
