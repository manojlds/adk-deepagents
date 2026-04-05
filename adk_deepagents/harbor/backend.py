"""HarborBackend — proxies all file operations into the Harbor task container.

All read/write/edit/glob/grep operations go through ``environment.exec()``,
so the deepagent's filesystem tools operate on the container's filesystem,
not the host.
"""

from __future__ import annotations

import base64
import json
import shlex

from harbor.environments.base import BaseEnvironment  # ty: ignore[unresolved-import]

from adk_deepagents.backends.protocol import (
    Backend,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    ReadResult,
    WriteResult,
)

_MAX_OUTPUT = 200_000  # bytes before truncation


class HarborBackend(Backend):
    """Routes deepagent file operations through Harbor's environment.exec().

    Sync abstract methods raise ``NotImplementedError`` — deepagent's async
    tools call the ``a*`` variants which use ``environment.exec()`` directly.
    """

    def __init__(self, environment: BaseEnvironment) -> None:
        self._env = environment

    # ------------------------------------------------------------------
    # Sync stubs — not called by deepagent async tools
    # ------------------------------------------------------------------

    def ls_info(self, path: str) -> list[FileInfo]:
        raise NotImplementedError("Use async methods with HarborBackend")

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        raise NotImplementedError("Use async methods with HarborBackend")

    def write(self, file_path: str, content: str) -> WriteResult:
        raise NotImplementedError("Use async methods with HarborBackend")

    def edit(
        self, file_path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> EditResult:
        raise NotImplementedError("Use async methods with HarborBackend")

    def grep_raw(
        self, pattern: str, path: str | None = None, glob: str | None = None
    ) -> list[GrepMatch]:
        raise NotImplementedError("Use async methods with HarborBackend")

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        raise NotImplementedError("Use async methods with HarborBackend")

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        raise NotImplementedError("Use async methods with HarborBackend")

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        raise NotImplementedError("Use async methods with HarborBackend")

    # ------------------------------------------------------------------
    # Async implementations — all proxy through environment.exec()
    # ------------------------------------------------------------------

    async def _exec(self, command: str, timeout: int = 30) -> tuple[str, str]:
        """Execute a command in the container, return (stdout, stderr)."""
        result = await self._env.exec(command=command, timeout_sec=timeout)
        return result.stdout or "", result.stderr or ""

    async def als_info(self, path: str) -> list[FileInfo]:
        stdout, _ = await self._exec(
            f"find {shlex.quote(path)} -maxdepth 1 -printf '%p\\t%y\\t%s\\n' 2>/dev/null || true"
        )
        entries: list[FileInfo] = []
        for line in stdout.strip().splitlines():
            parts = line.split("\t")
            if len(parts) < 2 or parts[0] == path:
                continue
            entry: FileInfo = {"path": parts[0], "is_dir": parts[1] == "d"}
            if len(parts) >= 3 and parts[2].isdigit():
                entry["size"] = int(parts[2])
            entries.append(entry)
        return entries

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        # Use sed for line-based offset/limit
        start = offset + 1
        end = offset + limit
        stdout, stderr = await self._exec(
            f"sed -n '{start},{end}p' {shlex.quote(file_path)} 2>/dev/null",
        )
        if stderr and not stdout:
            return ReadResult(error="file_not_found", path=file_path)
        return ReadResult(content=stdout, path=file_path)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        # Base64-encode content to safely pass arbitrary text through shell
        encoded = base64.b64encode(content.encode()).decode()
        _, stderr = await self._exec(
            f"mkdir -p $(dirname {shlex.quote(file_path)}) && "
            f"printf '%s' {shlex.quote(encoded)} | base64 -d > {shlex.quote(file_path)}"
        )
        if stderr:
            return WriteResult(error="invalid_path", path=file_path)
        return WriteResult(path=file_path)

    async def aedit(
        self, file_path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> EditResult:
        old_b64 = base64.b64encode(old_string.encode()).decode()
        new_b64 = base64.b64encode(new_string.encode()).decode()
        replace_all_flag = "True" if replace_all else "False"

        script = (
            "import base64, sys; "
            f"old=base64.b64decode('{old_b64}').decode(); "
            f"new=base64.b64decode('{new_b64}').decode(); "
            f"content=open({shlex.quote(file_path)}).read(); "
            "count=content.count(old); "
            "print(count) if count > 0 else (print('ERR_NOT_FOUND') or sys.exit(1)); "
            f"replaced=content.replace(old, new) if {replace_all_flag} else content.replace(old, new, 1); "
            f"open({shlex.quote(file_path)}, 'w').write(replaced)"
        )
        stdout, stderr = await self._exec(f"python3 -c {shlex.quote(script)}")

        if "ERR_NOT_FOUND" in stdout or stderr:
            return EditResult(error="old_string not found in file", path=file_path)
        try:
            occurrences = int(stdout.strip())
        except ValueError:
            occurrences = None
        return EditResult(path=file_path, occurrences=occurrences)

    async def agrep_raw(
        self, pattern: str, path: str | None = None, glob: str | None = None
    ) -> list[GrepMatch]:
        target = shlex.quote(path) if path else "."
        glob_flag = f"--include={shlex.quote(glob)}" if glob else ""
        stdout, _ = await self._exec(
            f"grep -rn {glob_flag} {shlex.quote(pattern)} {target} 2>/dev/null || true"
        )
        matches: list[GrepMatch] = []
        for line in stdout.strip().splitlines()[:500]:  # cap results
            # format: path:line:text
            parts = line.split(":", 2)
            if len(parts) == 3:
                try:
                    matches.append(GrepMatch(path=parts[0], line=int(parts[1]), text=parts[2]))
                except ValueError:
                    continue
        return matches

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        script = (
            f"import glob as g, os, json; "
            f"matches=g.glob({shlex.quote(str(path) + '/' + pattern)}, recursive=True); "
            "print(json.dumps([{'path': m, 'is_dir': os.path.isdir(m)} for m in matches[:500]]))"
        )
        stdout, _ = await self._exec(f"python3 -c {shlex.quote(script)}")
        try:
            return [FileInfo(**item) for item in json.loads(stdout)]
        except Exception:
            return []

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        results = []
        for name, content in files:
            encoded = base64.b64encode(content).decode()
            _, stderr = await self._exec(
                f"printf '%s' {shlex.quote(encoded)} | base64 -d > {shlex.quote(name)}"
            )
            results.append(FileUploadResponse(path=name, error="invalid_path" if stderr else None))
        return results

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        results = []
        for path in paths:
            stdout, stderr = await self._exec(f"base64 {shlex.quote(path)} 2>/dev/null")
            if stderr or not stdout:
                results.append(FileDownloadResponse(path=path, error="file_not_found"))
            else:
                try:
                    results.append(
                        FileDownloadResponse(path=path, content=base64.b64decode(stdout.strip()))
                    )
                except Exception:
                    results.append(FileDownloadResponse(path=path, error="file_not_found"))
        return results
