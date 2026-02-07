"""Backend protocol - abstract interface for file storage and operations.

Ported from deepagents.backends.protocol with adaptations for ADK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Literal, TypeAlias, TypedDict

# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

FileOperationError = Literal[
    "file_not_found",
    "permission_denied",
    "is_directory",
    "invalid_path",
]

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class FileInfo(TypedDict, total=False):
    """Metadata for a file or directory entry."""

    path: str  # required
    is_dir: bool
    size: int
    modified_at: str


# Require `path` at minimum
FileInfo.__required_keys__ = frozenset({"path"})


class GrepMatch(TypedDict):
    """A single grep search match."""

    path: str
    line: int
    text: str


class FileData(TypedDict, total=False):
    """Internal representation of a file stored in state."""

    content: list[str]  # lines
    created_at: str  # ISO 8601
    modified_at: str  # ISO 8601


@dataclass
class WriteResult:
    """Result of a write operation."""

    error: FileOperationError | None = None
    path: str = ""
    files_update: dict[str, FileData] | None = None


@dataclass
class EditResult:
    """Result of an edit (string replacement) operation."""

    error: FileOperationError | str | None = None
    path: str = ""
    files_update: dict[str, FileData] | None = None
    occurrences: int | None = None


@dataclass
class FileDownloadResponse:
    """Response from downloading a file."""

    path: str
    content: bytes | None = None
    error: FileOperationError | None = None


@dataclass
class FileUploadResponse:
    """Response from uploading a file."""

    path: str
    error: FileOperationError | None = None


@dataclass
class ExecuteResponse:
    """Response from executing a shell command."""

    output: str = ""
    exit_code: int | None = None
    truncated: bool = False


# ---------------------------------------------------------------------------
# Backend abstract base class
# ---------------------------------------------------------------------------


class Backend(ABC):
    """Abstract backend for file operations.

    Mirrors the ``BackendProtocol`` from deepagents but uses a plain ABC
    instead of a ``runtime_checkpointer``-aware protocol.
    """

    @abstractmethod
    def ls_info(self, path: str) -> list[FileInfo]:
        """List files/directories at *path*."""

    @abstractmethod
    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read file content with optional pagination."""

    @abstractmethod
    def write(self, file_path: str, content: str) -> WriteResult:
        """Write *content* to *file_path* (create-only, no overwrites)."""

    @abstractmethod
    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace *old_string* with *new_string* in *file_path*."""

    @abstractmethod
    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Search for *pattern* in files. Returns matches or formatted string."""

    @abstractmethod
    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Find files matching a glob *pattern*."""

    @abstractmethod
    def upload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Upload files (name, content) pairs."""

    @abstractmethod
    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files by path."""


class SandboxBackend(Backend):
    """Backend that also supports shell command execution."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for this sandbox."""

    @abstractmethod
    def execute(self, command: str) -> ExecuteResponse:
        """Execute a shell *command* and return the result."""


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

BackendFactory: TypeAlias = Callable[[dict], Backend]
"""A callable that creates a Backend from session state dict."""
