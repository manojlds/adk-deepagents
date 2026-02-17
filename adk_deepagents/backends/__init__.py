"""Backend abstraction for file storage and operations."""

from adk_deepagents.backends.composite import CompositeBackend
from adk_deepagents.backends.filesystem import FilesystemBackend
from adk_deepagents.backends.protocol import (
    Backend,
    BackendFactory,
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    SandboxBackend,
    WriteResult,
)
from adk_deepagents.backends.state import StateBackend
from adk_deepagents.backends.store import StoreBackend

__all__ = [
    "Backend",
    "BackendFactory",
    "CompositeBackend",
    "EditResult",
    "ExecuteResponse",
    "FileDownloadResponse",
    "FilesystemBackend",
    "FileInfo",
    "FileUploadResponse",
    "GrepMatch",
    "SandboxBackend",
    "StateBackend",
    "StoreBackend",
    "WriteResult",
]
