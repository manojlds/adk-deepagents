# Backends

Backends control how the agent reads and writes files. Every filesystem tool (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`) delegates to a backend for the actual storage operation.

## Overview

The backend system has three layers:

1. **`Backend` ABC** — Abstract base class defining the file operation interface
2. **Concrete backends** — `StateBackend`, `FilesystemBackend`, `CompositeBackend`, `StoreBackend`
3. **`BackendFactory`** — A callable `(state: dict) -> Backend` for deferred construction

```
Backend (ABC)
├── StateBackend       — files in session.state["files"]
├── FilesystemBackend  — files on local disk
├── CompositeBackend   — path-prefix routing to child backends
└── StoreBackend       — shared dict for cross-session persistence
```

## Backend ABC

All backends implement the `Backend` abstract base class defined in `adk_deepagents.backends.protocol`:

```python
from abc import ABC, abstractmethod

class Backend(ABC):
    @abstractmethod
    def ls_info(self, path: str) -> list[FileInfo]: ...

    @abstractmethod
    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str: ...

    @abstractmethod
    def write(self, file_path: str, content: str) -> WriteResult: ...

    @abstractmethod
    def edit(self, file_path: str, old_string: str, new_string: str,
             replace_all: bool = False) -> EditResult: ...

    @abstractmethod
    def grep_raw(self, pattern: str, path: str | None = None,
                 glob: str | None = None) -> list[GrepMatch] | str: ...

    @abstractmethod
    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]: ...

    @abstractmethod
    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]: ...

    @abstractmethod
    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]: ...
```

Each sync method has an async counterpart (`als_info`, `aread`, `awrite`, `aedit`, `agrep_raw`, `aglob_info`). The default implementations delegate to the sync method via `asyncio.to_thread()`. In-memory backends (`StateBackend`, `StoreBackend`) override these to call the sync method directly (no I/O overhead).

### Protocol Methods

| Method | Description |
|--------|-------------|
| `ls_info(path)` | List files and directories at `path`. Returns `list[FileInfo]`. |
| `read(file_path, offset, limit)` | Read file contents with pagination. Returns formatted string with line numbers. |
| `write(file_path, content)` | Create a new file (no overwrites). Returns `WriteResult`. |
| `edit(file_path, old_string, new_string, replace_all)` | Replace text in a file. Returns `EditResult`. |
| `grep_raw(pattern, path, glob)` | Search for literal text pattern. Returns `list[GrepMatch]` or formatted string. |
| `glob_info(pattern, path)` | Find files matching a glob pattern. Returns `list[FileInfo]`. |
| `upload_files(files)` | Upload `(name, content_bytes)` pairs. Returns `list[FileUploadResponse]`. |
| `download_files(paths)` | Download files by path. Returns `list[FileDownloadResponse]`. |

## Data Types

All data types are defined in `adk_deepagents.backends.protocol` and re-exported from `adk_deepagents.backends`.

### FileInfo

Metadata for a file or directory entry. Used by `ls_info()` and `glob_info()`.

```python
class FileInfo(TypedDict, total=False):
    path: str       # required — absolute path
    is_dir: bool    # True for directories
    size: int       # file size in bytes
    modified_at: str  # ISO 8601 timestamp
```

### GrepMatch

A single grep search match. Used by `grep_raw()`.

```python
class GrepMatch(TypedDict):
    path: str   # file path
    line: int   # line number (1-based)
    text: str   # matching line content
```

### FileData

Internal representation of a file stored in state (used by `StateBackend` and `StoreBackend`).

```python
class FileData(TypedDict, total=False):
    content: list[str]   # lines of text
    created_at: str      # ISO 8601
    modified_at: str     # ISO 8601
    _binary: str         # base64-encoded binary content
```

### WriteResult

Result of a `write()` operation.

```python
@dataclass
class WriteResult:
    error: FileOperationError | None = None
    path: str = ""
    files_update: dict[str, FileData] | None = None
```

- `error` — One of `"file_not_found"`, `"permission_denied"`, `"is_directory"`, `"invalid_path"`, `"already_exists"`, or `None` on success.
- `path` — The normalized path of the written file.
- `files_update` — A dict of `{path: FileData}` to merge into `session.state["files"]`. This is `None` for `FilesystemBackend` (which persists directly to disk) and populated for `StateBackend` / `StoreBackend`.

### EditResult

Result of an `edit()` operation.

```python
@dataclass
class EditResult:
    error: FileOperationError | str | None = None
    path: str = ""
    files_update: dict[str, FileData] | None = None
    occurrences: int | None = None
```

- `occurrences` — Number of replacements made.
- `error` — Can be a `FileOperationError` literal or a descriptive string (e.g., `"old_string not found in file content"`).

### FileDownloadResponse

Response from `download_files()`.

```python
@dataclass
class FileDownloadResponse:
    path: str
    content: bytes | None = None
    error: FileOperationError | None = None
```

### ExecuteResponse

Response from shell command execution (used by `SandboxBackend`).

```python
@dataclass
class ExecuteResponse:
    output: str = ""
    exit_code: int | None = None
    truncated: bool = False
```

---

## StateBackend

**In-memory file storage backed by the ADK session state dict.**

Files are stored as `FileData` dicts under `state["files"]`. They persist for the duration of the session and are lost when the session ends.

### How It Works

- Constructor receives the session state dict (from `tool_context.state` or a factory).
- All file operations read/write `state["files"]`.
- Write and edit operations return `files_update` dicts that must be merged into `state["files"]` (handled automatically by the filesystem tools via `_apply_files_update()`).

### When to Use

- **Ephemeral agents** — prototyping, testing, demos
- **No filesystem access needed** — agent works entirely in-memory
- **Sandboxed environments** — where local disk access isn't available

### Example

```python
from adk_deepagents import create_deep_agent

# StateBackend is the default — no backend argument needed
agent = create_deep_agent(
    name="ephemeral_agent",
    instruction="You are a helpful assistant.",
)
```

Or explicitly:

```python
from adk_deepagents.backends.state import StateBackend

def my_backend_factory(state):
    return StateBackend(state)

agent = create_deep_agent(
    name="ephemeral_agent",
    instruction="You are a helpful assistant.",
    backend=my_backend_factory,
)
```

### How `files_update` Merging Works

When an agent writes or edits a file with `StateBackend`, the result includes a `files_update` dict:

```python
result = backend.write("/hello.txt", "Hello, world!")
# result.files_update == {"/hello.txt": {"content": ["Hello, world!"], ...}}
```

The filesystem tool automatically merges this into session state:

```python
files = tool_context.state.get("files", {})
files.update(result.files_update)
tool_context.state["files"] = files
```

This ensures changes persist in the ADK session state dict for subsequent tool calls.

---

## FilesystemBackend

**Reads and writes to the local filesystem.**

Files are persisted directly to disk. `WriteResult.files_update` and `EditResult.files_update` are always `None` — no state merging is needed.

### Constructor Parameters

```python
FilesystemBackend(
    root_dir: str | Path | None = None,  # Root directory (default: cwd)
    virtual_mode: bool = False,           # Enforce path containment
    max_file_size_mb: float = 10,         # Max file size for reads
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `root_dir` | Current working directory | Base directory for all file operations |
| `virtual_mode` | `False` | When `True`, all paths are resolved relative to `root_dir` and cannot escape it. Paths starting with `/` are treated as relative to the root. |
| `max_file_size_mb` | `10` | Maximum file size in MB for read operations. Files exceeding this are rejected. |

### Virtual Mode

When `virtual_mode=True`:

- The path `/src/main.py` resolves to `{root_dir}/src/main.py`
- Path traversal (`../`) is rejected with a `ValueError`
- All `FileInfo.path` values in results use virtual paths (e.g., `/src/main.py`)

When `virtual_mode=False` (default):

- Absolute paths are used as-is
- Relative paths are resolved relative to `root_dir`

### When to Use

- **Working with real project files** — reading and editing source code
- **Persistent storage** — files survive across sessions
- **Local development** — full access to the host filesystem

### Grep: Ripgrep Integration

`FilesystemBackend.grep_raw()` uses a two-tier strategy:

1. **Ripgrep (`rg`)** — If the `rg` command is available on the system, it's used for fast, JSON-formatted search with `--json -F` (literal/fixed-string matching). Results are parsed and returned as `list[GrepMatch]`.
2. **Python fallback** — If `rg` is not installed or fails, a pure-Python implementation walks the file tree and searches line-by-line.

### Example

```python
from adk_deepagents import create_deep_agent
from adk_deepagents.backends.filesystem import FilesystemBackend

agent = create_deep_agent(
    name="project_agent",
    instruction="You are a coding assistant for this project.",
    backend=FilesystemBackend(
        root_dir="/home/user/my-project",
        virtual_mode=True,
        max_file_size_mb=5,
    ),
)
```

With a `BackendFactory` (so the backend is created per-session):

```python
from adk_deepagents.backends.filesystem import FilesystemBackend

def project_backend_factory(state):
    return FilesystemBackend(root_dir="/home/user/my-project", virtual_mode=True)

agent = create_deep_agent(
    name="project_agent",
    backend=project_backend_factory,
)
```

---

## CompositeBackend

**Routes file operations to different backends based on path prefixes.**

### How It Works

- Each route is a `(prefix, backend)` pair.
- When a file operation is requested, the path is matched against the routes **longest prefix first**.
- If no route matches, the `default` backend is used.
- For cross-backend operations (`grep`, `glob`), all relevant backends are queried and results are merged (with deduplication for `glob`).

### Constructor Parameters

```python
CompositeBackend(
    default: Backend,                       # Fallback backend
    routes: dict[str, Backend] | None = None,  # Prefix → backend mapping
)
```

### When to Use

- **Mix of in-memory and filesystem storage** — e.g., workspace files on disk, scratch files in memory
- **Multi-project routing** — different directories backed by different storage
- **Selective persistence** — some paths persist to disk, others are ephemeral

### How Grep/Glob Span Multiple Backends

When `path` is `None` or `"/"`, `grep_raw()` and `glob_info()` query **all** backends (default + all routes) and merge the results:

- `grep_raw()` — concatenates all `GrepMatch` lists
- `glob_info()` — merges all `FileInfo` lists, deduplicating by path

When `path` is a specific directory, only the backend matching that path is queried.

### Example

```python
from adk_deepagents import create_deep_agent
from adk_deepagents.backends.state import StateBackend
from adk_deepagents.backends.filesystem import FilesystemBackend
from adk_deepagents.backends.composite import CompositeBackend

def composite_factory(state):
    return CompositeBackend(
        default=StateBackend(state),
        routes={
            "/workspace": FilesystemBackend(
                root_dir="/home/user/workspace",
                virtual_mode=True,
            ),
            "/workspace/data": FilesystemBackend(
                root_dir="/data/shared",
                virtual_mode=True,
            ),
        },
    )

agent = create_deep_agent(
    name="multi_backend_agent",
    instruction="You have access to a workspace on disk and in-memory scratch space.",
    backend=composite_factory,
)
```

In this setup:

- `/workspace/data/input.csv` → routes to the `/data/shared` filesystem (longest prefix match wins)
- `/workspace/src/main.py` → routes to the `/home/user/workspace` filesystem
- `/scratch/notes.md` → routes to the in-memory `StateBackend`
- `grep("TODO")` with no path → searches all three backends

---

## StoreBackend

**Cross-session persistence via a shared dict-based store.**

Unlike `StateBackend` (which is tied to a single session's state dict), `StoreBackend` uses a shared mutable dict that can be passed to multiple sessions, enabling agents to share data across threads and sessions.

### How It Works

- Files are stored under `store["files"]` in a shared dict.
- A `namespace` prefix provides path isolation — different agents or workflows can operate in isolated namespaces while sharing the same underlying store.
- Namespace prefixing/stripping is transparent: the agent sees paths like `/readme.md`, but internally they're stored as `/{namespace}/readme.md`.

### Constructor Parameters

```python
StoreBackend(
    store: dict[str, Any],         # Shared mutable dict
    namespace: str | None = None,  # Optional namespace prefix
)
```

### Namespace Prefixing and Stripping

When `namespace="project1"`:

| Agent sees | Stored as |
|-----------|-----------|
| `/readme.md` | `/project1/readme.md` |
| `/src/main.py` | `/project1/src/main.py` |
| `/` (ls) | Lists only files under `/project1/` |

The namespace prefix is automatically added on write/read and stripped from results. The agent never sees the namespace prefix.

### Cross-Session Persistence

The key feature of `StoreBackend` is that the same `store` dict can be shared across multiple sessions:

```python
shared_store = {}

# Session A writes a file
backend_a = StoreBackend(shared_store, namespace="project")
result = backend_a.write("/readme.md", "# My Project")
shared_store["files"].update(result.files_update)

# Session B reads the same file
backend_b = StoreBackend(shared_store, namespace="project")
content = backend_b.read("/readme.md")
# content contains "# My Project"
```

### When to Use

- **Sharing data across sessions** — multiple conversations accessing the same files
- **Multi-agent workflows** — agents in different sessions collaborating on shared files
- **Persistent in-memory storage** — when you want in-memory semantics but across sessions

### Example

```python
from adk_deepagents import create_deep_agent
from adk_deepagents.backends.store import StoreBackend

# Shared store — pass this to all sessions that need to share data
shared_store = {}

def store_backend_factory(state):
    return StoreBackend(shared_store, namespace="my_project")

agent = create_deep_agent(
    name="persistent_agent",
    instruction="You are a coding assistant with persistent file storage.",
    backend=store_backend_factory,
)
```

---

## Custom Backend

You can implement the `Backend` ABC to create a custom storage backend.

### How to Implement

Subclass `Backend` and implement all abstract methods:

```python
from adk_deepagents.backends.protocol import (
    Backend,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)


class S3Backend(Backend):
    """Example: Store files in an S3 bucket."""

    def __init__(self, bucket: str, prefix: str = ""):
        self._bucket = bucket
        self._prefix = prefix
        # Initialize your S3 client here

    def ls_info(self, path: str) -> list[FileInfo]:
        # List objects in the S3 bucket under the given prefix
        # Return a list of FileInfo dicts
        ...

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        # Fetch the object from S3, split into lines, apply pagination
        # Return formatted string with line numbers
        ...

    def write(self, file_path: str, content: str) -> WriteResult:
        # Put object to S3
        # Return WriteResult with files_update=None (direct persistence)
        ...

    def edit(
        self, file_path: str, old_string: str, new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        # Fetch object, perform string replacement, put back
        # Use perform_string_replacement() from backends.utils
        ...

    def grep_raw(
        self, pattern: str, path: str | None = None, glob: str | None = None,
    ) -> list[GrepMatch] | str:
        # Search objects for the pattern
        ...

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        # List objects matching the glob pattern
        ...

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        # Upload raw bytes to S3
        ...

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        # Download raw bytes from S3
        ...
```

### Tips

- Use `adk_deepagents.backends.utils` for shared utilities: `normalize_path()`, `validate_path()`, `format_content_with_line_numbers()`, `format_read_response()`, `perform_string_replacement()`, `glob_search_files()`, `grep_matches_from_files()`.
- If your backend does direct persistence (like `FilesystemBackend`), set `files_update=None` in `WriteResult` and `EditResult`.
- If your backend uses in-memory state (like `StateBackend`), populate `files_update` so the filesystem tools can merge changes into `tool_context.state["files"]`.
- Override the async methods (`als_info`, `aread`, etc.) if your backend has native async I/O. Otherwise, the default `asyncio.to_thread()` wrappers will be used.

### Using Your Custom Backend

```python
agent = create_deep_agent(
    name="s3_agent",
    backend=S3Backend(bucket="my-bucket", prefix="agents/workspace/"),
)
```

Or with a factory:

```python
def s3_factory(state):
    return S3Backend(bucket="my-bucket", prefix="agents/workspace/")

agent = create_deep_agent(
    name="s3_agent",
    backend=s3_factory,
)
```

---

## BackendFactory

A `BackendFactory` is a callable with the signature `(state: dict) -> Backend`. It's used for deferred backend construction — the backend is created when a tool is first invoked, using the session state dict.

```python
from adk_deepagents.backends.protocol import BackendFactory

# Type alias:
BackendFactory = Callable[[dict], Backend]
```

### When to Use

- **Session-dependent backends** — when the backend needs access to session state (like `StateBackend`)
- **Composite setups** — when you need to create a `StateBackend` from state and combine it with a `FilesystemBackend`
- **Lazy initialization** — deferring expensive backend construction until first use

### How It Works

When you pass a `BackendFactory` as the `backend` parameter to `create_deep_agent()`, it's stored in `state["_backend_factory"]` during the `before_agent_callback`. When a filesystem tool runs, it resolves the backend:

1. Check `tool_context.state["_backend"]` — if set, use it
2. Otherwise, call `tool_context.state["_backend_factory"](tool_context.state)` to create one
3. Cache the result in `tool_context.state["_backend"]`

### Examples

```python
from adk_deepagents.backends.state import StateBackend
from adk_deepagents.backends.filesystem import FilesystemBackend
from adk_deepagents.backends.composite import CompositeBackend

# Simple factory
def state_factory(state: dict) -> StateBackend:
    return StateBackend(state)

# Composite factory
def composite_factory(state: dict) -> CompositeBackend:
    return CompositeBackend(
        default=StateBackend(state),
        routes={
            "/project": FilesystemBackend(root_dir="./project", virtual_mode=True),
        },
    )

agent = create_deep_agent(backend=composite_factory)
```

If you pass a concrete `Backend` instance instead of a factory, `create_deep_agent()` wraps it in a factory automatically:

```python
# These are equivalent:
agent = create_deep_agent(backend=FilesystemBackend(root_dir="."))

# Internally becomes:
def wrapped_factory(_state, _b=backend):
    return _b
```
