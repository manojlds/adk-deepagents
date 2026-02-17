"""Tests for StoreBackend."""

import asyncio

import pytest

from adk_deepagents.backends.store import StoreBackend
from adk_deepagents.backends.utils import create_file_data

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def shared_store() -> dict:
    """A shared store dict pre-populated with files in the 'proj' namespace."""
    return {
        "files": {
            "/proj/hello.txt": create_file_data("Hello, World!"),
            "/proj/src/main.py": create_file_data("def main():\n    print('hello')\n"),
            "/proj/src/utils.py": create_file_data("def add(a, b):\n    return a + b\n"),
            "/proj/docs/readme.md": create_file_data("# My Project\n\nA description."),
        }
    }


@pytest.fixture
def store_backend(shared_store) -> StoreBackend:
    """A StoreBackend with namespace 'proj' and pre-populated files."""
    return StoreBackend(shared_store, namespace="proj")


@pytest.fixture
def no_ns_store() -> dict:
    """A shared store with no namespace."""
    return {
        "files": {
            "/hello.txt": create_file_data("Hello, World!"),
            "/src/main.py": create_file_data("def main():\n    print('hello')\n"),
        }
    }


@pytest.fixture
def no_ns_backend(no_ns_store) -> StoreBackend:
    """A StoreBackend without namespace."""
    return StoreBackend(no_ns_store)


# ---------------------------------------------------------------------------
# ls_info
# ---------------------------------------------------------------------------


class TestStoreBackendLs:
    def test_ls_root(self, store_backend):
        entries = store_backend.ls_info("/")
        paths = [e["path"] for e in entries]
        assert "/docs" in paths
        assert "/hello.txt" in paths
        assert "/src" in paths

    def test_ls_subdirectory(self, store_backend):
        entries = store_backend.ls_info("/src")
        paths = [e["path"] for e in entries]
        assert "/src/main.py" in paths
        assert "/src/utils.py" in paths
        assert len(entries) == 2

    def test_ls_file(self, store_backend):
        entries = store_backend.ls_info("/hello.txt")
        assert len(entries) == 1
        assert entries[0]["path"] == "/hello.txt"
        assert entries[0]["is_dir"] is False

    def test_ls_nonexistent(self, store_backend):
        entries = store_backend.ls_info("/nonexistent")
        assert entries == []

    def test_ls_no_namespace(self, no_ns_backend):
        entries = no_ns_backend.ls_info("/")
        paths = [e["path"] for e in entries]
        assert "/hello.txt" in paths
        assert "/src" in paths


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


class TestStoreBackendRead:
    def test_read_existing(self, store_backend):
        content = store_backend.read("/hello.txt")
        assert "Hello, World!" in content

    def test_read_with_line_numbers(self, store_backend):
        content = store_backend.read("/src/main.py")
        assert "1\t" in content or "1" in content

    def test_read_nonexistent(self, store_backend):
        content = store_backend.read("/missing.txt")
        assert "Error" in content or "not found" in content

    def test_read_with_offset(self, store_backend):
        content = store_backend.read("/src/main.py", offset=1, limit=1)
        assert "print" in content


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


class TestStoreBackendWrite:
    def test_write_new_file(self, store_backend):
        result = store_backend.write("/new.txt", "new content")
        assert result.error is None
        assert result.path == "/new.txt"
        assert result.files_update is not None

    def test_write_existing_file_fails(self, store_backend):
        result = store_backend.write("/hello.txt", "overwrite")
        assert result.error == "already_exists"

    def test_write_updates_store(self, shared_store):
        backend = StoreBackend(shared_store, namespace="proj")
        result = backend.write("/created.txt", "hello")
        assert result.files_update is not None
        # Apply update
        shared_store["files"].update(result.files_update)
        content = backend.read("/created.txt")
        assert "hello" in content


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


class TestStoreBackendEdit:
    def test_edit_existing(self, store_backend):
        result = store_backend.edit("/hello.txt", "World", "Store")
        assert result.error is None
        assert result.occurrences == 1
        assert result.files_update is not None

    def test_edit_nonexistent(self, store_backend):
        result = store_backend.edit("/missing.txt", "a", "b")
        assert result.error is not None

    def test_edit_pattern_not_found(self, store_backend):
        result = store_backend.edit("/hello.txt", "ZZZZZ", "replacement")
        assert result.error is not None

    def test_edit_replace_all(self, shared_store):
        shared_store["files"]["/proj/repeat.txt"] = create_file_data("foo foo foo")
        backend = StoreBackend(shared_store, namespace="proj")
        result = backend.edit("/repeat.txt", "foo", "bar", replace_all=True)
        assert result.error is None
        assert result.occurrences == 3


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


class TestStoreBackendGrep:
    def test_grep_finds_matches(self, store_backend):
        matches = store_backend.grep_raw("def")
        assert isinstance(matches, list)
        assert len(matches) >= 2

    def test_grep_with_path(self, store_backend):
        matches = store_backend.grep_raw("def", path="/src/main.py")
        assert isinstance(matches, list)
        assert all(m["path"] == "/src/main.py" for m in matches)

    def test_grep_no_matches(self, store_backend):
        matches = store_backend.grep_raw("nonexistent_pattern_xyz")
        assert isinstance(matches, list)
        assert len(matches) == 0


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------


class TestStoreBackendGlob:
    def test_glob_python_files(self, store_backend):
        entries = store_backend.glob_info("**/*.py", "/")
        paths = [e["path"] for e in entries]
        assert "/src/main.py" in paths
        assert "/src/utils.py" in paths

    def test_glob_markdown_files(self, store_backend):
        entries = store_backend.glob_info("**/*.md", "/")
        paths = [e["path"] for e in entries]
        assert "/docs/readme.md" in paths

    def test_glob_no_matches(self, store_backend):
        entries = store_backend.glob_info("**/*.rs", "/")
        assert len(entries) == 0


# ---------------------------------------------------------------------------
# download / upload
# ---------------------------------------------------------------------------


class TestStoreBackendDownload:
    def test_download_existing(self, store_backend):
        results = store_backend.download_files(["/hello.txt"])
        assert len(results) == 1
        assert results[0].content is not None
        assert b"Hello" in results[0].content

    def test_download_nonexistent(self, store_backend):
        results = store_backend.download_files(["/missing.txt"])
        assert results[0].error is not None


class TestStoreBackendUpload:
    def test_upload_new_file(self, store_backend):
        results = store_backend.upload_files([("uploaded.txt", b"uploaded content")])
        assert len(results) == 1
        assert results[0].error is None

    def test_upload_existing_fails(self, store_backend):
        results = store_backend.upload_files([("/hello.txt", b"duplicate")])
        assert results[0].error == "already_exists"


# ---------------------------------------------------------------------------
# Cross-thread persistence
# ---------------------------------------------------------------------------


class TestStoreBackendCrossThread:
    def test_cross_session_read(self):
        """Files written in session A can be read in session B."""
        shared = {}
        backend_a = StoreBackend(shared, namespace="shared")
        backend_b = StoreBackend(shared, namespace="shared")

        # Write via backend_a
        result = backend_a.write("/data.txt", "shared data")
        assert result.error is None
        shared["files"].update(result.files_update)

        # Read via backend_b
        content = backend_b.read("/data.txt")
        assert "shared data" in content

    def test_namespace_isolation(self):
        """Different namespaces cannot see each other's files."""
        shared = {}
        backend_ns1 = StoreBackend(shared, namespace="ns1")
        backend_ns2 = StoreBackend(shared, namespace="ns2")

        result = backend_ns1.write("/secret.txt", "ns1 data")
        shared["files"].update(result.files_update)

        # ns2 should not see ns1's file
        content = backend_ns2.read("/secret.txt")
        assert "not found" in content

        # ns1 should see it
        content = backend_ns1.read("/secret.txt")
        assert "ns1 data" in content

    def test_no_namespace_shared(self):
        """Without namespace, files are in the global scope."""
        shared = {}
        backend_a = StoreBackend(shared)
        backend_b = StoreBackend(shared)

        result = backend_a.write("/global.txt", "global data")
        shared["files"].update(result.files_update)

        content = backend_b.read("/global.txt")
        assert "global data" in content

    def test_empty_store_initializes_files(self):
        """An empty store dict gets a 'files' key automatically."""
        shared = {}
        StoreBackend(shared)
        assert "files" in shared


# ---------------------------------------------------------------------------
# CompositeBackend routing
# ---------------------------------------------------------------------------


class TestStoreBackendCompositeRouting:
    def test_composite_routes_to_store(self):
        """CompositeBackend can route to StoreBackend by path prefix."""
        from adk_deepagents.backends.composite import CompositeBackend
        from adk_deepagents.backends.state import StateBackend

        shared_store = {}
        store = StoreBackend(shared_store, namespace="store")
        state = StateBackend({"files": {}})

        composite = CompositeBackend(default=state, routes={"/store": store})

        # Write to /store prefix → StoreBackend
        result = composite.write("/store/data.txt", "stored content")
        assert result.error is None
        shared_store["files"].update(result.files_update)

        # Read back via composite
        content = composite.read("/store/data.txt")
        assert "stored content" in content

        # Write to default → StateBackend
        result_default = composite.write("/local.txt", "local content")
        assert result_default.error is None


# ---------------------------------------------------------------------------
# Async method tests
# ---------------------------------------------------------------------------


class TestStoreBackendAsyncMethods:
    """Verify async overrides produce the same results as sync and
    do NOT use asyncio.to_thread.
    """

    async def test_als_info(self, store_backend):
        sync_result = store_backend.ls_info("/")
        async_result = await store_backend.als_info("/")
        assert async_result == sync_result

    async def test_aread(self, store_backend):
        sync_result = store_backend.read("/hello.txt")
        async_result = await store_backend.aread("/hello.txt")
        assert async_result == sync_result

    async def test_aread_with_offset(self, store_backend):
        sync_result = store_backend.read("/src/main.py", offset=1, limit=1)
        async_result = await store_backend.aread("/src/main.py", offset=1, limit=1)
        assert async_result == sync_result

    async def test_awrite(self, store_backend):
        result = await store_backend.awrite("/async_file.txt", "async content")
        assert result.error is None
        assert result.path == "/async_file.txt"
        assert result.files_update is not None

    async def test_awrite_existing_fails(self, store_backend):
        result = await store_backend.awrite("/hello.txt", "overwrite")
        assert result.error == "already_exists"

    async def test_aedit(self, store_backend):
        result = await store_backend.aedit("/hello.txt", "World", "Async")
        assert result.error is None
        assert result.occurrences == 1
        assert result.files_update is not None

    async def test_aedit_nonexistent(self, store_backend):
        result = await store_backend.aedit("/missing.txt", "a", "b")
        assert result.error is not None

    async def test_agrep_raw(self, store_backend):
        sync_result = store_backend.grep_raw("def")
        async_result = await store_backend.agrep_raw("def")
        assert async_result == sync_result

    async def test_agrep_raw_with_path(self, store_backend):
        sync_result = store_backend.grep_raw("def", path="/src/main.py")
        async_result = await store_backend.agrep_raw("def", path="/src/main.py")
        assert async_result == sync_result

    async def test_aglob_info(self, store_backend):
        sync_result = store_backend.glob_info("**/*.py", "/")
        async_result = await store_backend.aglob_info("**/*.py", "/")
        assert async_result == sync_result

    async def test_no_asyncio_to_thread(self, store_backend, monkeypatch):
        """Verify StoreBackend async methods do NOT call asyncio.to_thread."""
        original_to_thread = asyncio.to_thread
        called = False

        async def patched_to_thread(*args, **kwargs):
            nonlocal called
            called = True
            return await original_to_thread(*args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", patched_to_thread)

        await store_backend.als_info("/")
        await store_backend.aread("/hello.txt")
        await store_backend.awrite("/no_thread.txt", "test")
        await store_backend.aedit("/hello.txt", "World", "Direct")
        await store_backend.agrep_raw("def")
        await store_backend.aglob_info("**/*.py")

        assert not called, "StoreBackend async methods should not use asyncio.to_thread"
