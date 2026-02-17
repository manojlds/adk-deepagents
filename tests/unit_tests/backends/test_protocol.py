"""Tests for backends/protocol.py dataclasses and types."""

import pytest

from adk_deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)


def test_file_info_required_keys():
    fi = FileInfo(path="/test.txt")
    assert fi["path"] == "/test.txt"


def test_file_info_optional_keys():
    fi = FileInfo(path="/test.txt", is_dir=False, size=100, modified_at="2025-01-01")
    assert fi["is_dir"] is False
    assert fi["size"] == 100


def test_grep_match():
    gm = GrepMatch(path="/file.py", line=10, text="hello world")
    assert gm["path"] == "/file.py"
    assert gm["line"] == 10


def test_write_result_defaults():
    wr = WriteResult()
    assert wr.error is None
    assert wr.path == ""
    assert wr.files_update is None


def test_write_result_with_update():
    wr = WriteResult(
        path="/new.txt",
        files_update={"/new.txt": FileData(content=["hi"], created_at="t", modified_at="t")},
    )
    assert wr.error is None
    assert wr.files_update is not None
    assert "/new.txt" in wr.files_update


def test_edit_result():
    er = EditResult(path="/file.txt", occurrences=3)
    assert er.occurrences == 3
    assert er.error is None


def test_execute_response():
    er = ExecuteResponse(output="done", exit_code=0, truncated=False)
    assert er.output == "done"
    assert er.exit_code == 0


def test_file_download_response():
    fdr = FileDownloadResponse(path="/file.txt", content=b"hello")
    assert fdr.content == b"hello"
    assert fdr.error is None


def test_file_upload_response():
    fur = FileUploadResponse(path="/file.txt")
    assert fur.error is None


# ---------------------------------------------------------------------------
# Async method tests (Backend ABC default wrappers via asyncio.to_thread)
# ---------------------------------------------------------------------------


class TestBackendAsyncMethods:
    """Verify that the default async wrappers on Backend produce
    the same results as their sync counterparts.

    Uses StateBackend as a concrete Backend implementation.
    """

    @pytest.fixture
    def backend(self, state_backend):
        return state_backend

    async def test_als_info(self, backend):
        sync_result = backend.ls_info("/")
        async_result = await backend.als_info("/")
        assert async_result == sync_result

    async def test_aread(self, backend):
        sync_result = backend.read("/hello.txt")
        async_result = await backend.aread("/hello.txt")
        assert async_result == sync_result

    async def test_aread_with_offset(self, backend):
        sync_result = backend.read("/src/main.py", offset=1, limit=1)
        async_result = await backend.aread("/src/main.py", offset=1, limit=1)
        assert async_result == sync_result

    async def test_awrite(self, backend):
        sync_result = backend.write("/async_new.txt", "async content")
        async_result = await backend.awrite("/async_new2.txt", "async content")
        assert async_result.error == sync_result.error
        assert async_result.files_update is not None

    async def test_aedit(self, backend):
        sync_result = backend.edit("/src/main.py", "hello", "sync_val")
        assert sync_result.error is None
        # Apply sync update so aedit can see it
        backend._state["files"].update(sync_result.files_update)
        async_result = await backend.aedit("/src/main.py", "sync_val", "async_val")
        assert async_result.error is None
        assert async_result.occurrences == sync_result.occurrences

    async def test_agrep_raw(self, backend):
        sync_result = backend.grep_raw("def")
        async_result = await backend.agrep_raw("def")
        assert async_result == sync_result

    async def test_aglob_info(self, backend):
        sync_result = backend.glob_info("**/*.py", "/")
        async_result = await backend.aglob_info("**/*.py", "/")
        assert async_result == sync_result
