"""Tests for backends/protocol.py dataclasses and types."""

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
