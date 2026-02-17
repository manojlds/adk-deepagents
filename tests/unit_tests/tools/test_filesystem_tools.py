"""Tests for filesystem tools."""

import base64
from unittest.mock import MagicMock

from adk_deepagents.backends.protocol import FileDownloadResponse
from adk_deepagents.tools.filesystem import (
    edit_file,
    glob,
    grep,
    ls,
    read_file,
    write_file,
)


class TestLs:
    def test_ls_root(self, mock_tool_context):
        result = ls("/", mock_tool_context)
        assert result["status"] == "success"
        paths = [e["path"] for e in result["entries"]]
        assert "/src" in paths
        assert "/hello.txt" in paths

    def test_ls_invalid_path(self, mock_tool_context):
        result = ls("../escape", mock_tool_context)
        assert result["status"] == "error"


class TestReadFile:
    def test_read_existing(self, mock_tool_context):
        result = read_file("/hello.txt", mock_tool_context)
        assert result["status"] == "success"
        assert "Hello" in result["content"]

    def test_read_nonexistent(self, mock_tool_context):
        result = read_file("/missing.txt", mock_tool_context)
        assert result["status"] == "error"

    def test_read_traversal_blocked(self, mock_tool_context):
        result = read_file("/../etc/passwd", mock_tool_context)
        assert result["status"] == "error"

    def test_read_image_png(self, mock_tool_context):
        raw_bytes = b"\x89PNG\r\n\x1a\nfakedata"
        backend = mock_tool_context.state["_backend"]
        backend.download_files = MagicMock(
            return_value=[FileDownloadResponse(path="/photo.png", content=raw_bytes)]
        )
        result = read_file("/photo.png", mock_tool_context)
        assert result["status"] == "success"
        content = result["content"]
        assert content["type"] == "image"
        assert content["media_type"] == "image/png"
        assert content["data"] == base64.b64encode(raw_bytes).decode("ascii")

    def test_read_image_jpeg(self, mock_tool_context):
        raw_bytes = b"\xff\xd8\xff\xe0jpegdata"
        backend = mock_tool_context.state["_backend"]
        backend.download_files = MagicMock(
            return_value=[FileDownloadResponse(path="/photo.jpg", content=raw_bytes)]
        )
        result = read_file("/photo.jpg", mock_tool_context)
        assert result["status"] == "success"
        assert result["content"]["media_type"] == "image/jpeg"

    def test_read_image_not_found(self, mock_tool_context):
        backend = mock_tool_context.state["_backend"]
        backend.download_files = MagicMock(
            return_value=[FileDownloadResponse(path="/missing.png", error="file_not_found")]
        )
        result = read_file("/missing.png", mock_tool_context)
        assert result["status"] == "error"

    def test_read_non_image_unchanged(self, mock_tool_context):
        result = read_file("/hello.txt", mock_tool_context)
        assert result["status"] == "success"
        assert isinstance(result["content"], str)


class TestWriteFile:
    def test_write_new(self, mock_tool_context):
        result = write_file("/new_file.txt", "new content", mock_tool_context)
        assert result["status"] == "success"
        assert result["path"] == "/new_file.txt"

    def test_write_existing_fails(self, mock_tool_context):
        result = write_file("/hello.txt", "overwrite", mock_tool_context)
        assert result["status"] == "error"

    def test_write_invalid_path(self, mock_tool_context):
        result = write_file("~/.ssh/keys", "content", mock_tool_context)
        assert result["status"] == "error"


class TestEditFile:
    def test_edit_existing(self, mock_tool_context):
        result = edit_file("/hello.txt", "World", "ADK", mock_tool_context)
        assert result["status"] == "success"
        assert result["occurrences"] == 1

    def test_edit_nonexistent(self, mock_tool_context):
        result = edit_file("/missing.txt", "a", "b", mock_tool_context)
        assert result["status"] == "error"

    def test_edit_pattern_not_found(self, mock_tool_context):
        result = edit_file("/hello.txt", "ZZZZZ", "replacement", mock_tool_context)
        assert result["status"] == "error"


class TestGlob:
    def test_glob_python(self, mock_tool_context):
        result = glob("**/*.py", mock_tool_context)
        assert result["status"] == "success"
        paths = [e["path"] for e in result["entries"]]
        assert "/src/main.py" in paths

    def test_glob_no_matches(self, mock_tool_context):
        result = glob("**/*.rs", mock_tool_context)
        assert result["status"] == "success"
        assert len(result["entries"]) == 0


class TestGrep:
    def test_grep_finds_pattern(self, mock_tool_context):
        result = grep("def", mock_tool_context)
        assert result["status"] == "success"
        assert "main.py" in result["result"] or "utils.py" in result["result"]

    def test_grep_no_matches(self, mock_tool_context):
        result = grep("xyznonexistent", mock_tool_context)
        assert result["status"] == "success"
        assert "No matches" in result["result"]

    def test_grep_with_path(self, mock_tool_context):
        result = grep("def", mock_tool_context, path="/src/main.py")
        assert result["status"] == "success"
