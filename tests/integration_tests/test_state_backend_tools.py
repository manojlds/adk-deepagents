"""Integration tests — StateBackend + filesystem tools end-to-end.

Verifies that the filesystem tool functions (write_file, read_file, edit_file,
ls, glob, grep) work correctly when backed by a StateBackend via the
mock_tool_context fixture.  No API key or LLM required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from adk_deepagents.backends.state import StateBackend
from adk_deepagents.tools.filesystem import edit_file, glob, grep, ls, read_file, write_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx():
    """A fresh mock ToolContext with an empty StateBackend."""
    state: dict = {"files": {}}
    mock = MagicMock()
    mock.state = state
    mock.state["_backend"] = StateBackend(state)
    return mock


def _write(ctx, path: str, content: str) -> dict:
    """Write a file and return the tool result dict."""
    return write_file(path, content, ctx)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWriteReadRoundtrip:
    def test_write_then_read_roundtrip(self, ctx):
        result = _write(ctx, "/hello.txt", "Hello, World!")
        assert result["status"] == "success"

        read_result = read_file("/hello.txt", ctx)
        assert read_result["status"] == "success"
        assert "Hello, World!" in read_result["content"]

    def test_write_then_edit_then_read(self, ctx):
        _write(ctx, "/greet.txt", "Hello, World!")

        edit_result = edit_file("/greet.txt", "World", "Universe", ctx)
        assert edit_result["status"] == "success"
        assert edit_result["occurrences"] == 1

        read_result = read_file("/greet.txt", ctx)
        assert read_result["status"] == "success"
        assert "Hello, Universe!" in read_result["content"]

    def test_write_duplicate_fails(self, ctx):
        _write(ctx, "/dup.txt", "first")
        result = _write(ctx, "/dup.txt", "second")
        assert result["status"] == "error"
        assert "already exists" in result["message"].lower()

    def test_edit_nonexistent_fails(self, ctx):
        result = edit_file("/nope.txt", "a", "b", ctx)
        assert result["status"] == "error"
        assert "file_not_found" in result["message"]

    def test_edit_ambiguous_fails(self, ctx):
        _write(ctx, "/repeat.txt", "aaa")
        result = edit_file("/repeat.txt", "a", "b", ctx)
        assert result["status"] == "error"
        assert "appears 3 times" in result["message"]

    def test_edit_replace_all(self, ctx):
        _write(ctx, "/repeat.txt", "aaa")
        result = edit_file("/repeat.txt", "a", "b", ctx, replace_all=True)
        assert result["status"] == "success"
        assert result["occurrences"] == 3

        read_result = read_file("/repeat.txt", ctx)
        assert "bbb" in read_result["content"]


class TestListGlobGrep:
    def test_ls_after_write(self, ctx):
        _write(ctx, "/src/a.py", "a")
        _write(ctx, "/src/b.py", "b")
        _write(ctx, "/src/sub/c.py", "c")

        result = ls("/src", ctx)
        assert result["status"] == "success"
        paths = [e["path"] for e in result["entries"]]
        assert "/src/a.py" in paths
        assert "/src/b.py" in paths
        # Sub-directory should appear as a directory entry
        assert "/src/sub" in paths

    def test_glob_after_write(self, ctx):
        _write(ctx, "/proj/main.py", "main")
        _write(ctx, "/proj/utils.py", "utils")
        _write(ctx, "/proj/README.md", "readme")

        result = glob("**/*.py", ctx, path="/")
        assert result["status"] == "success"
        paths = [e["path"] for e in result["entries"]]
        assert "/proj/main.py" in paths
        assert "/proj/utils.py" in paths
        assert "/proj/README.md" not in paths

    def test_grep_after_write(self, ctx):
        _write(ctx, "/a.txt", "hello world")
        _write(ctx, "/b.txt", "goodbye world")
        _write(ctx, "/c.txt", "nothing here")

        result = grep("world", ctx)
        assert result["status"] == "success"
        assert "/a.txt" in result["result"]
        assert "/b.txt" in result["result"]
        assert "/c.txt" not in result["result"]

    def test_grep_output_modes(self, ctx):
        _write(ctx, "/x.txt", "foo\nbar\nfoo")

        content_result = grep("foo", ctx, output_mode="content")
        assert content_result["status"] == "success"
        assert "/x.txt:1:foo" in content_result["result"]
        assert "/x.txt:3:foo" in content_result["result"]

        fwm_result = grep("foo", ctx, output_mode="files_with_matches")
        assert fwm_result["status"] == "success"
        assert "/x.txt" in fwm_result["result"]

        count_result = grep("foo", ctx, output_mode="count")
        assert count_result["status"] == "success"
        assert "/x.txt: 2" in count_result["result"]


class TestPagination:
    def test_read_file_pagination(self, ctx):
        lines = "\n".join(f"line {i}" for i in range(100))
        _write(ctx, "/big.txt", lines)

        result = read_file("/big.txt", ctx, offset=0, limit=10)
        assert result["status"] == "success"
        assert "line 0" in result["content"]
        assert "line 9" in result["content"]
        # Should mention remaining lines
        assert "more lines" in result["content"]

        result2 = read_file("/big.txt", ctx, offset=90, limit=20)
        assert result2["status"] == "success"
        assert "line 99" in result2["content"]


class TestPathValidation:
    def test_path_validation_rejects_traversal(self, ctx):
        result = _write(ctx, "/../../etc/passwd", "bad")
        assert result["status"] == "error"
        assert "traversal" in result["message"].lower()

    def test_path_normalization(self, ctx):
        result = _write(ctx, "no_slash.txt", "content")
        assert result["status"] == "success"
        assert result["path"] == "/no_slash.txt"

        read_result = read_file("/no_slash.txt", ctx)
        assert read_result["status"] == "success"
        assert "content" in read_result["content"]
