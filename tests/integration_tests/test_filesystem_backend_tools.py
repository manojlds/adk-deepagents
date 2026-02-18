"""Integration tests — FilesystemBackend + filesystem tools with real disk I/O.

Uses ``tmp_path`` to create a temporary directory and verifies that
filesystem tools operate correctly against real files on disk.
No API key or LLM required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from adk_deepagents.backends.filesystem import FilesystemBackend
from adk_deepagents.tools.filesystem import edit_file, glob, grep, ls, read_file, write_file

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fs_tool_context(tmp_path):
    """A mock ToolContext backed by a FilesystemBackend in virtual mode."""
    ctx = MagicMock()
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    ctx.state = {"_backend": backend}
    return ctx, tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWriteReadRoundtrip:
    def test_write_read_roundtrip(self, fs_tool_context):
        ctx, tmp_path = fs_tool_context

        result = write_file("/hello.txt", "Hello from disk!", ctx)
        assert result["status"] == "success"

        read_result = read_file("/hello.txt", ctx)
        assert read_result["status"] == "success"
        assert "Hello from disk!" in read_result["content"]

        # Verify on actual filesystem
        assert (tmp_path / "hello.txt").read_text() == "Hello from disk!"


class TestEditFileOnDisk:
    def test_edit_file_on_disk(self, fs_tool_context):
        ctx, tmp_path = fs_tool_context

        write_file("/doc.txt", "old content here", ctx)

        result = edit_file("/doc.txt", "old content", "new content", ctx)
        assert result["status"] == "success"
        assert result["occurrences"] == 1

        # Verify on disk
        assert "new content here" in (tmp_path / "doc.txt").read_text()


class TestLsDirectory:
    def test_ls_directory(self, fs_tool_context):
        ctx, tmp_path = fs_tool_context

        # Create files directly on disk
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "c.py").write_text("c")

        result = ls("/", ctx)
        assert result["status"] == "success"
        paths = [e["path"] for e in result["entries"]]
        assert "/a.py" in paths
        assert "/b.py" in paths
        assert "/sub" in paths


class TestGlobPatternMatching:
    def test_glob_pattern_matching(self, fs_tool_context):
        ctx, tmp_path = fs_tool_context

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("main")
        (tmp_path / "src" / "utils.py").write_text("utils")
        (tmp_path / "src" / "readme.md").write_text("readme")

        result = glob("**/*.py", ctx, path="/")
        assert result["status"] == "success"
        paths = [e["path"] for e in result["entries"]]
        assert any("main.py" in p for p in paths)
        assert any("utils.py" in p for p in paths)
        assert not any("readme.md" in p for p in paths)


class TestGrepSearch:
    def test_grep_search(self, fs_tool_context):
        ctx, tmp_path = fs_tool_context

        (tmp_path / "a.txt").write_text("hello world\n")
        (tmp_path / "b.txt").write_text("goodbye world\n")
        (tmp_path / "c.txt").write_text("nothing here\n")

        result = grep("world", ctx)
        assert result["status"] == "success"
        assert "a.txt" in result["result"]
        assert "b.txt" in result["result"]
        assert "c.txt" not in result["result"]


class TestVirtualModePathContainment:
    def test_virtual_mode_path_containment(self, fs_tool_context):
        ctx, _ = fs_tool_context

        result = write_file("/../../../etc/passwd", "bad", ctx)
        assert result["status"] == "error"


class TestNonVirtualMode:
    def test_non_virtual_mode(self, tmp_path):
        ctx = MagicMock()
        backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
        ctx.state = {"_backend": backend}

        # Write using an absolute path inside tmp_path
        abs_path = str(tmp_path / "abs_file.txt")
        result = write_file(abs_path, "absolute content", ctx)
        assert result["status"] == "success"

        read_result = read_file(abs_path, ctx)
        assert read_result["status"] == "success"
        assert "absolute content" in read_result["content"]


class TestLargeFileSizeLimit:
    def test_large_file_size_limit(self, tmp_path):
        ctx = MagicMock()
        backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True, max_file_size_mb=0.001)
        ctx.state = {"_backend": backend}

        # Write a file that exceeds the tiny limit (~1 KB)
        large_content = "x" * 2000
        (tmp_path / "large.txt").write_text(large_content)

        result = read_file("/large.txt", ctx)
        assert result["status"] == "error"
        assert "too large" in result["message"].lower()
