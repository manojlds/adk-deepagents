"""Integration tests — CompositeBackend routing across multiple backends.

Verifies that CompositeBackend correctly routes operations by path prefix,
applies longest-prefix matching, falls back to the default backend, and
merges results for grep/glob across backends.
No API key or LLM required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from adk_deepagents.backends.composite import CompositeBackend
from adk_deepagents.backends.filesystem import FilesystemBackend
from adk_deepagents.backends.state import StateBackend
from adk_deepagents.backends.utils import create_file_data
from adk_deepagents.tools.filesystem import glob, grep, read_file, write_file

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def composite_ctx(tmp_path):
    """A mock ToolContext with a CompositeBackend routing /workspace to
    FilesystemBackend and everything else to StateBackend."""
    state: dict = {"files": {}}
    state_backend = StateBackend(state)
    fs_backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)

    composite = CompositeBackend(
        default=state_backend,
        routes={"/workspace": fs_backend},
    )

    ctx = MagicMock()
    ctx.state = state
    ctx.state["_backend"] = composite
    return ctx, tmp_path, state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRouteByPrefix:
    def test_route_by_prefix(self, composite_ctx):
        ctx, tmp_path, _ = composite_ctx

        # Write to /workspace → filesystem (path is resolved relative to root)
        result_ws = write_file("/workspace/file.txt", "on disk", ctx)
        assert result_ws["status"] == "success"
        assert (tmp_path / "workspace" / "file.txt").exists()

        # Write to /other → state
        result_other = write_file("/other/file.txt", "in state", ctx)
        assert result_other["status"] == "success"
        assert not (tmp_path / "other" / "file.txt").exists()


class TestLongestPrefixWins:
    def test_longest_prefix_wins(self, tmp_path):
        state: dict = {"files": {}}
        state_backend = StateBackend(state)

        short_dir = tmp_path / "short"
        short_dir.mkdir()
        long_dir = tmp_path / "long"
        long_dir.mkdir()

        short_backend = FilesystemBackend(root_dir=short_dir, virtual_mode=True)
        long_backend = FilesystemBackend(root_dir=long_dir, virtual_mode=True)

        composite = CompositeBackend(
            default=state_backend,
            routes={
                "/a": short_backend,
                "/a/b": long_backend,
            },
        )

        ctx = MagicMock()
        ctx.state = state
        ctx.state["_backend"] = composite

        # /a/b/c/file.txt should route to long_backend (/a/b)
        # FilesystemBackend resolves the full path relative to root_dir
        write_file("/a/b/c/file.txt", "deep", ctx)
        assert (long_dir / "a" / "b" / "c" / "file.txt").exists()
        assert not (short_dir / "b" / "c" / "file.txt").exists()

        # /a/file.txt should route to short_backend (/a)
        write_file("/a/file.txt", "shallow", ctx)
        assert (short_dir / "a" / "file.txt").exists()


class TestDefaultFallback:
    def test_default_fallback(self, composite_ctx):
        ctx, tmp_path, state = composite_ctx

        result = write_file("/unrouted/data.txt", "default land", ctx)
        assert result["status"] == "success"
        # Should be in state, not on disk
        assert "/unrouted/data.txt" in state["files"]
        assert not (tmp_path / "unrouted" / "data.txt").exists()


class TestGrepSpansBackends:
    def test_grep_spans_backends(self, composite_ctx):
        ctx, tmp_path, state = composite_ctx

        # File on filesystem backend
        (tmp_path / "search.txt").write_text("needle in haystack\n")

        # File in state backend
        state["files"]["/memo.txt"] = create_file_data("needle in a memo")

        result = grep("needle", ctx)
        assert result["status"] == "success"
        # Both should appear
        assert "memo.txt" in result["result"]
        # filesystem match (path depends on ripgrep availability)
        assert "search.txt" in result["result"]


class TestGlobSpansBackends:
    def test_glob_spans_backends(self, composite_ctx):
        ctx, tmp_path, state = composite_ctx

        # Filesystem backend
        (tmp_path / "app.py").write_text("app")

        # State backend
        state["files"]["/lib.py"] = create_file_data("lib")

        result = glob("**/*.py", ctx, path="/")
        assert result["status"] == "success"
        paths = [e["path"] for e in result["entries"]]
        assert "/lib.py" in paths
        # The workspace file should also appear
        assert any("app.py" in p for p in paths)

        # No duplicates
        assert len(paths) == len(set(paths))


class TestWriteToCorrectBackend:
    def test_write_to_correct_backend(self, composite_ctx):
        ctx, tmp_path, state = composite_ctx

        write_file("/workspace/code.py", "print('hi')", ctx)
        write_file("/notes/todo.txt", "do stuff", ctx)

        assert (tmp_path / "workspace" / "code.py").exists()
        assert "/notes/todo.txt" in state["files"]


class TestReadFromCorrectBackend:
    def test_read_from_correct_backend(self, composite_ctx):
        ctx, tmp_path, state = composite_ctx

        # FilesystemBackend resolves /workspace/disk.txt to <root>/workspace/disk.txt
        ws_dir = tmp_path / "workspace"
        ws_dir.mkdir(parents=True, exist_ok=True)
        (ws_dir / "disk.txt").write_text("from disk")
        state["files"]["/mem.txt"] = create_file_data("from memory")

        disk_result = read_file("/workspace/disk.txt", ctx)
        assert disk_result["status"] == "success"
        assert "from disk" in disk_result["content"]

        mem_result = read_file("/mem.txt", ctx)
        assert mem_result["status"] == "success"
        assert "from memory" in mem_result["content"]
