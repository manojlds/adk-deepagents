"""Tests for FilesystemBackend."""

import os
import tempfile
from pathlib import Path

import pytest

from adk_deepagents.backends.filesystem import FilesystemBackend


@pytest.fixture
def tmp_root(tmp_path):
    """Create a temp directory with some test files."""
    (tmp_path / "hello.txt").write_text("Hello, World!")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main():\n    print('hello')\n")
    (tmp_path / "src" / "utils.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "readme.md").write_text("# My Project\n\nA description.")
    return tmp_path


@pytest.fixture
def fs_backend(tmp_root):
    """A FilesystemBackend in virtual mode rooted at tmp_root."""
    return FilesystemBackend(root_dir=tmp_root, virtual_mode=True)


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------


class TestLs:
    def test_ls_root(self, fs_backend):
        entries = fs_backend.ls_info("/")
        names = [Path(e["path"]).name for e in entries]
        assert "docs" in names
        assert "hello.txt" in names
        assert "src" in names

    def test_ls_subdirectory(self, fs_backend):
        entries = fs_backend.ls_info("/src")
        names = [Path(e["path"]).name for e in entries]
        assert "main.py" in names
        assert "utils.py" in names
        assert len(entries) == 2

    def test_ls_file(self, fs_backend):
        entries = fs_backend.ls_info("/hello.txt")
        assert len(entries) == 1
        assert entries[0]["is_dir"] is False
        assert entries[0]["size"] == 13  # "Hello, World!" is 13 bytes

    def test_ls_nonexistent(self, fs_backend):
        entries = fs_backend.ls_info("/nonexistent")
        assert entries == []

    def test_ls_has_modified_at(self, fs_backend):
        entries = fs_backend.ls_info("/hello.txt")
        assert "modified_at" in entries[0]
        assert entries[0]["modified_at"]  # non-empty string


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


class TestRead:
    def test_read_existing(self, fs_backend):
        content = fs_backend.read("/hello.txt")
        assert "Hello, World!" in content

    def test_read_with_line_numbers(self, fs_backend):
        content = fs_backend.read("/src/main.py")
        # Should contain line numbers
        assert "1" in content
        assert "def main" in content

    def test_read_nonexistent(self, fs_backend):
        content = fs_backend.read("/missing.txt")
        assert "Error" in content

    def test_read_with_offset(self, fs_backend):
        content = fs_backend.read("/src/main.py", offset=1, limit=1)
        assert "print" in content

    def test_read_directory_gives_error(self, fs_backend):
        content = fs_backend.read("/src")
        assert "Error" in content

    def test_read_empty_file(self, tmp_root):
        (tmp_root / "empty.txt").write_text("")
        backend = FilesystemBackend(root_dir=tmp_root, virtual_mode=True)
        content = backend.read("/empty.txt")
        assert "empty" in content.lower() or "System reminder" in content


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


class TestWrite:
    def test_write_new_file(self, fs_backend, tmp_root):
        result = fs_backend.write("/new.txt", "new content")
        assert result.error is None
        assert result.path == "/new.txt"
        # files_update is None for filesystem backend
        assert result.files_update is None
        # Verify file was actually written
        assert (tmp_root / "new.txt").read_text() == "new content"

    def test_write_existing_file_fails(self, fs_backend):
        result = fs_backend.write("/hello.txt", "overwrite")
        assert result.error is not None

    def test_write_creates_parent_dirs(self, fs_backend, tmp_root):
        result = fs_backend.write("/deep/nested/file.txt", "nested content")
        assert result.error is None
        assert (tmp_root / "deep" / "nested" / "file.txt").read_text() == "nested content"


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


class TestEdit:
    def test_edit_existing(self, fs_backend, tmp_root):
        result = fs_backend.edit("/hello.txt", "World", "ADK")
        assert result.error is None
        assert result.occurrences == 1
        assert result.files_update is None  # filesystem backend
        assert (tmp_root / "hello.txt").read_text() == "Hello, ADK!"

    def test_edit_nonexistent(self, fs_backend):
        result = fs_backend.edit("/missing.txt", "a", "b")
        assert result.error is not None

    def test_edit_pattern_not_found(self, fs_backend):
        result = fs_backend.edit("/hello.txt", "ZZZZZ", "replacement")
        assert result.error is not None

    def test_edit_multiple_without_replace_all(self, tmp_root):
        (tmp_root / "repeat.txt").write_text("foo foo foo")
        backend = FilesystemBackend(root_dir=tmp_root, virtual_mode=True)
        result = backend.edit("/repeat.txt", "foo", "bar")
        assert result.error is not None

    def test_edit_replace_all(self, tmp_root):
        (tmp_root / "repeat.txt").write_text("foo foo foo")
        backend = FilesystemBackend(root_dir=tmp_root, virtual_mode=True)
        result = backend.edit("/repeat.txt", "foo", "bar", replace_all=True)
        assert result.error is None
        assert result.occurrences == 3
        assert (tmp_root / "repeat.txt").read_text() == "bar bar bar"


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


class TestGrep:
    def test_grep_finds_matches(self, fs_backend):
        matches = fs_backend.grep_raw("def")
        assert isinstance(matches, list)
        assert len(matches) >= 2  # main.py and utils.py

    def test_grep_with_path(self, fs_backend):
        matches = fs_backend.grep_raw("def", path="/src/main.py")
        assert isinstance(matches, list)
        assert len(matches) >= 1
        assert all("main.py" in m["path"] for m in matches)

    def test_grep_no_matches(self, fs_backend):
        matches = fs_backend.grep_raw("nonexistent_pattern_xyz")
        assert isinstance(matches, list)
        assert len(matches) == 0

    def test_grep_with_glob_filter(self, fs_backend):
        matches = fs_backend.grep_raw("def", glob="*.py")
        assert isinstance(matches, list)
        assert len(matches) >= 2
        assert all(m["path"].endswith(".py") for m in matches)


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------


class TestGlob:
    def test_glob_python_files(self, fs_backend):
        entries = fs_backend.glob_info("*.py", "/src")
        assert len(entries) == 2
        paths = [e["path"] for e in entries]
        assert any("main.py" in p for p in paths)
        assert any("utils.py" in p for p in paths)

    def test_glob_recursive(self, fs_backend):
        entries = fs_backend.glob_info("**/*.py", "/")
        assert len(entries) >= 2

    def test_glob_markdown(self, fs_backend):
        entries = fs_backend.glob_info("*.md", "/docs")
        assert len(entries) == 1
        assert any("readme.md" in e["path"] for e in entries)

    def test_glob_no_matches(self, fs_backend):
        entries = fs_backend.glob_info("*.rs", "/")
        assert len(entries) == 0


# ---------------------------------------------------------------------------
# download / upload
# ---------------------------------------------------------------------------


class TestDownloadUpload:
    def test_download_existing(self, fs_backend):
        results = fs_backend.download_files(["/hello.txt"])
        assert len(results) == 1
        assert results[0].content is not None
        assert b"Hello" in results[0].content

    def test_download_nonexistent(self, fs_backend):
        results = fs_backend.download_files(["/missing.txt"])
        assert results[0].error is not None

    def test_upload(self, fs_backend, tmp_root):
        results = fs_backend.upload_files([("/uploaded.txt", b"uploaded content")])
        assert len(results) == 1
        assert results[0].error is None
        assert (tmp_root / "uploaded.txt").read_bytes() == b"uploaded content"


# ---------------------------------------------------------------------------
# virtual mode security
# ---------------------------------------------------------------------------


class TestVirtualMode:
    def test_escape_prevented(self, tmp_root):
        backend = FilesystemBackend(root_dir=tmp_root, virtual_mode=True)
        with pytest.raises(ValueError, match="escapes root"):
            backend._resolve_path("/../../../etc/passwd")

    def test_normal_path_works(self, tmp_root):
        backend = FilesystemBackend(root_dir=tmp_root, virtual_mode=True)
        resolved = backend._resolve_path("/hello.txt")
        assert resolved == tmp_root / "hello.txt"

    def test_root_path_resolves_to_root(self, tmp_root):
        backend = FilesystemBackend(root_dir=tmp_root, virtual_mode=True)
        resolved = backend._resolve_path("/")
        assert resolved == tmp_root


# ---------------------------------------------------------------------------
# non-virtual mode
# ---------------------------------------------------------------------------


class TestNonVirtualMode:
    def test_absolute_path(self, tmp_root):
        backend = FilesystemBackend(root_dir=tmp_root, virtual_mode=False)
        file_path = str(tmp_root / "hello.txt")
        content = backend.read(file_path)
        assert "Hello, World!" in content

    def test_relative_path(self, tmp_root):
        backend = FilesystemBackend(root_dir=tmp_root, virtual_mode=False)
        content = backend.read("hello.txt")
        assert "Hello, World!" in content
