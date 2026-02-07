"""Tests for StateBackend."""

import pytest

from adk_deepagents.backends.state import StateBackend
from adk_deepagents.backends.utils import create_file_data


class TestStateBackendLs:
    def test_ls_root(self, state_backend):
        entries = state_backend.ls_info("/")
        paths = [e["path"] for e in entries]
        assert "/docs" in paths
        assert "/hello.txt" in paths
        assert "/src" in paths

    def test_ls_subdirectory(self, state_backend):
        entries = state_backend.ls_info("/src")
        paths = [e["path"] for e in entries]
        assert "/src/main.py" in paths
        assert "/src/utils.py" in paths
        assert len(entries) == 2

    def test_ls_file(self, state_backend):
        entries = state_backend.ls_info("/hello.txt")
        assert len(entries) == 1
        assert entries[0]["path"] == "/hello.txt"
        assert entries[0]["is_dir"] is False

    def test_ls_nonexistent(self, state_backend):
        entries = state_backend.ls_info("/nonexistent")
        assert entries == []


class TestStateBackendRead:
    def test_read_existing(self, state_backend):
        content = state_backend.read("/hello.txt")
        assert "Hello, World!" in content

    def test_read_with_line_numbers(self, state_backend):
        content = state_backend.read("/src/main.py")
        assert "1\t" in content or "1" in content

    def test_read_nonexistent(self, state_backend):
        content = state_backend.read("/missing.txt")
        assert "Error" in content or "not found" in content

    def test_read_with_offset(self, state_backend):
        content = state_backend.read("/src/main.py", offset=1, limit=1)
        assert "print" in content


class TestStateBackendWrite:
    def test_write_new_file(self, state_backend):
        result = state_backend.write("/new.txt", "new content")
        assert result.error is None
        assert result.path == "/new.txt"
        assert result.files_update is not None
        assert "/new.txt" in result.files_update

    def test_write_existing_file_fails(self, state_backend):
        result = state_backend.write("/hello.txt", "overwrite")
        assert result.error is not None

    def test_write_updates_state(self, populated_state):
        backend = StateBackend(populated_state)
        result = backend.write("/created.txt", "hello")
        assert result.files_update is not None
        # Apply the update to simulate what the tool does
        populated_state["files"].update(result.files_update)
        content = backend.read("/created.txt")
        assert "hello" in content


class TestStateBackendEdit:
    def test_edit_existing(self, state_backend):
        result = state_backend.edit("/hello.txt", "World", "ADK")
        assert result.error is None
        assert result.occurrences == 1
        assert result.files_update is not None

    def test_edit_nonexistent(self, state_backend):
        result = state_backend.edit("/missing.txt", "a", "b")
        assert result.error is not None

    def test_edit_pattern_not_found(self, state_backend):
        result = state_backend.edit("/hello.txt", "ZZZZZ", "replacement")
        assert result.error is not None

    def test_edit_multiple_occurrences_no_replace_all(self, populated_state):
        populated_state["files"]["/repeat.txt"] = create_file_data("foo foo foo")
        backend = StateBackend(populated_state)
        result = backend.edit("/repeat.txt", "foo", "bar")
        assert result.error is not None  # Multiple occurrences without replace_all

    def test_edit_replace_all(self, populated_state):
        populated_state["files"]["/repeat.txt"] = create_file_data("foo foo foo")
        backend = StateBackend(populated_state)
        result = backend.edit("/repeat.txt", "foo", "bar", replace_all=True)
        assert result.error is None
        assert result.occurrences == 3


class TestStateBackendGrep:
    def test_grep_finds_matches(self, state_backend):
        matches = state_backend.grep_raw("def")
        assert isinstance(matches, list)
        assert len(matches) >= 2  # main.py and utils.py both have "def"

    def test_grep_with_path(self, state_backend):
        matches = state_backend.grep_raw("def", path="/src/main.py")
        assert isinstance(matches, list)
        assert all(m["path"] == "/src/main.py" for m in matches)

    def test_grep_no_matches(self, state_backend):
        matches = state_backend.grep_raw("nonexistent_pattern_xyz")
        assert isinstance(matches, list)
        assert len(matches) == 0


class TestStateBackendGlob:
    def test_glob_python_files(self, state_backend):
        entries = state_backend.glob_info("**/*.py", "/")
        paths = [e["path"] for e in entries]
        assert "/src/main.py" in paths
        assert "/src/utils.py" in paths

    def test_glob_markdown_files(self, state_backend):
        entries = state_backend.glob_info("**/*.md", "/")
        paths = [e["path"] for e in entries]
        assert "/docs/readme.md" in paths

    def test_glob_no_matches(self, state_backend):
        entries = state_backend.glob_info("**/*.rs", "/")
        assert len(entries) == 0


class TestStateBackendDownload:
    def test_download_existing(self, state_backend):
        results = state_backend.download_files(["/hello.txt"])
        assert len(results) == 1
        assert results[0].content is not None
        assert b"Hello" in results[0].content

    def test_download_nonexistent(self, state_backend):
        results = state_backend.download_files(["/missing.txt"])
        assert results[0].error is not None


class TestStateBackendUpload:
    def test_upload_not_supported(self, state_backend):
        with pytest.raises(NotImplementedError):
            state_backend.upload_files([("test.txt", b"content")])
