"""Tests for CompositeBackend."""

from __future__ import annotations

from adk_deepagents.backends.composite import CompositeBackend
from adk_deepagents.backends.protocol import Backend
from adk_deepagents.backends.state import StateBackend
from adk_deepagents.backends.utils import create_file_data


def _make_state(files: dict | None = None) -> dict:
    return {"files": files or {}}


def _make_backend(files: dict | None = None) -> StateBackend:
    return StateBackend(_make_state(files))


# ---------------------------------------------------------------------------
# Route resolution
# ---------------------------------------------------------------------------


class TestRouteResolution:
    def test_default_backend_for_unmatched_path(self):
        default = _make_backend({"/file.txt": create_file_data("hello")})
        composite = CompositeBackend(default=default)
        result = composite.read("/file.txt")
        assert "hello" in result

    def test_route_matches_exact_prefix(self):
        default = _make_backend()
        ws = _make_backend({"/workspace/file.txt": create_file_data("ws content")})
        composite = CompositeBackend(default=default, routes={"/workspace": ws})
        result = composite.read("/workspace/file.txt")
        assert "ws content" in result

    def test_route_matches_subpath(self):
        default = _make_backend()
        ws = _make_backend({"/workspace/sub/file.txt": create_file_data("deep")})
        composite = CompositeBackend(default=default, routes={"/workspace": ws})
        result = composite.read("/workspace/sub/file.txt")
        assert "deep" in result

    def test_longer_prefix_wins(self):
        default = _make_backend()
        ws = _make_backend({"/workspace/data/f.txt": create_file_data("ws")})
        ws_data = _make_backend({"/workspace/data/f.txt": create_file_data("ws_data")})
        composite = CompositeBackend(
            default=default,
            routes={
                "/workspace": ws,
                "/workspace/data": ws_data,
            },
        )
        result = composite.read("/workspace/data/f.txt")
        assert "ws_data" in result

    def test_fallback_to_default(self):
        default = _make_backend({"/other.txt": create_file_data("default")})
        ws = _make_backend()
        composite = CompositeBackend(default=default, routes={"/workspace": ws})
        result = composite.read("/other.txt")
        assert "default" in result


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


class TestCompositeOperations:
    def test_write(self):
        default = _make_backend()
        composite = CompositeBackend(default=default)
        result = composite.write("/new.txt", "content")
        assert result.error is None

    def test_write_to_route(self):
        default = _make_backend()
        ws = _make_backend()
        composite = CompositeBackend(default=default, routes={"/workspace": ws})
        result = composite.write("/workspace/new.txt", "ws content")
        assert result.error is None
        assert result.path == "/workspace/new.txt"
        # StateBackend.write returns files_update but doesn't persist directly
        # (persistence is handled by the tool layer via _apply_files_update)
        assert result.files_update is not None
        assert "/workspace/new.txt" in result.files_update

    def test_edit(self):
        default = _make_backend({"/file.txt": create_file_data("old text here")})
        composite = CompositeBackend(default=default)
        result = composite.edit("/file.txt", "old", "new")
        assert result.error is None

    def test_ls_info(self):
        default = _make_backend(
            {
                "/dir/a.txt": create_file_data("a"),
                "/dir/b.txt": create_file_data("b"),
            }
        )
        composite = CompositeBackend(default=default)
        files = composite.ls_info("/dir")
        assert len(files) == 2

    def test_glob_info_merges_backends(self):
        default = _make_backend({"/a.py": create_file_data("a")})
        ws = _make_backend({"/workspace/b.py": create_file_data("b")})
        composite = CompositeBackend(default=default, routes={"/workspace": ws})
        results = composite.glob_info("**/*.py", "/")
        paths = {r["path"] for r in results}
        assert "/a.py" in paths
        assert "/workspace/b.py" in paths

    def test_grep_raw_merges_backends(self):
        default = _make_backend({"/a.txt": create_file_data("hello world")})
        ws = _make_backend({"/workspace/b.txt": create_file_data("hello there")})
        composite = CompositeBackend(default=default, routes={"/workspace": ws})
        results = composite.grep_raw("hello")
        assert isinstance(results, list)
        assert len(results) == 2

    def test_upload_files_routes_correctly(self):
        """Test that upload_files routes to the correct backend.

        StateBackend doesn't support upload_files, so we verify the
        routing logic by checking it calls the correct backend.
        """
        from unittest.mock import MagicMock

        default = MagicMock(spec=Backend)
        default.upload_files.return_value = [MagicMock(path="/file.txt", error=None)]
        ws = MagicMock(spec=Backend)
        ws.upload_files.return_value = [MagicMock(path="/workspace/file.txt", error=None)]

        composite = CompositeBackend(default=default, routes={"/workspace": ws})
        responses = composite.upload_files(
            [
                ("/file.txt", b"default"),
                ("/workspace/file.txt", b"ws"),
            ]
        )
        assert len(responses) == 2
        default.upload_files.assert_called_once()
        ws.upload_files.assert_called_once()

    def test_download_files(self):
        default = _make_backend({"/a.txt": create_file_data("content_a")})
        ws = _make_backend({"/workspace/b.txt": create_file_data("content_b")})
        composite = CompositeBackend(default=default, routes={"/workspace": ws})
        responses = composite.download_files(["/a.txt", "/workspace/b.txt"])
        assert len(responses) == 2
        assert responses[0].content is not None
        assert responses[1].content is not None


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestCompositeProperties:
    def test_default_property(self):
        default = _make_backend()
        composite = CompositeBackend(default=default)
        assert composite.default is default

    def test_routes_property(self):
        default = _make_backend()
        ws = _make_backend()
        composite = CompositeBackend(default=default, routes={"/workspace": ws})
        assert len(composite.routes) == 1
        assert composite.routes[0][0] == "/workspace"

    def test_no_routes(self):
        default = _make_backend()
        composite = CompositeBackend(default=default)
        assert composite.routes == []
