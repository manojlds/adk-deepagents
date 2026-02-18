"""Integration tests — StoreBackend cross-session persistence.

Verifies that StoreBackend can share data between separate instances via a
shared store dict, and that namespace isolation works correctly.
No API key or LLM required.
"""

from __future__ import annotations

import pytest

from adk_deepagents.backends.store import StoreBackend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCrossSessionSharing:
    def test_write_in_one_backend_read_in_another(self):
        shared_store: dict = {}

        backend_a = StoreBackend(shared_store, namespace="project")
        backend_b = StoreBackend(shared_store, namespace="project")

        result = backend_a.write("/readme.md", "# Hello")
        assert result.error is None
        # Apply files_update to the shared store
        shared_store["files"].update(result.files_update)

        content = backend_b.read("/readme.md")
        assert "# Hello" in content


class TestNamespaceIsolation:
    def test_namespace_isolation(self):
        shared_store: dict = {}

        backend_alpha = StoreBackend(shared_store, namespace="alpha")
        backend_beta = StoreBackend(shared_store, namespace="beta")

        result_a = backend_alpha.write("/secret.txt", "alpha data")
        shared_store["files"].update(result_a.files_update)

        result_b = backend_beta.write("/secret.txt", "beta data")
        shared_store["files"].update(result_b.files_update)

        alpha_content = backend_alpha.read("/secret.txt")
        assert "alpha data" in alpha_content

        beta_content = backend_beta.read("/secret.txt")
        assert "beta data" in beta_content


class TestNamespacePathPrefixing:
    def test_namespace_path_prefixing(self):
        shared_store: dict = {}
        backend = StoreBackend(shared_store, namespace="ns")

        result = backend.write("/file.txt", "content")
        shared_store["files"].update(result.files_update)

        # Internal storage should be prefixed
        assert "/ns/file.txt" in shared_store["files"]

        # External API should use unprefixed path
        assert result.path == "/file.txt"

        content = backend.read("/file.txt")
        assert "content" in content


class TestLsWithNamespace:
    def test_ls_with_namespace(self):
        shared_store: dict = {}
        backend = StoreBackend(shared_store, namespace="proj")

        for name in ("a.py", "b.py"):
            result = backend.write(f"/src/{name}", f"# {name}")
            shared_store["files"].update(result.files_update)

        entries = backend.ls_info("/src")
        paths = [e["path"] for e in entries]
        assert "/src/a.py" in paths
        assert "/src/b.py" in paths


class TestGlobWithNamespace:
    def test_glob_with_namespace(self):
        shared_store: dict = {}
        backend = StoreBackend(shared_store, namespace="proj")

        for name in ("main.py", "utils.py", "readme.md"):
            result = backend.write(f"/{name}", f"# {name}")
            shared_store["files"].update(result.files_update)

        entries = backend.glob_info("**/*.py", "/")
        paths = [e["path"] for e in entries]
        assert "/main.py" in paths
        assert "/utils.py" in paths
        assert "/readme.md" not in paths


class TestGrepWithNamespace:
    def test_grep_with_namespace(self):
        shared_store: dict = {}
        backend = StoreBackend(shared_store, namespace="proj")

        result_a = backend.write("/a.txt", "needle in a")
        shared_store["files"].update(result_a.files_update)
        result_b = backend.write("/b.txt", "no match")
        shared_store["files"].update(result_b.files_update)

        matches = backend.grep_raw("needle")
        assert isinstance(matches, list)
        match_paths = [m["path"] for m in matches]
        # Should return external (namespace-stripped) paths
        assert "/a.txt" in match_paths
        assert "/b.txt" not in match_paths
