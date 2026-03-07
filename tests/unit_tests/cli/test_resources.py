"""Unit tests for CLI memory/skills discovery and backend resource mapping."""

from __future__ import annotations

from pathlib import Path

from adk_deepagents.cli.config import bootstrap_cli_home, resolve_cli_paths
from adk_deepagents.cli.resources import (
    MemoryMappedFilesystemBackend,
    discover_cli_agent_resources,
)


def test_discover_cli_agent_resources_memory_precedence_global_then_project(tmp_path: Path) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("project memory", encoding="utf-8")

    paths = resolve_cli_paths(home_dir)
    bootstrap_cli_home(paths)

    resources = discover_cli_agent_resources(paths=paths, agent_name="demo", cwd=workspace)

    assert resources.memory_sources == (
        "global://profiles/demo/AGENTS.md",
        "project://AGENTS.md",
    )
    assert (
        resources.memory_source_paths["global://profiles/demo/AGENTS.md"]
        == (home_dir / "profiles" / "demo" / "AGENTS.md").resolve()
    )
    assert (
        resources.memory_source_paths["project://AGENTS.md"] == (workspace / "AGENTS.md").resolve()
    )


def test_discover_cli_agent_resources_skills_precedence_global_then_project(tmp_path: Path) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    paths = resolve_cli_paths(home_dir)
    bootstrap_cli_home(paths)

    global_skills = home_dir / "profiles" / "demo" / "skills"
    project_skills = workspace / "skills"
    global_skills.mkdir(parents=True)
    project_skills.mkdir(parents=True)

    resources = discover_cli_agent_resources(paths=paths, agent_name="demo", cwd=workspace)

    assert resources.skills_dirs == (
        str(global_skills.resolve()),
        str(project_skills.resolve()),
    )


def test_discover_cli_agent_resources_omits_missing_project_resources(tmp_path: Path) -> None:
    home_dir = tmp_path / ".adk-deepagents"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    paths = resolve_cli_paths(home_dir)
    bootstrap_cli_home(paths)

    resources = discover_cli_agent_resources(paths=paths, agent_name="demo", cwd=workspace)

    assert resources.memory_sources == ("global://profiles/demo/AGENTS.md",)
    assert resources.skills_dirs == ()


def test_memory_mapped_filesystem_backend_reads_mapped_memory_sources(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_memory = tmp_path / "outside" / "AGENTS.md"
    outside_memory.parent.mkdir(parents=True)
    outside_memory.write_text("global memory", encoding="utf-8")

    backend = MemoryMappedFilesystemBackend(
        root_dir=workspace,
        memory_source_paths={"global://profiles/demo/AGENTS.md": outside_memory},
    )

    responses = backend.download_files(["global://profiles/demo/AGENTS.md"])

    assert len(responses) == 1
    assert responses[0].path == "global://profiles/demo/AGENTS.md"
    assert responses[0].error is None
    assert responses[0].content == b"global memory"


def test_memory_mapped_filesystem_backend_keeps_workspace_download_behavior(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("project memory", encoding="utf-8")

    backend = MemoryMappedFilesystemBackend(root_dir=workspace)

    responses = backend.download_files(["/AGENTS.md"])

    assert len(responses) == 1
    assert responses[0].path == "/AGENTS.md"
    assert responses[0].error is None
    assert responses[0].content == b"project memory"
