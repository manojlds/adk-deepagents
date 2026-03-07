"""CLI memory/skills discovery and backend helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from adk_deepagents.backends.filesystem import FilesystemBackend
from adk_deepagents.backends.protocol import FileDownloadResponse
from adk_deepagents.cli.config import PROFILE_MEMORY_FILENAME, CliPaths, ensure_profile_memory

PROFILE_SKILLS_DIRNAME = "skills"
PROJECT_MEMORY_FILENAME = "AGENTS.md"
PROJECT_SKILLS_DIRNAME = "skills"


@dataclass(frozen=True)
class CliAgentResources:
    """Resolved memory/skills resources for a CLI agent run."""

    memory_sources: tuple[str, ...]
    memory_source_paths: Mapping[str, Path]
    skills_dirs: tuple[str, ...]


def discover_cli_agent_resources(
    *, paths: CliPaths, agent_name: str, cwd: Path
) -> CliAgentResources:
    """Resolve memory and skills discovery paths with deterministic precedence.

    Precedence order:
    1. Global profile resources under ``~/.adk-deepagents/profiles/<agent>/``
    2. Project resources under the current working directory

    Project entries are appended after global entries to give project context
    the final precedence in the system prompt / skills discovery order.
    """
    normalized_cwd = cwd.resolve()

    profile_memory_path = ensure_profile_memory(paths, agent_name).resolve()
    project_memory_path = (normalized_cwd / PROJECT_MEMORY_FILENAME).resolve()

    memory_sources: list[str] = []
    memory_source_paths: dict[str, Path] = {}
    seen_memory_paths: set[Path] = set()

    global_memory_source = f"global://profiles/{agent_name}/{PROFILE_MEMORY_FILENAME}"
    if profile_memory_path not in seen_memory_paths:
        seen_memory_paths.add(profile_memory_path)
        memory_sources.append(global_memory_source)
        memory_source_paths[global_memory_source] = profile_memory_path

    project_memory_source = f"project://{PROJECT_MEMORY_FILENAME}"
    if project_memory_path.is_file() and project_memory_path not in seen_memory_paths:
        seen_memory_paths.add(project_memory_path)
        memory_sources.append(project_memory_source)
        memory_source_paths[project_memory_source] = project_memory_path

    skills_dirs: list[str] = []
    seen_skills_dirs: set[str] = set()
    skills_candidates = [
        (paths.profiles_dir / agent_name / PROFILE_SKILLS_DIRNAME).resolve(),
        (normalized_cwd / PROJECT_SKILLS_DIRNAME).resolve(),
    ]

    for skills_dir in skills_candidates:
        if not skills_dir.is_dir():
            continue

        key = str(skills_dir)
        if key in seen_skills_dirs:
            continue

        seen_skills_dirs.add(key)
        skills_dirs.append(key)

    return CliAgentResources(
        memory_sources=tuple(memory_sources),
        memory_source_paths=memory_source_paths,
        skills_dirs=tuple(skills_dirs),
    )


def build_missing_skills_dependency_error(skills_dirs: Sequence[str]) -> RuntimeError:
    """Create an actionable error for missing optional skills dependencies."""
    requested = ", ".join(skills_dirs) if skills_dirs else "(none)"
    return RuntimeError(
        "Skills discovery was requested but optional dependency 'adk-skills-agent' is missing. "
        f"Requested skills directories: {requested}. "
        'Install optional support with: pip install "adk-deepagents[skills]"'
    )


class MemoryMappedFilesystemBackend(FilesystemBackend):
    """Workspace filesystem backend with explicit memory-source path mappings.

    CLI memory may include files outside the workspace (for example global
    profile memory under ``~/.adk-deepagents``). This backend keeps normal tool
    operations sandboxed to the workspace while allowing memory downloads from
    explicitly mapped source keys.
    """

    def __init__(
        self,
        *,
        root_dir: Path,
        memory_source_paths: Mapping[str, Path] | None = None,
    ) -> None:
        super().__init__(root_dir=root_dir, virtual_mode=True)
        self._memory_source_paths = {
            source: path.expanduser().resolve()
            for source, path in (memory_source_paths or {}).items()
        }

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        results: list[FileDownloadResponse] = []
        for source in paths:
            mapped_path = self._memory_source_paths.get(source)
            if mapped_path is None:
                results.extend(super().download_files([source]))
                continue

            results.append(self._download_mapped_source(source=source, path=mapped_path))

        return results

    @staticmethod
    def _download_mapped_source(*, source: str, path: Path) -> FileDownloadResponse:
        if not path.exists():
            return FileDownloadResponse(path=source, error="file_not_found")
        if path.is_dir():
            return FileDownloadResponse(path=source, error="is_directory")

        try:
            content = path.read_bytes()
        except PermissionError:
            return FileDownloadResponse(path=source, error="permission_denied")
        except OSError:
            return FileDownloadResponse(path=source, error="file_not_found")

        return FileDownloadResponse(path=source, content=content)
