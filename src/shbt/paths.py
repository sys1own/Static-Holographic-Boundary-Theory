from __future__ import annotations

from pathlib import Path


class ProjectPaths:
    # Target: Static-Holographic-Boundary-Theory-main/
    ROOT = Path(__file__).resolve().parents[2]
    SRC = ROOT / "src"
    CONFIG = ROOT / "config"
    PHYSICS_PROFILES = CONFIG / "physics_profiles"
    EXTERNAL_TRIGGERS = PHYSICS_PROFILES / "external_triggers"
    SCRIPTS = ROOT / "scripts"
    PAPERS = ROOT / "papers"
    RESULTS = ROOT / "results"

    @classmethod
    def ensure_dirs(cls) -> None:
        cls.RESULTS.mkdir(parents=True, exist_ok=True)


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ProjectPaths.ROOT
PAPERS_DIR = ProjectPaths.PAPERS


def resolve_resource_path(resource_type: str, filename: str | None = None) -> Path:
    """Resolve shared repository resources.

    Supports both the newer ``(resource_type, filename)`` form for data/config
    lookups and the existing single-argument filename form used throughout the
    manuscript/audit stack.
    """

    if filename is not None:
        if resource_type in {"data", "physics_profiles"}:
            return ProjectPaths.PHYSICS_PROFILES / filename
        if resource_type == "config":
            return ProjectPaths.CONFIG / filename
        return ProjectPaths.ROOT / filename

    relative_path = Path(resource_type)
    candidates = (
        PACKAGE_ROOT / relative_path,
        PACKAGE_ROOT / "audit" / relative_path,
        PACKAGE_ROOT / "core" / relative_path,
        PAPERS_DIR / relative_path,
        PAPERS_DIR / relative_path.name,
        REPO_ROOT / relative_path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


__all__ = [
    "PACKAGE_ROOT",
    "PAPERS_DIR",
    "ProjectPaths",
    "REPO_ROOT",
    "resolve_resource_path",
]
