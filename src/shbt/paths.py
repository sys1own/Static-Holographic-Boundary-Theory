from __future__ import annotations

from pathlib import Path


class ProjectPaths:
    ROOT = Path(__file__).resolve().parents[2]
    SRC = ROOT / "src"
    DATA = ROOT / "data"
    CONFIG = ROOT / "config"
    SCRIPTS = ROOT / "scripts"
    PAPERS = ROOT / "papers"
    RESULTS = ROOT / "results"

    @classmethod
    def ensure_dirs(cls) -> None:
        cls.RESULTS.mkdir(parents=True, exist_ok=True)


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ProjectPaths.ROOT
PAPERS_DIR = ProjectPaths.PAPERS


def resolve_resource_path(*relative_parts: str) -> Path:
    relative_path = Path(*relative_parts)
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
