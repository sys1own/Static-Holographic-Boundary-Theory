from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
PAPERS_DIR = REPO_ROOT / "papers"


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


__all__ = ["PAPERS_DIR", "PACKAGE_ROOT", "REPO_ROOT", "resolve_resource_path"]
