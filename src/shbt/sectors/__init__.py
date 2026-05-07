from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ComplexitySectorAudit",
    "PrimeSyncAudit",
    "PrimeSyncWindow",
    "ShannonEntropyLimitAudit",
    "build_complexity_sector_audit",
    "build_complexity_sector_report",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module("shbt.sectors.complexity_sector")
        return getattr(module, name)
    raise AttributeError(f"module 'shbt.sectors' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
