from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DEFAULT_BRANCH",
    "DEFAULT_PRECISION",
    "EvolutionaryEngine",
    "UniverseFactory",
    "bootstrap",
    "discover_kernel_from_bitlogic",
    "scan_boundary_configurations",
]


def __getattr__(name: str) -> Any:
    if name in {"DEFAULT_PRECISION", "EvolutionaryEngine", "UniverseFactory"}:
        module = import_module("shbt.core.evolutionary_engine")
        return getattr(module, name)
    if name in {"DEFAULT_BRANCH", "discover_kernel_from_bitlogic", "scan_boundary_configurations"}:
        module = import_module("shbt.core.bootstrap")
        return getattr(module, name)
    if name == "bootstrap":
        return import_module("shbt.core.bootstrap")
    raise AttributeError(f"module 'shbt.core' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
