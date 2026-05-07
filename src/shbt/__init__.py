from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any


_SRC_PACKAGE_DIR = Path(__file__).resolve().parent.parent / "src" / "shbt"
__path__ = [str(_SRC_PACKAGE_DIR)]
__all__ = ["DEFAULT_PRECISION", "EvolutionaryEngine", "UniverseFactory"]


def __getattr__(name: str) -> Any:
    if name in {"DEFAULT_PRECISION", "EvolutionaryEngine", "UniverseFactory"}:
        module = import_module("shbt.core.evolutionary_engine")
        return getattr(module, name)
    raise AttributeError(f"module 'shbt' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
