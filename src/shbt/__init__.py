from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["DEFAULT_PRECISION", "UniverseFactory"]


def __getattr__(name: str) -> Any:
    if name in {"DEFAULT_PRECISION", "UniverseFactory"}:
        module = import_module("shbt.core.derivation_api")
        return getattr(module, name)
    raise AttributeError(f"module 'shbt' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
