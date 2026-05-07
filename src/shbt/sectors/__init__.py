from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "BiologicalComplexityAudit": "shbt.sectors.complexity",
    "BiologicalInformationProcessingAudit": "shbt.sectors.complexity",
    "BiologicalIsomorphism": "shbt.sectors.complexity",
    "BoundaryRenderPatch": "shbt.sectors.gravity",
    "ComplexitySectorAudit": "shbt.sectors.complexity_sector",
    "DnaErrorCorrectionAudit": "shbt.sectors.complexity",
    "FibonacciPhyllotaxisAudit": "shbt.sectors.complexity",
    "GravitySectorAudit": "shbt.sectors.gravity",
    "MarkovCollar": "shbt.sectors.gravity",
    "ObserverTuple": "shbt.sectors.gravity",
    "PrimeSyncAudit": "shbt.sectors.complexity_sector",
    "PrimeSyncWindow": "shbt.sectors.complexity_sector",
    "ShannonEntropyLimitAudit": "shbt.sectors.complexity_sector",
    "build_biological_complexity_audit": "shbt.sectors.complexity",
    "build_biological_complexity_report": "shbt.sectors.complexity",
    "build_complexity_sector_audit": "shbt.sectors.complexity_sector",
    "build_complexity_sector_report": "shbt.sectors.complexity_sector",
    "build_gravity_sector_audit": "shbt.sectors.gravity",
    "build_gravity_sector_report": "shbt.sectors.gravity",
    "build_markov_collar": "shbt.sectors.gravity",
    "derive_gravitational_acceleration": "shbt.sectors.gravity",
    "render_boundary_on_observer_horizon": "shbt.sectors.gravity",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'shbt.sectors' has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    return list(__all__)
