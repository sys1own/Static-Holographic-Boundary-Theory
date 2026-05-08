from __future__ import annotations

"""Compatibility export for the SHBT derivation engine.

The programmatic factory now lives in ``shbt.core.derivation_api``. This
module preserves the historical ``shbt.core.evolutionary_engine`` import path
while keeping ``EvolutionaryEngine`` and ``UniverseFactory`` as exact aliases of
the stabilized base factory.
"""

from shbt.core import derivation_api as _derivation_api
from shbt.core.rigidity_kernel import stabilize_boundary


UniverseFactory = _derivation_api.UniverseFactory
EvolutionaryEngine = UniverseFactory

for _name in _derivation_api.__all__:
    _value = getattr(_derivation_api, _name)
    if _name == "UniverseFactory":
        globals()[_name] = UniverseFactory
    elif _name in {"build_derivation_ledger", "build_lambda_ledger"}:
        globals()[_name] = stabilize_boundary(_value)
    else:
        globals()[_name] = _value

__all__ = [*_derivation_api.__all__, "EvolutionaryEngine", "UniverseFactory"]
