from __future__ import annotations

"""Compatibility export for the SHBT derivation engine.

The programmatic factory now lives in ``shbt.core.derivation_api``. This
module preserves the historical ``shbt.core.evolutionary_engine`` import path
while keeping ``EvolutionaryEngine`` and ``UniverseFactory`` as exact aliases of
the stabilized base factory.
"""

from shbt.core import derivation_api as _derivation_api


EvolutionaryEngine = _derivation_api.UniverseFactory
UniverseFactory = _derivation_api.UniverseFactory

for _name in _derivation_api.__all__:
    globals()[_name] = getattr(_derivation_api, _name)

globals()["EvolutionaryEngine"] = EvolutionaryEngine
globals()["UniverseFactory"] = UniverseFactory

del _name

__all__ = list(dict.fromkeys([*_derivation_api.__all__, "EvolutionaryEngine", "UniverseFactory"]))
