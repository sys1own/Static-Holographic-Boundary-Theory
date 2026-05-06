from __future__ import annotations

"""Compatibility export for the SHBT derivation engine.

This module preserves the requested ``evolutionary_engine`` import path while
reusing the programmatic derivation implementation hosted in
``shbt.core.derivation_api`` and enforcing boundary-stabilized parity checks
around every public derivation entrypoint.
"""

from shbt.core import derivation_api as _derivation_api
from shbt.core.holographic_error_stabilizer import stabilize_boundary, stabilize_classmethods


EvolutionaryEngine = stabilize_classmethods(_derivation_api.UniverseFactory)

for _name in _derivation_api.__all__:
    _value = getattr(_derivation_api, _name)
    if _name == "UniverseFactory":
        globals()[_name] = EvolutionaryEngine
    elif _name in {"build_derivation_ledger", "build_lambda_ledger"}:
        globals()[_name] = stabilize_boundary(_value)
    else:
        globals()[_name] = _value
del _name

__all__ = [*_derivation_api.__all__, "EvolutionaryEngine"]
