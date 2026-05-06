from __future__ import annotations

"""Compatibility export for the SHBT derivation engine.

This module preserves the requested ``evolutionary_engine`` import path while
reusing the programmatic derivation implementation hosted in
``shbt.core.derivation_api``.
"""

from shbt.core import derivation_api as _derivation_api


EvolutionaryEngine = _derivation_api.UniverseFactory

for _name in _derivation_api.__all__:
    globals()[_name] = getattr(_derivation_api, _name)
del _name

__all__ = [*_derivation_api.__all__, "EvolutionaryEngine"]
