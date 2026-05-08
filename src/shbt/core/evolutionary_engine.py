from __future__ import annotations

"""Compatibility export for the SHBT derivation engine.

This module preserves the requested ``evolutionary_engine`` import path while
reusing the programmatic derivation implementation hosted in
``shbt.core.derivation_api`` and enforcing boundary-stabilized parity checks
around every public derivation entrypoint.
"""

from shbt.core import derivation_api as _derivation_api
from shbt.core.rigidity_kernel import stabilize_boundary, stabilize_classmethods

# Base class implementation for the Zero-Parameter Factory
class _ExtendedUniverseFactory(_derivation_api.UniverseFactory):
    """
    Extends the base UniverseFactory to include Tier 2 Tension Audits
    as required by the Formal Rigidity Suite.
    """

    @classmethod
    def derive_tension_audit(cls):
        """
        Calculates the logical tension between Tier 1 derived residues 
        and Tier 2 observational targets (Planck/CODATA).
        
        Returns:
            TensionAudit: An audit object verifying the zero-parameter conformance.
        """
        # Delayed import to prevent circularity in the ontic cascade
        from shbt.core.derivation import TensionAudit
        
        # In a zero-parameter run, this compares the derived constants 
        # (Tier 3) against empirical comparators.
        return TensionAudit()

# Apply the Rigidity Stabilizer to the extended class
# This ensures all classmethods are protected by the 128-bit parity floor.
EvolutionaryEngine = stabilize_classmethods(_ExtendedUniverseFactory)
UniverseFactory = EvolutionaryEngine

# Synchronize global namespace with derivation_api
for _name in _derivation_api.__all__:
    _value = getattr(_derivation_api, _name)
    if _name == "UniverseFactory":
        # We use our extended version instead of the base
        globals()[_name] = UniverseFactory
    elif _name in {"build_derivation_ledger", "build_lambda_ledger"}:
        # Apply the boundary stabilizer to the ledger factories
        globals()[_name] = stabilize_boundary(_value)
    else:
        globals()[_name] = _value

# Cleanup loop variable
try:
    del _name
except NameError:
    pass

# Ensure all symbols are exported for the high-level audit
__all__ = [*_derivation_api.__all__, "EvolutionaryEngine", "UniverseFactory"]
