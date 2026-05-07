from __future__ import annotations

from dataclasses import dataclass

from shbt.constants import (
    LEPTON_LEVEL,
    PARENT_LEVEL,
    QUARK_LEVEL,
    SU2_DIMENSION,
    SU2_DUAL_COXETER,
    SU3_DIMENSION,
    SU3_DUAL_COXETER,
)


@dataclass(frozen=True)
class GKOCentralChargeAudit:
    parent_level: int
    lepton_level: int
    quark_level: int
    parent_su3_central_charge: float
    parent_su2_central_charge: float
    visible_su3_central_charge: float
    visible_su2_central_charge: float
    c_dark_residue: float

    @property
    def orthogonality_verified(self) -> bool:
        return self.c_dark_residue > 0.0


def wzw_central_charge(level: int, dimension: int, dual_coxeter: int) -> float:
    resolved_level = float(level)
    denominator = resolved_level + float(dual_coxeter)
    if denominator <= 0.0:
        raise ValueError("WZW central charge requires k + h^∨ > 0.")
    return float(resolved_level * float(dimension) / denominator)


def gko_c_dark_residue(
    *,
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
) -> float:
    r"""Return the rigid GKO central-charge residue ``c_dark``.

    The residue is the orthogonal parent-visible difference
    ``[c(SU(3)_K)+c(SU(2)_K)]-[c(SU(3)_{k_q})+c(SU(2)_{k_\ell})]``.
    """

    parent_su3_central_charge = wzw_central_charge(parent_level, SU3_DIMENSION, SU3_DUAL_COXETER)
    parent_su2_central_charge = wzw_central_charge(parent_level, SU2_DIMENSION, SU2_DUAL_COXETER)
    visible_su3_central_charge = wzw_central_charge(quark_level, SU3_DIMENSION, SU3_DUAL_COXETER)
    visible_su2_central_charge = wzw_central_charge(lepton_level, SU2_DIMENSION, SU2_DUAL_COXETER)
    return float(
        (parent_su3_central_charge + parent_su2_central_charge)
        - (visible_su3_central_charge + visible_su2_central_charge)
    )


def verify_gko_orthogonality(
    *,
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
) -> GKOCentralChargeAudit:
    """Audit the rigid GKO central-charge residue for the selected branch."""

    parent_su3_central_charge = wzw_central_charge(parent_level, SU3_DIMENSION, SU3_DUAL_COXETER)
    parent_su2_central_charge = wzw_central_charge(parent_level, SU2_DIMENSION, SU2_DUAL_COXETER)
    visible_su3_central_charge = wzw_central_charge(quark_level, SU3_DIMENSION, SU3_DUAL_COXETER)
    visible_su2_central_charge = wzw_central_charge(lepton_level, SU2_DIMENSION, SU2_DUAL_COXETER)
    c_dark_residue = gko_c_dark_residue(
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )
    return GKOCentralChargeAudit(
        parent_level=int(parent_level),
        lepton_level=int(lepton_level),
        quark_level=int(quark_level),
        parent_su3_central_charge=float(parent_su3_central_charge),
        parent_su2_central_charge=float(parent_su2_central_charge),
        visible_su3_central_charge=float(visible_su3_central_charge),
        visible_su2_central_charge=float(visible_su2_central_charge),
        c_dark_residue=float(c_dark_residue),
    )


__all__ = [
    "GKOCentralChargeAudit",
    "gko_c_dark_residue",
    "verify_gko_orthogonality",
    "wzw_central_charge",
]
