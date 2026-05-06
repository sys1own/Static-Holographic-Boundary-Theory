from __future__ import annotations

"""Strong-interaction residue bookkeeping for the SHBT branch audit.

The selected branch closes the visible-color bookkeeping only when the quark
triplet loading, the lepton support, and the fixed parent level all preserve a
zero framing residue. This module packages those statements as a small QCD
residue audit that downstream baryogenesis and proton-stability proofs can use.
"""

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from shbt.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL


@dataclass(frozen=True)
class ColorChargeAudit:
    branch: tuple[int, int, int]
    quark_branching_index: Fraction
    color_triplet_fraction: Fraction
    color_residue: Fraction

    @property
    def singlet_locked(self) -> bool:
        return self.color_residue == 0


@dataclass(frozen=True)
class GluonFluxTubeAudit:
    branch: tuple[int, int, int]
    lepton_gap: Fraction
    quark_gap: Fraction
    framing_residue: Fraction
    flux_tube_loading: Decimal

    @property
    def confinement_locked(self) -> bool:
        return self.framing_residue == 0


@dataclass(frozen=True)
class QCDResidueAudit:
    branch: tuple[int, int, int]
    color_charge: ColorChargeAudit
    flux_tube: GluonFluxTubeAudit
    baryon_violation_channel: str

    @property
    def baryon_asymmetry_channel_locked(self) -> bool:
        return self.color_charge.singlet_locked and self.flux_tube.confinement_locked

    @property
    def proton_stability_channel_locked(self) -> bool:
        return self.baryon_asymmetry_channel_locked


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(str(value))


def _distance_to_integer_fraction(value: Fraction) -> Fraction:
    denominator = value.denominator
    remainder = Fraction(value.numerator % denominator, denominator)
    return min(remainder, Fraction(1, 1) - remainder)


def derive_color_charge_audit(
    *,
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
) -> ColorChargeAudit:
    quark_index = Fraction(int(parent_level), 3 * int(quark_level))
    color_fraction = Fraction(3 * int(quark_level), int(parent_level))
    return ColorChargeAudit(
        branch=(int(lepton_level), int(quark_level), int(parent_level)),
        quark_branching_index=quark_index,
        color_triplet_fraction=color_fraction,
        color_residue=_distance_to_integer_fraction(quark_index),
    )


def derive_gluon_flux_tube_audit(
    *,
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
) -> GluonFluxTubeAudit:
    lepton_gap = _distance_to_integer_fraction(Fraction(int(parent_level), 2 * int(lepton_level)))
    quark_gap = _distance_to_integer_fraction(Fraction(int(parent_level), 3 * int(quark_level)))
    framing_residue = lepton_gap if lepton_gap >= quark_gap else quark_gap
    flux_tube_loading = _decimal(Fraction(int(parent_level), int(lepton_level) * int(quark_level)))
    return GluonFluxTubeAudit(
        branch=(int(lepton_level), int(quark_level), int(parent_level)),
        lepton_gap=lepton_gap,
        quark_gap=quark_gap,
        framing_residue=framing_residue,
        flux_tube_loading=flux_tube_loading,
    )


def build_qcd_residue_audit(
    *,
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
) -> QCDResidueAudit:
    color_charge = derive_color_charge_audit(
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )
    flux_tube = derive_gluon_flux_tube_audit(
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
    )
    return QCDResidueAudit(
        branch=(int(lepton_level), int(quark_level), int(parent_level)),
        color_charge=color_charge,
        flux_tube=flux_tube,
        baryon_violation_channel=r"\overline{\mathbf{126}}_H",
    )


__all__ = [
    "ColorChargeAudit",
    "GluonFluxTubeAudit",
    "QCDResidueAudit",
    "build_qcd_residue_audit",
    "derive_color_charge_audit",
    "derive_gluon_flux_tube_audit",
]
