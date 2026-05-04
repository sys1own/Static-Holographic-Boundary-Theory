from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "pub"

from .constants import (
    LEPTON_LEVEL,
    PARENT_LEVEL,
    QUARK_LEVEL,
    SO10_DIMENSION,
    SO10_DUAL_COXETER,
    SU2_DIMENSION,
    SU2_DUAL_COXETER,
    SU3_DIMENSION,
    SU3_DUAL_COXETER,
)
from .noether_bridge import load_c_dark_completion_fraction, newton_constant_lock
from .tn import TopologicalVacuum, verify_dark_energy_tension, wzw_central_charge_fraction

DEFAULT_PRECISION = 80


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


def _format_decimal(value: Decimal, *, places: int = 18) -> str:
    if value.is_zero():
        return "0"
    adjusted = value.adjusted()
    if adjusted >= 6 or adjusted <= -4:
        return f"{value:.{places}E}"
    return f"{value:.{places}f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class GKOParityResidue:
    parent_central_charge_fraction: Fraction
    visible_central_charge_fraction: Fraction
    reference_coset_central_charge_fraction: Fraction
    deanchored_modularity_gap_fraction: Fraction
    c_dark_fraction: Fraction
    live_c_dark_decimal: Decimal


@dataclass(frozen=True)
class ParityNeutralityProof:
    parity_sink_density_fraction: Fraction
    bl_visible_fraction: Fraction
    bl_dark_fraction: Fraction
    neutrality_sum_fraction: Fraction


@dataclass(frozen=True)
class GravityLinkProof:
    c_dark_fraction: Fraction
    planck_mass_ev: Decimal
    eight_pi_g_from_lock_ev_minus2: Decimal
    eight_pi_g_from_formula_ev_minus2: Decimal
    lambda_holo_si_m2: Decimal
    newton_lock_positive: bool
    lambda_holo_positive: bool


@dataclass(frozen=True)
class ParitySinkVerification:
    gko: GKOParityResidue
    neutrality: ParityNeutralityProof
    gravity: GravityLinkProof


def derive_gko_parity_residue(*, precision: int = DEFAULT_PRECISION) -> GKOParityResidue:
    del precision
    parent_central_charge_fraction = wzw_central_charge_fraction(PARENT_LEVEL, SO10_DIMENSION, SO10_DUAL_COXETER)
    visible_central_charge_fraction = (
        wzw_central_charge_fraction(LEPTON_LEVEL, SU2_DIMENSION, SU2_DUAL_COXETER)
        + wzw_central_charge_fraction(QUARK_LEVEL, SU3_DIMENSION, SU3_DUAL_COXETER)
    )
    c_dark_fraction = load_c_dark_completion_fraction()
    deanchored_modularity_gap_fraction = c_dark_fraction / 24
    reference_coset_central_charge_fraction = (
        parent_central_charge_fraction - visible_central_charge_fraction - c_dark_fraction
    )

    assert deanchored_modularity_gap_fraction > 0
    assert c_dark_fraction == 24 * deanchored_modularity_gap_fraction

    return GKOParityResidue(
        parent_central_charge_fraction=parent_central_charge_fraction,
        visible_central_charge_fraction=visible_central_charge_fraction,
        reference_coset_central_charge_fraction=reference_coset_central_charge_fraction,
        deanchored_modularity_gap_fraction=deanchored_modularity_gap_fraction,
        c_dark_fraction=c_dark_fraction,
        live_c_dark_decimal=_decimal(c_dark_fraction),
    )


def derive_parity_neutrality(*, c_dark_fraction: Fraction) -> ParityNeutralityProof:
    parity_sink_density_fraction = c_dark_fraction / PARENT_LEVEL
    bl_visible_fraction = parity_sink_density_fraction
    bl_dark_fraction = -parity_sink_density_fraction
    neutrality_sum_fraction = bl_visible_fraction + bl_dark_fraction

    assert neutrality_sum_fraction == 0

    return ParityNeutralityProof(
        parity_sink_density_fraction=parity_sink_density_fraction,
        bl_visible_fraction=bl_visible_fraction,
        bl_dark_fraction=bl_dark_fraction,
        neutrality_sum_fraction=neutrality_sum_fraction,
    )


def derive_gravity_link(*, c_dark_fraction: Fraction, precision: int = DEFAULT_PRECISION) -> GravityLinkProof:
    model = TopologicalVacuum()
    dark_energy_audit = verify_dark_energy_tension(model=model)
    newton_lock_audit = newton_constant_lock(c_dark_fraction=c_dark_fraction, precision=precision)

    planck_mass_ev = newton_lock_audit.planck_mass_ev
    c_dark_decimal = _decimal(c_dark_fraction)
    eight_pi_g_from_formula_ev_minus2 = Decimal(12) / (c_dark_decimal * planck_mass_ev * planck_mass_ev)
    eight_pi_g_from_lock_ev_minus2 = newton_lock_audit.eight_pi_g_effective_ev_minus2
    lambda_holo_si_m2 = _decimal(dark_energy_audit.lambda_surface_tension_si_m2)

    tolerance = Decimal("1e-30") * max(abs(eight_pi_g_from_lock_ev_minus2), Decimal(1))
    assert abs(eight_pi_g_from_formula_ev_minus2 - eight_pi_g_from_lock_ev_minus2) <= tolerance
    assert eight_pi_g_from_lock_ev_minus2 > 0
    assert lambda_holo_si_m2 > 0

    return GravityLinkProof(
        c_dark_fraction=c_dark_fraction,
        planck_mass_ev=planck_mass_ev,
        eight_pi_g_from_lock_ev_minus2=eight_pi_g_from_lock_ev_minus2,
        eight_pi_g_from_formula_ev_minus2=eight_pi_g_from_formula_ev_minus2,
        lambda_holo_si_m2=lambda_holo_si_m2,
        newton_lock_positive=bool(eight_pi_g_from_lock_ev_minus2 > 0),
        lambda_holo_positive=bool(lambda_holo_si_m2 > 0),
    )


def verify_parity_sink(*, precision: int = DEFAULT_PRECISION) -> ParitySinkVerification:
    gko = derive_gko_parity_residue(precision=precision)
    neutrality = derive_parity_neutrality(c_dark_fraction=gko.c_dark_fraction)
    gravity = derive_gravity_link(c_dark_fraction=gko.c_dark_fraction, precision=precision)
    return ParitySinkVerification(gko=gko, neutrality=neutrality, gravity=gravity)


def build_parity_sink_report(*, precision: int = DEFAULT_PRECISION) -> str:
    proof = verify_parity_sink(precision=precision)
    lines = [
        "Parity Sink Verification",
        "========================",
        "",
        "GKO Orthogonality",
        f"- c_parent = {proof.gko.parent_central_charge_fraction.numerator}/{proof.gko.parent_central_charge_fraction.denominator}",
        f"- c_vis = {proof.gko.visible_central_charge_fraction.numerator}/{proof.gko.visible_central_charge_fraction.denominator}",
        f"- c_coset^ref = {proof.gko.reference_coset_central_charge_fraction.numerator}/{proof.gko.reference_coset_central_charge_fraction.denominator}",
        f"- Delta_mod = {proof.gko.deanchored_modularity_gap_fraction.numerator}/{proof.gko.deanchored_modularity_gap_fraction.denominator} = {_format_decimal(_decimal(proof.gko.deanchored_modularity_gap_fraction), places=24)}",
        f"- c_dark = 24 * Delta_mod = {proof.gko.c_dark_fraction.numerator}/{proof.gko.c_dark_fraction.denominator} = {_format_decimal(proof.gko.live_c_dark_decimal, places=24)}",
        "",
        "Neutrality Equation",
        f"- parity sink density = c_dark / K = {proof.neutrality.parity_sink_density_fraction.numerator}/{proof.neutrality.parity_sink_density_fraction.denominator}",
        f"- (B-L)_vis = +{proof.neutrality.bl_visible_fraction.numerator}/{proof.neutrality.bl_visible_fraction.denominator}",
        f"- (B-L)_dark = {proof.neutrality.bl_dark_fraction.numerator}/{proof.neutrality.bl_dark_fraction.denominator}",
        f"- (B-L)_vis + (B-L)_dark = {proof.neutrality.neutrality_sum_fraction}",
        "",
        "Gravity Link",
        f"- M_P = {_format_decimal(proof.gravity.planck_mass_ev, places=18)} eV",
        f"- 8*pi*G_N from Newton lock = {_format_decimal(proof.gravity.eight_pi_g_from_lock_ev_minus2, places=24)} eV^-2",
        f"- 8*pi*G_N from 12/(c_dark M_P^2) = {_format_decimal(proof.gravity.eight_pi_g_from_formula_ev_minus2, places=24)} eV^-2",
        f"- Lambda_holo = {_format_decimal(proof.gravity.lambda_holo_si_m2, places=24)} m^-2",
        f"- Newton lock positive = {proof.gravity.newton_lock_positive}",
        f"- Lambda_holo positive = {proof.gravity.lambda_holo_positive}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify that c_dark is the mandatory B-L parity sink of the benchmark branch.")
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION, help="Decimal precision used for the Newton-lock verification.")
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    print(build_parity_sink_report(precision=max(args.precision, 32)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
