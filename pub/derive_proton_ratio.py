from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Sequence

from scipy.constants import electron_mass, proton_mass

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "pub"

from . import algebra
from .derive_universe import derive_kappa_d5 as derive_decimal_kappa_d5
from .physics_engine import quark_branching_pressure
from .tn import (
    LEPTON_LEVEL,
    PARENT_LEVEL,
    QUARK_LEVEL,
    RANK_DIFFERENCE,
    SU2_DIMENSION,
    SU2_DUAL_COXETER,
    SU3_DIMENSION,
    SU3_DUAL_COXETER,
    quark_branching_index,
    wzw_central_charge_fraction,
)

DEFAULT_PRECISION = 50
DEFAULT_RELATIVE_TOLERANCE = Decimal("1e-3")
_GUARD_DIGITS = 12
LEPTON_CENTRAL_CHARGE_FRACTION = Fraction(39, 14)
QUARK_CENTRAL_CHARGE_FRACTION = Fraction(64, 11)
BRANCH_PIXEL_VOLUME_FRACTION = Fraction(3, 13)
BRANCH_PIXEL_VOLUME_INVERSE_FRACTION = Fraction(
    BRANCH_PIXEL_VOLUME_FRACTION.denominator,
    BRANCH_PIXEL_VOLUME_FRACTION.numerator,
)
BRANCH_CENTRAL_CHARGE_RATIO_FRACTION = QUARK_CENTRAL_CHARGE_FRACTION / LEPTON_CENTRAL_CHARGE_FRACTION
BRANCH_STRUCTURAL_PREFRACTOR_FRACTION = BRANCH_CENTRAL_CHARGE_RATIO_FRACTION * BRANCH_PIXEL_VOLUME_INVERSE_FRACTION


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


def _fraction_to_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _format_decimal(value: Decimal, *, places: int = 18) -> str:
    if value.is_zero():
        return "0"
    adjusted = value.adjusted()
    if adjusted >= 6 or adjusted <= -4:
        return f"{value:.{places}E}"
    return f"{value:.{places}f}".rstrip("0").rstrip(".")


def decimal_cuberoot(value: Decimal, *, precision: int = DEFAULT_PRECISION) -> Decimal:
    if value <= 0:
        raise ValueError("Cube root requires a positive Decimal.")
    with localcontext() as context:
        context.prec = precision + _GUARD_DIGITS
        guess = Decimal(str(float(value) ** (1.0 / 3.0)))
        if guess <= 0:
            guess = Decimal(1)
        threshold = Decimal(1).scaleb(-(precision + _GUARD_DIGITS // 2))
        while True:
            next_guess = (Decimal(2) * guess + value / (guess * guess)) / Decimal(3)
            if abs(next_guess - guess) <= threshold:
                context.prec = precision
                return +next_guess
            guess = next_guess


@dataclass(frozen=True)
class CentralChargeGeometry:
    lepton_central_charge_fraction: Fraction
    quark_central_charge_fraction: Fraction
    central_charge_ratio_fraction: Fraction
    central_charge_ratio_decimal: Decimal
    quark_branching_index: int
    branch_pixel_simplex_volume_fraction: Fraction
    inverse_pixel_volume_fraction: Fraction
    inverse_pixel_volume_decimal: Decimal

    @property
    def structural_prefactor_fraction(self) -> Fraction:
        return self.central_charge_ratio_fraction * self.inverse_pixel_volume_fraction

    @property
    def structural_prefactor_decimal(self) -> Decimal:
        return self.central_charge_ratio_decimal * self.inverse_pixel_volume_decimal

    @property
    def branch_pixel_simplex_volume_decimal(self) -> Decimal:
        return _fraction_to_decimal(self.branch_pixel_simplex_volume_fraction)


@dataclass(frozen=True)
class VacuumPressureDerivation:
    visible_reference_entry_magnitude: Decimal
    vacuum_pressure: Decimal


@dataclass(frozen=True)
class CodataMassRatioAudit:
    proton_mass_kg: Decimal
    electron_mass_kg: Decimal
    mass_ratio: Decimal


@dataclass(frozen=True)
class ProtonRatioDerivation:
    geometry: CentralChargeGeometry
    vacuum_pressure: VacuumPressureDerivation
    structural_prefactor: Decimal
    kappa_d5: Decimal
    kappa_d5_cuberoot: Decimal
    geometric_friction_factor: Decimal
    pressure_loading: Decimal
    su3_branching_pressure_scale: Decimal
    mu_audit: Decimal
    codata_audit: CodataMassRatioAudit
    absolute_delta: Decimal
    relative_error: Decimal
    tolerance: Decimal

    @property
    def structural_mu(self) -> Decimal:
        return self.mu_audit

    @property
    def derived_mu(self) -> Decimal:
        return self.mu_audit


def derive_central_charge_geometry() -> CentralChargeGeometry:
    lepton_central_charge_fraction = wzw_central_charge_fraction(LEPTON_LEVEL, SU2_DIMENSION, SU2_DUAL_COXETER)
    quark_central_charge_fraction = wzw_central_charge_fraction(QUARK_LEVEL, SU3_DIMENSION, SU3_DUAL_COXETER)
    central_charge_ratio_fraction = quark_central_charge_fraction / lepton_central_charge_fraction
    central_charge_ratio_decimal = _fraction_to_decimal(central_charge_ratio_fraction)
    resolved_quark_branching_index = quark_branching_index(PARENT_LEVEL, QUARK_LEVEL)
    branch_pixel_simplex_volume_fraction = Fraction(SU3_DUAL_COXETER, resolved_quark_branching_index)
    inverse_pixel_volume_fraction = Fraction(
        branch_pixel_simplex_volume_fraction.denominator,
        branch_pixel_simplex_volume_fraction.numerator,
    )
    inverse_pixel_volume_decimal = _fraction_to_decimal(inverse_pixel_volume_fraction)

    assert lepton_central_charge_fraction == LEPTON_CENTRAL_CHARGE_FRACTION
    assert quark_central_charge_fraction == QUARK_CENTRAL_CHARGE_FRACTION
    assert central_charge_ratio_fraction == BRANCH_CENTRAL_CHARGE_RATIO_FRACTION
    assert branch_pixel_simplex_volume_fraction == BRANCH_PIXEL_VOLUME_FRACTION
    assert inverse_pixel_volume_fraction == BRANCH_PIXEL_VOLUME_INVERSE_FRACTION

    return CentralChargeGeometry(
        lepton_central_charge_fraction=lepton_central_charge_fraction,
        quark_central_charge_fraction=quark_central_charge_fraction,
        central_charge_ratio_fraction=central_charge_ratio_fraction,
        central_charge_ratio_decimal=central_charge_ratio_decimal,
        quark_branching_index=resolved_quark_branching_index,
        branch_pixel_simplex_volume_fraction=branch_pixel_simplex_volume_fraction,
        inverse_pixel_volume_fraction=inverse_pixel_volume_fraction,
        inverse_pixel_volume_decimal=inverse_pixel_volume_decimal,
    )


def derive_vacuum_pressure() -> VacuumPressureDerivation:
    visible_block = algebra.su3_low_weight_block(QUARK_LEVEL)
    reference_entry_magnitude = abs(complex(visible_block[0, 0]))
    vacuum_pressure = quark_branching_pressure(visible_block, RANK_DIFFERENCE)
    return VacuumPressureDerivation(
        visible_reference_entry_magnitude=_decimal(reference_entry_magnitude),
        vacuum_pressure=_decimal(vacuum_pressure),
    )


def derive_codata_mass_ratio() -> CodataMassRatioAudit:
    proton_mass_kg = _decimal(proton_mass)
    electron_mass_kg = _decimal(electron_mass)
    mass_ratio = proton_mass_kg / electron_mass_kg
    return CodataMassRatioAudit(
        proton_mass_kg=proton_mass_kg,
        electron_mass_kg=electron_mass_kg,
        mass_ratio=mass_ratio,
    )


def derive_proton_ratio(
    *,
    precision: int = DEFAULT_PRECISION,
    tolerance: Decimal = DEFAULT_RELATIVE_TOLERANCE,
) -> ProtonRatioDerivation:
    geometry = derive_central_charge_geometry()
    vacuum_pressure = derive_vacuum_pressure()
    codata_audit = derive_codata_mass_ratio()

    with localcontext() as context:
        context.prec = precision + _GUARD_DIGITS
        quark_central_charge = _fraction_to_decimal(geometry.quark_central_charge_fraction)
        lepton_central_charge = _fraction_to_decimal(geometry.lepton_central_charge_fraction)
        branch_pixel_volume = geometry.branch_pixel_simplex_volume_decimal
        structural_prefactor = _fraction_to_decimal(BRANCH_STRUCTURAL_PREFRACTOR_FRACTION)
        kappa_d5 = derive_decimal_kappa_d5(precision=context.prec).kappa
        kappa_d5_cuberoot = decimal_cuberoot(kappa_d5, precision=context.prec)
        geometric_friction_factor = (Decimal(1) - kappa_d5) * kappa_d5_cuberoot
        pressure_loading = (vacuum_pressure.vacuum_pressure * vacuum_pressure.vacuum_pressure) / geometric_friction_factor
        su3_branching_pressure_scale = pressure_loading / branch_pixel_volume
        mu_audit = (quark_central_charge / lepton_central_charge) * su3_branching_pressure_scale
        absolute_delta = abs(mu_audit - codata_audit.mass_ratio)
        relative_error = absolute_delta / codata_audit.mass_ratio
        context.prec = precision

    assert geometry.structural_prefactor_fraction == BRANCH_STRUCTURAL_PREFRACTOR_FRACTION, (
        "SO(10)_312 structural prefactor drift: the branch-fixed proton/electron ratio no longer closes to the expected "
        f"(c_q/c_l) * V_px^(-1) factor {BRANCH_STRUCTURAL_PREFRACTOR_FRACTION}."
    )
    assert relative_error <= tolerance, (
        "Branch proton/electron residue no longer matches the CODATA value within the one-copy dictionary tolerance: "
        f"predicted {mu_audit}, CODATA {codata_audit.mass_ratio}, relative error {relative_error}."
    )

    return ProtonRatioDerivation(
        geometry=geometry,
        vacuum_pressure=vacuum_pressure,
        structural_prefactor=+structural_prefactor,
        kappa_d5=+kappa_d5,
        kappa_d5_cuberoot=+kappa_d5_cuberoot,
        geometric_friction_factor=+geometric_friction_factor,
        pressure_loading=+pressure_loading,
        su3_branching_pressure_scale=+su3_branching_pressure_scale,
        mu_audit=+mu_audit,
        codata_audit=codata_audit,
        absolute_delta=+absolute_delta,
        relative_error=+relative_error,
        tolerance=+tolerance,
    )


def build_proton_ratio_ledger(*, precision: int = DEFAULT_PRECISION) -> str:
    derivation = derive_proton_ratio(precision=precision)
    geometry = derivation.geometry
    pressure = derivation.vacuum_pressure
    codata = derivation.codata_audit

    lines = [
        "Proton Ratio Ledger",
        "===================",
        "",
        "Branch Geometry",
        f"- k_l = {LEPTON_LEVEL}",
        f"- k_q = {QUARK_LEVEL}",
        f"- K = {PARENT_LEVEL}",
        f"- c_l = 3*k_l/(k_l + 2) = {geometry.lepton_central_charge_fraction.numerator}/{geometry.lepton_central_charge_fraction.denominator}",
        f"- c_q = 8*k_q/(k_q + 3) = {geometry.quark_central_charge_fraction.numerator}/{geometry.quark_central_charge_fraction.denominator}",
        f"- c_q/c_l = {geometry.central_charge_ratio_fraction.numerator}/{geometry.central_charge_ratio_fraction.denominator}",
        f"- I_Q = K/(3*k_q) = {geometry.quark_branching_index}",
        f"- V_px = h^vee_SU(3)/I_Q = {geometry.branch_pixel_simplex_volume_fraction.numerator}/{geometry.branch_pixel_simplex_volume_fraction.denominator}",
        f"- V_px^(-1) = {geometry.inverse_pixel_volume_fraction.numerator}/{geometry.inverse_pixel_volume_fraction.denominator}",
        f"- (c_q/c_l) * V_px^(-1) = {geometry.structural_prefactor_fraction.numerator}/{geometry.structural_prefactor_fraction.denominator}",
        "",
        "SU(3)_8 Branching Pressure",
        f"- |S_00^(low)| = {_format_decimal(pressure.visible_reference_entry_magnitude, places=24)}",
        f"- Pi_vac = -(Delta r/8) * log|S_00^(low)| = {_format_decimal(pressure.vacuum_pressure, places=24)}",
        f"- (c_q/c_l) * V_px^(-1) = {_format_decimal(derivation.structural_prefactor, places=24)}",
        f"- kappa_D5 = {_format_decimal(derivation.kappa_d5, places=24)}",
        f"- kappa_D5^(1/3) = {_format_decimal(derivation.kappa_d5_cuberoot, places=24)}",
        f"- geometric friction = (1-kappa_D5) * kappa_D5^(1/3) = {_format_decimal(derivation.geometric_friction_factor, places=24)}",
        f"- P_mu = Pi_vac^2 / [(1-kappa_D5) * kappa_D5^(1/3)] = {_format_decimal(derivation.pressure_loading, places=24)}",
        f"- Pi_branch^(SU(3)_8) = V_px^(-1) * P_mu = {_format_decimal(derivation.su3_branching_pressure_scale, places=24)}",
        "",
        "Mu Audit",
        f"- mu_struct = mu_audit = (c_q/c_l) * Pi_branch^(SU(3)_8) = ({geometry.central_charge_ratio_fraction.numerator}/{geometry.central_charge_ratio_fraction.denominator}) * {_format_decimal(derivation.su3_branching_pressure_scale, places=24)} = {_format_decimal(derivation.mu_audit, places=24)}",
        f"- equivalently, mu_struct = mu_audit = (c_q/c_l) * V_px^(-1) * Pi_vac^2 / [(1-kappa_D5) * kappa_D5^(1/3)] = ({geometry.structural_prefactor_fraction.numerator}/{geometry.structural_prefactor_fraction.denominator}) * P_mu",
        "",
        "Audit vs. CODATA",
        f"- scipy.constants.proton_mass = {_format_decimal(codata.proton_mass_kg, places=24)} kg",
        f"- scipy.constants.electron_mass = {_format_decimal(codata.electron_mass_kg, places=24)} kg",
        f"- m_p/m_e via scipy.constants.proton_mass / electron_mass = {_format_decimal(codata.mass_ratio, places=12)}",
        f"- audit minus CODATA = {_format_decimal(derivation.mu_audit - codata.mass_ratio, places=24)}",
        f"- |audit - CODATA| = {_format_decimal(derivation.absolute_delta, places=24)}",
        f"- relative error = {_format_decimal(derivation.relative_error, places=24)}",
        f"- tolerance = {_format_decimal(derivation.tolerance, places=12)}",
        f"- within one-copy tolerance = {derivation.relative_error <= derivation.tolerance}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive the proton-to-electron mass ratio from branch geometry.")
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION, help="Decimal precision used for the structural ledger.")
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    print(build_proton_ratio_ledger(precision=max(args.precision, DEFAULT_PRECISION)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
