from __future__ import annotations

"""Programmatic derivation API for the SHBT benchmark ledgers.

This module consolidates the mathematical content previously embedded in the
standalone ``scripts/derive_universe.py`` and ``scripts/derive_lambda.py``
entrypoints into an importable core API. External researchers can now generate
the same benchmark derivation ledgers directly from Python via
``UniverseFactory.generate_ledger(...)`` or by calling the individual
derivation methods exposed on the class.
"""

import json
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Literal

from scipy.constants import electron_mass, proton_mass

from shbt.constants import (
    BENCHMARK_DIAGNOSTICS_FILENAME,
    G_SM,
    HBAR_EV_SECONDS,
    HOLOGRAPHIC_BITS,
    KAPPA_D5,
    LEPTON_LEVEL,
    PARENT_LEVEL,
    PLANCK_LENGTH_M,
    PLANCK_MASS_EV,
    QUARK_LEVEL,
    RANK_DIFFERENCE,
    SU2_DIMENSION,
    SU2_DUAL_COXETER,
    SU3_DIMENSION,
    SU3_DUAL_COXETER,
)
from shbt.core import algebra
from shbt.core.engine import (
    quark_branching_index,
    surface_tension_gauge_alpha_inverse,
    topological_mass_coordinate_ev,
    topological_newton_coordinate_ev_minus2,
    topological_planck_mass_ev,
    wzw_central_charge_fraction,
)
from shbt.core.ontic_cascade import LogicRelation, OnticAxioms, OnticCascade, evaluate_ontic_cascade
from shbt.core.rigidity_kernel import stabilize_boundary, stabilize_classmethods
from shbt.main import (
    derive_cosmology_anchor,
    holographic_lambda_scaling_identity_si_m2,
    holographic_surface_tension_lambda_si_m2,
    lambda_si_m2_to_ev2,
    verify_unity_of_scale,
)
from shbt.paths import ProjectPaths
from shbt.physics_engine import quark_branching_pressure
from shbt.verification.comparators import (
    CODATA_FINE_STRUCTURE_ALPHA_INVERSE as COMPARATOR_CODATA_FINE_STRUCTURE_ALPHA_INVERSE,
    CODATA_PROTON_TO_ELECTRON_MASS_RATIO as COMPARATOR_CODATA_PROTON_TO_ELECTRON_MASS_RATIO,
    PLANCK2018_LAMBDA_SI_M2 as COMPARATOR_PLANCK2018_LAMBDA_SI_M2,
)


DEFAULT_PRECISION = 50
_GUARD_DIGITS = 12
D5_WEIGHT_SIMPLEX_HYPERAREA_FRACTION = Fraction(160, 1521)
LIVE_BENCHMARK_AUDIT_SKIPPED_MESSAGE = "Benchmark artifacts not found; skipping live comparison audit"
DEFAULT_RELATIVE_TOLERANCE = Decimal("1e-3")
LEPTON_CENTRAL_CHARGE_FRACTION = Fraction(39, 14)
QUARK_CENTRAL_CHARGE_FRACTION = Fraction(64, 11)
BRANCH_PIXEL_VOLUME_FRACTION = Fraction(3, 13)
BRANCH_PIXEL_VOLUME_INVERSE_FRACTION = Fraction(
    BRANCH_PIXEL_VOLUME_FRACTION.denominator,
    BRANCH_PIXEL_VOLUME_FRACTION.numerator,
)
BRANCH_STRUCTURAL_PREFRACTOR_FRACTION = (
    QUARK_CENTRAL_CHARGE_FRACTION / LEPTON_CENTRAL_CHARGE_FRACTION
) * BRANCH_PIXEL_VOLUME_INVERSE_FRACTION


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


@lru_cache(maxsize=None)
def decimal_pi(precision: int = DEFAULT_PRECISION) -> Decimal:
    with localcontext() as context:
        context.prec = precision + _GUARD_DIGITS
        one = Decimal(1)
        two = Decimal(2)
        four = Decimal(4)
        a_term = one
        b_term = one / two.sqrt()
        t_term = Decimal("0.25")
        p_term = one
        pi_value = Decimal(0)
        for _ in range(8):
            a_next = (a_term + b_term) / two
            b_term = (a_term * b_term).sqrt()
            delta = a_term - a_next
            t_term -= p_term * delta * delta
            a_term = a_next
            p_term *= two
            pi_value = ((a_term + b_term) * (a_term + b_term)) / (four * t_term)
        context.prec = precision
        return +pi_value


def decimal_sin(value: Decimal, *, precision: int = DEFAULT_PRECISION) -> Decimal:
    with localcontext() as context:
        context.prec = precision + _GUARD_DIGITS
        term = value
        result = value
        threshold = Decimal(1).scaleb(-(precision + _GUARD_DIGITS // 2))
        index = 1
        while True:
            denominator = Decimal((2 * index) * (2 * index + 1))
            term *= -(value * value) / denominator
            result += term
            if abs(term) <= threshold:
                break
            index += 1
        context.prec = precision
        return +result


def su2_total_quantum_dimension_decimal(level: int, *, precision: int = DEFAULT_PRECISION) -> Decimal:
    with localcontext() as context:
        context.prec = precision + _GUARD_DIGITS
        level_plus_two = Decimal(level + 2)
        prefactor = (level_plus_two / Decimal(2)).sqrt()
        theta = decimal_pi(context.prec) / level_plus_two
        dimension = prefactor / decimal_sin(theta, precision=context.prec)
        context.prec = precision
        return +dimension


def quarter_power_inverse(value: Decimal, *, precision: int = DEFAULT_PRECISION) -> Decimal:
    with localcontext() as context:
        context.prec = precision + _GUARD_DIGITS
        if value <= 0:
            raise ValueError("Quarter-power inverse requires a positive Decimal.")
        result = Decimal(1) / value.sqrt().sqrt()
        context.prec = precision
        return +result


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
class AlphaSurfaceDerivation:
    visible_support: int | Fraction
    level_density_ratio: Fraction
    alpha_inverse_fraction: Fraction
    alpha_inverse_decimal: Decimal
    live_alpha_inverse: Decimal
    codata_alpha_inverse: Decimal
    ontic_cascade: OnticCascade | None = None


@dataclass(frozen=True)
class ObservationalBoundaryConditionDerivation:
    hbar_ev_seconds: Decimal


@dataclass(frozen=True)
class KappaDerivation:
    simplex_fraction: Fraction
    sqrt_ten: Decimal
    weight_simplex_hyperarea: Decimal
    su2_total_quantum_dimension: Decimal
    beta: Decimal
    eta_spin: Decimal
    kappa: Decimal
    live_kappa: Decimal


@dataclass(frozen=True)
class MassBridgeDerivation:
    branch_planck_mass_ev: Decimal
    configured_planck_mass_ev: Decimal
    holographic_bits: Decimal
    holographic_bits_quarter_inverse: Decimal
    neutrino_floor_ev: Decimal
    neutrino_floor_mev: Decimal
    live_neutrino_floor_ev: Decimal


@dataclass(frozen=True)
class UnityOfScaleDerivation:
    branch_planck_mass_ev: Decimal
    holographic_bits: Decimal
    lambda_holo_ev2: Decimal
    g_newton_topological_ev_minus2: Decimal
    g_newton_inverse_square_ev_minus2: Decimal
    g_newton_bridge_drift: Decimal
    kappa_fourth: Decimal
    expanded_theorem_rhs_ev2: Decimal
    reduced_theorem_rhs_ev2: Decimal
    identity_ratio: Decimal
    epsilon_lambda: Decimal
    decimal_tolerance: Decimal
    register_noise_floor: Decimal

    @property
    def passed(self) -> bool:
        return self.epsilon_lambda <= self.decimal_tolerance


@dataclass(frozen=True)
class LambdaSurfaceDerivation:
    holographic_bits: Decimal
    planck_length_m: Decimal
    scaling_identity_si_m2: Decimal
    scaling_identity_ev2: Decimal
    lambda_holo_si_m2: Decimal
    lambda_holo_ev2: Decimal
    live_scaling_identity_si_m2: Decimal
    live_lambda_holo_si_m2: Decimal
    live_lambda_holo_ev2: Decimal
    scaling_identity_drift_si_m2: Decimal
    lambda_holo_drift_si_m2: Decimal
    lambda_holo_ev2_drift: Decimal
    surface_tension_prefactor: Decimal
    anchor_lambda_si_m2: Decimal
    anchor_lambda_ev2: Decimal
    anchor_ratio: Decimal
    deviation_percent: Decimal


@dataclass(frozen=True)
class TopologicalVacuum:
    lepton_level: int
    quark_level: int
    parent_level: int
    generation_count: int = G_SM

    @property
    def branch(self) -> tuple[int, int, int]:
        return (self.lepton_level, self.quark_level, self.parent_level)

    @property
    def visible_support(self) -> int:
        return int(self.lepton_level) + int(self.quark_level)

    def to_ontic_axioms(self) -> OnticAxioms:
        return OnticAxioms(
            lepton_level=int(self.lepton_level),
            quark_level=int(self.quark_level),
            parent_level=int(self.parent_level),
            generation_count=int(self.generation_count),
        )


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


@dataclass(frozen=True)
class PhysicalLedger:
    vacuum: TopologicalVacuum
    observational_boundary_conditions: ObservationalBoundaryConditionDerivation
    alpha_surface: AlphaSurfaceDerivation
    kappa: KappaDerivation
    proton_ratio: ProtonRatioDerivation
    mass_bridge: MassBridgeDerivation
    unity_of_scale: UnityOfScaleDerivation

    @property
    def alpha_inverse_decimal(self) -> Decimal:
        return self.alpha_surface.alpha_inverse_decimal

    @property
    def mu_residue(self) -> Decimal:
        return self.proton_ratio.mu_audit

    @property
    def neutrino_floor_mev(self) -> Decimal:
        return self.mass_bridge.neutrino_floor_mev

    @property
    def residues(self) -> dict[str, int | Decimal | bool]:
        return {
            "k_l": int(self.vacuum.lepton_level),
            "k_q": int(self.vacuum.quark_level),
            "K": int(self.vacuum.parent_level),
            "LEPTON_LEVEL": int(self.vacuum.lepton_level),
            "QUARK_LEVEL": int(self.vacuum.quark_level),
            "PARENT_LEVEL": int(self.vacuum.parent_level),
            "alpha_inverse_decimal": self.alpha_inverse_decimal,
            "mu_residue": self.mu_residue,
            "neutrino_floor_mev": self.neutrino_floor_mev,
            "epsilon_lambda": self.unity_of_scale.epsilon_lambda,
            "decimal_tolerance": self.unity_of_scale.decimal_tolerance,
            "register_noise_floor": self.unity_of_scale.register_noise_floor,
            "decimal_passed": self.unity_of_scale.passed,
        }


@dataclass(frozen=True)
class CheckedInUnityPayload:
    source_path: str
    epsilon_lambda: Decimal
    exact_epsilon_lambda: Decimal
    numerical_residual: Decimal
    register_noise_floor: Decimal
    exact_register_noise_floor: Decimal
    passed: bool

    @property
    def comparison_tuple(self) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal, bool]:
        return (
            self.epsilon_lambda,
            self.exact_epsilon_lambda,
            self.numerical_residual,
            self.register_noise_floor,
            self.exact_register_noise_floor,
            self.passed,
        )


class UniverseFactory:
    """Importable façade for the SHBT universe/lambda derivation ledgers."""

    @classmethod
    def benchmark_vacuum(cls) -> TopologicalVacuum:
        del cls
        return TopologicalVacuum(
            lepton_level=LEPTON_LEVEL,
            quark_level=QUARK_LEVEL,
            parent_level=PARENT_LEVEL,
            generation_count=G_SM,
        )

    @classmethod
    def derive_observational_boundary_conditions(cls) -> ObservationalBoundaryConditionDerivation:
        del cls
        return ObservationalBoundaryConditionDerivation(hbar_ev_seconds=_decimal(HBAR_EV_SECONDS))

    @classmethod
    def derive_alpha_surface(
        cls,
        *,
        precision: int = DEFAULT_PRECISION,
        vacuum: TopologicalVacuum | None = None,
        logic_relation: LogicRelation | None = None,
    ) -> AlphaSurfaceDerivation:
        resolved_vacuum = cls.benchmark_vacuum() if vacuum is None else vacuum
        ontic_cascade = cls.evaluate_ontic_cascade(
            precision=precision,
            vacuum=resolved_vacuum,
            logic_relation=logic_relation,
        )
        visible_support = ontic_cascade.visible_support
        level_density_ratio = ontic_cascade.level_density_ratio
        alpha_inverse_fraction = ontic_cascade.alpha_inverse_fraction
        alpha_inverse_decimal = ontic_cascade.alpha_inverse_decimal
        live_alpha_inverse = _decimal(
            surface_tension_gauge_alpha_inverse(
                parent_level=resolved_vacuum.parent_level,
                lepton_level=resolved_vacuum.lepton_level,
                quark_level=resolved_vacuum.quark_level,
                generation_count=resolved_vacuum.generation_count,
                visible_support=visible_support,
            )
        )
        codata_alpha_inverse = _decimal(COMPARATOR_CODATA_FINE_STRUCTURE_ALPHA_INVERSE)
        return AlphaSurfaceDerivation(
            visible_support=visible_support,
            level_density_ratio=level_density_ratio,
            alpha_inverse_fraction=alpha_inverse_fraction,
            alpha_inverse_decimal=alpha_inverse_decimal,
            live_alpha_inverse=live_alpha_inverse,
            codata_alpha_inverse=codata_alpha_inverse,
            ontic_cascade=ontic_cascade,
        )

    @classmethod
    def evaluate_ontic_cascade(
        cls,
        *,
        precision: int = DEFAULT_PRECISION,
        vacuum: TopologicalVacuum | None = None,
        logic_relation: LogicRelation | None = None,
    ) -> OnticCascade:
        del cls
        resolved_vacuum = UniverseFactory.benchmark_vacuum() if vacuum is None else vacuum
        return evaluate_ontic_cascade(
            resolved_vacuum.to_ontic_axioms(),
            precision=precision,
            logic_relation=logic_relation,
        )

    @classmethod
    def derive_kappa_d5(
        cls,
        *,
        precision: int = DEFAULT_PRECISION,
        vacuum: TopologicalVacuum | None = None,
    ) -> KappaDerivation:
        resolved_vacuum = cls.benchmark_vacuum() if vacuum is None else vacuum
        with localcontext() as context:
            context.prec = precision + _GUARD_DIGITS
            simplex_fraction = D5_WEIGHT_SIMPLEX_HYPERAREA_FRACTION
            sqrt_ten = Decimal(10).sqrt()
            weight_simplex_hyperarea = _fraction_to_decimal(simplex_fraction) * sqrt_ten
            total_quantum_dimension = su2_total_quantum_dimension_decimal(resolved_vacuum.lepton_level, precision=context.prec)
            beta = total_quantum_dimension.ln() / Decimal(2)
            eta_spin = (_decimal(Fraction(347, 1)) - _decimal(Fraction(8, 1)) * beta * beta) / _decimal(Fraction(351, 1))
            kappa = (_decimal(Fraction(16, 5)) * weight_simplex_hyperarea * eta_spin).sqrt()
            context.prec = precision
            return KappaDerivation(
                simplex_fraction=simplex_fraction,
                sqrt_ten=+sqrt_ten,
                weight_simplex_hyperarea=+weight_simplex_hyperarea,
                su2_total_quantum_dimension=+total_quantum_dimension,
                beta=+beta,
                eta_spin=+eta_spin,
                kappa=+kappa,
                live_kappa=_decimal(KAPPA_D5),
            )

    @classmethod
    def derive_central_charge_geometry(
        cls,
        *,
        vacuum: TopologicalVacuum | None = None,
    ) -> CentralChargeGeometry:
        resolved_vacuum = cls.benchmark_vacuum() if vacuum is None else vacuum
        lepton_central_charge_fraction = wzw_central_charge_fraction(
            resolved_vacuum.lepton_level,
            SU2_DIMENSION,
            SU2_DUAL_COXETER,
        )
        quark_central_charge_fraction = wzw_central_charge_fraction(
            resolved_vacuum.quark_level,
            SU3_DIMENSION,
            SU3_DUAL_COXETER,
        )
        central_charge_ratio_fraction = quark_central_charge_fraction / lepton_central_charge_fraction
        central_charge_ratio_decimal = _fraction_to_decimal(central_charge_ratio_fraction)
        resolved_quark_branching_index = quark_branching_index(
            resolved_vacuum.parent_level,
            resolved_vacuum.quark_level,
        )
        branch_pixel_simplex_volume_fraction = Fraction(SU3_DUAL_COXETER, resolved_quark_branching_index)
        inverse_pixel_volume_fraction = Fraction(
            branch_pixel_simplex_volume_fraction.denominator,
            branch_pixel_simplex_volume_fraction.numerator,
        )
        inverse_pixel_volume_decimal = _fraction_to_decimal(inverse_pixel_volume_fraction)

        if resolved_vacuum.branch == cls.benchmark_vacuum().branch:
            assert lepton_central_charge_fraction == LEPTON_CENTRAL_CHARGE_FRACTION
            assert quark_central_charge_fraction == QUARK_CENTRAL_CHARGE_FRACTION
            assert central_charge_ratio_fraction == QUARK_CENTRAL_CHARGE_FRACTION / LEPTON_CENTRAL_CHARGE_FRACTION
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

    @classmethod
    def derive_vacuum_pressure(
        cls,
        *,
        vacuum: TopologicalVacuum | None = None,
    ) -> VacuumPressureDerivation:
        resolved_vacuum = cls.benchmark_vacuum() if vacuum is None else vacuum
        visible_block = algebra.su3_low_weight_block(resolved_vacuum.quark_level)
        reference_entry_magnitude = abs(complex(visible_block[0, 0]))
        vacuum_pressure = quark_branching_pressure(visible_block, RANK_DIFFERENCE)
        return VacuumPressureDerivation(
            visible_reference_entry_magnitude=_decimal(reference_entry_magnitude),
            vacuum_pressure=_decimal(vacuum_pressure),
        )

    @classmethod
    def derive_codata_mass_ratio(cls) -> CodataMassRatioAudit:
        del cls
        proton_mass_kg = _decimal(proton_mass)
        electron_mass_kg = _decimal(electron_mass)
        mass_ratio = _decimal(COMPARATOR_CODATA_PROTON_TO_ELECTRON_MASS_RATIO)
        return CodataMassRatioAudit(
            proton_mass_kg=proton_mass_kg,
            electron_mass_kg=electron_mass_kg,
            mass_ratio=mass_ratio,
        )

    @classmethod
    def derive_proton_ratio(
        cls,
        *,
        precision: int = DEFAULT_PRECISION,
        kappa: Decimal | None = None,
        tolerance: Decimal = DEFAULT_RELATIVE_TOLERANCE,
        vacuum: TopologicalVacuum | None = None,
    ) -> ProtonRatioDerivation:
        resolved_vacuum = cls.benchmark_vacuum() if vacuum is None else vacuum
        geometry = cls.derive_central_charge_geometry(vacuum=resolved_vacuum)
        vacuum_pressure = cls.derive_vacuum_pressure(vacuum=resolved_vacuum)
        codata_audit = cls.derive_codata_mass_ratio()

        with localcontext() as context:
            context.prec = precision + _GUARD_DIGITS
            central_charge_ratio = geometry.central_charge_ratio_decimal
            inverse_pixel_volume = geometry.inverse_pixel_volume_decimal
            structural_prefactor = central_charge_ratio * inverse_pixel_volume
            resolved_kappa = cls.derive_kappa_d5(precision=context.prec, vacuum=resolved_vacuum).kappa if kappa is None else _decimal(kappa)
            kappa_d5_cuberoot = decimal_cuberoot(resolved_kappa, precision=context.prec)
            geometric_friction_factor = (Decimal(1) - resolved_kappa) * kappa_d5_cuberoot
            pressure_loading = (vacuum_pressure.vacuum_pressure * vacuum_pressure.vacuum_pressure) / geometric_friction_factor
            su3_branching_pressure_scale = inverse_pixel_volume * pressure_loading
            mu_audit = structural_prefactor * pressure_loading
            absolute_delta = abs(mu_audit - codata_audit.mass_ratio)
            relative_error = absolute_delta / codata_audit.mass_ratio
            context.prec = precision

        if resolved_vacuum.branch == cls.benchmark_vacuum().branch:
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
            kappa_d5=+resolved_kappa,
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

    @classmethod
    def derive_mass_bridge(
        cls,
        *,
        precision: int = DEFAULT_PRECISION,
        kappa: Decimal | None = None,
    ) -> MassBridgeDerivation:
        with localcontext() as context:
            context.prec = precision + _GUARD_DIGITS
            resolved_kappa = cls.derive_kappa_d5(precision=context.prec).kappa if kappa is None else _decimal(kappa)
            branch_planck_mass_ev = _decimal(topological_planck_mass_ev())
            configured_planck_mass_ev = _decimal(PLANCK_MASS_EV)
            holographic_bits = _decimal(HOLOGRAPHIC_BITS)
            holographic_bits_quarter_inverse = quarter_power_inverse(holographic_bits, precision=context.prec)
            neutrino_floor_ev = resolved_kappa * branch_planck_mass_ev * holographic_bits_quarter_inverse
            neutrino_floor_mev = neutrino_floor_ev * Decimal(1000)
            live_neutrino_floor_ev = _decimal(
                topological_mass_coordinate_ev(
                    bit_count=HOLOGRAPHIC_BITS,
                    kappa_geometric=float(resolved_kappa),
                )
            )
            context.prec = precision
            return MassBridgeDerivation(
                branch_planck_mass_ev=+branch_planck_mass_ev,
                configured_planck_mass_ev=+configured_planck_mass_ev,
                holographic_bits=+holographic_bits,
                holographic_bits_quarter_inverse=+holographic_bits_quarter_inverse,
                neutrino_floor_ev=+neutrino_floor_ev,
                neutrino_floor_mev=+neutrino_floor_mev,
                live_neutrino_floor_ev=+live_neutrino_floor_ev,
            )

    @classmethod
    def derive_unity_of_scale(
        cls,
        *,
        precision: int = DEFAULT_PRECISION,
        kappa: Decimal | None = None,
        mass_bridge: MassBridgeDerivation | None = None,
    ) -> UnityOfScaleDerivation:
        with localcontext() as context:
            context.prec = precision + _GUARD_DIGITS
            resolved_kappa = cls.derive_kappa_d5(precision=context.prec).kappa if kappa is None else _decimal(kappa)
            resolved_mass_bridge = cls.derive_mass_bridge(precision=context.prec, kappa=resolved_kappa) if mass_bridge is None else mass_bridge
            three_pi = Decimal(3) * decimal_pi(context.prec)
            branch_planck_mass_ev = resolved_mass_bridge.branch_planck_mass_ev
            holographic_bits = resolved_mass_bridge.holographic_bits
            lambda_holo_ev2 = three_pi * branch_planck_mass_ev * branch_planck_mass_ev / holographic_bits
            g_newton_topological_ev_minus2 = _decimal(
                topological_newton_coordinate_ev_minus2(branch_planck_mass_ev=float(branch_planck_mass_ev))
            )
            g_newton_inverse_square_ev_minus2 = Decimal(1) / (branch_planck_mass_ev * branch_planck_mass_ev)
            g_newton_bridge_drift = abs(g_newton_topological_ev_minus2 - g_newton_inverse_square_ev_minus2)
            kappa_fourth = resolved_kappa**4
            expanded_theorem_rhs_ev2 = (
                (three_pi / kappa_fourth)
                * g_newton_inverse_square_ev_minus2
                * (resolved_mass_bridge.neutrino_floor_ev**4)
            )
            reduced_theorem_rhs_ev2 = three_pi * branch_planck_mass_ev * branch_planck_mass_ev / holographic_bits
            identity_ratio = lambda_holo_ev2 / expanded_theorem_rhs_ev2
            epsilon_lambda = abs(Decimal(1) - identity_ratio)
            decimal_tolerance = Decimal(1).scaleb(-(precision - _GUARD_DIGITS))
            register_noise_floor = Decimal(1) / holographic_bits
            context.prec = precision
            derivation = UnityOfScaleDerivation(
                branch_planck_mass_ev=+branch_planck_mass_ev,
                holographic_bits=+holographic_bits,
                lambda_holo_ev2=+lambda_holo_ev2,
                g_newton_topological_ev_minus2=+g_newton_topological_ev_minus2,
                g_newton_inverse_square_ev_minus2=+g_newton_inverse_square_ev_minus2,
                g_newton_bridge_drift=+g_newton_bridge_drift,
                kappa_fourth=+kappa_fourth,
                expanded_theorem_rhs_ev2=+expanded_theorem_rhs_ev2,
                reduced_theorem_rhs_ev2=+reduced_theorem_rhs_ev2,
                identity_ratio=+identity_ratio,
                epsilon_lambda=+epsilon_lambda,
                decimal_tolerance=+decimal_tolerance,
                register_noise_floor=+register_noise_floor,
            )
        assert derivation.passed, (
            "Unity of Scale Identity no longer closes within the Decimal audit tolerance: "
            f"epsilon_Lambda={derivation.epsilon_lambda}, tolerance={derivation.decimal_tolerance}."
        )
        return derivation

    @classmethod
    def derive_lambda_surface(cls, *, precision: int = DEFAULT_PRECISION) -> LambdaSurfaceDerivation:
        del cls
        with localcontext() as context:
            context.prec = precision + _GUARD_DIGITS
            holographic_bits = _decimal(HOLOGRAPHIC_BITS)
            planck_length_m = _decimal(PLANCK_LENGTH_M)
            scaling_identity_si_m2 = Decimal(1) / (holographic_bits * planck_length_m * planck_length_m)
            scaling_identity_ev2 = _decimal(lambda_si_m2_to_ev2(float(scaling_identity_si_m2)))
            surface_tension_prefactor = Decimal(3) * decimal_pi(context.prec)
            lambda_holo_si_m2 = surface_tension_prefactor * scaling_identity_si_m2
            lambda_holo_ev2 = _decimal(lambda_si_m2_to_ev2(float(lambda_holo_si_m2)))

            live_scaling_identity_si_m2 = _decimal(holographic_lambda_scaling_identity_si_m2())
            live_lambda_holo_si_m2 = _decimal(holographic_surface_tension_lambda_si_m2())
            live_lambda_holo_ev2 = _decimal(lambda_si_m2_to_ev2(float(live_lambda_holo_si_m2)))

            anchor_lambda_si_m2 = _decimal(COMPARATOR_PLANCK2018_LAMBDA_SI_M2)
            anchor_lambda_ev2 = _decimal(lambda_si_m2_to_ev2(COMPARATOR_PLANCK2018_LAMBDA_SI_M2))
            anchor_ratio = lambda_holo_si_m2 / anchor_lambda_si_m2
            deviation_percent = Decimal(100) * abs(anchor_ratio - Decimal(1))

            scaling_identity_drift_si_m2 = abs(scaling_identity_si_m2 - live_scaling_identity_si_m2)
            lambda_holo_drift_si_m2 = abs(lambda_holo_si_m2 - live_lambda_holo_si_m2)
            lambda_holo_ev2_drift = abs(lambda_holo_ev2 - live_lambda_holo_ev2)

            context.prec = precision
            return LambdaSurfaceDerivation(
                holographic_bits=+holographic_bits,
                planck_length_m=+planck_length_m,
                scaling_identity_si_m2=+scaling_identity_si_m2,
                scaling_identity_ev2=+scaling_identity_ev2,
                lambda_holo_si_m2=+lambda_holo_si_m2,
                lambda_holo_ev2=+lambda_holo_ev2,
                live_scaling_identity_si_m2=+live_scaling_identity_si_m2,
                live_lambda_holo_si_m2=+live_lambda_holo_si_m2,
                live_lambda_holo_ev2=+live_lambda_holo_ev2,
                scaling_identity_drift_si_m2=+scaling_identity_drift_si_m2,
                lambda_holo_drift_si_m2=+lambda_holo_drift_si_m2,
                lambda_holo_ev2_drift=+lambda_holo_ev2_drift,
                surface_tension_prefactor=+surface_tension_prefactor,
                anchor_lambda_si_m2=+anchor_lambda_si_m2,
                anchor_lambda_ev2=+anchor_lambda_ev2,
                anchor_ratio=+anchor_ratio,
                deviation_percent=+deviation_percent,
            )

    @classmethod
    def _checked_in_diagnostics_paths(cls) -> tuple[Path, ...]:
        del cls
        results_dir = ProjectPaths.RESULTS
        if not results_dir.is_dir():
            return ()
        preferred_paths = (
            results_dir / "final" / BENCHMARK_DIAGNOSTICS_FILENAME,
            results_dir / BENCHMARK_DIAGNOSTICS_FILENAME,
        )
        existing_preferred_paths = tuple(path for path in preferred_paths if path.is_file())
        if existing_preferred_paths:
            return existing_preferred_paths
        for candidate in sorted(results_dir.rglob(BENCHMARK_DIAGNOSTICS_FILENAME)):
            if candidate.is_file():
                return (candidate,)
        return ()

    @classmethod
    def _display_diagnostics_path(cls, diagnostics_path: Path) -> str:
        del cls
        try:
            return str(diagnostics_path.relative_to(ProjectPaths.ROOT))
        except ValueError:
            return str(diagnostics_path)

    @classmethod
    def load_checked_in_unity_payloads(cls) -> list[CheckedInUnityPayload]:
        payloads: list[CheckedInUnityPayload] = []
        diagnostics_paths = cls._checked_in_diagnostics_paths()

        if not diagnostics_paths:
            print(LIVE_BENCHMARK_AUDIT_SKIPPED_MESSAGE)
            return []

        for diagnostics_path in diagnostics_paths:
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"), parse_float=Decimal)

            unity_payload = diagnostics.get("unity_of_scale_identity")
            if not isinstance(unity_payload, dict):
                raise KeyError(
                    f"Checked-in benchmark diagnostics at {diagnostics_path} do not expose a unity_of_scale_identity payload."
                )

            payloads.append(
                CheckedInUnityPayload(
                    source_path=cls._display_diagnostics_path(diagnostics_path),
                    epsilon_lambda=_decimal(unity_payload["epsilon_lambda"]),
                    exact_epsilon_lambda=_decimal(unity_payload["exact_epsilon_lambda"]),
                    numerical_residual=_decimal(unity_payload["numerical_residual"]),
                    register_noise_floor=_decimal(unity_payload["register_noise_floor"]),
                    exact_register_noise_floor=_decimal(unity_payload["exact_register_noise_floor"]),
                    passed=bool(unity_payload["passed"]),
                )
            )

        if not payloads:
            print(LIVE_BENCHMARK_AUDIT_SKIPPED_MESSAGE)
            return []

        reference_payload = payloads[0]
        for payload in payloads[1:]:
            assert payload.comparison_tuple == reference_payload.comparison_tuple, (
                "Checked-in benchmark diagnostics no longer mirror the same unity-of-scale payload: "
                f"{reference_payload.source_path} != {payload.source_path}."
            )

        return payloads

    @classmethod
    def calculate_physical_ledger(cls, *, precision: int = DEFAULT_PRECISION) -> PhysicalLedger:
        resolved_precision = max(int(precision), DEFAULT_PRECISION)
        vacuum = TopologicalVacuum(
            lepton_level=LEPTON_LEVEL,
            quark_level=QUARK_LEVEL,
            parent_level=PARENT_LEVEL,
            generation_count=G_SM,
        )
        observational_boundary_conditions = cls.derive_observational_boundary_conditions()
        alpha_surface = cls.derive_alpha_surface(precision=resolved_precision, vacuum=vacuum)
        kappa = cls.derive_kappa_d5(precision=resolved_precision, vacuum=vacuum)
        proton_ratio = cls.derive_proton_ratio(
            precision=resolved_precision,
            kappa=kappa.kappa,
            vacuum=vacuum,
        )
        mass_bridge = cls.derive_mass_bridge(precision=resolved_precision, kappa=kappa.kappa)
        unity_of_scale = cls.derive_unity_of_scale(
            precision=resolved_precision,
            kappa=kappa.kappa,
            mass_bridge=mass_bridge,
        )
        return PhysicalLedger(
            vacuum=vacuum,
            observational_boundary_conditions=observational_boundary_conditions,
            alpha_surface=alpha_surface,
            kappa=kappa,
            proton_ratio=proton_ratio,
            mass_bridge=mass_bridge,
            unity_of_scale=unity_of_scale,
        )

    @classmethod
    def generate_residual_payload(
        cls,
        *,
        precision: int = DEFAULT_PRECISION,
    ) -> dict[str, object]:
        from shbt.main import build_quantified_two_loop_residuals

        resolved_precision = max(int(precision), DEFAULT_PRECISION)
        physical_ledger = cls.calculate_physical_ledger(precision=resolved_precision)
        lambda_surface = cls.derive_lambda_surface(precision=resolved_precision)

        benchmark_tuple = [
            int(physical_ledger.vacuum.lepton_level),
            int(physical_ledger.vacuum.quark_level),
            int(physical_ledger.vacuum.parent_level),
        ]
        derivation_residues = {
            "precision": resolved_precision,
            "benchmark_tuple": benchmark_tuple,
            "alpha_inverse_decimal": float(physical_ledger.alpha_surface.alpha_inverse_decimal),
            "codata_alpha_inverse": float(physical_ledger.alpha_surface.codata_alpha_inverse),
            "m_nu_ev": float(physical_ledger.mass_bridge.neutrino_floor_ev),
            "m_nu_mev": float(physical_ledger.mass_bridge.neutrino_floor_mev),
            "epsilon_lambda": float(physical_ledger.unity_of_scale.epsilon_lambda),
            "decimal_tolerance": float(physical_ledger.unity_of_scale.decimal_tolerance),
            "register_noise_floor": float(physical_ledger.unity_of_scale.register_noise_floor),
            "lambda_holo_si_m2": float(lambda_surface.lambda_holo_si_m2),
            "lambda_obs_si_m2": float(lambda_surface.anchor_lambda_si_m2),
        }

        payload = dict(build_quantified_two_loop_residuals())
        payload["artifact"] = "Quantified Two-Loop Residuals"
        payload["benchmark_tuple"] = benchmark_tuple
        payload["derivation_residues"] = derivation_residues
        return payload

    @classmethod
    def build_residue_dictionary(cls, *, precision: int = DEFAULT_PRECISION) -> dict[str, int | Decimal | bool]:
        return cls.calculate_physical_ledger(precision=precision).residues

    @classmethod
    def build_derivation_ledger(cls, *, precision: int = DEFAULT_PRECISION) -> str:
        physical_ledger = cls.calculate_physical_ledger(precision=precision)
        vacuum = physical_ledger.vacuum
        obc = physical_ledger.observational_boundary_conditions
        alpha = physical_ledger.alpha_surface
        kappa = physical_ledger.kappa
        proton_ratio = physical_ledger.proton_ratio
        mass = physical_ledger.mass_bridge
        unity = physical_ledger.unity_of_scale
        tension_audit = cls.derive_tension_audit(precision=precision)

        lines = [
            "Derivation Ledger",
            "=================",
            "",
            "Branch Integers",
            f"- k_l = {vacuum.lepton_level}",
            f"- k_q = {vacuum.quark_level}",
            f"- K = {vacuum.parent_level}",
            f"- G_SM = {vacuum.generation_count}",
            "",
            "Observational Boundary Conditions",
            f"- hbar [OBC] = {_format_decimal(obc.hbar_ev_seconds, places=24)} eV s",
            "",
            "Alpha Surface Inverse",
            f"- visible support = k_l + k_q = {alpha.visible_support}",
            f"- level-density ratio = K/(k_l + k_q) = {alpha.level_density_ratio.numerator}/{alpha.level_density_ratio.denominator}",
            f"- alpha_surf^-1 = G_SM * K/(k_l + k_q) = {alpha.alpha_inverse_fraction.numerator}/{alpha.alpha_inverse_fraction.denominator}",
            f"- alpha_surf^-1 [decimal] = {_format_decimal(alpha.alpha_inverse_decimal, places=24)}",
            f"- tn.py live check = {_format_decimal(alpha.live_alpha_inverse, places=24)}",
            f"- CODATA alpha^-1 = {_format_decimal(alpha.codata_alpha_inverse, places=12)}",
            f"- unresolved Two-Loop Residual = alpha_surf^-1 - alpha_CODATA^-1 = {_format_decimal(alpha.alpha_inverse_decimal - alpha.codata_alpha_inverse, places=24)}",
            "",
            "D5 Hyperarea Invariant",
            f"- simplex hyperarea prefactor = {kappa.simplex_fraction.numerator}/{kappa.simplex_fraction.denominator}",
            f"- sqrt(10) = {_format_decimal(kappa.sqrt_ten, places=24)}",
            f"- R_01^(par/vis) = 160*sqrt(10)/1521 = {_format_decimal(kappa.weight_simplex_hyperarea, places=24)}",
            f"- D_SU(2)({vacuum.lepton_level}) = sqrt((k_l+2)/2)/sin(pi/(k_l+2)) = {_format_decimal(kappa.su2_total_quantum_dimension, places=24)}",
            f"- beta = 1/2 ln D_SU(2) = {_format_decimal(kappa.beta, places=24)}",
            f"- eta_spin = (347 - 8 beta^2)/351 = {_format_decimal(kappa.eta_spin, places=24)}",
            f"- kappa_D5 = sqrt((16/5) * R_01^(par/vis) * eta_spin) = {_format_decimal(kappa.kappa, places=24)}",
            f"- tn.py benchmark kappa = {_format_decimal(kappa.live_kappa, places=24)}",
            f"- derivation drift = {_format_decimal(kappa.kappa - kappa.live_kappa, places=24)}",
            "",
            "Mu Audit",
            f"- c_l = 3*k_l/(k_l + 2) = {proton_ratio.geometry.lepton_central_charge_fraction.numerator}/{proton_ratio.geometry.lepton_central_charge_fraction.denominator}",
            f"- c_q = 8*k_q/(k_q + 3) = {proton_ratio.geometry.quark_central_charge_fraction.numerator}/{proton_ratio.geometry.quark_central_charge_fraction.denominator}",
            f"- c_q/c_l = {proton_ratio.geometry.central_charge_ratio_fraction.numerator}/{proton_ratio.geometry.central_charge_ratio_fraction.denominator}",
            f"- V_px = h^vee_SU(3)/I_Q = {proton_ratio.geometry.branch_pixel_simplex_volume_fraction.numerator}/{proton_ratio.geometry.branch_pixel_simplex_volume_fraction.denominator}",
            f"- V_px^(-1) = {proton_ratio.geometry.inverse_pixel_volume_fraction.numerator}/{proton_ratio.geometry.inverse_pixel_volume_fraction.denominator}",
            f"- Pi_vac = -(Delta r/8) * log|S_00^(low)| = {_format_decimal(proton_ratio.vacuum_pressure.vacuum_pressure, places=24)}",
            f"- mu_struct = (c_q/c_l) * V_px^(-1) * Pi_vac^2 / [(1-kappa_D5) * kappa_D5^(1/3)] = {_format_decimal(proton_ratio.mu_audit, places=24)}",
            f"- CODATA m_p/m_e = {_format_decimal(proton_ratio.codata_audit.mass_ratio, places=12)}",
            f"- relative error = {_format_decimal(proton_ratio.relative_error, places=24)}",
            "",
            "Tier-2 Conformance Audit",
            f"- chi^2 = {_format_decimal(tension_audit.chi_squared, places=24)}",
            f"- reduced chi^2 = {_format_decimal(tension_audit.reduced_chi_squared, places=24)}",
            f"- RMS pull = {_format_decimal(tension_audit.rms_pull, places=24)}",
            "",
            "Finite-Capacity Mass Bridge",
            f"- topological_planck_mass_ev() = {_format_decimal(mass.branch_planck_mass_ev, places=18)} eV",
            f"- PLANCK_MASS_EV from tn.py = {_format_decimal(mass.configured_planck_mass_ev, places=18)} eV",
            f"- HOLOGRAPHIC_BITS = {_format_decimal(mass.holographic_bits, places=18)}",
            f"- N_holo^(-1/4) = {_format_decimal(mass.holographic_bits_quarter_inverse, places=24)}",
            f"- m_nu = kappa_D5 * M_P * N_holo^(-1/4) = {_format_decimal(mass.neutrino_floor_ev, places=24)} eV",
            f"- neutrino floor = {_format_decimal(mass.neutrino_floor_mev, places=24)} meV",
            f"- tn.py live mass bridge = {_format_decimal(mass.live_neutrino_floor_ev, places=24)} eV",
            f"- mass-bridge drift = {_format_decimal(mass.neutrino_floor_ev - mass.live_neutrino_floor_ev, places=24)} eV",
            "",
            "Unity of Scale Identity",
            f"- Lambda_holo(lhs) = 3*pi*M_P^2/N = {_format_decimal(unity.lambda_holo_ev2, places=24)} eV^2",
            f"- G_N = topological_newton_coordinate_ev_minus2(M_P^bridge) = {_format_decimal(unity.g_newton_topological_ev_minus2, places=24)} eV^(-2)",
            f"- direct M_P^(-2) cross-check = {_format_decimal(unity.g_newton_inverse_square_ev_minus2, places=24)} eV^(-2)",
            f"- Newton-lock drift = {_format_decimal(unity.g_newton_bridge_drift, places=24)} eV^(-2)",
            f"- kappa_D5^4 = {_format_decimal(unity.kappa_fourth, places=24)}",
            f"- Lambda_holo(rhs, expanded) = (3*pi/kappa_D5^4) * G_N * m_nu^4 = {_format_decimal(unity.expanded_theorem_rhs_ev2, places=24)} eV^2",
            f"- Lambda_holo(rhs, reduced) = 3*pi*M_P^2/N = {_format_decimal(unity.reduced_theorem_rhs_ev2, places=24)} eV^2",
            f"- lhs/rhs(expanded) = {_format_decimal(unity.identity_ratio, places=24)}",
            f"- epsilon_Lambda = {_format_decimal(unity.epsilon_lambda, places=24)}",
            f"- Decimal audit tolerance = {_format_decimal(unity.decimal_tolerance, places=24)}",
            f"- register noise floor = 1/N = {_format_decimal(unity.register_noise_floor, places=24)}",
            f"- closes in Decimal arithmetic = {unity.passed}",
        ]
        return "\n".join(lines)

    @classmethod
    def build_lambda_ledger(cls, *, precision: int = DEFAULT_PRECISION) -> str:
        lambda_surface = cls.derive_lambda_surface(precision=precision)
        kappa = cls.derive_kappa_d5(precision=precision)
        mass = cls.derive_mass_bridge(precision=precision, kappa=kappa.kappa)
        decimal_unity = cls.derive_unity_of_scale(precision=precision, kappa=kappa.kappa, mass_bridge=mass)
        live_unity = verify_unity_of_scale()
        checked_in_payloads = cls.load_checked_in_unity_payloads()

        live_epsilon_lambda = _decimal(live_unity["epsilon_lambda"])
        live_exact_epsilon_lambda = _decimal(live_unity["exact_epsilon_lambda"])
        live_numerical_residual = _decimal(live_unity["numerical_residual"])
        live_register_noise_floor = _decimal(live_unity["register_noise_floor"])
        live_exact_register_noise_floor = _decimal(live_unity["exact_register_noise_floor"])
        live_passed = bool(live_unity["passed"])

        for payload in checked_in_payloads:
            assert payload.epsilon_lambda == live_epsilon_lambda, (
                "Checked-in epsilon_Lambda no longer matches the live verifier payload: "
                f"{payload.source_path} reports {payload.epsilon_lambda}, live tn.py reports {live_epsilon_lambda}."
            )
            assert payload.exact_epsilon_lambda == live_exact_epsilon_lambda, (
                "Checked-in exact epsilon_Lambda no longer matches the live verifier payload: "
                f"{payload.source_path} reports {payload.exact_epsilon_lambda}, live tn.py reports {live_exact_epsilon_lambda}."
            )
            assert payload.numerical_residual == live_numerical_residual, (
                "Checked-in numerical unity residual no longer matches the live verifier payload: "
                f"{payload.source_path} reports {payload.numerical_residual}, live tn.py reports {live_numerical_residual}."
            )
            assert payload.register_noise_floor == live_register_noise_floor, (
                "Checked-in register noise floor no longer matches the live verifier payload: "
                f"{payload.source_path} reports {payload.register_noise_floor}, live tn.py reports {live_register_noise_floor}."
            )
            assert payload.exact_register_noise_floor == live_exact_register_noise_floor, (
                "Checked-in exact register noise floor no longer matches the live verifier payload: "
                f"{payload.source_path} reports {payload.exact_register_noise_floor}, live tn.py reports {live_exact_register_noise_floor}."
            )
            assert payload.passed == live_passed, (
                "Checked-in unity-of-scale pass/fail flag no longer matches the live verifier payload: "
                f"{payload.source_path} reports {payload.passed}, live tn.py reports {live_passed}."
            )

        lines = [
            "Lambda Ledger",
            "=============",
            "",
            "Branch Integers",
            f"- k_l = {LEPTON_LEVEL}",
            f"- k_q = {QUARK_LEVEL}",
            f"- K = {PARENT_LEVEL}",
            "",
            "Holographic Surface Tension",
            f"- N_holo = {_format_decimal(lambda_surface.holographic_bits, places=18)}",
            f"- L_P = {_format_decimal(lambda_surface.planck_length_m, places=24)} m",
            f"- 1/(L_P^2 N_holo) = {_format_decimal(lambda_surface.scaling_identity_si_m2, places=24)} m^(-2)",
            f"- 1/(L_P^2 N_holo) [eV^2] = {_format_decimal(lambda_surface.scaling_identity_ev2, places=24)} eV^2",
            f"- surface-tension prefactor = 3*pi = {_format_decimal(lambda_surface.surface_tension_prefactor, places=24)}",
            f"- Lambda_holo = 3*pi/(L_P^2 N_holo) = {_format_decimal(lambda_surface.lambda_holo_si_m2, places=24)} m^(-2)",
            f"- Lambda_holo [eV^2] = {_format_decimal(lambda_surface.lambda_holo_ev2, places=24)} eV^2",
            f"- tn.py live 1/(L_P^2 N_holo) = {_format_decimal(lambda_surface.live_scaling_identity_si_m2, places=24)} m^(-2)",
            f"- tn.py live Lambda_holo = {_format_decimal(lambda_surface.live_lambda_holo_si_m2, places=24)} m^(-2)",
            f"- tn.py live Lambda_holo [eV^2] = {_format_decimal(lambda_surface.live_lambda_holo_ev2, places=24)} eV^2",
            f"- scaling-identity drift = {_format_decimal(lambda_surface.scaling_identity_drift_si_m2, places=24)} m^(-2)",
            f"- Lambda_holo drift [m^(-2)] = {_format_decimal(lambda_surface.lambda_holo_drift_si_m2, places=24)} m^(-2)",
            f"- Lambda_holo drift [eV^2] = {_format_decimal(lambda_surface.lambda_holo_ev2_drift, places=24)} eV^2",
            "",
            "Tier-2 Comparator",
            f"- Lambda_obs = {_format_decimal(lambda_surface.anchor_lambda_si_m2, places=24)} m^(-2)",
            f"- Lambda_obs [eV^2] = {_format_decimal(lambda_surface.anchor_lambda_ev2, places=24)} eV^2",
            f"- Lambda_holo/Lambda_obs = {_format_decimal(lambda_surface.anchor_ratio, places=24)}",
            f"- surface-tension deviation = {_format_decimal(lambda_surface.deviation_percent, places=24)} %",
            "",
            "Unity of Scale Closure",
            f"- kappa_D5 = {_format_decimal(kappa.kappa, places=24)}",
            f"- m_nu = kappa_D5 * M_P * N_holo^(-1/4) = {_format_decimal(mass.neutrino_floor_ev, places=24)} eV",
            f"- G_N = {_format_decimal(decimal_unity.g_newton_topological_ev_minus2, places=24)} eV^(-2)",
            f"- Lambda_holo(lhs) = {_format_decimal(decimal_unity.lambda_holo_ev2, places=24)} eV^2",
            f"- Lambda_holo(rhs) = (3*pi/kappa_D5^4) * G_N * m_nu^4 = {_format_decimal(decimal_unity.expanded_theorem_rhs_ev2, places=24)} eV^2",
            f"- Decimal spot-check lhs/rhs = {_format_decimal(decimal_unity.identity_ratio, places=24)}",
            f"- Decimal spot-check epsilon_Lambda = {_format_decimal(decimal_unity.epsilon_lambda, places=24)}",
            f"- Decimal audit tolerance = {_format_decimal(decimal_unity.decimal_tolerance, places=24)}",
            f"- tn.py exact epsilon_Lambda = {_format_decimal(live_exact_epsilon_lambda, places=24)}",
            f"- tn.py numerical residual = {_format_decimal(live_numerical_residual, places=24)}",
            f"- register noise floor = {_format_decimal(live_register_noise_floor, places=24)}",
            f"- exact register noise floor = {_format_decimal(live_exact_register_noise_floor, places=24)}",
            f"- closes in benchmark verifier = {live_passed}",
            "",
            "Checked-In Benchmark Payloads",
        ]

        if checked_in_payloads:
            lines.append(f"- mirrored checked-in payloads = {len(checked_in_payloads)}")
            for payload in checked_in_payloads:
                lines.append(
                    "- "
                    f"{payload.source_path}: epsilon_Lambda={_format_decimal(payload.epsilon_lambda, places=24)}, "
                    f"exact_epsilon_Lambda={_format_decimal(payload.exact_epsilon_lambda, places=24)}, "
                    f"numerical_residual={_format_decimal(payload.numerical_residual, places=24)}, "
                    f"register_noise_floor={_format_decimal(payload.register_noise_floor, places=24)}, "
                    f"passed={payload.passed}"
                )

            lines.extend(
                [
                    f"- live exact epsilon_Lambda - checked-in exact epsilon_Lambda = {_format_decimal(live_exact_epsilon_lambda - checked_in_payloads[0].exact_epsilon_lambda, places=24)}",
                    f"- live numerical residual - checked-in numerical residual = {_format_decimal(live_numerical_residual - checked_in_payloads[0].numerical_residual, places=24)}",
                    f"- live register noise floor - checked-in register noise floor = {_format_decimal(live_register_noise_floor - checked_in_payloads[0].register_noise_floor, places=24)}",
                    "- checked-in payloads mirror the live unity-of-scale benchmark = True",
                ]
            )
        else:
            lines.append("- mirrored checked-in payloads = 0")

        return "\n".join(lines)

    @classmethod
    def generate_ledger(
        cls,
        *,
        kind: Literal["universe", "derivation", "lambda", "cosmological_constant"] = "universe",
        precision: int = DEFAULT_PRECISION,
    ) -> str:
        resolved_precision = max(int(precision), DEFAULT_PRECISION)
        if kind in {"universe", "derivation"}:
            return cls.build_derivation_ledger(precision=resolved_precision)
        if kind in {"lambda", "cosmological_constant"}:
            return cls.build_lambda_ledger(precision=resolved_precision)
        raise ValueError(f"Unknown ledger kind: {kind}")

    @classmethod
    def derive_tension_audit(cls, *, precision: int = DEFAULT_PRECISION):
        """Compare branch-fixed Tier 3 residues against Tier 2 comparators."""
        from shbt.core.derivation import TensionAudit, build_tension_audit, build_tension_component
        from shbt.verification.comparators import (
            CODATA_ALPHA_INVERSE_COMPARATOR,
            CODATA_PROTON_ELECTRON_MASS_RATIO_COMPARATOR,
            PLANCK2018_H0_COMPARATOR,
            PLANCK2018_LAMBDA_COMPARATOR,
        )

        resolved_precision = max(int(precision), DEFAULT_PRECISION)
        physical_ledger = cls.calculate_physical_ledger(precision=resolved_precision)
        lambda_surface = cls.derive_lambda_surface(precision=resolved_precision)
        components = (
            build_tension_component(
                label=CODATA_ALPHA_INVERSE_COMPARATOR.label,
                predicted_value=physical_ledger.alpha_surface.alpha_inverse_decimal,
                comparator=CODATA_ALPHA_INVERSE_COMPARATOR,
            ),
            build_tension_component(
                label=CODATA_PROTON_ELECTRON_MASS_RATIO_COMPARATOR.label,
                predicted_value=physical_ledger.proton_ratio.mu_audit,
                comparator=CODATA_PROTON_ELECTRON_MASS_RATIO_COMPARATOR,
            ),
            build_tension_component(
                label=PLANCK2018_H0_COMPARATOR.label,
                predicted_value=PLANCK2018_H0_COMPARATOR.value,
                comparator=PLANCK2018_H0_COMPARATOR,
            ),
            build_tension_component(
                label=PLANCK2018_LAMBDA_COMPARATOR.label,
                predicted_value=lambda_surface.lambda_holo_si_m2,
                comparator=PLANCK2018_LAMBDA_COMPARATOR,
            ),
        )
        audit: TensionAudit = build_tension_audit(
            label="Zero-Parameter Tier-2 conformance audit",
            benchmark_branch=physical_ledger.vacuum.branch,
            components=components,
        )
        return audit


def build_derivation_ledger(*, precision: int = DEFAULT_PRECISION) -> str:
    return UniverseFactory.build_derivation_ledger(precision=precision)


def build_lambda_ledger(*, precision: int = DEFAULT_PRECISION) -> str:
    return UniverseFactory.build_lambda_ledger(precision=precision)


UniverseFactory = stabilize_classmethods(UniverseFactory)
build_derivation_ledger = stabilize_boundary(build_derivation_ledger)
build_lambda_ledger = stabilize_boundary(build_lambda_ledger)


__all__ = [
    "AlphaSurfaceDerivation",
    "CentralChargeGeometry",
    "CheckedInUnityPayload",
    "CodataMassRatioAudit",
    "DEFAULT_PRECISION",
    "DEFAULT_RELATIVE_TOLERANCE",
    "KappaDerivation",
    "LambdaSurfaceDerivation",
    "MassBridgeDerivation",
    "ObservationalBoundaryConditionDerivation",
    "OnticAxioms",
    "OnticCascade",
    "PhysicalLedger",
    "ProtonRatioDerivation",
    "TopologicalVacuum",
    "UnityOfScaleDerivation",
    "VacuumPressureDerivation",
    "UniverseFactory",
    "build_derivation_ledger",
    "build_lambda_ledger",
    "decimal_cuberoot",
    "decimal_pi",
    "decimal_sin",
    "quarter_power_inverse",
    "su2_total_quantum_dimension_decimal",
]
