from __future__ import annotations

"""Stochastic uncertainty propagation for branch-fixed Tier 3 residues.

This module treats microscopic boundary fluctuations as a stress test on the
same finite-capacity boundary data used elsewhere in SHBT. The engine samples
ultra-small perturbations of the ordered bit-loading sequence, propagates the
resulting entropy / register jitter into bulk-facing derived residues, and
reports confidence intervals on the drift away from the benchmark branch.
"""

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np

if __package__ in (None, ""):
    import sys

    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.constants import (
    AUDIT_TOLERANCE,
    BENCHMARK_C_DARK_RESIDUE,
    BENCHMARK_C_DARK_RESIDUE_FRACTION,
    BENCHMARK_REFERENCE_COSET_CENTRAL_CHARGE,
    BENCHMARK_REFERENCE_COSET_CENTRAL_CHARGE_FRACTION,
    CODATA_FINE_STRUCTURE_ALPHA_INVERSE,
    DEFAULT_RANDOM_SEED,
    G_SM,
    GEOMETRIC_KAPPA,
    HOLOGRAPHIC_BITS,
    HOLOGRAPHIC_BITS_FRACTIONAL_SIGMA,
    LEPTON_LEVEL,
    PARENT_LEVEL,
    PLANCK_HOLOGRAPHIC_BITS,
    QUARK_LEVEL,
    R_GUT,
    SO10_DIMENSION,
    SO10_DUAL_COXETER,
    SU2_DIMENSION,
    SU2_DUAL_COXETER,
    SU3_DIMENSION,
    SU3_DUAL_COXETER,
)
from shbt.core.derivation_api import (
    DEFAULT_PRECISION as DERIVATION_DEFAULT_PRECISION,
    UniverseFactory,
    decimal_pi,
    su2_total_quantum_dimension_decimal,
)
from shbt.core.temporal_emergence_kernel import ManifoldSliceLoadingMap, map_manifold_slice_bit_loading_density


CoordinateKey = Literal["lepton_level", "quark_level", "parent_level"]
DecimalMatrix = tuple[tuple[Decimal, ...], ...]

DEFAULT_PRECISION = max(int(DERIVATION_DEFAULT_PRECISION), 160)
_GUARD_DIGITS = 24
_D5_WEIGHT_SIMPLEX_HYPERAREA_FRACTION = Fraction(160, 1521)


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(str(value))


def _fraction_to_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _matrix_total(matrix: DecimalMatrix) -> Decimal:
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
        return sum((sum(row, Decimal("0")) for row in matrix), Decimal("0"))


def _normalize_matrix(matrix: DecimalMatrix, *, precision: int) -> DecimalMatrix:
    total = _matrix_total(matrix)
    if total <= 0:
        raise ValueError("Matrix total must be positive.")
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        normalized = tuple(
            tuple(value / total for value in row)
            for row in matrix
        )
    return normalized


def _matrix_entropy(matrix: DecimalMatrix, *, precision: int) -> Decimal:
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        entropy = Decimal("0")
        for row in matrix:
            for value in row:
                if value <= 0:
                    continue
                entropy -= value * value.ln()
        context.prec = max(int(precision), DEFAULT_PRECISION)
        return +entropy


def _quantile(values: tuple[Decimal, ...], q: Decimal | float) -> Decimal:
    if not values:
        raise ValueError("At least one value is required to compute a quantile.")
    if len(values) == 1:
        return values[0]
    resolved_q = float(q)
    if not 0.0 <= resolved_q <= 1.0:
        raise ValueError("Quantile must lie in [0, 1].")
    sorted_values = tuple(sorted(values))
    position = resolved_q * float(len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
        weight = Decimal(str(position - lower_index))
        return +(lower_value + (upper_value - lower_value) * weight)


def _continuous_wzw_central_charge(
    level: Decimal | Fraction | float | int | str,
    dimension: int,
    dual_coxeter: int,
    *,
    precision: int,
) -> Decimal:
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        resolved_level = _decimal(level)
        denominator = resolved_level + Decimal(int(dual_coxeter))
        if denominator <= 0:
            raise ValueError("WZW central charge requires k + h^∨ > 0.")
        context.prec = max(int(precision), DEFAULT_PRECISION)
        return +(resolved_level * Decimal(int(dimension)) / denominator)


def _continuous_alpha_inverse(
    *,
    lepton_level: Decimal,
    quark_level: Decimal,
    parent_level: Decimal,
    generation_count: int = G_SM,
    precision: int,
) -> Decimal:
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        visible_support = lepton_level + quark_level
        if visible_support <= 0:
            raise ValueError("Visible support must remain positive.")
        context.prec = max(int(precision), DEFAULT_PRECISION)
        return +(Decimal(int(generation_count)) * parent_level / visible_support)


def _continuous_r_gut(*, lepton_level: Decimal, quark_level: Decimal, precision: int) -> Decimal:
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        denominator = lepton_level + Decimal(int(SU2_DUAL_COXETER))
        if denominator <= 0:
            raise ValueError("R_GUT denominator must remain positive.")
        context.prec = max(int(precision), DEFAULT_PRECISION)
        return +(quark_level / denominator)


def _continuous_c_dark(
    *,
    lepton_level: Decimal,
    quark_level: Decimal,
    parent_level: Decimal,
    precision: int,
) -> Decimal:
    parent_su3 = _continuous_wzw_central_charge(parent_level, SU3_DIMENSION, SU3_DUAL_COXETER, precision=precision)
    parent_su2 = _continuous_wzw_central_charge(parent_level, SU2_DIMENSION, SU2_DUAL_COXETER, precision=precision)
    visible_su3 = _continuous_wzw_central_charge(quark_level, SU3_DIMENSION, SU3_DUAL_COXETER, precision=precision)
    visible_su2 = _continuous_wzw_central_charge(lepton_level, SU2_DIMENSION, SU2_DUAL_COXETER, precision=precision)
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION)
        return +(parent_su3 + parent_su2 - visible_su3 - visible_su2)


def _continuous_reference_coset_central_charge(
    *,
    lepton_level: Decimal,
    quark_level: Decimal,
    parent_level: Decimal,
    precision: int,
) -> Decimal:
    parent_so10 = _continuous_wzw_central_charge(parent_level, SO10_DIMENSION, SO10_DUAL_COXETER, precision=precision)
    visible_su3 = _continuous_wzw_central_charge(quark_level, SU3_DIMENSION, SU3_DUAL_COXETER, precision=precision)
    visible_su2 = _continuous_wzw_central_charge(lepton_level, SU2_DIMENSION, SU2_DUAL_COXETER, precision=precision)
    c_dark = _continuous_c_dark(
        lepton_level=lepton_level,
        quark_level=quark_level,
        parent_level=parent_level,
        precision=precision,
    )
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION)
        return +(parent_so10 - (visible_su2 + visible_su3) - c_dark)


def _continuous_kappa_d5(*, lepton_level: Decimal, precision: int) -> Decimal:
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        simplex_factor = _fraction_to_decimal(_D5_WEIGHT_SIMPLEX_HYPERAREA_FRACTION)
        sqrt_ten = Decimal(10).sqrt()
        weight_simplex_hyperarea = simplex_factor * sqrt_ten
        total_quantum_dimension = su2_total_quantum_dimension_decimal(lepton_level, precision=context.prec)
        beta = total_quantum_dimension.ln() / Decimal(2)
        eta_spin = (Decimal(347) - Decimal(8) * beta * beta) / Decimal(351)
        kappa = (Decimal(16) * weight_simplex_hyperarea * eta_spin / Decimal(5)).sqrt()
        context.prec = max(int(precision), DEFAULT_PRECISION)
        return +kappa


def _effective_newton_coordinate_ev_minus2(
    *,
    c_dark: Decimal,
    branch_planck_mass_ev: Decimal,
    precision: int,
) -> Decimal:
    if c_dark <= 0:
        raise ValueError("c_dark must remain positive to derive the effective Newton coordinate.")
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        three = Decimal(3)
        two = Decimal(2)
        two_pi = two * decimal_pi(context.prec)
        denominator = two_pi * c_dark * branch_planck_mass_ev * branch_planck_mass_ev
        context.prec = max(int(precision), DEFAULT_PRECISION)
        return +(three / denominator)


@dataclass(frozen=True)
class BoundaryFluctuationSample:
    sample_index: int
    lepton_drift: Decimal
    quark_drift: Decimal
    parent_drift: Decimal
    sequence_jitter: Decimal
    entropy_jitter: Decimal
    effective_holographic_bits: Decimal
    residues: dict[str, Decimal]

    @property
    def combined_jitter(self) -> Decimal:
        return self.sequence_jitter + self.entropy_jitter


@dataclass(frozen=True)
class ResidueConfidenceInterval:
    name: str
    baseline: Decimal
    lower: Decimal
    median: Decimal
    upper: Decimal
    lower_shift: Decimal
    median_shift: Decimal
    upper_shift: Decimal
    max_abs_shift: Decimal
    max_fractional_shift: Decimal
    robust_against_audit_tolerance: bool


@dataclass(frozen=True)
class JitterTransferFunction:
    drift_scale: Decimal
    alpha_inverse_shifts: dict[CoordinateKey, Decimal]
    alpha_inverse_gradients: dict[CoordinateKey, Decimal]
    g_effective_shifts: dict[CoordinateKey, Decimal]
    g_effective_gradients: dict[CoordinateKey, Decimal]


@dataclass(frozen=True)
class StochasticSensitivityAudit:
    sample_count: int
    bit_loading_jitter_scale: Decimal
    integer_drift_scale: Decimal
    confidence_level: Decimal
    audit_tolerance: Decimal
    baseline_residues: dict[str, Decimal]
    jitter_transfer_function: JitterTransferFunction
    confidence_intervals: dict[str, ResidueConfidenceInterval]
    samples: tuple[BoundaryFluctuationSample, ...]

    @property
    def max_fractional_shift(self) -> Decimal:
        if not self.confidence_intervals:
            return Decimal("0")
        return max(interval.max_fractional_shift for interval in self.confidence_intervals.values())

    @property
    def all_tier3_intervals_robust(self) -> bool:
        return bool(self.confidence_intervals) and all(
            interval.robust_against_audit_tolerance for interval in self.confidence_intervals.values()
        )


class StochasticSensitivityEngine:
    def __init__(
        self,
        *,
        sample_count: int = 256,
        bit_loading_jitter_scale: Decimal | Fraction | float | int | str = Decimal("1e-77"),
        integer_drift_scale: Decimal | Fraction | float | int | str = Decimal("1e-80"),
        confidence_level: Decimal | Fraction | float | int | str = Decimal("0.95"),
        seed: int = DEFAULT_RANDOM_SEED,
        precision: int = DEFAULT_PRECISION,
    ) -> None:
        self.sample_count = max(int(sample_count), 1)
        self.bit_loading_jitter_scale = abs(_decimal(bit_loading_jitter_scale))
        self.integer_drift_scale = abs(_decimal(integer_drift_scale))
        self.confidence_level = _decimal(confidence_level)
        self.seed = int(seed)
        self.precision = max(int(precision), DEFAULT_PRECISION)
        if not Decimal("0") < self.confidence_level < Decimal("1"):
            raise ValueError("confidence_level must lie strictly between 0 and 1.")
        self._rng = np.random.default_rng(self.seed)

    @cached_property
    def baseline_loading_map(self) -> ManifoldSliceLoadingMap:
        return map_manifold_slice_bit_loading_density(
            lepton_level=LEPTON_LEVEL,
            quark_level=QUARK_LEVEL,
            precision=self.precision,
        )

    @cached_property
    def baseline_loading_entropy(self) -> Decimal:
        return _matrix_entropy(self.baseline_loading_map.loading_density, precision=self.precision)

    @cached_property
    def baseline_planck_mass_ev(self) -> Decimal:
        return UniverseFactory.derive_mass_bridge(precision=self.precision).branch_planck_mass_ev

    @cached_property
    def baseline_residues(self) -> dict[str, Decimal]:
        return self._residue_snapshot(
            lepton_level=_decimal(LEPTON_LEVEL),
            quark_level=_decimal(QUARK_LEVEL),
            parent_level=_decimal(PARENT_LEVEL),
            effective_holographic_bits=_decimal(HOLOGRAPHIC_BITS),
        )

    def _gaussian_decimal(self, scale: Decimal) -> Decimal:
        return _decimal(self._rng.normal(loc=0.0, scale=float(scale)))

    def _perturb_loading_density(self, loading_density: DecimalMatrix) -> DecimalMatrix:
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            perturbed_rows: list[tuple[Decimal, ...]] = []
            for row in loading_density:
                perturbed_row: list[Decimal] = []
                for value in row:
                    perturbation = self._gaussian_decimal(self.bit_loading_jitter_scale)
                    perturbed_row.append(value * (Decimal("1") + perturbation))
                perturbed_rows.append(tuple(perturbed_row))
        return _normalize_matrix(tuple(perturbed_rows), precision=self.precision)

    def _sequence_jitter(self, perturbed_loading: DecimalMatrix) -> Decimal:
        ordered_coordinates = self.baseline_loading_map.dominant_loading_sequence
        if len(ordered_coordinates) <= 1:
            return Decimal("0")
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            count = len(ordered_coordinates)
            denominator = Decimal(count - 1)
            jitter = Decimal("0")
            for index, (row_index, column_index) in enumerate(ordered_coordinates):
                weight = (Decimal(2 * index) - denominator) / denominator
                delta = (
                    perturbed_loading[row_index][column_index]
                    - self.baseline_loading_map.loading_density[row_index][column_index]
                )
                jitter += weight * delta
            context.prec = self.precision
            return +jitter

    def _entropy_jitter(self, perturbed_loading: DecimalMatrix) -> Decimal:
        sample_entropy = _matrix_entropy(perturbed_loading, precision=self.precision)
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            if self.baseline_loading_entropy == 0:
                return Decimal("0")
            context.prec = self.precision
            return +(sample_entropy / self.baseline_loading_entropy - Decimal("1"))

    def _residue_snapshot(
        self,
        *,
        lepton_level: Decimal,
        quark_level: Decimal,
        parent_level: Decimal,
        effective_holographic_bits: Decimal,
    ) -> dict[str, Decimal]:
        alpha_inverse = _continuous_alpha_inverse(
            lepton_level=lepton_level,
            quark_level=quark_level,
            parent_level=parent_level,
            generation_count=G_SM,
            precision=self.precision,
        )
        kappa = _continuous_kappa_d5(lepton_level=lepton_level, precision=self.precision)
        r_gut = _continuous_r_gut(
            lepton_level=lepton_level,
            quark_level=quark_level,
            precision=self.precision,
        )
        c_dark = _continuous_c_dark(
            lepton_level=lepton_level,
            quark_level=quark_level,
            parent_level=parent_level,
            precision=self.precision,
        )
        reference_coset = _continuous_reference_coset_central_charge(
            lepton_level=lepton_level,
            quark_level=quark_level,
            parent_level=parent_level,
            precision=self.precision,
        )
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            neutrino_floor_mev = (
                kappa
                * self.baseline_planck_mass_ev
                / effective_holographic_bits.sqrt().sqrt()
                * Decimal(1000)
            )
            fractional_sigma = _decimal(HOLOGRAPHIC_BITS_FRACTIONAL_SIGMA) + abs(
                effective_holographic_bits / _decimal(HOLOGRAPHIC_BITS) - Decimal("1")
            )
            alpha_delta = alpha_inverse - _decimal(CODATA_FINE_STRUCTURE_ALPHA_INVERSE)
            context.prec = self.precision
            return {
                "alpha_inverse_decimal": +alpha_inverse,
                "alpha_inverse_codata_delta": +alpha_delta,
                "GEOMETRIC_KAPPA": +kappa,
                "PLANCK_HOLOGRAPHIC_BITS": +effective_holographic_bits,
                "HOLOGRAPHIC_BITS": +effective_holographic_bits,
                "HOLOGRAPHIC_BITS_FRACTIONAL_SIGMA": +fractional_sigma,
                "R_GUT": +r_gut,
                "BENCHMARK_C_DARK_RESIDUE_FRACTION": +c_dark,
                "BENCHMARK_C_DARK_RESIDUE": +c_dark,
                "BENCHMARK_REFERENCE_COSET_CENTRAL_CHARGE_FRACTION": +reference_coset,
                "BENCHMARK_REFERENCE_COSET_CENTRAL_CHARGE": +reference_coset,
                "g_effective_ev_minus2": _effective_newton_coordinate_ev_minus2(
                    c_dark=c_dark,
                    branch_planck_mass_ev=self.baseline_planck_mass_ev,
                    precision=self.precision,
                ),
                "neutrino_floor_mev": +neutrino_floor_mev,
            }

    def _sample(self, sample_index: int) -> BoundaryFluctuationSample:
        lepton_drift = self._gaussian_decimal(self.integer_drift_scale)
        quark_drift = self._gaussian_decimal(self.integer_drift_scale)
        parent_drift = self._gaussian_decimal(self.integer_drift_scale)
        perturbed_loading = self._perturb_loading_density(self.baseline_loading_map.loading_density)
        sequence_jitter = self._sequence_jitter(perturbed_loading)
        entropy_jitter = self._entropy_jitter(perturbed_loading)
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            combined_jitter = sequence_jitter + entropy_jitter
            effective_holographic_bits = _decimal(HOLOGRAPHIC_BITS) * (Decimal("1") + combined_jitter)
            context.prec = self.precision
            combined_jitter = +combined_jitter
            effective_holographic_bits = +effective_holographic_bits
            residues = self._residue_snapshot(
                lepton_level=_decimal(LEPTON_LEVEL) + lepton_drift,
                quark_level=_decimal(QUARK_LEVEL) + quark_drift,
                parent_level=_decimal(PARENT_LEVEL) + parent_drift,
                effective_holographic_bits=effective_holographic_bits,
            )
        return BoundaryFluctuationSample(
            sample_index=sample_index,
            lepton_drift=lepton_drift,
            quark_drift=quark_drift,
            parent_drift=parent_drift,
            sequence_jitter=sequence_jitter,
            entropy_jitter=entropy_jitter,
            effective_holographic_bits=effective_holographic_bits,
            residues=residues,
        )

    def run_monte_carlo(self) -> tuple[BoundaryFluctuationSample, ...]:
        return tuple(self._sample(sample_index) for sample_index in range(self.sample_count))

    def calculate_jitter_transfer_function(self) -> JitterTransferFunction:
        baseline_alpha = self.baseline_residues["alpha_inverse_decimal"]
        baseline_g = self.baseline_residues["g_effective_ev_minus2"]
        alpha_shifts: dict[CoordinateKey, Decimal] = {}
        alpha_gradients: dict[CoordinateKey, Decimal] = {}
        g_shifts: dict[CoordinateKey, Decimal] = {}
        g_gradients: dict[CoordinateKey, Decimal] = {}
        for coordinate_name in ("lepton_level", "quark_level", "parent_level"):
            with localcontext() as context:
                context.prec = self.precision + _GUARD_DIGITS
                lepton = _decimal(LEPTON_LEVEL)
                quark = _decimal(QUARK_LEVEL)
                parent = _decimal(PARENT_LEVEL)
                if coordinate_name == "lepton_level":
                    lepton += self.integer_drift_scale
                elif coordinate_name == "quark_level":
                    quark += self.integer_drift_scale
                else:
                    parent += self.integer_drift_scale
            shifted = self._residue_snapshot(
                lepton_level=lepton,
                quark_level=quark,
                parent_level=parent,
                effective_holographic_bits=_decimal(HOLOGRAPHIC_BITS),
            )
            alpha_shift = shifted["alpha_inverse_decimal"] - baseline_alpha
            g_shift = shifted["g_effective_ev_minus2"] - baseline_g
            alpha_shifts[coordinate_name] = alpha_shift
            alpha_gradients[coordinate_name] = alpha_shift / self.integer_drift_scale
            g_shifts[coordinate_name] = g_shift
            g_gradients[coordinate_name] = g_shift / self.integer_drift_scale
        return JitterTransferFunction(
            drift_scale=self.integer_drift_scale,
            alpha_inverse_shifts=alpha_shifts,
            alpha_inverse_gradients=alpha_gradients,
            g_effective_shifts=g_shifts,
            g_effective_gradients=g_gradients,
        )

    def audit_confidence_intervals(self) -> StochasticSensitivityAudit:
        samples = self.run_monte_carlo()
        lower_q = (Decimal("1") - self.confidence_level) / Decimal("2")
        upper_q = Decimal("1") - lower_q
        intervals: dict[str, ResidueConfidenceInterval] = {}
        for residue_name, baseline_value in self.baseline_residues.items():
            with localcontext() as context:
                context.prec = self.precision + _GUARD_DIGITS
                values = tuple(sample.residues[residue_name] for sample in samples)
                lower = _quantile(values, lower_q)
                median = _quantile(values, Decimal("0.5"))
                upper = _quantile(values, upper_q)
                lower_shift = lower - baseline_value
                median_shift = median - baseline_value
                upper_shift = upper - baseline_value
                max_abs_shift = max(abs(lower_shift), abs(upper_shift))
                if baseline_value == 0:
                    max_fractional_shift = max_abs_shift
                else:
                    max_fractional_shift = max_abs_shift / abs(baseline_value)
                context.prec = self.precision
                lower = +lower
                median = +median
                upper = +upper
                lower_shift = +lower_shift
                median_shift = +median_shift
                upper_shift = +upper_shift
                max_abs_shift = +max_abs_shift
                max_fractional_shift = +max_fractional_shift
            intervals[residue_name] = ResidueConfidenceInterval(
                name=residue_name,
                baseline=baseline_value,
                lower=lower,
                median=median,
                upper=upper,
                lower_shift=lower_shift,
                median_shift=median_shift,
                upper_shift=upper_shift,
                max_abs_shift=max_abs_shift,
                max_fractional_shift=max_fractional_shift,
                robust_against_audit_tolerance=max_fractional_shift < AUDIT_TOLERANCE,
            )
        return StochasticSensitivityAudit(
            sample_count=self.sample_count,
            bit_loading_jitter_scale=self.bit_loading_jitter_scale,
            integer_drift_scale=self.integer_drift_scale,
            confidence_level=self.confidence_level,
            audit_tolerance=AUDIT_TOLERANCE,
            baseline_residues=dict(self.baseline_residues),
            jitter_transfer_function=self.calculate_jitter_transfer_function(),
            confidence_intervals=intervals,
            samples=samples,
        )


def build_stochastic_sensitivity_audit(
    *,
    sample_count: int = 256,
    bit_loading_jitter_scale: Decimal | Fraction | float | int | str = Decimal("1e-77"),
    integer_drift_scale: Decimal | Fraction | float | int | str = Decimal("1e-80"),
    confidence_level: Decimal | Fraction | float | int | str = Decimal("0.95"),
    seed: int = DEFAULT_RANDOM_SEED,
    precision: int = DEFAULT_PRECISION,
) -> StochasticSensitivityAudit:
    return StochasticSensitivityEngine(
        sample_count=sample_count,
        bit_loading_jitter_scale=bit_loading_jitter_scale,
        integer_drift_scale=integer_drift_scale,
        confidence_level=confidence_level,
        seed=seed,
        precision=precision,
    ).audit_confidence_intervals()


__all__ = [
    "BoundaryFluctuationSample",
    "DEFAULT_PRECISION",
    "JitterTransferFunction",
    "ResidueConfidenceInterval",
    "StochasticSensitivityAudit",
    "StochasticSensitivityEngine",
    "build_stochastic_sensitivity_audit",
]
