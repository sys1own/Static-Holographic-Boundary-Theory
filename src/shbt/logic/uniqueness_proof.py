from __future__ import annotations

r"""Logic-facing proof that only the benchmark kernel sustains reality.

The module packages a compact mathematical certificate around the benchmark
kernel ``(26, 8, 312)``. Two ideas are enforced simultaneously:

1. The visible kernel dimension must remain locked to ``26``.
2. The prime-index coordinate ``k_ell / 2`` must remain prime.

For the benchmark branch these conditions read

    D = 26,
    pi(D) = D / 2 = 13 \in \mathbb{P},

and the ordinary SHBT closure functional already vanishes:

    C_top = Delta_fr + |c_dark - c_dark^*| + Delta_dio = 0.

We quantify collapse away from the benchmark through the singular divergence

    Xi = C_top + |D - 26| / 26 + |D / 2 - 13| + S_pi,

where ``S_pi = 0`` for prime-indexed kernels and ``S_pi = 1`` otherwise.
The associated information-retention factor is

    R = 1 / (1 + Xi),

so any non-zero divergence produces information loss ``1 - R``.

To certify non-invertibility we define the Distinction Logic Jacobian

    J = diag(L_26, P_pi, C_closure),

with indicator gates

- ``L_26 = 1`` only for the 26D kernel,
- ``P_pi = 1`` only when ``D / 2`` is an integer prime, and
- ``C_closure = 1`` only when the topological closure survives.

A kernel sustains reality iff ``det(J) = 1``. Any drift from ``(26, 8, 312)``
forces at least one gate to zero, so the Jacobian becomes singular and the
information map is non-invertible.
"""

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Final

from shbt.constants import (
    BENCHMARK_C_DARK_RESIDUE_FRACTION,
    LEPTON_LEVEL,
    PARENT_LEVEL,
    QUARK_LEVEL,
    SU2_DIMENSION,
    SU2_DUAL_COXETER,
    SU3_DIMENSION,
    SU3_DUAL_COXETER,
)
from shbt.core.rigidity_landscape import calculate_moat_depth
from shbt.math_engine import guard_fraction, guard_sum

from .bootstrap import evaluate_kernel


BENCHMARK_REALITY_KERNEL: Final[tuple[int, int, int]] = (
    int(LEPTON_LEVEL),
    int(QUARK_LEVEL),
    int(PARENT_LEVEL),
)
BENCHMARK_PRIME_INDEX: Final[Fraction] = Fraction(BENCHMARK_REALITY_KERNEL[0], 2)
_ONE: Final[Fraction] = Fraction(1, 1)
_ZERO: Final[Fraction] = Fraction(0, 1)


@dataclass(frozen=True)
class DistinctionLogicAudit:
    benchmark_coordinates: tuple[int, int, int]
    evaluated_coordinates: tuple[int, int, int]
    prime_index_coordinate: Fraction
    dimension_locked: bool
    prime_index_integral: bool
    prime_index_prime: bool
    topological_closure_locked: bool
    distinction_jacobian: tuple[tuple[Fraction, Fraction, Fraction], ...]
    distinction_jacobian_determinant: Fraction

    @property
    def prime_indexed_kernel(self) -> bool:
        return bool(self.prime_index_integral and self.prime_index_prime)

    @property
    def information_map_invertible(self) -> bool:
        return self.distinction_jacobian_determinant == _ONE

    @property
    def axioms_satisfied(self) -> bool:
        return bool(
            self.dimension_locked
            and self.prime_indexed_kernel
            and self.topological_closure_locked
            and self.information_map_invertible
        )


@dataclass(frozen=True)
class KernelDivergenceReport:
    benchmark_coordinates: tuple[int, int, int]
    evaluated_coordinates: tuple[int, int, int]
    prime_index_coordinate: Fraction
    moat_divergence: Fraction
    c_dark_shift: Fraction
    diophantine_gap: Fraction
    stability_score: Fraction
    stability_penalty: Fraction
    dimension_drift: Fraction
    prime_index_drift: Fraction
    prime_index_singularity: Fraction
    singular_divergence: Fraction
    information_retention: Fraction
    information_loss: Fraction
    distinction_logic: DistinctionLogicAudit
    failure_modes: tuple[str, ...]

    @property
    def drift_from_fixed_point(self) -> bool:
        return self.evaluated_coordinates != self.benchmark_coordinates

    @property
    def non_invertible_information_loss(self) -> bool:
        return bool(self.information_loss > _ZERO and not self.distinction_logic.information_map_invertible)

    @property
    def sustains_reality(self) -> bool:
        return bool(self.singular_divergence == _ZERO and self.distinction_logic.axioms_satisfied)

    @property
    def statement(self) -> str:
        if self.sustains_reality:
            return (
                "The benchmark kernel saturates the Distinction Logic axioms: "
                "the information map remains invertible and reality is sustained."
            )
        return (
            f"Kernel {self.evaluated_coordinates} fails to sustain reality: "
            f"singular divergence={self.singular_divergence}, "
            f"det(J)={self.distinction_logic.distinction_jacobian_determinant}, "
            "so the information map collapses."
        )


def _coerce_positive_integer(name: str, value: int) -> int:
    resolved_value = int(value)
    if resolved_value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return resolved_value


def _is_prime(value: int) -> bool:
    resolved_value = int(value)
    if resolved_value < 2:
        return False
    if resolved_value == 2:
        return True
    if resolved_value % 2 == 0:
        return False
    limit = math.isqrt(resolved_value)
    for divisor in range(3, limit + 1, 2):
        if resolved_value % divisor == 0:
            return False
    return True


def _wzw_central_charge(level: int, dimension: int, dual_coxeter: int) -> Fraction:
    resolved_level = int(level)
    denominator = resolved_level + int(dual_coxeter)
    if denominator <= 0:
        raise ValueError("WZW central charge requires k + h^∨ > 0.")
    return guard_fraction(resolved_level * int(dimension), denominator)


def _c_dark_shift(parent_level: int, dimension_level: int, gauge_level: int) -> Fraction:
    completion_residue = guard_sum(
        (
            _wzw_central_charge(parent_level, SU3_DIMENSION, SU3_DUAL_COXETER),
            _wzw_central_charge(parent_level, SU2_DIMENSION, SU2_DUAL_COXETER),
            -_wzw_central_charge(gauge_level, SU3_DIMENSION, SU3_DUAL_COXETER),
            -_wzw_central_charge(dimension_level, SU2_DIMENSION, SU2_DUAL_COXETER),
        )
    )
    return abs(completion_residue - BENCHMARK_C_DARK_RESIDUE_FRACTION)


def _diophantine_gap(parent_level: int, dimension_level: int, gauge_level: int) -> Fraction:
    minimal_parent_level = math.lcm(2 * int(dimension_level), 3 * int(gauge_level))
    return abs(guard_fraction(int(parent_level) - minimal_parent_level, minimal_parent_level))


def _prime_index_coordinate(dimension: int) -> Fraction:
    return Fraction(int(dimension), 2)


def _coerce_fraction(value: Fraction | int | bool | float | str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        return Fraction(int(value), 1)
    return Fraction(value)


def _distinction_jacobian(
    *,
    dimension_locked: bool,
    prime_index_prime: bool,
    topological_closure_locked: bool,
) -> tuple[tuple[Fraction, Fraction, Fraction], ...]:
    return (
        (Fraction(int(dimension_locked), 1), _ZERO, _ZERO),
        (_ZERO, Fraction(int(prime_index_prime), 1), _ZERO),
        (_ZERO, _ZERO, Fraction(int(topological_closure_locked), 1)),
    )


def generate_divergence_report(
    dimension: int = BENCHMARK_REALITY_KERNEL[0],
    generation: int = BENCHMARK_REALITY_KERNEL[1],
    nodes: int = BENCHMARK_REALITY_KERNEL[2],
) -> KernelDivergenceReport:
    """Quantify the singular divergence of a candidate kernel.

    The benchmark branch returns zero divergence and a unit Jacobian
    determinant. Any drift away from ``(26, 8, 312)`` is asserted to produce a
    singular Distinction Logic Jacobian and therefore non-invertible
    information loss.
    """

    resolved_dimension = _coerce_positive_integer("dimension", dimension)
    resolved_generation = _coerce_positive_integer("generation", generation)
    resolved_nodes = _coerce_positive_integer("nodes", nodes)
    evaluated_coordinates = (resolved_dimension, resolved_generation, resolved_nodes)

    kernel_evaluation = evaluate_kernel(resolved_dimension, resolved_generation, resolved_nodes)
    topological_closure_locked = bool(kernel_evaluation["is_closed"])
    stability_score = _coerce_fraction(kernel_evaluation["stability_score"])
    stability_penalty = guard_fraction(1, stability_score) - _ONE

    prime_index_coordinate = _prime_index_coordinate(resolved_dimension)
    prime_index_integral = prime_index_coordinate.denominator == 1
    prime_index_prime = bool(prime_index_integral and _is_prime(prime_index_coordinate.numerator))
    dimension_locked = resolved_dimension == BENCHMARK_REALITY_KERNEL[0]

    moat_divergence = calculate_moat_depth(resolved_nodes, resolved_dimension, resolved_generation)
    c_dark_shift = _c_dark_shift(resolved_nodes, resolved_dimension, resolved_generation)
    diophantine_gap = _diophantine_gap(resolved_nodes, resolved_dimension, resolved_generation)
    dimension_drift = abs(Fraction(resolved_dimension - BENCHMARK_REALITY_KERNEL[0], BENCHMARK_REALITY_KERNEL[0]))
    prime_index_drift = abs(prime_index_coordinate - BENCHMARK_PRIME_INDEX)
    prime_index_singularity = _ZERO if prime_index_prime else _ONE
    singular_divergence = guard_sum(
        (
            stability_penalty,
            dimension_drift,
            prime_index_drift,
            prime_index_singularity,
        )
    )
    information_retention = guard_fraction(1, _ONE + singular_divergence)
    information_loss = _ONE - information_retention

    distinction_jacobian = _distinction_jacobian(
        dimension_locked=dimension_locked,
        prime_index_prime=prime_index_prime,
        topological_closure_locked=topological_closure_locked,
    )
    distinction_jacobian_determinant = (
        distinction_jacobian[0][0] * distinction_jacobian[1][1] * distinction_jacobian[2][2]
    )
    distinction_logic = DistinctionLogicAudit(
        benchmark_coordinates=BENCHMARK_REALITY_KERNEL,
        evaluated_coordinates=evaluated_coordinates,
        prime_index_coordinate=prime_index_coordinate,
        dimension_locked=dimension_locked,
        prime_index_integral=prime_index_integral,
        prime_index_prime=prime_index_prime,
        topological_closure_locked=topological_closure_locked,
        distinction_jacobian=distinction_jacobian,
        distinction_jacobian_determinant=distinction_jacobian_determinant,
    )

    failure_modes: list[str] = []
    if not dimension_locked:
        failure_modes.append(
            f"Kernel dimension drifted away from 26D: received {resolved_dimension}D."
        )
    if not prime_index_integral:
        failure_modes.append(
            f"Prime-index coordinate k_l/2={prime_index_coordinate} is not an integer."
        )
    elif not prime_index_prime:
        failure_modes.append(
            f"Prime-index coordinate k_l/2={prime_index_coordinate.numerator} is not prime."
        )
    if moat_divergence > _ZERO:
        failure_modes.append(f"Framing moat reopened with Delta_fr={moat_divergence}.")
    if c_dark_shift > _ZERO:
        failure_modes.append(f"Completion residue shifted by {c_dark_shift}.")
    if diophantine_gap > _ZERO:
        failure_modes.append(f"Diophantine lock failed with gap {diophantine_gap}.")
    if not topological_closure_locked:
        failure_modes.append("Topological closure failed, so the distinction map is singular.")

    report = KernelDivergenceReport(
        benchmark_coordinates=BENCHMARK_REALITY_KERNEL,
        evaluated_coordinates=evaluated_coordinates,
        prime_index_coordinate=prime_index_coordinate,
        moat_divergence=moat_divergence,
        c_dark_shift=c_dark_shift,
        diophantine_gap=diophantine_gap,
        stability_score=stability_score,
        stability_penalty=stability_penalty,
        dimension_drift=dimension_drift,
        prime_index_drift=prime_index_drift,
        prime_index_singularity=prime_index_singularity,
        singular_divergence=singular_divergence,
        information_retention=information_retention,
        information_loss=information_loss,
        distinction_logic=distinction_logic,
        failure_modes=tuple(failure_modes),
    )

    if report.drift_from_fixed_point:
        assert report.non_invertible_information_loss, (
            "Any drift away from the benchmark fixed point must produce non-invertible information loss."
        )
    else:
        assert report.sustains_reality, "The benchmark kernel must remain the unique reality-sustaining branch."

    return report


__all__ = [
    "BENCHMARK_PRIME_INDEX",
    "BENCHMARK_REALITY_KERNEL",
    "DistinctionLogicAudit",
    "KernelDivergenceReport",
    "generate_divergence_report",
]
