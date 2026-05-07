from __future__ import annotations

"""Sensitivity audit for alternative and non-prime indexed kernels.

This module evaluates candidate kernels against the recursive ontic cascade and
the Axiom IX closure predicates. The benchmark branch `(26, 8, 312)` is the
only retained survivor; alternative-dimension or non-prime indexed kernels are
reported as finite, non-singular divergences rather than hidden fit knobs.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from shbt.constants import G_SM, LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.ontic_cascade import DEFAULT_PRECISION, LogicRelation, OnticAxioms, OnticCascade, evaluate_ontic_cascade


BenchmarkKernel = tuple[int, int, int]
BENCHMARK_KERNEL: BenchmarkKernel = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))


@dataclass(frozen=True)
class KernelSensitivityPoint:
    kernel: BenchmarkKernel
    cascade: OnticCascade
    alpha_shift_fraction: Fraction
    alpha_shift_decimal: Decimal

    @property
    def boundary_dimension(self) -> int:
        return int(self.kernel[0])

    @property
    def alternative_dimension(self) -> bool:
        return self.boundary_dimension != BENCHMARK_KERNEL[0]

    @property
    def dimension_index(self) -> Fraction:
        return self.cascade.dimension_index

    @property
    def prime_indexed(self) -> bool:
        return self.cascade.prime_indexed

    @property
    def axiom_ix_closure(self) -> bool:
        return self.cascade.axiom_ix.topological_closure

    @property
    def closure_tensor_amplitude(self) -> Decimal:
        return self.cascade.axiom_ix.closure_tensor_amplitude

    @property
    def failure_modes(self) -> tuple[str, ...]:
        return self.cascade.axiom_ix.failure_modes

    @property
    def non_singular_divergence(self) -> bool:
        return bool(
            not self.axiom_ix_closure
            and self.alpha_shift_decimal.is_finite()
            and self.closure_tensor_amplitude.is_finite()
        )


@dataclass(frozen=True)
class KernelSensitivityAudit:
    benchmark_kernel: BenchmarkKernel
    benchmark_cascade: OnticCascade
    evaluated_points: tuple[KernelSensitivityPoint, ...]

    @property
    def axiom_ix_survivors(self) -> tuple[BenchmarkKernel, ...]:
        return tuple(point.kernel for point in self.evaluated_points if point.axiom_ix_closure)

    @property
    def axiom_ix_survivor_count(self) -> int:
        return len(self.axiom_ix_survivors)

    @property
    def unique_axiom_ix_survivor(self) -> BenchmarkKernel | None:
        survivors = self.axiom_ix_survivors
        return survivors[0] if len(survivors) == 1 else None


def _normalize_kernel(candidate: Sequence[int]) -> BenchmarkKernel:
    if len(candidate) != 3:
        raise ValueError("Kernel sensitivity audits require (lepton_level, quark_level, parent_level) triples.")
    return (int(candidate[0]), int(candidate[1]), int(candidate[2]))


def audit_kernel_sensitivity(
    kernels: Sequence[Sequence[int]],
    *,
    generation_count: int = G_SM,
    logic_relation: LogicRelation | None = None,
    precision: int = DEFAULT_PRECISION,
) -> KernelSensitivityAudit:
    if not kernels:
        raise ValueError("Kernel sensitivity audits require at least one kernel.")

    resolved_precision = max(int(precision), DEFAULT_PRECISION)
    benchmark_cascade = evaluate_ontic_cascade(
        OnticAxioms(
            lepton_level=BENCHMARK_KERNEL[0],
            quark_level=BENCHMARK_KERNEL[1],
            parent_level=BENCHMARK_KERNEL[2],
            generation_count=int(generation_count),
        ),
        logic_relation=logic_relation,
        precision=resolved_precision,
    )

    points: list[KernelSensitivityPoint] = []
    seen: set[BenchmarkKernel] = set()
    for candidate in kernels:
        kernel = _normalize_kernel(candidate)
        if kernel in seen:
            continue
        seen.add(kernel)
        cascade = evaluate_ontic_cascade(
            OnticAxioms(
                lepton_level=kernel[0],
                quark_level=kernel[1],
                parent_level=kernel[2],
                generation_count=int(generation_count),
            ),
            logic_relation=logic_relation,
            precision=resolved_precision,
        )
        points.append(
            KernelSensitivityPoint(
                kernel=kernel,
                cascade=cascade,
                alpha_shift_fraction=abs(cascade.alpha_inverse_fraction - benchmark_cascade.alpha_inverse_fraction),
                alpha_shift_decimal=abs(cascade.alpha_inverse_decimal - benchmark_cascade.alpha_inverse_decimal),
            )
        )

    return KernelSensitivityAudit(
        benchmark_kernel=BENCHMARK_KERNEL,
        benchmark_cascade=benchmark_cascade,
        evaluated_points=tuple(points),
    )


def audit_alternative_dimension_kernels(
    boundary_dimensions: Sequence[int],
    *,
    quark_level: int = QUARK_LEVEL,
    parent_level: int = PARENT_LEVEL,
    generation_count: int = G_SM,
    logic_relation: LogicRelation | None = None,
    precision: int = DEFAULT_PRECISION,
) -> KernelSensitivityAudit:
    kernels = [
        (int(boundary_dimension), int(quark_level), int(parent_level))
        for boundary_dimension in boundary_dimensions
    ]
    return audit_kernel_sensitivity(
        kernels,
        generation_count=generation_count,
        logic_relation=logic_relation,
        precision=precision,
    )


def assert_unique_axiom_ix_survivor(audit: KernelSensitivityAudit) -> BenchmarkKernel:
    survivor = audit.unique_axiom_ix_survivor
    if survivor is None:
        raise AssertionError(
            "Axiom IX sensitivity audit failed to isolate a unique survivor: "
            f"{audit.axiom_ix_survivors}."
        )
    return survivor


__all__ = [
    "BENCHMARK_KERNEL",
    "KernelSensitivityAudit",
    "KernelSensitivityPoint",
    "assert_unique_axiom_ix_survivor",
    "audit_alternative_dimension_kernels",
    "audit_kernel_sensitivity",
]
