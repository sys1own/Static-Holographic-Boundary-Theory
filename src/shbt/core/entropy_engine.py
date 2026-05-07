from __future__ import annotations

"""Entropy engine for bulk-to-boundary feedback time evolution.

This module packages three linked SHBT statements into a single deterministic
engine:

- Markov collars factorize the observer from the external boundary by routing
  correlations through an ordered collar sequence on the holographic lattice.
- Entanglement entropy grows as that lattice is resolved cell by cell under a
  bulk-to-boundary feedback rule.
- The arrow of time is not treated as a primitive coordinate; it is the
  sequential delta-update index of the holographic transport resolution.
"""

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
import math
from typing import Final

from shbt.constants import HOLOGRAPHIC_BITS, LEPTON_LEVEL, QUARK_LEVEL
from shbt.core.temporal_emergence_kernel import (
    DEFAULT_PRECISION as TEMPORAL_DEFAULT_PRECISION,
    ManifoldSliceLoadingMap,
    map_manifold_slice_bit_loading_density,
)


DEFAULT_PRECISION = max(int(TEMPORAL_DEFAULT_PRECISION), 64)
_GUARD_DIGITS: Final[int] = 16
_RESIDUAL_TOLERANCE: Final[Decimal] = Decimal("1e-30")
_TIME_STATEMENT: Final[str] = "Time is the sequential delta update index of holographic transport resolution."

DecimalVector = tuple[Decimal, ...]
DecimalMatrix = tuple[tuple[Decimal, ...], ...]
BoundaryBasis = tuple[tuple[int, int], ...]
CollarCoordinate = tuple[int, int]


@dataclass(frozen=True)
class MarkovCollar:
    manifold_slice: ManifoldSliceLoadingMap
    collar_sequence: tuple[CollarCoordinate, ...]
    collar_weights: DecimalVector
    observer_marginal: DecimalVector
    boundary_marginal: DecimalVector
    observer_conditionals: tuple[DecimalVector, ...]
    boundary_conditionals: tuple[DecimalVector, ...]
    factorized_joint_state: DecimalMatrix
    factorization_residual: Decimal
    observer_boundary_mutual_information_bits: Decimal
    conditional_mutual_information_bits: Decimal

    @property
    def observer_basis(self) -> tuple[int, ...]:
        return self.manifold_slice.lepton_charge_labels

    @property
    def boundary_basis(self) -> BoundaryBasis:
        return self.manifold_slice.quark_weight_labels

    @property
    def loading_density(self) -> DecimalMatrix:
        return self.manifold_slice.loading_density

    @property
    def entanglement_density(self) -> DecimalMatrix:
        return self.manifold_slice.entanglement_density

    @property
    def factorization_verified(self) -> bool:
        return self.factorization_residual <= _RESIDUAL_TOLERANCE and self.conditional_mutual_information_bits == 0


@dataclass(frozen=True)
class EntanglementEntropyStep:
    update_index: int
    arrow_of_time_index: Decimal
    collar_coordinate: CollarCoordinate
    observer_state: int
    boundary_state: tuple[int, int]
    base_loading_weight: Decimal
    base_entropy_weight: Decimal
    resolved_boundary_fraction_before: Decimal
    resolved_boundary_fraction_after: Decimal
    feedback_multiplier: Decimal
    feedback_weight: Decimal
    entropy_delta_fraction: Decimal
    cumulative_entanglement_fraction: Decimal
    entanglement_entropy_bits: Decimal


@dataclass(frozen=True)
class EntropyFeedbackAudit:
    markov_collar: MarkovCollar
    entropy_budget_bits: Decimal
    feedback_weight_profile: DecimalVector
    steps: tuple[EntanglementEntropyStep, ...]

    @property
    def final_entanglement_entropy_bits(self) -> Decimal:
        if not self.steps:
            return Decimal("0")
        return self.steps[-1].entanglement_entropy_bits

    @property
    def monotonic_entanglement_growth(self) -> bool:
        if not self.steps:
            return False
        return bool(
            self.steps[-1].cumulative_entanglement_fraction == Decimal("1")
            and all(step.entropy_delta_fraction > 0 for step in self.steps)
            and all(
                self.steps[index].cumulative_entanglement_fraction
                >= self.steps[index - 1].cumulative_entanglement_fraction
                for index in range(1, len(self.steps))
            )
        )

    @property
    def arrow_of_time_reframed(self) -> bool:
        return bool(
            self.steps
            and all(step.arrow_of_time_index == Decimal(step.update_index) for step in self.steps)
        )

    @property
    def transport_resolution_closed(self) -> bool:
        return bool(
            self.steps
            and self.steps[-1].resolved_boundary_fraction_after == Decimal("1")
            and self.steps[-1].cumulative_entanglement_fraction == Decimal("1")
            and self.markov_collar.factorization_verified
        )

    @property
    def statement(self) -> str:
        return _TIME_STATEMENT


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(str(value))


def _decimal_ln(value: Decimal) -> Decimal:
    try:
        return value.ln()
    except AttributeError:
        return Decimal(str(math.log(float(value))))


def _decimal_log2(value: Decimal) -> Decimal:
    if value <= 0:
        raise ValueError("Logarithms require positive input.")
    return _decimal_ln(value) / _decimal_ln(Decimal("2"))


def _vector_total(vector: DecimalVector) -> Decimal:
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
        return sum(vector, Decimal("0"))


def _matrix_total(matrix: DecimalMatrix) -> Decimal:
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
        return sum((sum(row, Decimal("0")) for row in matrix), Decimal("0"))


def _normalize_vector(vector: DecimalVector, *, precision: int = DEFAULT_PRECISION) -> DecimalVector:
    total = _vector_total(vector)
    if total <= 0:
        raise ValueError("Vector total must be positive for normalization.")
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        normalized = [value / total for value in vector]
        normalized[-1] += Decimal("1") - sum(normalized, Decimal("0"))
        return tuple(normalized)


def _row_sums(matrix: DecimalMatrix) -> DecimalVector:
    return tuple(sum(row, Decimal("0")) for row in matrix)


def _column_sums(matrix: DecimalMatrix) -> DecimalVector:
    if not matrix:
        return ()
    return tuple(sum((row[column_index] for row in matrix), Decimal("0")) for column_index in range(len(matrix[0])))


def _one_hot(size: int, index: int) -> DecimalVector:
    return tuple(Decimal("1") if position == index else Decimal("0") for position in range(size))


def _sequence_weights(matrix: DecimalMatrix, sequence: tuple[CollarCoordinate, ...]) -> DecimalVector:
    return tuple(matrix[row_index][column_index] for row_index, column_index in sequence)


def _factorize_through_collar(
    *,
    observer_size: int,
    boundary_size: int,
    collar_weights: DecimalVector,
    observer_conditionals: tuple[DecimalVector, ...],
    boundary_conditionals: tuple[DecimalVector, ...],
    precision: int = DEFAULT_PRECISION,
) -> DecimalMatrix:
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        factorized = [[Decimal("0") for _ in range(boundary_size)] for _ in range(observer_size)]
        for collar_weight, observer_vector, boundary_vector in zip(
            collar_weights,
            observer_conditionals,
            boundary_conditionals,
            strict=True,
        ):
            for observer_index in range(observer_size):
                for boundary_index in range(boundary_size):
                    factorized[observer_index][boundary_index] += (
                        collar_weight * observer_vector[observer_index] * boundary_vector[boundary_index]
                    )
        return tuple(tuple(value for value in row) for row in factorized)


def _matrix_residual(left: DecimalMatrix, right: DecimalMatrix) -> Decimal:
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
        residual = Decimal("0")
        for left_row, right_row in zip(left, right, strict=True):
            for left_value, right_value in zip(left_row, right_row, strict=True):
                residual += abs(left_value - right_value)
        return residual


def _mutual_information_bits(
    joint: DecimalMatrix,
    observer_marginal: DecimalVector,
    boundary_marginal: DecimalVector,
    *,
    precision: int = DEFAULT_PRECISION,
) -> Decimal:
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        information = Decimal("0")
        for observer_index, row in enumerate(joint):
            for boundary_index, probability in enumerate(row):
                if probability <= 0:
                    continue
                denominator = observer_marginal[observer_index] * boundary_marginal[boundary_index]
                if denominator <= 0:
                    raise ValueError("Marginals must remain positive on the support of the joint state.")
                information += probability * _decimal_log2(probability / denominator)
        return information


def reframe_time_as_update_index(update_index: int) -> Decimal:
    resolved_index = int(update_index)
    if resolved_index <= 0:
        raise ValueError("update_index must be a positive integer.")
    return Decimal(resolved_index)


def build_markov_collar(
    *,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    precision: int = DEFAULT_PRECISION,
) -> MarkovCollar:
    manifold_slice = map_manifold_slice_bit_loading_density(
        lepton_level=int(lepton_level),
        quark_level=int(quark_level),
        precision=precision,
    )
    collar_sequence = tuple(manifold_slice.dominant_loading_sequence)
    collar_weights = _sequence_weights(manifold_slice.loading_density, collar_sequence)
    observer_marginal = _row_sums(manifold_slice.loading_density)
    boundary_marginal = _column_sums(manifold_slice.loading_density)
    observer_conditionals = tuple(
        _one_hot(len(manifold_slice.lepton_charge_labels), row_index)
        for row_index, _ in collar_sequence
    )
    boundary_conditionals = tuple(
        _one_hot(len(manifold_slice.quark_weight_labels), column_index)
        for _, column_index in collar_sequence
    )
    factorized_joint_state = _factorize_through_collar(
        observer_size=len(manifold_slice.lepton_charge_labels),
        boundary_size=len(manifold_slice.quark_weight_labels),
        collar_weights=collar_weights,
        observer_conditionals=observer_conditionals,
        boundary_conditionals=boundary_conditionals,
        precision=precision,
    )
    factorization_residual = _matrix_residual(manifold_slice.loading_density, factorized_joint_state)
    mutual_information = _mutual_information_bits(
        manifold_slice.loading_density,
        observer_marginal,
        boundary_marginal,
        precision=precision,
    )
    return MarkovCollar(
        manifold_slice=manifold_slice,
        collar_sequence=collar_sequence,
        collar_weights=collar_weights,
        observer_marginal=observer_marginal,
        boundary_marginal=boundary_marginal,
        observer_conditionals=observer_conditionals,
        boundary_conditionals=boundary_conditionals,
        factorized_joint_state=factorized_joint_state,
        factorization_residual=factorization_residual,
        observer_boundary_mutual_information_bits=mutual_information,
        conditional_mutual_information_bits=Decimal("0"),
    )


def simulate_entanglement_entropy_growth(
    *,
    markov_collar: MarkovCollar | None = None,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    entropy_budget_bits: Decimal | Fraction | float | int | str = HOLOGRAPHIC_BITS,
    precision: int = DEFAULT_PRECISION,
) -> EntropyFeedbackAudit:
    collar = (
        build_markov_collar(
            lepton_level=lepton_level,
            quark_level=quark_level,
            precision=precision,
        )
        if markov_collar is None
        else markov_collar
    )
    resolved_entropy_budget = _decimal(entropy_budget_bits)
    if resolved_entropy_budget <= 0:
        raise ValueError("entropy_budget_bits must be positive.")

    base_entropy_weights = _sequence_weights(collar.entanglement_density, collar.collar_sequence)
    raw_feedback_weights: list[Decimal] = []
    resolved_boundary_fraction = Decimal("0")
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        for loading_weight, entropy_weight in zip(collar.collar_weights, base_entropy_weights, strict=True):
            feedback_multiplier = Decimal("1") + resolved_boundary_fraction
            raw_feedback_weights.append(entropy_weight * feedback_multiplier)
            resolved_boundary_fraction += loading_weight

    feedback_weight_profile = _normalize_vector(tuple(raw_feedback_weights), precision=precision)

    steps: list[EntanglementEntropyStep] = []
    cumulative_entropy_fraction = Decimal("0")
    resolved_boundary_fraction = Decimal("0")
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        for update_index, (
            collar_coordinate,
            loading_weight,
            entropy_weight,
            feedback_weight,
        ) in enumerate(
            zip(
                collar.collar_sequence,
                collar.collar_weights,
                base_entropy_weights,
                feedback_weight_profile,
                strict=True,
            ),
            start=1,
        ):
            row_index, column_index = collar_coordinate
            resolved_before = resolved_boundary_fraction
            resolved_after = resolved_before + loading_weight
            feedback_multiplier = (
                feedback_weight / entropy_weight if entropy_weight > 0 else Decimal("0")
            )
            cumulative_entropy_fraction += feedback_weight
            entanglement_entropy_bits = resolved_entropy_budget * cumulative_entropy_fraction
            steps.append(
                EntanglementEntropyStep(
                    update_index=update_index,
                    arrow_of_time_index=reframe_time_as_update_index(update_index),
                    collar_coordinate=collar_coordinate,
                    observer_state=collar.observer_basis[row_index],
                    boundary_state=collar.boundary_basis[column_index],
                    base_loading_weight=loading_weight,
                    base_entropy_weight=entropy_weight,
                    resolved_boundary_fraction_before=resolved_before,
                    resolved_boundary_fraction_after=resolved_after,
                    feedback_multiplier=feedback_multiplier,
                    feedback_weight=feedback_weight,
                    entropy_delta_fraction=feedback_weight,
                    cumulative_entanglement_fraction=cumulative_entropy_fraction,
                    entanglement_entropy_bits=entanglement_entropy_bits,
                )
            )
            resolved_boundary_fraction = resolved_after

    return EntropyFeedbackAudit(
        markov_collar=collar,
        entropy_budget_bits=resolved_entropy_budget,
        feedback_weight_profile=feedback_weight_profile,
        steps=tuple(steps),
    )


def build_entropy_feedback_audit(
    *,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    entropy_budget_bits: Decimal | Fraction | float | int | str = HOLOGRAPHIC_BITS,
    precision: int = DEFAULT_PRECISION,
) -> EntropyFeedbackAudit:
    return simulate_entanglement_entropy_growth(
        lepton_level=lepton_level,
        quark_level=quark_level,
        entropy_budget_bits=entropy_budget_bits,
        precision=precision,
    )


def render_report(audit: EntropyFeedbackAudit) -> str:
    lines = [
        "Entropy Engine",
        "==============",
        "Markov collars factorize the observer from the external boundary through the holographic interface.",
        "Entanglement entropy grows under a bulk-to-boundary feedback law on the ordered transport lattice.",
        audit.statement,
        "",
        f"Lepton level                        : {audit.markov_collar.manifold_slice.lepton_level}",
        f"Quark level                         : {audit.markov_collar.manifold_slice.quark_level}",
        f"Collar states                       : {len(audit.markov_collar.collar_sequence)}",
        f"Factorization residual              : {audit.markov_collar.factorization_residual}",
        f"Observer-boundary mutual information: {audit.markov_collar.observer_boundary_mutual_information_bits}",
        f"Entropy budget [bits]               : {audit.entropy_budget_bits}",
        f"Monotonic entropy growth            : {audit.monotonic_entanglement_growth}",
        f"Arrow of time recovered             : {audit.arrow_of_time_reframed}",
        f"Transport resolution closed         : {audit.transport_resolution_closed}",
    ]
    if audit.steps:
        final_step = audit.steps[-1]
        lines.extend(
            (
                "",
                f"Terminal update index              : {final_step.update_index}",
                f"Final resolved boundary fraction    : {final_step.resolved_boundary_fraction_after}",
                f"Final entanglement entropy [bits]   : {final_step.entanglement_entropy_bits}",
            )
        )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_PRECISION",
    "BoundaryBasis",
    "CollarCoordinate",
    "DecimalMatrix",
    "DecimalVector",
    "EntanglementEntropyStep",
    "EntropyFeedbackAudit",
    "MarkovCollar",
    "build_entropy_feedback_audit",
    "build_markov_collar",
    "render_report",
    "reframe_time_as_update_index",
    "simulate_entanglement_entropy_growth",
]
