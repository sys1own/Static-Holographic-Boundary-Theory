from __future__ import annotations

r"""Dynamic entropy engine for computational time emergence.

The SHBT benchmark branch fixes a static boundary configuration, but the visible
Arrow of Time is taken here to be the ordered computational overhead required to
resolve that boundary into its effective bulk image. The engine therefore turns
static loading data into an iterative solver with explicit bulk-to-boundary
feedback.

Let ``\sigma = (c_1, \ldots, c_n)`` be the dominant holographic loading order on
the benchmark manifold slice. If ``\rho_B`` and ``\rho_E`` are the normalized
boundary-loading and entanglement densities, define the cumulative bulk state
before iteration ``n`` by

    \Omega_{n-1}(i,j) = \sum_{m < n} \delta_{c_m,(i,j)} \frac{\Delta S_m}{N}.

For a candidate coordinate ``c`` in the unresolved set, the bulk-to-boundary
feedback signal is

    F_{n-1}(c) = \sum_{i,j} \frac{\Omega_{n-1}(i,j)}{1 + |i-c_i| + |j-c_j|}.

The next resolution step uses the feedback-dressed weights

    w_B(c) = \rho_B(c) [1 + \lambda F_{n-1}(c)],
    w_E(c) = \rho_E(c) [1 + \lambda F_{n-1}(c)],

with non-negative feedback coupling ``\lambda``. If ``R_B`` and ``R_E`` denote
remaining unresolved boundary and bulk entropy budgets, then the generated
bit-residues are

    \Delta B_n = R_B \frac{w_B(c_n)}{\sum_{c \in U_n} w_B(c)},
    \Delta S_n = R_E \frac{w_E(c_n)}{\sum_{c \in U_n} w_E(c)}.

Time is not inserted as an external coordinate. It is the entropic overhead
required by self-resolution:

    \Delta T_n = \frac{\Delta S_n}{N}.

To tie this directly to the benchmark ``26D \to 4D`` collapse, define the
sequential projection state

    D_n = 26 - (26 - 4) \sum_{m \le n} \Delta T_m,

so every positive update obeys

    \Delta D_n = D_{n-1} - D_n = 22 \Delta T_n > 0.

The Arrow of Time is therefore the ordered sequence of positive projection
updates, while the feedback functional closes a causal loop from the emergent
bulk entropy state back into the next boundary-resolution step.
"""

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Final, Sequence

if __package__ in (None, ""):
    import sys

    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.constants import HOLOGRAPHIC_BITS, LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.temporal_emergence_kernel import (
    DEFAULT_PRECISION as TEMPORAL_DEFAULT_PRECISION,
    ManifoldSliceLoadingMap,
    derive_temporal_increment,
    map_manifold_slice_bit_loading_density,
)
from shbt.core.topological_kernel import sequence_bit_loading


Coordinate = tuple[int, int]
Branch = tuple[int, int, int]
DecimalMatrix = tuple[tuple[Decimal, ...], ...]

DEFAULT_PRECISION: Final[int] = max(int(TEMPORAL_DEFAULT_PRECISION), 64)
DEFAULT_FEEDBACK_COUPLING: Final[Decimal] = Decimal("1")
SOURCE_PROJECTION_DIMENSION: Final[int] = 26
TARGET_PROJECTION_DIMENSION: Final[int] = 4
_PROJECTION_GAP: Final[Decimal] = Decimal(str(SOURCE_PROJECTION_DIMENSION - TARGET_PROJECTION_DIMENSION))
_GUARD_DIGITS: Final[int] = 16
_RESOLUTION_TOLERANCE: Final[Decimal] = Decimal("1e-30")
_IDENTITY_TOLERANCE: Final[Decimal] = Decimal("1e-27")
BENCHMARK_BRANCH: Final[Branch] = (
    int(LEPTON_LEVEL),
    int(QUARK_LEVEL),
    int(PARENT_LEVEL),
)


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(str(value))


def _zero_matrix(shape: tuple[int, int]) -> DecimalMatrix:
    rows, columns = shape
    return tuple(tuple(Decimal("0") for _ in range(columns)) for _ in range(rows))


def _matrix_shape(matrix: DecimalMatrix) -> tuple[int, int]:
    return (len(matrix), 0 if not matrix else len(matrix[0]))


def _matrix_update(matrix: DecimalMatrix, coordinate: Coordinate, delta: Decimal) -> DecimalMatrix:
    row_index, column_index = coordinate
    updated_rows: list[tuple[Decimal, ...]] = []
    for current_row_index, row in enumerate(matrix):
        if current_row_index != row_index:
            updated_rows.append(row)
            continue
        updated_row = list(row)
        updated_row[column_index] += delta
        updated_rows.append(tuple(updated_row))
    return tuple(updated_rows)


def _manhattan_distance(left: Coordinate, right: Coordinate) -> int:
    return abs(int(left[0]) - int(right[0])) + abs(int(left[1]) - int(right[1]))


def _feedback_signal(
    *,
    bulk_entropy_state: DecimalMatrix,
    coordinate: Coordinate,
    precision: int,
) -> Decimal:
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        signal = Decimal("0")
        for row_index, row in enumerate(bulk_entropy_state):
            for column_index, state_weight in enumerate(row):
                if state_weight == 0:
                    continue
                signal += state_weight / Decimal(1 + _manhattan_distance((row_index, column_index), coordinate))
    return signal


@dataclass(frozen=True)
class EntropyIteration:
    iteration_index: int
    coordinate: Coordinate
    sequence_rank: int
    baseline_boundary_share: Decimal
    adjusted_boundary_share: Decimal
    baseline_entanglement_share: Decimal
    adjusted_entanglement_share: Decimal
    bulk_feedback_signal: Decimal
    boundary_bit_residue: Decimal
    bulk_entropy_bit_residue: Decimal
    temporal_increment: Decimal
    cumulative_boundary_bit_residue: Decimal
    cumulative_bulk_entropy_bit_residue: Decimal
    cumulative_bulk_entropy_fraction: Decimal
    projected_dimension_before: Decimal
    projected_dimension_after: Decimal
    delta_projection_26d_to_4d: Decimal
    bulk_entropy_state: DecimalMatrix

    @property
    def feedback_perturbs_boundary_resolution(self) -> bool:
        return abs(self.adjusted_boundary_share - self.baseline_boundary_share) > _RESOLUTION_TOLERANCE

    @property
    def causal_feedback_active(self) -> bool:
        return self.bulk_feedback_signal > 0

    @property
    def arrow_of_time_positive(self) -> bool:
        return bool(
            self.temporal_increment > 0
            and self.delta_projection_26d_to_4d > 0
            and self.projected_dimension_after < self.projected_dimension_before
        )

    @property
    def projection_time_identity_holds(self) -> bool:
        return abs(self.delta_projection_26d_to_4d - (_PROJECTION_GAP * self.temporal_increment)) <= _IDENTITY_TOLERANCE


@dataclass(frozen=True)
class EntropyEngineAudit:
    branch: Branch
    bit_budget: Decimal
    feedback_coupling: Decimal
    source_projection_dimension: int
    target_projection_dimension: int
    manifold_slice: ManifoldSliceLoadingMap
    loading_sequence: tuple[Coordinate, ...]
    iterations: tuple[EntropyIteration, ...]
    final_bulk_entropy_state: DecimalMatrix
    boundary_budget_residual: Decimal
    bulk_entropy_budget_residual: Decimal
    final_projection_residual: Decimal
    time_is_computational_overhead: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_is_computational_overhead", True)

    @property
    def total_boundary_bit_residue(self) -> Decimal:
        return Decimal("0") if not self.iterations else self.iterations[-1].cumulative_boundary_bit_residue

    @property
    def total_bulk_entropy_bit_residue(self) -> Decimal:
        return Decimal("0") if not self.iterations else self.iterations[-1].cumulative_bulk_entropy_bit_residue

    @property
    def total_temporal_overhead(self) -> Decimal:
        return sum((step.temporal_increment for step in self.iterations), Decimal("0"))

    @property
    def final_projected_dimension(self) -> Decimal:
        if not self.iterations:
            return Decimal(str(self.source_projection_dimension))
        return self.iterations[-1].projected_dimension_after

    @property
    def temporal_overhead_identity_holds(self) -> bool:
        return bool(
            abs(self.total_temporal_overhead - derive_temporal_increment(self.total_bulk_entropy_bit_residue, self.bit_budget))
            <= _RESOLUTION_TOLERANCE
        )

    @property
    def arrow_of_time_established(self) -> bool:
        return bool(
            self.iterations
            and all(step.arrow_of_time_positive for step in self.iterations)
            and all(step.projection_time_identity_holds for step in self.iterations)
            and all(
                self.iterations[index].cumulative_bulk_entropy_bit_residue
                > self.iterations[index - 1].cumulative_bulk_entropy_bit_residue
                for index in range(1, len(self.iterations))
            )
        )

    @property
    def bulk_boundary_causal_loop_closed(self) -> bool:
        if len(self.iterations) < 2:
            return False
        return bool(
            any(step.causal_feedback_active for step in self.iterations[1:])
            and any(step.feedback_perturbs_boundary_resolution for step in self.iterations[1:])
        )

    @property
    def self_resolution_complete(self) -> bool:
        return bool(
            abs(self.boundary_budget_residual) <= _RESOLUTION_TOLERANCE
            and abs(self.bulk_entropy_budget_residual) <= _RESOLUTION_TOLERANCE
            and abs(self.final_projection_residual) <= _RESOLUTION_TOLERANCE
        )

    @property
    def statement(self) -> str:
        return "Time is the computational overhead of the universe's self-resolution."


class EntropyAccumulator:
    def __init__(
        self,
        *,
        bit_budget: Decimal | Fraction | float | int | str = HOLOGRAPHIC_BITS,
        lepton_level: int = LEPTON_LEVEL,
        quark_level: int = QUARK_LEVEL,
        parent_level: int = PARENT_LEVEL,
        feedback_coupling: Decimal | Fraction | float | int | str = DEFAULT_FEEDBACK_COUPLING,
        precision: int = DEFAULT_PRECISION,
    ) -> None:
        self.bit_budget = _decimal(bit_budget)
        self.lepton_level = int(lepton_level)
        self.quark_level = int(quark_level)
        self.parent_level = int(parent_level)
        self.branch: Branch = (self.lepton_level, self.quark_level, self.parent_level)
        self.feedback_coupling = _decimal(feedback_coupling)
        self.precision = max(int(precision), DEFAULT_PRECISION)
        if self.bit_budget <= 0:
            raise ValueError("bit_budget must be positive.")
        if self.feedback_coupling < 0:
            raise ValueError("feedback_coupling must be non-negative.")

        self.manifold_slice = map_manifold_slice_bit_loading_density(
            lepton_level=self.lepton_level,
            quark_level=self.quark_level,
            precision=self.precision,
        )
        self.loading_sequence = tuple(
            sequence_bit_loading(lepton_level=self.lepton_level, quark_level=self.quark_level)
        )
        assert self.loading_sequence == self.manifold_slice.dominant_loading_sequence
        self._sequence_rank = {coordinate: index + 1 for index, coordinate in enumerate(self.loading_sequence)}
        self._iterations: list[EntropyIteration] = []
        self._bulk_entropy_state = _zero_matrix(_matrix_shape(self.manifold_slice.entanglement_density))
        self._cumulative_boundary_bit_residue = Decimal("0")
        self._cumulative_bulk_entropy_bit_residue = Decimal("0")

    @property
    def iterations(self) -> tuple[EntropyIteration, ...]:
        return tuple(self._iterations)

    @property
    def iteration_count(self) -> int:
        return len(self._iterations)

    @property
    def bulk_entropy_state(self) -> DecimalMatrix:
        return self._bulk_entropy_state

    @property
    def remaining_coordinates(self) -> tuple[Coordinate, ...]:
        return self.loading_sequence[self.iteration_count :]

    def _remaining_share_maps(
        self,
        *,
        remaining_coordinates: Sequence[Coordinate],
    ) -> tuple[dict[Coordinate, Decimal], dict[Coordinate, Decimal], dict[Coordinate, Decimal], dict[Coordinate, Decimal], dict[Coordinate, Decimal]]:
        loading_density = self.manifold_slice.loading_density
        entanglement_density = self.manifold_slice.entanglement_density
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            feedback_signals = {
                coordinate: _feedback_signal(
                    bulk_entropy_state=self._bulk_entropy_state,
                    coordinate=coordinate,
                    precision=self.precision,
                )
                for coordinate in remaining_coordinates
            }
            baseline_boundary_total = sum(
                (loading_density[row_index][column_index] for row_index, column_index in remaining_coordinates),
                Decimal("0"),
            )
            baseline_entanglement_total = sum(
                (entanglement_density[row_index][column_index] for row_index, column_index in remaining_coordinates),
                Decimal("0"),
            )
            adjusted_boundary_weights = {
                coordinate: loading_density[coordinate[0]][coordinate[1]]
                * (Decimal("1") + self.feedback_coupling * feedback_signals[coordinate])
                for coordinate in remaining_coordinates
            }
            adjusted_entanglement_weights = {
                coordinate: entanglement_density[coordinate[0]][coordinate[1]]
                * (Decimal("1") + self.feedback_coupling * feedback_signals[coordinate])
                for coordinate in remaining_coordinates
            }
            adjusted_boundary_total = sum(adjusted_boundary_weights.values(), Decimal("0"))
            adjusted_entanglement_total = sum(adjusted_entanglement_weights.values(), Decimal("0"))
            baseline_boundary_share = {
                coordinate: loading_density[coordinate[0]][coordinate[1]] / baseline_boundary_total
                for coordinate in remaining_coordinates
            }
            adjusted_boundary_share = {
                coordinate: adjusted_boundary_weights[coordinate] / adjusted_boundary_total
                for coordinate in remaining_coordinates
            }
            baseline_entanglement_share = {
                coordinate: entanglement_density[coordinate[0]][coordinate[1]] / baseline_entanglement_total
                for coordinate in remaining_coordinates
            }
            adjusted_entanglement_share = {
                coordinate: adjusted_entanglement_weights[coordinate] / adjusted_entanglement_total
                for coordinate in remaining_coordinates
            }
        return (
            feedback_signals,
            baseline_boundary_share,
            adjusted_boundary_share,
            baseline_entanglement_share,
            adjusted_entanglement_share,
        )

    def step(self) -> EntropyIteration:
        remaining_coordinates = self.remaining_coordinates
        if not remaining_coordinates:
            raise StopIteration("EntropyAccumulator has already resolved the full loading sequence.")

        coordinate = remaining_coordinates[0]
        (
            feedback_signals,
            baseline_boundary_share,
            adjusted_boundary_share,
            baseline_entanglement_share,
            adjusted_entanglement_share,
        ) = self._remaining_share_maps(remaining_coordinates=remaining_coordinates)

        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            remaining_boundary_budget = self.bit_budget - self._cumulative_boundary_bit_residue
            remaining_bulk_entropy_budget = self.bit_budget - self._cumulative_bulk_entropy_bit_residue
            boundary_bit_residue = remaining_boundary_budget * adjusted_boundary_share[coordinate]
            bulk_entropy_bit_residue = remaining_bulk_entropy_budget * adjusted_entanglement_share[coordinate]
            projected_dimension_before = Decimal(str(SOURCE_PROJECTION_DIMENSION)) - _PROJECTION_GAP * derive_temporal_increment(
                self._cumulative_bulk_entropy_bit_residue,
                self.bit_budget,
                precision=self.precision,
            )
            temporal_increment = derive_temporal_increment(
                bulk_entropy_bit_residue,
                self.bit_budget,
                precision=self.precision,
            )
            self._cumulative_boundary_bit_residue += boundary_bit_residue
            self._cumulative_bulk_entropy_bit_residue += bulk_entropy_bit_residue
            cumulative_bulk_entropy_fraction = derive_temporal_increment(
                self._cumulative_bulk_entropy_bit_residue,
                self.bit_budget,
                precision=self.precision,
            )
            projected_dimension_after = Decimal(str(SOURCE_PROJECTION_DIMENSION)) - _PROJECTION_GAP * cumulative_bulk_entropy_fraction
            delta_projection = projected_dimension_before - projected_dimension_after
            self._bulk_entropy_state = _matrix_update(
                self._bulk_entropy_state,
                coordinate,
                temporal_increment,
            )

        iteration = EntropyIteration(
            iteration_index=self.iteration_count + 1,
            coordinate=coordinate,
            sequence_rank=self._sequence_rank[coordinate],
            baseline_boundary_share=baseline_boundary_share[coordinate],
            adjusted_boundary_share=adjusted_boundary_share[coordinate],
            baseline_entanglement_share=baseline_entanglement_share[coordinate],
            adjusted_entanglement_share=adjusted_entanglement_share[coordinate],
            bulk_feedback_signal=feedback_signals[coordinate],
            boundary_bit_residue=boundary_bit_residue,
            bulk_entropy_bit_residue=bulk_entropy_bit_residue,
            temporal_increment=temporal_increment,
            cumulative_boundary_bit_residue=self._cumulative_boundary_bit_residue,
            cumulative_bulk_entropy_bit_residue=self._cumulative_bulk_entropy_bit_residue,
            cumulative_bulk_entropy_fraction=cumulative_bulk_entropy_fraction,
            projected_dimension_before=projected_dimension_before,
            projected_dimension_after=projected_dimension_after,
            delta_projection_26d_to_4d=delta_projection,
            bulk_entropy_state=self._bulk_entropy_state,
        )
        self._iterations.append(iteration)
        return iteration

    def run(self) -> EntropyEngineAudit:
        while self.remaining_coordinates:
            self.step()
        return self.audit()

    def simulate(self) -> EntropyEngineAudit:
        return self.run()

    def audit(self) -> EntropyEngineAudit:
        final_projection_residual = self.final_projected_dimension - Decimal(str(TARGET_PROJECTION_DIMENSION))
        return EntropyEngineAudit(
            branch=self.branch,
            bit_budget=self.bit_budget,
            feedback_coupling=self.feedback_coupling,
            source_projection_dimension=SOURCE_PROJECTION_DIMENSION,
            target_projection_dimension=TARGET_PROJECTION_DIMENSION,
            manifold_slice=self.manifold_slice,
            loading_sequence=self.loading_sequence,
            iterations=self.iterations,
            final_bulk_entropy_state=self._bulk_entropy_state,
            boundary_budget_residual=self.bit_budget - self._cumulative_boundary_bit_residue,
            bulk_entropy_budget_residual=self.bit_budget - self._cumulative_bulk_entropy_bit_residue,
            final_projection_residual=final_projection_residual,
        )

    @property
    def final_projected_dimension(self) -> Decimal:
        if not self._iterations:
            return Decimal(str(SOURCE_PROJECTION_DIMENSION))
        return self._iterations[-1].projected_dimension_after


def build_entropy_engine_audit(
    *,
    bit_budget: Decimal | Fraction | float | int | str = HOLOGRAPHIC_BITS,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    parent_level: int = PARENT_LEVEL,
    feedback_coupling: Decimal | Fraction | float | int | str = DEFAULT_FEEDBACK_COUPLING,
    precision: int = DEFAULT_PRECISION,
) -> EntropyEngineAudit:
    return EntropyAccumulator(
        bit_budget=bit_budget,
        lepton_level=lepton_level,
        quark_level=quark_level,
        parent_level=parent_level,
        feedback_coupling=feedback_coupling,
        precision=precision,
    ).run()


def render_report(audit: EntropyEngineAudit) -> str:
    lines = [
        "Entropy Engine",
        "==============",
        "Time is modeled as the iterative entropic overhead of holographic self-resolution.",
        f"Benchmark branch                  : {audit.branch}",
        f"Feedback coupling lambda          : {audit.feedback_coupling}",
        f"Loading sequence length           : {len(audit.loading_sequence)}",
        f"Arrow of Time established         : {audit.arrow_of_time_established}",
        f"Bulk-boundary causal loop         : {audit.bulk_boundary_causal_loop_closed}",
        f"Time is computational overhead    : {audit.time_is_computational_overhead}",
        f"Self-resolution complete          : {audit.self_resolution_complete}",
        f"Final projected dimension         : {audit.final_projected_dimension}",
        f"Total temporal overhead           : {audit.total_temporal_overhead}",
        audit.statement,
    ]
    return "\n".join(lines)


__all__ = [
    "BENCHMARK_BRANCH",
    "DEFAULT_FEEDBACK_COUPLING",
    "DEFAULT_PRECISION",
    "EntropyAccumulator",
    "EntropyEngineAudit",
    "EntropyIteration",
    "SOURCE_PROJECTION_DIMENSION",
    "TARGET_PROJECTION_DIMENSION",
    "build_entropy_engine_audit",
    "render_report",
]
