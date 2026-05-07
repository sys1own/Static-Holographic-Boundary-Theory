from __future__ import annotations

"""Computational-overhead solver for the SHBT 26D -> 4D projection.

This module removes the last "static" interpretation of time from the solver
stack. Time is treated here as the iteration counter required to reconcile the
branch-fixed 26-dimensional boundary with the emergent 4-dimensional bulk. At

- iteration ``0`` the universe is initialized by the ``main()`` call,
- every subsequent solver iteration contributes a non-negative entropy-derived
  bit residue, and
- the arrow of time is simply the ordered sequence of those reconciliation
  computations.
"""

import argparse
from dataclasses import dataclass
from decimal import Decimal
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

from shbt.constants import BULK_SPACETIME_DIMENSION, HOLOGRAPHIC_BITS, LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.causal_execution import ObservableTarget, TransportChecker
from shbt.core.causal_resolution import CausalResolutionAudit, RecursiveTemporalFrame, TemporalStateResolver
from shbt.core.temporal_emergence_kernel import DEFAULT_PRECISION, derive_temporal_increment


Coordinate = tuple[int, int]
Branch = tuple[int, int, int]


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(str(value))


@dataclass(frozen=True)
class EntropyResidueStep:
    iteration_index: int
    time_counter: int
    coordinate: Coordinate
    delta_runtime_overhead_ticks: int
    delta_entanglement_entropy: Decimal
    delta_bit_residue: Decimal
    cumulative_entanglement_entropy: Decimal
    cumulative_bit_residue: Decimal
    pending_coordinate_count: int
    equilibrium_preserved: bool

    @property
    def statement(self) -> str:
        return "Each solver iteration deposits a bit residue required to reconcile boundary and bulk."

    @property
    def advances_time(self) -> bool:
        return self.time_counter == self.iteration_index


@dataclass(frozen=True)
class SolverIterationFrame:
    iteration_index: int
    time_counter: int
    branch: Branch
    boundary_dimension: int
    bulk_dimension: int
    recursive_depth: int
    recursive_frame: RecursiveTemporalFrame
    entropy_residue: EntropyResidueStep

    @property
    def coordinate(self) -> Coordinate:
        return self.recursive_frame.coordinate

    @property
    def runtime_overhead_ticks(self) -> int:
        return self.recursive_frame.runtime_overhead_ticks

    @property
    def maintains_bulk_boundary_equilibrium(self) -> bool:
        return bool(
            self.recursive_frame.maintains_bulk_boundary_equilibrium
            and self.entropy_residue.equilibrium_preserved
        )

    @property
    def statement(self) -> str:
        return "The present universe is the current recursive frame indexed by the solver iteration counter."


@dataclass(frozen=True)
class EntropyAccumulator:
    bit_budget: Decimal | Fraction | float | int | str = HOLOGRAPHIC_BITS
    precision: int = DEFAULT_PRECISION

    def __post_init__(self) -> None:
        resolved_bit_budget = _decimal(self.bit_budget)
        if resolved_bit_budget <= 0:
            raise ValueError("bit_budget must be positive.")
        object.__setattr__(self, "bit_budget", resolved_bit_budget)
        object.__setattr__(self, "precision", max(int(self.precision), DEFAULT_PRECISION))

    def accumulate(self, audit: CausalResolutionAudit) -> tuple[EntropyResidueStep, ...]:
        cumulative_entanglement_entropy = Decimal("0")
        cumulative_bit_residue = Decimal("0")
        entropy_steps: list[EntropyResidueStep] = []

        for delta_update in audit.delta_updates:
            delta_entanglement_entropy = delta_update.delta_entanglement_weight
            delta_bit_residue = derive_temporal_increment(
                delta_entanglement_entropy,
                self.bit_budget,
                precision=self.precision,
            )
            cumulative_entanglement_entropy += delta_entanglement_entropy
            cumulative_bit_residue += delta_bit_residue
            entropy_steps.append(
                EntropyResidueStep(
                    iteration_index=int(delta_update.frame_index),
                    time_counter=int(delta_update.frame_index),
                    coordinate=delta_update.coordinate,
                    delta_runtime_overhead_ticks=int(delta_update.delta_runtime_overhead_ticks),
                    delta_entanglement_entropy=delta_entanglement_entropy,
                    delta_bit_residue=delta_bit_residue,
                    cumulative_entanglement_entropy=cumulative_entanglement_entropy,
                    cumulative_bit_residue=cumulative_bit_residue,
                    pending_coordinate_count=int(delta_update.pending_coordinate_count),
                    equilibrium_preserved=bool(delta_update.maintains_bulk_boundary_equilibrium),
                )
            )
        return tuple(entropy_steps)

    def __call__(self, audit: CausalResolutionAudit) -> tuple[EntropyResidueStep, ...]:
        return self.accumulate(audit)


@dataclass(frozen=True)
class SolverAudit:
    branch: Branch
    boundary_dimension: int
    bulk_dimension: int
    bit_budget: Decimal
    precision: int
    causal_resolution: CausalResolutionAudit
    entropy_residues: tuple[EntropyResidueStep, ...]
    frames: tuple[SolverIterationFrame, ...]

    @property
    def current_state(self) -> SolverIterationFrame | None:
        return None if not self.frames else self.frames[-1]

    @property
    def current_time_counter(self) -> int:
        return 0 if self.current_state is None else int(self.current_state.time_counter)

    @property
    def total_iterations(self) -> int:
        return len(self.frames)

    @property
    def total_bit_residue(self) -> Decimal:
        return Decimal("0") if not self.entropy_residues else self.entropy_residues[-1].cumulative_bit_residue

    @property
    def total_entropy(self) -> Decimal:
        return Decimal("0") if not self.entropy_residues else self.entropy_residues[-1].cumulative_entanglement_entropy

    @property
    def arrow_of_time_is_iteration_counter(self) -> bool:
        return bool(
            self.frames
            and [frame.time_counter for frame in self.frames] == list(range(1, len(self.frames) + 1))
            and [frame.time_counter for frame in self.frames] == [frame.iteration_index for frame in self.frames]
        )

    @property
    def time_is_computational_overhead(self) -> bool:
        return bool(
            self.frames
            and all(frame.runtime_overhead_ticks == frame.time_counter for frame in self.frames)
            and all(step.delta_runtime_overhead_ticks == 1 for step in self.entropy_residues)
        )

    @property
    def entropy_accumulates_monotonically(self) -> bool:
        return bool(
            self.entropy_residues
            and all(
                self.entropy_residues[index].cumulative_bit_residue >= self.entropy_residues[index - 1].cumulative_bit_residue
                and self.entropy_residues[index].cumulative_entanglement_entropy >= self.entropy_residues[index - 1].cumulative_entanglement_entropy
                for index in range(1, len(self.entropy_residues))
            )
            and all(step.delta_bit_residue >= 0 for step in self.entropy_residues)
        )

    @property
    def maintains_bulk_boundary_equilibrium(self) -> bool:
        return bool(self.frames and all(frame.maintains_bulk_boundary_equilibrium for frame in self.frames))

    @property
    def big_bang_statement(self) -> str:
        return self.causal_resolution.big_bang_statement

    @property
    def time_definition(self) -> str:
        return "Time is the iteration counter of the holographic transport solver."

    @property
    def entropy_definition(self) -> str:
        return "Each iteration generates a bit residue that records the solver overhead of the 26D -> 4D projection."

    @property
    def computational_overhead_statement(self) -> str:
        return "Time is the sequence of computations required to reconcile the boundary with the bulk."

    @property
    def statement(self) -> str:
        return "The arrow of time is the ordered computation count of the holographic transport solver."


@dataclass(frozen=True)
class HolographicTransportSolver(TemporalStateResolver):
    """Resolve SHBT time as solver-iteration overhead plus entropy residues."""

    boundary_dimension: int = LEPTON_LEVEL
    bulk_dimension: int = BULK_SPACETIME_DIMENSION

    def resolve(
        self,
        *,
        observable_targets: Sequence[ObservableTarget] = (),
        transport_checker: TransportChecker | None = None,
        max_solver_iterations: int | None = None,
    ) -> SolverAudit:
        causal_audit = super().resolve(
            observable_targets=observable_targets,
            transport_checker=transport_checker,
            max_solver_iterations=max_solver_iterations,
        )
        accumulator = EntropyAccumulator(bit_budget=self.bit_budget, precision=self.precision)
        entropy_residues = accumulator.accumulate(causal_audit)
        frames = tuple(
            SolverIterationFrame(
                iteration_index=int(recursive_frame.frame_index),
                time_counter=int(recursive_frame.frame_index),
                branch=causal_audit.branch,
                boundary_dimension=int(self.boundary_dimension),
                bulk_dimension=int(self.bulk_dimension),
                recursive_depth=int(recursive_frame.recursive_depth),
                recursive_frame=recursive_frame,
                entropy_residue=entropy_residue,
            )
            for recursive_frame, entropy_residue in zip(causal_audit.recursive_frames, entropy_residues)
        )
        return SolverAudit(
            branch=causal_audit.branch,
            boundary_dimension=int(self.boundary_dimension),
            bulk_dimension=int(self.bulk_dimension),
            bit_budget=_decimal(self.bit_budget),
            precision=max(int(self.precision), DEFAULT_PRECISION),
            causal_resolution=causal_audit,
            entropy_residues=frames and tuple(frame.entropy_residue for frame in frames) or (),
            frames=frames,
        )

    def resolve_frames(
        self,
        *,
        observable_targets: Sequence[ObservableTarget] = (),
        transport_checker: TransportChecker | None = None,
        max_solver_iterations: int | None = None,
    ) -> tuple[SolverIterationFrame, ...]:
        return self.resolve(
            observable_targets=observable_targets,
            transport_checker=transport_checker,
            max_solver_iterations=max_solver_iterations,
        ).frames

    def resolve_current_state(
        self,
        *,
        observable_targets: Sequence[ObservableTarget] = (),
        transport_checker: TransportChecker | None = None,
        max_solver_iterations: int | None = None,
    ) -> SolverIterationFrame | None:
        return self.resolve(
            observable_targets=observable_targets,
            transport_checker=transport_checker,
            max_solver_iterations=max_solver_iterations,
        ).current_state


def build_solver_audit(
    *,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    parent_level: int = PARENT_LEVEL,
    bit_budget: Decimal | Fraction | float | int | str = HOLOGRAPHIC_BITS,
    precision: int = DEFAULT_PRECISION,
    observable_targets: Sequence[ObservableTarget] = (),
    transport_checker: TransportChecker | None = None,
    max_solver_iterations: int | None = None,
) -> SolverAudit:
    solver = HolographicTransportSolver(
        lepton_level=lepton_level,
        quark_level=quark_level,
        parent_level=parent_level,
        bit_budget=bit_budget,
        precision=precision,
    )
    return solver.resolve(
        observable_targets=observable_targets,
        transport_checker=transport_checker,
        max_solver_iterations=max_solver_iterations,
    )


def build_solver_report(
    *,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    parent_level: int = PARENT_LEVEL,
    bit_budget: Decimal | Fraction | float | int | str = HOLOGRAPHIC_BITS,
    precision: int = DEFAULT_PRECISION,
    observable_targets: Sequence[ObservableTarget] = (),
    transport_checker: TransportChecker | None = None,
    max_solver_iterations: int | None = None,
) -> str:
    audit = build_solver_audit(
        lepton_level=lepton_level,
        quark_level=quark_level,
        parent_level=parent_level,
        bit_budget=bit_budget,
        precision=precision,
        observable_targets=observable_targets,
        transport_checker=transport_checker,
        max_solver_iterations=max_solver_iterations,
    )
    lines = [
        "Holographic Solver Audit",
        "========================",
        f"Branch                       : {audit.branch}",
        f"Boundary dimension           : {audit.boundary_dimension}",
        f"Bulk dimension               : {audit.bulk_dimension}",
        f"Total iterations             : {audit.total_iterations}",
        f"Current time counter         : {audit.current_time_counter}",
        f"Arrow of time established    : {audit.arrow_of_time_is_iteration_counter}",
        f"Time is computational work   : {audit.time_is_computational_overhead}",
        f"Entropy accumulates          : {audit.entropy_accumulates_monotonically}",
        f"Bulk-boundary equilibrium    : {audit.maintains_bulk_boundary_equilibrium}",
        f"Total bit residue            : {audit.total_bit_residue}",
        audit.big_bang_statement,
        audit.time_definition,
        audit.entropy_definition,
        audit.computational_overhead_statement,
        audit.statement,
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Treat SHBT time as solver-iteration overhead.")
    parser.add_argument(
        "--max-solver-iterations",
        type=int,
        default=None,
        help="Optional cap on the number of solver iterations to execute.",
    )
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    print(build_solver_report(max_solver_iterations=args.max_solver_iterations))
    return 0


__all__ = [
    "EntropyAccumulator",
    "EntropyResidueStep",
    "HolographicTransportSolver",
    "SolverAudit",
    "SolverIterationFrame",
    "build_solver_audit",
    "build_solver_report",
    "main",
]
