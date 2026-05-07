from __future__ import annotations

"""Sequential causal-resolution layer for the SHBT transport solver.

The bulk remains branch-fixed, but what is operationally perceived as time is
reframed here as the ordered sequence of delta updates performed by the
holographic transport solver to preserve bulk-boundary equilibrium. The Big
Bang is represented as the initial ``main()`` call that instantiates the
resolution loop, and each present state of the universe is treated as a
recursive frame in that transport-resolution stack.
"""

import argparse
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Final, Mapping, Sequence

if __package__ in (None, ""):
    import sys

    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.constants import HOLOGRAPHIC_BITS, LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.causal_execution import (
    BENCHMARK_BRANCH,
    CausalExecutionAudit,
    GetManifestationEvent,
    ObservableTarget,
    OrderOfExecutionEngine,
    TransportChecker,
)
from shbt.core.temporal_emergence_kernel import (
    DEFAULT_PRECISION,
    ManifoldSliceLoadingMap,
    derive_temporal_increment,
)


Coordinate = tuple[int, int]
Branch = tuple[int, int, int]
_FRAME_EQUILIBRIUM_TOLERANCE: Final[Decimal] = Decimal("1e-15")


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(str(value))


@dataclass(frozen=True)
class BigBangCall:
    branch: Branch
    iteration_index: int = 0
    entrypoint: str = "main"
    event_type: str = "INITIAL_MAIN_CALL"
    payload: Mapping[str, object] = field(default_factory=dict)

    @property
    def is_initial_main_call(self) -> bool:
        return True


@dataclass(frozen=True)
class ResolutionStep:
    iteration_index: int
    causal_tick: int
    runtime_overhead_ticks: int
    coordinate: Coordinate
    branch: Branch
    sequence_rank: int
    transport_attempt: int
    transport_check_completed: bool
    transport_check_passed: bool
    loading_weight: Decimal
    entanglement_weight: Decimal
    resolved_coordinate_count: int
    manifested_event_count: int
    state_label: str = "boundary_transport_resolution"


@dataclass(frozen=True)
class EquilibriumDeltaUpdate:
    frame_index: int
    causal_tick: int
    coordinate: Coordinate
    delta_runtime_overhead_ticks: int
    delta_resolved_coordinates: int
    delta_manifestation_events: int
    delta_loading_weight: Decimal
    delta_entanglement_weight: Decimal
    delta_temporal_update: Decimal
    cumulative_loading_weight: Decimal
    cumulative_entanglement_weight: Decimal
    cumulative_temporal_update: Decimal
    equilibrium_temporal_target: Decimal
    equilibrium_residual: Decimal
    pending_coordinate_count: int
    equilibrium_preserved: bool

    @property
    def statement(self) -> str:
        return "A temporal tick is a delta update required to preserve bulk-boundary equilibrium."

    @property
    def advances_equilibrium(self) -> bool:
        return self.delta_resolved_coordinates > 0

    @property
    def maintains_bulk_boundary_equilibrium(self) -> bool:
        return bool(
            self.equilibrium_preserved
            and abs(self.equilibrium_residual) <= _FRAME_EQUILIBRIUM_TOLERANCE
        )


@dataclass(frozen=True)
class RecursiveTemporalFrame:
    frame_index: int
    recursive_depth: int
    branch: Branch
    frame_chain: tuple[int, ...]
    previous_frame_index: int | None
    previous_coordinate: Coordinate | None
    current_state: ResolutionStep
    delta_update: EquilibriumDeltaUpdate
    cumulative_manifested_event_count: int
    pending_coordinate_count: int
    state_label: str = "recursive_transport_frame"

    @property
    def is_present_state(self) -> bool:
        return self.frame_index == self.recursive_depth

    @property
    def iteration_index(self) -> int:
        return int(self.current_state.iteration_index)

    @property
    def causal_tick(self) -> int:
        return int(self.current_state.causal_tick)

    @property
    def runtime_overhead_ticks(self) -> int:
        return int(self.current_state.runtime_overhead_ticks)

    @property
    def coordinate(self) -> Coordinate:
        return self.current_state.coordinate

    @property
    def sequence_rank(self) -> int:
        return int(self.current_state.sequence_rank)

    @property
    def transport_attempt(self) -> int:
        return int(self.current_state.transport_attempt)

    @property
    def transport_check_completed(self) -> bool:
        return bool(self.current_state.transport_check_completed)

    @property
    def transport_check_passed(self) -> bool:
        return bool(self.current_state.transport_check_passed)

    @property
    def loading_weight(self) -> Decimal:
        return self.current_state.loading_weight

    @property
    def entanglement_weight(self) -> Decimal:
        return self.current_state.entanglement_weight

    @property
    def resolved_coordinate_count(self) -> int:
        return int(self.current_state.resolved_coordinate_count)

    @property
    def manifested_event_count(self) -> int:
        return int(self.current_state.manifested_event_count)

    @property
    def maintains_bulk_boundary_equilibrium(self) -> bool:
        return self.delta_update.maintains_bulk_boundary_equilibrium

    @property
    def statement(self) -> str:
        return "Dynamic reality is the current recursive frame produced by the transport solver."


@dataclass(frozen=True)
class CausalResolutionAudit:
    branch: Branch
    bit_budget: Decimal
    precision: int
    big_bang: BigBangCall
    manifold_slice: ManifoldSliceLoadingMap
    loading_sequence: tuple[Coordinate, ...]
    resolution_steps: tuple[ResolutionStep, ...]
    manifestation_events: tuple[GetManifestationEvent, ...]

    @property
    def total_resolution_iterations(self) -> int:
        return len(self.resolution_steps)

    @property
    def current_iteration_index(self) -> int:
        return 0 if not self.resolution_steps else int(self.resolution_steps[-1].iteration_index)

    @property
    def current_state(self) -> RecursiveTemporalFrame | None:
        return self.current_recursive_frame

    @property
    def current_resolution_step(self) -> ResolutionStep | None:
        return None if not self.resolution_steps else self.resolution_steps[-1]

    @property
    def delta_updates(self) -> tuple[EquilibriumDeltaUpdate, ...]:
        delta_updates: list[EquilibriumDeltaUpdate] = []
        prior_runtime_overhead_ticks = 0
        prior_resolved_coordinate_count = 0

        with localcontext() as context:
            context.prec = max(int(self.precision), DEFAULT_PRECISION) + 16
            cumulative_loading_weight = Decimal("0")
            cumulative_entanglement_weight = Decimal("0")
            cumulative_temporal_update = Decimal("0")

            for frame_index, step in enumerate(self.resolution_steps, start=1):
                delta_loading_weight = step.loading_weight if step.transport_check_completed else Decimal("0")
                delta_entanglement_weight = step.entanglement_weight if step.transport_check_completed else Decimal("0")
                delta_temporal_update = derive_temporal_increment(
                    delta_entanglement_weight,
                    self.bit_budget,
                    precision=self.precision,
                )
                cumulative_loading_weight += delta_loading_weight
                cumulative_entanglement_weight += delta_entanglement_weight
                cumulative_temporal_update += delta_temporal_update
                equilibrium_temporal_target = derive_temporal_increment(
                    cumulative_entanglement_weight,
                    self.bit_budget,
                    precision=self.precision,
                )
                equilibrium_residual = cumulative_temporal_update - equilibrium_temporal_target
                equilibrium_preserved = bool(
                    step.branch == self.branch
                    and (not step.transport_check_completed or step.transport_check_passed)
                    and abs(equilibrium_residual) <= _FRAME_EQUILIBRIUM_TOLERANCE
                )
                delta_updates.append(
                    EquilibriumDeltaUpdate(
                        frame_index=frame_index,
                        causal_tick=int(step.causal_tick),
                        coordinate=step.coordinate,
                        delta_runtime_overhead_ticks=int(step.runtime_overhead_ticks - prior_runtime_overhead_ticks),
                        delta_resolved_coordinates=int(step.resolved_coordinate_count - prior_resolved_coordinate_count),
                        delta_manifestation_events=int(step.manifested_event_count),
                        delta_loading_weight=delta_loading_weight,
                        delta_entanglement_weight=delta_entanglement_weight,
                        delta_temporal_update=delta_temporal_update,
                        cumulative_loading_weight=cumulative_loading_weight,
                        cumulative_entanglement_weight=cumulative_entanglement_weight,
                        cumulative_temporal_update=cumulative_temporal_update,
                        equilibrium_temporal_target=equilibrium_temporal_target,
                        equilibrium_residual=equilibrium_residual,
                        pending_coordinate_count=max(len(self.loading_sequence) - int(step.resolved_coordinate_count), 0),
                        equilibrium_preserved=equilibrium_preserved,
                    )
                )
                prior_runtime_overhead_ticks = int(step.runtime_overhead_ticks)
                prior_resolved_coordinate_count = int(step.resolved_coordinate_count)

        return tuple(delta_updates)

    @property
    def delta_update_count(self) -> int:
        return len(self.delta_updates)

    @property
    def completed_delta_updates(self) -> int:
        return 0 if not self.delta_updates else sum(int(update.advances_equilibrium) for update in self.delta_updates)

    @property
    def recursive_frames(self) -> tuple[RecursiveTemporalFrame, ...]:
        frames: list[RecursiveTemporalFrame] = []
        cumulative_manifested_event_count = 0
        for frame_index, (step, delta_update) in enumerate(zip(self.resolution_steps, self.delta_updates), start=1):
            cumulative_manifested_event_count += int(delta_update.delta_manifestation_events)
            previous_step = None if frame_index == 1 else self.resolution_steps[frame_index - 2]
            frames.append(
                RecursiveTemporalFrame(
                    frame_index=frame_index,
                    recursive_depth=frame_index,
                    branch=self.branch,
                    frame_chain=tuple(range(1, frame_index + 1)),
                    previous_frame_index=None if frame_index == 1 else frame_index - 1,
                    previous_coordinate=None if previous_step is None else previous_step.coordinate,
                    current_state=step,
                    delta_update=delta_update,
                    cumulative_manifested_event_count=cumulative_manifested_event_count,
                    pending_coordinate_count=delta_update.pending_coordinate_count,
                )
            )
        return tuple(frames)

    @property
    def current_recursive_frame(self) -> RecursiveTemporalFrame | None:
        return None if not self.recursive_frames else self.recursive_frames[-1]

    @property
    def maintains_bulk_boundary_equilibrium(self) -> bool:
        return bool(
            self.delta_updates
            and all(delta_update.maintains_bulk_boundary_equilibrium for delta_update in self.delta_updates)
        )

    @property
    def time_is_solver_overhead(self) -> bool:
        return bool(
            self.resolution_steps
            and all(step.runtime_overhead_ticks == step.iteration_index for step in self.resolution_steps)
        )

    @property
    def big_bang_statement(self) -> str:
        return "The Big Bang is the initial main() function call."

    @property
    def time_definition(self) -> str:
        return "Time is the sequence of delta updates required to maintain bulk-boundary equilibrium."

    @property
    def dynamic_reality_bridge(self) -> str:
        return "Dynamic reality is the recursive stack of present-state frames produced by the transport solver."

    @property
    def computational_overhead_statement(self) -> str:
        return "Time becomes the computational overhead of the solver."

    @property
    def current_state_of_universe(self) -> str:
        return (
            f"The current universe is iteration {self.current_iteration_index} "
            "of the boundary-transport solver."
        )

    @property
    def current_recursive_state_of_universe(self) -> str:
        current_frame = self.current_recursive_frame
        if current_frame is None:
            return "The current universe has not entered the recursive transport stack."
        return (
            f"The current universe is recursive frame {current_frame.frame_index} "
            "of the boundary-transport solver."
        )

    @property
    def time_is_sequential_resolution(self) -> bool:
        return bool(
            self.big_bang.iteration_index == 0
            and self.big_bang.is_initial_main_call
            and [step.iteration_index for step in self.resolution_steps]
            == list(range(1, len(self.resolution_steps) + 1))
            and [step.causal_tick for step in self.resolution_steps]
            == [step.iteration_index for step in self.resolution_steps]
            and [step.runtime_overhead_ticks for step in self.resolution_steps]
            == [step.iteration_index for step in self.resolution_steps]
        )

    @property
    def time_is_delta_update_sequence(self) -> bool:
        return bool(
            self.delta_updates
            and all(delta_update.delta_runtime_overhead_ticks == 1 for delta_update in self.delta_updates)
            and all(delta_update.maintains_bulk_boundary_equilibrium for delta_update in self.delta_updates)
            and self.completed_delta_updates == len(self.loading_sequence)
        )

    @property
    def statement(self) -> str:
        return "Time is the sequential resolution order of the holographic transport solver."


@dataclass(frozen=True)
class CausalExecutionLoop:
    lepton_level: int = LEPTON_LEVEL
    quark_level: int = QUARK_LEVEL
    parent_level: int = PARENT_LEVEL
    precision: int = DEFAULT_PRECISION
    branch: Branch = field(init=False)
    manifold_slice: ManifoldSliceLoadingMap = field(init=False)
    loading_sequence: tuple[Coordinate, ...] = field(init=False)
    _execution_engine: OrderOfExecutionEngine = field(init=False, repr=False)

    def __post_init__(self) -> None:
        engine = OrderOfExecutionEngine(
            lepton_level=int(self.lepton_level),
            quark_level=int(self.quark_level),
            parent_level=int(self.parent_level),
            precision=max(int(self.precision), DEFAULT_PRECISION),
        )
        object.__setattr__(self, "branch", engine.branch)
        object.__setattr__(self, "manifold_slice", engine.manifold_slice)
        object.__setattr__(self, "loading_sequence", engine.loading_sequence)
        object.__setattr__(self, "_execution_engine", engine)

    def resolve(
        self,
        *,
        observable_targets: Sequence[ObservableTarget] = (),
        transport_checker: TransportChecker | None = None,
        max_solver_iterations: int | None = None,
    ) -> CausalResolutionAudit:
        execution_audit = self._execution_engine.execute(
            observable_targets=observable_targets,
            transport_checker=transport_checker,
            max_runtime_overhead_ticks=max_solver_iterations,
        )
        return _build_resolution_audit_from_execution(
            execution_audit,
            bit_budget=HOLOGRAPHIC_BITS,
            precision=self.precision,
        )

    def resolve_recursive_frames(
        self,
        *,
        observable_targets: Sequence[ObservableTarget] = (),
        transport_checker: TransportChecker | None = None,
        max_solver_iterations: int | None = None,
    ) -> tuple[RecursiveTemporalFrame, ...]:
        return self.resolve(
            observable_targets=observable_targets,
            transport_checker=transport_checker,
            max_solver_iterations=max_solver_iterations,
        ).recursive_frames

    def resolve_current_state(
        self,
        *,
        observable_targets: Sequence[ObservableTarget] = (),
        transport_checker: TransportChecker | None = None,
        max_solver_iterations: int | None = None,
    ) -> RecursiveTemporalFrame | None:
        return self.resolve(
            observable_targets=observable_targets,
            transport_checker=transport_checker,
            max_solver_iterations=max_solver_iterations,
        ).current_recursive_frame


@dataclass(frozen=True)
class TemporalStateResolver(CausalExecutionLoop):
    """Resolve dynamic SHBT present-states as recursive transport frames."""

    bit_budget: Decimal | Fraction | float | int | str = HOLOGRAPHIC_BITS

    def __post_init__(self) -> None:
        super().__post_init__()
        resolved_bit_budget = _decimal(self.bit_budget)
        if resolved_bit_budget <= 0:
            raise ValueError("bit_budget must be positive.")
        object.__setattr__(self, "bit_budget", resolved_bit_budget)

    def resolve(
        self,
        *,
        observable_targets: Sequence[ObservableTarget] = (),
        transport_checker: TransportChecker | None = None,
        max_solver_iterations: int | None = None,
    ) -> CausalResolutionAudit:
        execution_audit = self._execution_engine.execute(
            observable_targets=observable_targets,
            transport_checker=transport_checker,
            max_runtime_overhead_ticks=max_solver_iterations,
        )
        return _build_resolution_audit_from_execution(
            execution_audit,
            bit_budget=self.bit_budget,
            precision=self.precision,
        )


def _build_resolution_audit_from_execution(
    execution_audit: CausalExecutionAudit,
    *,
    bit_budget: Decimal | Fraction | float | int | str = HOLOGRAPHIC_BITS,
    precision: int = DEFAULT_PRECISION,
) -> CausalResolutionAudit:
    resolved_bit_budget = _decimal(bit_budget)
    resolved_precision = max(int(precision), DEFAULT_PRECISION)
    resolved_coordinate_count = 0
    resolution_steps: list[ResolutionStep] = []
    for step in execution_audit.steps:
        if step.transport_check_completed:
            resolved_coordinate_count += 1
        resolution_steps.append(
            ResolutionStep(
                iteration_index=int(step.causal_tick),
                causal_tick=int(step.causal_tick),
                runtime_overhead_ticks=int(step.runtime_overhead_ticks),
                coordinate=step.coordinate,
                branch=step.branch,
                sequence_rank=int(step.sequence_rank),
                transport_attempt=int(step.transport_attempt),
                transport_check_completed=bool(step.transport_check_completed),
                transport_check_passed=bool(step.transport_check_passed),
                loading_weight=step.loading_weight,
                entanglement_weight=step.entanglement_weight,
                resolved_coordinate_count=int(resolved_coordinate_count),
                manifested_event_count=len(step.manifested_events),
            )
        )

    big_bang = BigBangCall(
        branch=execution_audit.branch,
        payload={
            "loading_sequence_length": len(execution_audit.loading_sequence),
            "entrypoint": "main",
            "bit_budget": resolved_bit_budget,
        },
    )
    return CausalResolutionAudit(
        branch=execution_audit.branch,
        bit_budget=resolved_bit_budget,
        precision=resolved_precision,
        big_bang=big_bang,
        manifold_slice=execution_audit.manifold_slice,
        loading_sequence=execution_audit.loading_sequence,
        resolution_steps=tuple(resolution_steps),
        manifestation_events=execution_audit.manifestation_events,
    )


def build_causal_resolution_audit(
    *,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    parent_level: int = PARENT_LEVEL,
    precision: int = DEFAULT_PRECISION,
    observable_targets: Sequence[ObservableTarget] = (),
    transport_checker: TransportChecker | None = None,
    max_solver_iterations: int | None = None,
) -> CausalResolutionAudit:
    """Build the sequential causal-resolution audit over the holographic solver."""

    loop = TemporalStateResolver(
        lepton_level=lepton_level,
        quark_level=quark_level,
        parent_level=parent_level,
        precision=precision,
    )
    return loop.resolve(
        observable_targets=observable_targets,
        transport_checker=transport_checker,
        max_solver_iterations=max_solver_iterations,
    )


def build_causal_resolution_report(
    *,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    parent_level: int = PARENT_LEVEL,
    precision: int = DEFAULT_PRECISION,
    observable_targets: Sequence[ObservableTarget] = (),
    transport_checker: TransportChecker | None = None,
    max_solver_iterations: int | None = None,
) -> str:
    audit = build_causal_resolution_audit(
        lepton_level=lepton_level,
        quark_level=quark_level,
        parent_level=parent_level,
        precision=precision,
        observable_targets=observable_targets,
        transport_checker=transport_checker,
        max_solver_iterations=max_solver_iterations,
    )
    lines = [
        "Causal Resolution Audit",
        "=======================",
        f"Branch                       : {audit.branch}",
        f"Big Bang event               : {audit.big_bang.event_type}",
        f"Big Bang entrypoint          : {audit.big_bang.entrypoint}()",
        f"Big Bang iteration           : {audit.big_bang.iteration_index}",
        f"Total resolution iterations  : {audit.total_resolution_iterations}",
        f"Current iteration index      : {audit.current_iteration_index}",
        f"Manifestation events         : {len(audit.manifestation_events)}",
        f"Arrow of time established    : {audit.time_is_sequential_resolution}",
        f"Delta-update time law        : {audit.time_is_delta_update_sequence}",
        f"Time is solver overhead      : {audit.time_is_solver_overhead}",
        f"Bulk-boundary equilibrium    : {audit.maintains_bulk_boundary_equilibrium}",
        f"Completed delta updates      : {audit.completed_delta_updates}",
        f"Recursive temporal frames    : {len(audit.recursive_frames)}",
        f"Current recursive depth      : {0 if audit.current_recursive_frame is None else audit.current_recursive_frame.recursive_depth}",
        audit.big_bang_statement,
        audit.current_state_of_universe,
        audit.current_recursive_state_of_universe,
        audit.time_definition,
        audit.dynamic_reality_bridge,
        audit.computational_overhead_statement,
        audit.statement,
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve SHBT time as sequential causal transport.")
    parser.add_argument(
        "--max-solver-iterations",
        type=int,
        default=None,
        help="Optional ceiling on solver iterations before resolution halts.",
    )
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    print(
        build_causal_resolution_report(
            max_solver_iterations=args.max_solver_iterations,
        )
    )
    return 0


__all__ = [
    "BENCHMARK_BRANCH",
    "BigBangCall",
    "CausalExecutionLoop",
    "CausalResolutionAudit",
    "EquilibriumDeltaUpdate",
    "ResolutionStep",
    "RecursiveTemporalFrame",
    "TemporalStateResolver",
    "build_causal_resolution_audit",
    "build_causal_resolution_report",
    "main",
]


assert BENCHMARK_BRANCH == (LEPTON_LEVEL, QUARK_LEVEL, PARENT_LEVEL)


if __name__ == "__main__":
    raise SystemExit(main())
