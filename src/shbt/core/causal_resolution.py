from __future__ import annotations

"""Sequential causal-resolution layer for the SHBT transport solver.

The bulk remains branch-fixed, but what is operationally perceived as time is
reframed here as the ordered sequence of refinement steps performed by the
holographic transport solver. The Big Bang is represented as the initial
``main()`` call that instantiates the resolution loop, and the present state of
the universe is the current iteration index reached by that loop.
"""

import argparse
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Final, Mapping, Sequence

if __package__ in (None, ""):
    import sys

    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.causal_execution import (
    BENCHMARK_BRANCH,
    CausalExecutionAudit,
    GetManifestationEvent,
    ObservableTarget,
    OrderOfExecutionEngine,
    TransportChecker,
)
from shbt.core.temporal_emergence_kernel import DEFAULT_PRECISION, ManifoldSliceLoadingMap


Coordinate = tuple[int, int]
Branch = tuple[int, int, int]


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
class CausalResolutionAudit:
    branch: Branch
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
    def current_state(self) -> ResolutionStep | None:
        return None if not self.resolution_steps else self.resolution_steps[-1]

    @property
    def big_bang_statement(self) -> str:
        return "The Big Bang is the initial main() function call."

    @property
    def current_state_of_universe(self) -> str:
        return (
            f"The current universe is iteration {self.current_iteration_index} "
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
        return _build_resolution_audit_from_execution(execution_audit)


def _build_resolution_audit_from_execution(execution_audit: CausalExecutionAudit) -> CausalResolutionAudit:
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
        },
    )
    return CausalResolutionAudit(
        branch=execution_audit.branch,
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

    loop = CausalExecutionLoop(
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
        audit.big_bang_statement,
        audit.current_state_of_universe,
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
    "ResolutionStep",
    "build_causal_resolution_audit",
    "build_causal_resolution_report",
    "main",
]


assert BENCHMARK_BRANCH == (LEPTON_LEVEL, QUARK_LEVEL, PARENT_LEVEL)


if __name__ == "__main__":
    raise SystemExit(main())
