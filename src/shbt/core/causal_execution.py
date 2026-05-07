from __future__ import annotations

"""Causal execution layer for the static SHBT block.

The benchmark bulk is static. Operational time is represented here as the
runtime overhead accumulated while the holographic bit-loading sequence is
executed in order. A physical observable acquires a ``GET`` manifestation event
only when the transport logic attached to its coordinate completes its check.
"""

from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
import inspect
from typing import Any, Callable, Final, Mapping, Sequence

from shbt.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.temporal_emergence_kernel import (
    DEFAULT_PRECISION,
    ManifoldSliceLoadingMap,
    map_manifold_slice_bit_loading_density,
)
from shbt.core.topological_kernel import sequence_bit_loading


Coordinate = tuple[int, int]
Branch = tuple[int, int, int]
TransportChecker = Callable[..., "TransportCheck"]

BENCHMARK_BRANCH: Final[Branch] = (
    int(LEPTON_LEVEL),
    int(QUARK_LEVEL),
    int(PARENT_LEVEL),
)
DEFAULT_MAX_RUNTIME_OVERHEAD_FACTOR: Final[int] = 8


@dataclass(frozen=True)
class ObservableTarget:
    observable_name: str
    coordinate: Coordinate
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TransportCheck:
    coordinate: Coordinate
    completed: bool
    passed: bool
    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GetManifestationEvent:
    observable_name: str
    coordinate: Coordinate
    branch: Branch
    causal_tick: int
    runtime_overhead_ticks: int
    transport_attempt: int
    transport_check_passed: bool
    payload: Mapping[str, object] = field(default_factory=dict)
    event_type: str = "GET"

    @property
    def observable_triggered(self) -> bool:
        return True


@dataclass(frozen=True)
class CausalExecutionStep:
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
    manifested_events: tuple[GetManifestationEvent, ...] = ()


@dataclass(frozen=True)
class CausalExecutionAudit:
    branch: Branch
    manifold_slice: ManifoldSliceLoadingMap
    loading_sequence: tuple[Coordinate, ...]
    steps: tuple[CausalExecutionStep, ...]
    manifestation_events: tuple[GetManifestationEvent, ...]

    @property
    def total_runtime_overhead_ticks(self) -> int:
        return len(self.steps)

    @property
    def completed_transport_checks(self) -> int:
        return sum(1 for step in self.steps if step.transport_check_completed)

    @property
    def time_is_runtime_overhead(self) -> bool:
        return bool(
            self.steps
            and self.steps[-1].causal_tick == self.total_runtime_overhead_ticks
            and all(step.causal_tick == step.runtime_overhead_ticks for step in self.steps)
        )

    @property
    def statement(self) -> str:
        return "Time is the runtime overhead of the Universal Code."


def _call_transport_checker(transport_checker: TransportChecker, **candidate_arguments: object) -> TransportCheck:
    try:
        parameters = inspect.signature(transport_checker).parameters
    except (TypeError, ValueError):
        return transport_checker(**candidate_arguments)

    supports_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    compatible_arguments = (
        candidate_arguments
        if supports_kwargs
        else {name: value for name, value in candidate_arguments.items() if name in parameters}
    )
    return transport_checker(**compatible_arguments)


def default_transport_check(
    *,
    coordinate: Coordinate,
    branch: Branch,
    transport_attempt: int,
    sequence_rank: int,
    loading_weight: Decimal,
    entanglement_weight: Decimal,
) -> TransportCheck:
    """Return the default completed transport check for a loading coordinate."""

    return TransportCheck(
        coordinate=coordinate,
        completed=True,
        passed=True,
        payload={
            "branch": branch,
            "sequence_rank": int(sequence_rank),
            "transport_attempt": int(transport_attempt),
            "loading_weight": loading_weight,
            "entanglement_weight": entanglement_weight,
        },
    )


def define_get_event(
    target: ObservableTarget,
    *,
    branch: Branch,
    causal_tick: int,
    runtime_overhead_ticks: int,
    transport_attempt: int,
    transport_check: TransportCheck,
) -> GetManifestationEvent:
    """Define the manifestation event emitted once a coordinate finishes checking."""

    if not transport_check.completed:
        raise ValueError("GET manifestation requires a completed transport check.")
    payload = dict(target.metadata)
    payload.update(dict(transport_check.payload))
    return GetManifestationEvent(
        observable_name=target.observable_name,
        coordinate=target.coordinate,
        branch=branch,
        causal_tick=int(causal_tick),
        runtime_overhead_ticks=int(runtime_overhead_ticks),
        transport_attempt=int(transport_attempt),
        transport_check_passed=bool(transport_check.passed),
        payload=payload,
    )


@dataclass(frozen=True)
class OrderOfExecutionEngine:
    lepton_level: int = LEPTON_LEVEL
    quark_level: int = QUARK_LEVEL
    parent_level: int = PARENT_LEVEL
    precision: int = DEFAULT_PRECISION
    branch: Branch = field(init=False)
    manifold_slice: ManifoldSliceLoadingMap = field(init=False)
    loading_sequence: tuple[Coordinate, ...] = field(init=False)
    _sequence_rank_map: Mapping[Coordinate, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        resolved_branch = (
            int(self.lepton_level),
            int(self.quark_level),
            int(self.parent_level),
        )
        if min(resolved_branch) <= 0:
            raise ValueError("Causal execution requires positive integer branch coordinates.")
        resolved_precision = max(int(self.precision), DEFAULT_PRECISION)
        manifold_slice = map_manifold_slice_bit_loading_density(
            lepton_level=resolved_branch[0],
            quark_level=resolved_branch[1],
            precision=resolved_precision,
        )
        loading_sequence = tuple(
            sequence_bit_loading(
                lepton_level=resolved_branch[0],
                quark_level=resolved_branch[1],
            )
        )
        assert loading_sequence == manifold_slice.dominant_loading_sequence
        object.__setattr__(self, "branch", resolved_branch)
        object.__setattr__(self, "manifold_slice", manifold_slice)
        object.__setattr__(self, "loading_sequence", loading_sequence)
        object.__setattr__(
            self,
            "_sequence_rank_map",
            {coordinate: index + 1 for index, coordinate in enumerate(loading_sequence)},
        )

    def execute(
        self,
        *,
        observable_targets: Sequence[ObservableTarget] = (),
        transport_checker: TransportChecker | None = None,
        max_runtime_overhead_ticks: int | None = None,
    ) -> CausalExecutionAudit:
        resolved_targets = tuple(observable_targets)
        resolved_checker = default_transport_check if transport_checker is None else transport_checker
        runtime_limit = (
            len(self.loading_sequence) * DEFAULT_MAX_RUNTIME_OVERHEAD_FACTOR
            if max_runtime_overhead_ticks is None
            else int(max_runtime_overhead_ticks)
        )
        if runtime_limit < len(self.loading_sequence):
            raise ValueError("max_runtime_overhead_ticks must cover at least one pass over the loading sequence.")

        pending_coordinates = deque(self.loading_sequence)
        attempts = {coordinate: 0 for coordinate in self.loading_sequence}
        steps: list[CausalExecutionStep] = []
        events: list[GetManifestationEvent] = []
        causal_tick = 0

        while pending_coordinates:
            causal_tick += 1
            if causal_tick > runtime_limit:
                raise RuntimeError(
                    "Causal execution exceeded the allotted runtime overhead before all transport checks completed."
                )

            coordinate = pending_coordinates.popleft()
            attempts[coordinate] += 1
            sequence_rank = int(self._sequence_rank_map[coordinate])
            loading_weight = self.manifold_slice.loading_density[coordinate[0]][coordinate[1]]
            entanglement_weight = self.manifold_slice.entanglement_density[coordinate[0]][coordinate[1]]
            transport_check = _call_transport_checker(
                resolved_checker,
                coordinate=coordinate,
                branch=self.branch,
                transport_attempt=attempts[coordinate],
                sequence_rank=sequence_rank,
                loading_weight=loading_weight,
                entanglement_weight=entanglement_weight,
            )
            if transport_check.coordinate != coordinate:
                raise ValueError(
                    f"Transport check returned coordinate {transport_check.coordinate}, expected {coordinate}."
                )

            manifested_events: tuple[GetManifestationEvent, ...] = ()
            if transport_check.completed:
                manifested_events = tuple(
                    define_get_event(
                        target,
                        branch=self.branch,
                        causal_tick=causal_tick,
                        runtime_overhead_ticks=causal_tick,
                        transport_attempt=attempts[coordinate],
                        transport_check=transport_check,
                    )
                    for target in resolved_targets
                    if target.coordinate == coordinate
                )
                events.extend(manifested_events)
            else:
                pending_coordinates.append(coordinate)

            steps.append(
                CausalExecutionStep(
                    causal_tick=causal_tick,
                    runtime_overhead_ticks=causal_tick,
                    coordinate=coordinate,
                    branch=self.branch,
                    sequence_rank=sequence_rank,
                    transport_attempt=attempts[coordinate],
                    transport_check_completed=bool(transport_check.completed),
                    transport_check_passed=bool(transport_check.passed),
                    loading_weight=loading_weight,
                    entanglement_weight=entanglement_weight,
                    manifested_events=manifested_events,
                )
            )

        return CausalExecutionAudit(
            branch=self.branch,
            manifold_slice=self.manifold_slice,
            loading_sequence=self.loading_sequence,
            steps=tuple(steps),
            manifestation_events=tuple(events),
        )


def build_causal_execution_audit(
    *,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    parent_level: int = PARENT_LEVEL,
    precision: int = DEFAULT_PRECISION,
    observable_targets: Sequence[ObservableTarget] = (),
    transport_checker: TransportChecker | None = None,
    max_runtime_overhead_ticks: int | None = None,
) -> CausalExecutionAudit:
    """Execute the causal layer over the ordered holographic bit-loading sequence."""

    engine = OrderOfExecutionEngine(
        lepton_level=lepton_level,
        quark_level=quark_level,
        parent_level=parent_level,
        precision=precision,
    )
    return engine.execute(
        observable_targets=observable_targets,
        transport_checker=transport_checker,
        max_runtime_overhead_ticks=max_runtime_overhead_ticks,
    )


__all__ = [
    "BENCHMARK_BRANCH",
    "CausalExecutionAudit",
    "CausalExecutionStep",
    "DEFAULT_MAX_RUNTIME_OVERHEAD_FACTOR",
    "GetManifestationEvent",
    "ObservableTarget",
    "OrderOfExecutionEngine",
    "TransportCheck",
    "build_causal_execution_audit",
    "default_transport_check",
    "define_get_event",
]
