from __future__ import annotations

import pytest

from shbt.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.causal_execution import (
    ObservableTarget,
    OrderOfExecutionEngine,
    TransportCheck,
    build_causal_execution_audit,
    define_get_event,
)
from shbt.core.topological_kernel import sequence_bit_loading


def test_order_of_execution_engine_identifies_time_with_loading_sequence_iteration() -> None:
    audit = build_causal_execution_audit()

    assert audit.branch == (LEPTON_LEVEL, QUARK_LEVEL, PARENT_LEVEL)
    assert audit.loading_sequence == sequence_bit_loading(lepton_level=26, quark_level=8)
    assert [step.coordinate for step in audit.steps] == list(audit.loading_sequence)
    assert [step.causal_tick for step in audit.steps] == list(range(1, len(audit.loading_sequence) + 1))
    assert audit.total_runtime_overhead_ticks == len(audit.loading_sequence)
    assert audit.completed_transport_checks == len(audit.loading_sequence)
    assert audit.time_is_runtime_overhead
    assert audit.statement == "Time is the runtime overhead of the Universal Code."


def test_get_manifestation_event_waits_for_transport_completion() -> None:
    engine = OrderOfExecutionEngine()
    target_coordinate = engine.loading_sequence[0]
    target = ObservableTarget("alpha(M_Z)", target_coordinate)

    def delayed_checker(*, coordinate, transport_attempt, **_kwargs):
        if coordinate == target_coordinate and transport_attempt == 1:
            return TransportCheck(coordinate=coordinate, completed=False, passed=False, payload={"status": "pending"})
        return TransportCheck(coordinate=coordinate, completed=True, passed=True, payload={"status": "complete"})

    audit = engine.execute(observable_targets=(target,), transport_checker=delayed_checker)

    assert audit.total_runtime_overhead_ticks == len(engine.loading_sequence) + 1
    assert len(audit.manifestation_events) == 1
    event = audit.manifestation_events[0]
    assert event.observable_name == "alpha(M_Z)"
    assert event.coordinate == target_coordinate
    assert event.causal_tick == len(engine.loading_sequence) + 1
    assert event.transport_attempt == 2
    assert event.observable_triggered
    assert event.payload["status"] == "complete"


def test_define_get_event_requires_completed_transport_check() -> None:
    target = ObservableTarget("mu", (0, 0))
    incomplete_check = TransportCheck(coordinate=(0, 0), completed=False, passed=False)

    with pytest.raises(ValueError, match="completed transport check"):
        define_get_event(
            target,
            branch=(LEPTON_LEVEL, QUARK_LEVEL, PARENT_LEVEL),
            causal_tick=1,
            runtime_overhead_ticks=1,
            transport_attempt=1,
            transport_check=incomplete_check,
        )


def test_default_transport_check_exposes_loading_context_in_get_event() -> None:
    engine = OrderOfExecutionEngine()
    target_coordinate = engine.loading_sequence[0]
    audit = engine.execute(observable_targets=(ObservableTarget("observable", target_coordinate),))

    event = audit.manifestation_events[0]
    assert event.payload["branch"] == (LEPTON_LEVEL, QUARK_LEVEL, PARENT_LEVEL)
    assert event.payload["sequence_rank"] == 1
    assert event.payload["transport_attempt"] == 1
    assert event.payload["loading_weight"] > 0
    assert event.payload["entanglement_weight"] > 0
