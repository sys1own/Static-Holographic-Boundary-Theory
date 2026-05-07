from __future__ import annotations

import pytest

from shbt.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.causal_execution import ObservableTarget, TransportCheck
from shbt.core.causal_resolution import (
    CausalExecutionLoop,
    TemporalStateResolver,
    build_causal_resolution_audit,
    build_causal_resolution_report,
)
from shbt.core.topological_kernel import sequence_bit_loading


def test_causal_resolution_establishes_arrow_of_time_from_initial_main_call() -> None:
    audit = build_causal_resolution_audit()

    assert audit.branch == (LEPTON_LEVEL, QUARK_LEVEL, PARENT_LEVEL)
    assert audit.big_bang.is_initial_main_call
    assert audit.big_bang.entrypoint == "main"
    assert audit.big_bang.iteration_index == 0
    assert audit.loading_sequence == sequence_bit_loading(lepton_level=26, quark_level=8)
    assert [step.coordinate for step in audit.resolution_steps] == list(audit.loading_sequence)
    assert [step.iteration_index for step in audit.resolution_steps] == list(range(1, len(audit.loading_sequence) + 1))
    assert audit.total_resolution_iterations == len(audit.loading_sequence)
    assert audit.current_iteration_index == len(audit.loading_sequence)
    assert audit.current_state is not None
    assert audit.current_state.is_present_state is True
    assert audit.current_state.coordinate == audit.loading_sequence[-1]
    assert audit.current_state.recursive_depth == len(audit.loading_sequence)
    assert audit.current_state.delta_update.advances_equilibrium is True
    assert audit.completed_delta_updates == len(audit.loading_sequence)
    assert audit.time_is_sequential_resolution
    assert audit.time_is_solver_overhead is True
    assert audit.big_bang_statement == "The Big Bang is the initial main() function call."
    assert audit.current_state_of_universe == (
        f"The current universe is iteration {len(audit.loading_sequence)} of the boundary-transport solver."
    )
    assert audit.current_recursive_state_of_universe == (
        f"The current universe is recursive frame {len(audit.loading_sequence)} of the boundary-transport solver."
    )
    assert audit.statement == "Time is the sequential resolution order of the holographic transport solver."


def test_causal_resolution_tracks_present_universe_as_current_solver_iteration() -> None:
    loop = CausalExecutionLoop()
    target_coordinate = loop.loading_sequence[0]
    target = ObservableTarget("alpha(M_Z)", target_coordinate)

    def delayed_checker(*, coordinate, transport_attempt, **_kwargs):
        if coordinate == target_coordinate and transport_attempt == 1:
            return TransportCheck(coordinate=coordinate, completed=False, passed=False, payload={"status": "pending"})
        return TransportCheck(coordinate=coordinate, completed=True, passed=True, payload={"status": "complete"})

    audit = loop.resolve(observable_targets=(target,), transport_checker=delayed_checker)

    assert audit.total_resolution_iterations == len(loop.loading_sequence) + 1
    assert audit.current_iteration_index == len(loop.loading_sequence) + 1
    assert audit.current_state is not None
    assert audit.current_state.coordinate == target_coordinate
    assert audit.current_state.transport_attempt == 2
    assert audit.current_state.resolved_coordinate_count == len(loop.loading_sequence)
    assert audit.current_state.manifested_event_count == 1
    assert audit.delta_updates[0].delta_resolved_coordinates == 0
    assert audit.delta_updates[0].delta_temporal_update == 0
    assert audit.current_state.maintains_bulk_boundary_equilibrium is True
    assert len(audit.manifestation_events) == 1
    assert audit.manifestation_events[0].causal_tick == len(loop.loading_sequence) + 1


def test_causal_resolution_report_exposes_big_bang_and_current_iteration_language() -> None:
    report = build_causal_resolution_report()

    assert "Causal Resolution Audit" in report
    assert "Big Bang entrypoint          : main()" in report
    assert "Arrow of time established    : True" in report
    assert "The Big Bang is the initial main() function call." in report
    assert "The current universe is iteration" in report
    assert "Time is solver overhead      : True" in report
    assert "Bulk-boundary equilibrium    : True" in report


def test_causal_resolution_defines_time_as_equilibrium_preserving_delta_updates() -> None:
    audit = build_causal_resolution_audit()

    assert audit.delta_update_count == len(audit.resolution_steps)
    assert audit.completed_delta_updates == len(audit.loading_sequence)
    assert audit.time_is_delta_update_sequence is True
    assert all(delta_update.delta_runtime_overhead_ticks == 1 for delta_update in audit.delta_updates)
    assert all(delta_update.maintains_bulk_boundary_equilibrium for delta_update in audit.delta_updates)
    assert all(delta_update.delta_temporal_update > 0 for delta_update in audit.delta_updates)
    assert audit.time_definition == (
        "Time is the sequence of delta updates required to maintain bulk-boundary equilibrium."
    )
    assert audit.computational_overhead_statement == "Time becomes the computational overhead of the solver."
    assert audit.current_recursive_state_of_universe == (
        f"The current universe is recursive frame {len(audit.loading_sequence)} of the boundary-transport solver."
    )


def test_temporal_state_resolver_treats_present_universe_as_recursive_frame() -> None:
    resolver = TemporalStateResolver()

    current_frame = resolver.resolve_current_state()

    assert current_frame is not None
    assert current_frame.frame_index == len(resolver.loading_sequence)
    assert current_frame.recursive_depth == len(resolver.loading_sequence)
    assert current_frame.frame_chain == tuple(range(1, len(resolver.loading_sequence) + 1))
    assert current_frame.current_state.coordinate == resolver.loading_sequence[-1]
    assert current_frame.delta_update.maintains_bulk_boundary_equilibrium is True
    assert current_frame.is_present_state is True
    assert current_frame.statement == "Dynamic reality is the current recursive frame produced by the transport solver."


def test_temporal_state_resolver_rejects_nonpositive_bit_budget() -> None:
    with pytest.raises(ValueError, match="bit_budget must be positive"):
        TemporalStateResolver(bit_budget=0)


def test_causal_resolution_report_mentions_recursive_frame_and_delta_update_language() -> None:
    report = build_causal_resolution_report()

    assert "Delta-update time law        : True" in report
    assert "Recursive temporal frames    :" in report
    assert "Time is the sequence of delta updates required to maintain bulk-boundary equilibrium." in report
    assert "Dynamic reality is the recursive stack of present-state frames produced by the transport solver." in report
    assert "Time becomes the computational overhead of the solver." in report
