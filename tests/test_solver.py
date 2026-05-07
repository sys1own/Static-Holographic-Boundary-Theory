from __future__ import annotations

import pytest

from shbt.core.solver import EntropyAccumulator, HolographicTransportSolver, build_solver_audit, build_solver_report


def test_solver_defines_arrow_of_time_as_iteration_counter() -> None:
    audit = build_solver_audit()

    assert audit.total_iterations > 0
    assert audit.current_time_counter == audit.total_iterations
    assert audit.arrow_of_time_is_iteration_counter is True
    assert [frame.time_counter for frame in audit.frames] == list(range(1, audit.total_iterations + 1))
    assert audit.time_definition == "Time is the iteration counter of the holographic transport solver."
    assert audit.time_is_computational_overhead is True


def test_entropy_accumulator_generates_bit_residue_for_each_solver_step() -> None:
    audit = build_solver_audit()
    accumulator = EntropyAccumulator(bit_budget=audit.bit_budget, precision=audit.precision)
    residues = accumulator.accumulate(audit.causal_resolution)

    assert len(residues) == audit.total_iterations
    assert all(residue.delta_runtime_overhead_ticks == 1 for residue in residues)
    assert all(residue.delta_bit_residue > 0 for residue in residues)
    assert residues[-1].cumulative_bit_residue == pytest.approx(audit.total_bit_residue, rel=0.0, abs=1.0e-30)
    assert audit.entropy_accumulates_monotonically is True
    assert audit.maintains_bulk_boundary_equilibrium is True


def test_holographic_transport_solver_resolves_current_recursive_state() -> None:
    solver = HolographicTransportSolver()
    current_state = solver.resolve_current_state()

    assert current_state is not None
    assert current_state.iteration_index == current_state.time_counter
    assert current_state.recursive_depth == current_state.iteration_index
    assert current_state.boundary_dimension == 26
    assert current_state.bulk_dimension == 4
    assert current_state.maintains_bulk_boundary_equilibrium is True
    assert current_state.statement == (
        "The present universe is the current recursive frame indexed by the solver iteration counter."
    )


def test_solver_report_mentions_computational_overhead_time_law() -> None:
    report = build_solver_report()

    assert "Holographic Solver Audit" in report
    assert "Arrow of time established    : True" in report
    assert "Time is computational work   : True" in report
    assert "Time is the iteration counter of the holographic transport solver." in report
    assert "Each iteration generates a bit residue that records the solver overhead of the 26D -> 4D projection." in report
    assert "Time is the sequence of computations required to reconcile the boundary with the bulk." in report
