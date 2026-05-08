from __future__ import annotations

from decimal import Decimal

from shbt.core.entropy_engine import (
    SOURCE_PROJECTION_DIMENSION,
    TARGET_PROJECTION_DIMENSION,
    EntropyAccumulator,
    build_entropy_engine_audit,
    render_report,
)


def test_entropy_engine_turns_static_loading_into_dynamic_temporal_audit() -> None:
    audit = build_entropy_engine_audit()

    assert audit.branch == (26, 8, 312)
    assert len(audit.iterations) == len(audit.loading_sequence) == 9
    assert audit.iterations[0].projected_dimension_before == Decimal(str(SOURCE_PROJECTION_DIMENSION))
    assert audit.final_projected_dimension == Decimal(str(TARGET_PROJECTION_DIMENSION))
    assert audit.total_temporal_overhead == Decimal("1")
    assert audit.arrow_of_time_established is True
    assert audit.bulk_boundary_causal_loop_closed is True
    assert audit.time_is_computational_overhead is True
    assert audit.temporal_overhead_identity_holds is True
    assert audit.self_resolution_complete is True
    assert audit.statement == "Time is the computational overhead of the universe's self-resolution."


def test_entropy_accumulator_tracks_stepwise_bit_residue_and_feedback_activation() -> None:
    accumulator = EntropyAccumulator()

    first = accumulator.step()
    second = accumulator.step()

    assert first.iteration_index == 1
    assert second.iteration_index == 2
    assert first.boundary_bit_residue > 0
    assert first.bulk_entropy_bit_residue > 0
    assert first.temporal_increment > 0
    assert first.bulk_feedback_signal == 0
    assert first.feedback_perturbs_boundary_resolution is False
    assert second.bulk_feedback_signal > 0
    assert second.feedback_perturbs_boundary_resolution is True
    assert second.cumulative_bulk_entropy_bit_residue > first.cumulative_bulk_entropy_bit_residue
    assert second.projected_dimension_after < first.projected_dimension_after


def test_zero_feedback_recovers_static_resolution_shares_without_causal_perturbation() -> None:
    audit = build_entropy_engine_audit(feedback_coupling=Decimal("0"))

    assert audit.arrow_of_time_established is True
    assert audit.time_is_computational_overhead is True
    assert audit.self_resolution_complete is True
    assert audit.bulk_boundary_causal_loop_closed is False
    assert all(step.bulk_feedback_signal >= 0 for step in audit.iterations)
    assert all(step.feedback_perturbs_boundary_resolution is False for step in audit.iterations)


def test_entropy_engine_report_mentions_causal_loop_and_overhead_language() -> None:
    report = render_report(build_entropy_engine_audit())

    assert "Entropy Engine" in report
    assert "Bulk-boundary causal loop         : True" in report
    assert "Time is computational overhead    : True" in report
    assert "Time is modeled as the iterative entropic overhead of holographic self-resolution." in report


def test_entropy_engine_audit_materializes_time_axiom_on_initialization() -> None:
    audit = build_entropy_engine_audit()

    assert audit.__dict__["time_is_computational_overhead"] is True
