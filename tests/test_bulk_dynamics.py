from __future__ import annotations

from decimal import Decimal

from shbt.core.bulk_dynamics import BulkDynamics, build_bulk_dynamics_audit


def test_calculate_bulk_flow_builds_monotonic_entropy_hierarchy() -> None:
    audit = BulkDynamics().calculate_bulk_flow(redshift=Decimal("0"))

    assert audit.source_block_dimension == 4
    assert audit.emergent_bulk_dimension == 3
    assert len(audit.layers) == 9
    assert audit.monotonic_entropy_growth
    assert all(layer.entropy_increases for layer in audit.layers)
    assert audit.layers[-1].cumulative_entanglement_density == Decimal("1")
    assert audit.total_entropy_production_rate == audit.temporal_point.total_entanglement_entropy_gradient
    assert audit.statement == (
        "A static 4D block produces a non-equilibrium 3D bulk through "
        "mandatory layer-by-layer entanglement growth."
    )


def test_bulk_flow_matches_temporal_emergence_kernel() -> None:
    audit = build_bulk_dynamics_audit(redshift=Decimal("0.5"))

    assert audit.arrow_of_time_positive
    assert audit.consistent_with_temporal_emergence_kernel
    assert audit.static_block_projects_bulk
    assert abs(audit.arrow_of_time_consistency_residual) <= Decimal("1e-15")
    assert abs(audit.entropy_to_time_identity_residual) <= Decimal("1e-15")
    assert abs(audit.metric_lock_residual) <= Decimal("1e-15")
    assert audit.layerwise_arrow_of_time_gradient_km_s_mpc == audit.temporal_point.derived_temporal_rate_km_s_mpc
