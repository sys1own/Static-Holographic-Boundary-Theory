from __future__ import annotations

from decimal import Decimal

from shbt.core.temporal_emergence_kernel import (
    build_temporal_emergence_audit,
    map_manifold_slice_bit_loading_density,
)
from shbt.core.topological_kernel import ModularKernel


def test_core_topological_kernel_wrapper_exposes_modular_kernel() -> None:
    lepton_block = ModularKernel(26, "lepton").restricted_block()
    quark_block = ModularKernel(8, "quark").restricted_block()

    assert lepton_block.shape == (3, 3)
    assert quark_block.shape == (3, 3)


def test_manifold_slice_density_maps_are_normalized() -> None:
    density_map = map_manifold_slice_bit_loading_density()

    assert density_map.loading_density_sum == Decimal("1")
    assert density_map.entanglement_density_sum == Decimal("1")
    assert len(density_map.dominant_loading_sequence) == 9


def test_temporal_emergence_exactly_locks_to_precision_cosmology() -> None:
    audit = build_temporal_emergence_audit()

    assert audit.verification.arrow_of_time_gradient_verified
    assert audit.verification.exact_metric_lock
    assert audit.verification.loading_law_consistent
    assert audit.verification.ubi_confirmed
    assert audit.verification.max_arrow_of_time_gradient_residual <= Decimal("1e-15")
    assert audit.verification.max_metric_rate_residual == Decimal("0")
    assert audit.verification.max_loading_derivative_residual < Decimal("1e-50")

    for sample in audit.samples:
        assert sample.bulk_metric_expansion.dot_a_over_a_km_s_mpc == sample.metric_expansion_rate_km_s_mpc
        assert sample.arrow_of_time_gradient_verified
        assert abs(sample.arrow_of_time_gradient_residual) <= Decimal("1e-15")
        assert abs(sample.arrow_of_time_gradient_km_s_mpc - sample.metric_expansion_rate_km_s_mpc) <= Decimal("1e-15")
        assert sample.derived_temporal_rate_km_s_mpc == sample.metric_expansion_rate_km_s_mpc
        assert sample.loading_law_consistent
