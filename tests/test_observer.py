from __future__ import annotations

from decimal import Decimal

import pytest

from shbt.constants import HOLOGRAPHIC_BITS
from shbt.core.observer import Observer


def test_observer_exposes_local_horizon_radius_and_entropy_capacity() -> None:
    observer = Observer()

    assert observer.observer_radius_m == Decimal("0")
    assert observer.local_horizon_radius_m == observer.global_horizon_radius_m
    assert float(observer.entropy_capacity_bits) == pytest.approx(
        float(HOLOGRAPHIC_BITS),
        rel=1.0e-12,
    )
    assert observer.bekenstein_hawking_entropy_bits > observer.entropy_capacity_bits


def test_observer_bulk_projection_grows_with_covariant_frame_shift() -> None:
    center = Observer()
    near_horizon = Observer(
        observer_radius_m=center.global_horizon_radius_m * Decimal("0.9")
    )

    center_audit = center.perceive_noether_bridge()
    near_horizon_audit = near_horizon.perceive_noether_bridge()

    assert near_horizon_audit.covariant_frame_shift > center_audit.covariant_frame_shift
    assert near_horizon_audit.bulk_weight > center_audit.bulk_weight
    assert near_horizon_audit.boundary_weight < center_audit.boundary_weight
    assert (
        near_horizon_audit.bulk_residues.completion_residue
        > center_audit.bulk_residues.completion_residue
    )
    assert (
        near_horizon_audit.bulk_residues.unity_closure_residue
        > center_audit.bulk_residues.unity_closure_residue
    )
    assert near_horizon_audit.residues_conserved


def test_observer_asserts_global_holographic_budget_conservation_after_motion() -> None:
    observer = Observer()
    audit = observer.move_to(observer.global_horizon_radius_m * Decimal("0.5"))

    assert audit.bit_budget_conserved
    assert observer.entropy_capacity_bits == audit.entropy_capacity_bits
    assert observer.local_horizon_radius_m == audit.local_horizon_radius_m
    assert audit.hidden_entropy_capacity_bits > 0
    observer.assert_global_bit_budget_conserved(audit)
