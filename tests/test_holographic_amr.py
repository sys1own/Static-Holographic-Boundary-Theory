from __future__ import annotations

from decimal import Decimal

from shbt.constants import HOLOGRAPHIC_BITS
from shbt.core.holographic_amr import (
    AdaptiveHolographicMesh,
    PLANCK_VOLUME_M3,
    UNITY_TOLERANCE,
    build_holographic_amr_audit,
)


def test_recursive_mesh_refiner_scales_bit_budget_with_volume() -> None:
    audit = build_holographic_amr_audit(
        observer_coordinate_volume_m3=Decimal("512"),
        reference_coordinate_volume_m3=Decimal("1"),
    )

    assert audit.mesh_depth == 3
    assert audit.root.coordinate_volume_m3 == Decimal("512")
    assert audit.levels[-1].coordinate_volume_m3 == Decimal("1")
    assert audit.root.bit_budget == Decimal(str(HOLOGRAPHIC_BITS)) * Decimal("512")
    assert all(level.bit_rot_free for level in audit.levels)
    assert all(level.parity_preserved for level in audit.levels)


def test_holographic_amr_preserves_unity_and_parity_under_coarsening() -> None:
    galactic_proxy_volume = PLANCK_VOLUME_M3 * (Decimal("8") ** 12)
    audit = build_holographic_amr_audit(observer_coordinate_volume_m3=galactic_proxy_volume)

    assert audit.passed
    assert audit.unity_locked
    assert audit.parity_preserved
    assert audit.bit_rot_free
    assert audit.max_epsilon_lambda <= UNITY_TOLERANCE
    assert audit.max_epsilon_lambda_noether_bridged <= UNITY_TOLERANCE
    assert audit.max_budget_relative_residual <= UNITY_TOLERANCE
    assert audit.max_parity_residual == Decimal("0")
    assert audit.statement == (
        "The Adaptive Holographic Mesh coarsens from Planckian to macroscopic scales "
        "without parity loss, bit rot, or Unity-of-Scale drift."
    )


def test_holographic_amr_matches_requested_partial_final_coarsening() -> None:
    mesh = AdaptiveHolographicMesh(reference_coordinate_volume_m3=Decimal("1"))
    audit = mesh.refine_mesh(observer_coordinate_volume_m3=Decimal("100"))

    assert audit.root.coordinate_volume_m3 == Decimal("100")
    assert audit.root.bit_budget == Decimal(str(HOLOGRAPHIC_BITS)) * Decimal("100")
    assert audit.levels[0].bit_rot_free
    assert audit.passed
