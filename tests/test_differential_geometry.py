from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest

from shbt.core import numerics
from shbt.core.differential_geometry import (
    build_metric_tensor,
    christoffel_symbols,
    coordinate_transform,
    evolve_riemannian_manifold,
    line_element,
    lower_index,
    metric_inner_product,
    project_static_block_to_bulk,
    raise_index,
    ricci_flow_step,
    ricci_scalar,
    ricci_tensor,
    riemann_curvature_tensor,
)


def test_metric_tensor_operations_raise_and_lower_indices() -> None:
    metric = build_metric_tensor([[2.0, 0.0], [0.0, 3.0]])

    assert metric_inner_product(metric, [1.0, 2.0], [1.0, 2.0]) == pytest.approx(14.0, rel=0.0, abs=1.0e-12)
    assert line_element(metric, [1.0, 2.0]) == pytest.approx(14.0, rel=0.0, abs=1.0e-12)
    assert np.allclose(lower_index(metric, [1.0, 2.0]), np.asarray([2.0, 6.0]))
    assert np.allclose(raise_index(metric, [2.0, 6.0]), np.asarray([1.0, 2.0]))


def test_flat_metric_has_zero_curvature() -> None:
    metric = build_metric_tensor(np.eye(2))
    metric_derivatives = np.zeros((2, 2, 2))
    gamma = christoffel_symbols(metric, metric_derivatives)
    gamma_derivatives = np.zeros((2, 2, 2, 2))
    riemann = riemann_curvature_tensor(gamma, gamma_derivatives)
    ricci = ricci_tensor(riemann)

    assert np.allclose(gamma, np.zeros((2, 2, 2)))
    assert np.allclose(riemann, np.zeros((2, 2, 2, 2)))
    assert np.allclose(ricci, np.zeros((2, 2)))
    assert ricci_scalar(metric, ricci) == pytest.approx(0.0, rel=0.0, abs=1.0e-12)


def test_riemannian_manifold_evolution_performs_ricci_flow_step() -> None:
    metric = build_metric_tensor(np.eye(2))
    ricci = np.asarray([[0.1, 0.0], [0.0, 0.1]])

    stepped = ricci_flow_step(metric, ricci, step_size=0.5)
    evolution = evolve_riemannian_manifold(metric, [ricci], step_size=0.5)

    assert np.allclose(stepped.components, np.asarray([[0.9, 0.0], [0.0, 0.9]]))
    assert evolution.step_count == 1
    assert np.allclose(evolution.final_state.metric.components, stepped.components)


def test_numerics_compatibility_module_reexports_real_helpers() -> None:
    assert numerics.require_real_scalar(3.0, label="scalar") == pytest.approx(3.0, rel=0.0, abs=1.0e-12)


def test_metric_tensor_transforms_static_block_coordinates_into_bulk_metric() -> None:
    static_metric = build_metric_tensor(np.diag([-1.0, 1.0, 2.0, 3.0]), label="static_block_metric")
    projector = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    transformed = coordinate_transform(static_metric, projector, label="projected_bulk_metric")

    assert transformed.dimension == 3
    assert np.allclose(transformed.components, np.diag([1.0, 2.0, 3.0]))
    assert transformed.positive_definite


def test_metric_tensor_bulk_projection_is_constrained_by_holographic_stabilizer() -> None:
    static_metric = build_metric_tensor(np.diag([-1.0, 1.0, 1.5, 2.0]), label="static_block_metric")
    projector = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    class _PassingStabilizer:
        def verify_bulk_integrity(self):
            return SimpleNamespace(
                passed=True,
                zero_energy_boundary_locked=True,
                equivalence_principle_preserved=True,
                torsion_projection_residual=Decimal("0"),
                detail="bulk integrity locked",
                benchmark_branch=(26, 8, 312),
            )

    projection = project_static_block_to_bulk(static_metric, projector, stabilizer=_PassingStabilizer())

    assert projection.stabilizer_passed
    assert projection.static_block_dimension == 4
    assert projection.projected_bulk_dimension == 3
    assert projection.benchmark_branch == (26, 8, 312)
    assert np.allclose(projection.bulk_metric.components, np.diag([1.0, 1.5, 2.0]))


def test_metric_tensor_bulk_projection_rejects_failed_stabilizer() -> None:
    static_metric = build_metric_tensor(np.diag([-1.0, 1.0, 1.5, 2.0]), label="static_block_metric")
    projector = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    class _FailingStabilizer:
        def verify_bulk_integrity(self):
            return SimpleNamespace(
                passed=False,
                zero_energy_boundary_locked=False,
                equivalence_principle_preserved=False,
                torsion_projection_residual=Decimal("0.125"),
                detail="bulk torsion amplitude=0.125",
                benchmark_branch=(26, 8, 312),
            )

    with pytest.raises(ValueError, match="HolographicStabilizer rejected bulk metric evolution"):
        static_metric.project_to_bulk(projector, stabilizer=_FailingStabilizer())
