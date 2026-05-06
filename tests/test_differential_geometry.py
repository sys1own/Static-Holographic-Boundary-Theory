from __future__ import annotations

import numpy as np
import pytest

from shbt.core import numerics
from shbt.core.differential_geometry import (
    build_metric_tensor,
    christoffel_symbols,
    evolve_riemannian_manifold,
    line_element,
    lower_index,
    metric_inner_product,
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
