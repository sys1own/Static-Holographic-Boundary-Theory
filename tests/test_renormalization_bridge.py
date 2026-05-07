from __future__ import annotations

import pytest

from shbt.constants import CODATA_FINE_STRUCTURE_ALPHA_INVERSE
from shbt.core.renormalization_bridge import CoolingFunction, ScaleDependentTransport, verify_scale_dependent_transport_experiment
from shbt.core.renormalization_map import BENCHMARK_UV_SCALE_GEV, CODATA_ALIGNMENT_ABS_TOLERANCE


def test_scale_dependent_transport_derives_ir_alpha_from_boundary_geometry() -> None:
    bridge = ScaleDependentTransport()

    assert bridge.branch == (26, 8, 312)
    assert bridge.boundary_dimension == 26
    assert bridge.bulk_dimension == 4
    assert bridge.alpha_uv_inverse == pytest.approx(137.64705882352942, rel=0.0, abs=1.0e-12)
    assert bridge.derive_ir_alpha_inverse() == pytest.approx(137.03628702174976, rel=0.0, abs=1.0e-12)
    assert bridge.derive_ir_alpha() == pytest.approx(1.0 / 137.03628702174976, rel=0.0, abs=1.0e-15)


def test_scale_dependent_transport_simulates_monotonic_uv_to_ir_cooling() -> None:
    bridge = ScaleDependentTransport(ir_scale_gev=0.0)
    trajectory = bridge.simulate_transport(sample_count=9)

    assert trajectory.branch == bridge.branch
    assert len(trajectory.points) == 9
    assert trajectory.points[0].energy_gev == pytest.approx(BENCHMARK_UV_SCALE_GEV)
    assert trajectory.points[-1].energy_gev == pytest.approx(0.0)
    assert trajectory.points[0].cooling_fraction == pytest.approx(0.0)
    assert trajectory.points[-1].cooling_fraction == pytest.approx(1.0)
    assert trajectory.points[0].transported_alpha_inverse == pytest.approx(bridge.alpha_uv_inverse, rel=0.0, abs=1.0e-12)
    assert trajectory.ir_alpha_inverse == pytest.approx(bridge.derive_ir_alpha_inverse(), rel=0.0, abs=1.0e-12)
    assert trajectory.monotonic_information_loss is True
    assert trajectory.monotonic_alpha_inverse_cooling is True


def test_cooling_function_projects_boundary_bit_density_into_bulk() -> None:
    bridge = ScaleDependentTransport()

    assert isinstance(bridge.cooling_operator, CoolingFunction)
    assert bridge.cooling_operator.discarded_dimensions == 22
    assert bridge.cooling_operator.projection_ratio == pytest.approx(4.0 / 26.0)

    uv_projection = bridge.cooling_operator.project(BENCHMARK_UV_SCALE_GEV)
    ir_projection = bridge.cooling_operator.project(0.0)

    assert uv_projection.smearing_fraction == pytest.approx(0.0)
    assert ir_projection.smearing_fraction == pytest.approx(1.0)
    assert uv_projection.information_loss_inverse == pytest.approx(0.0, rel=0.0, abs=1.0e-15)
    assert ir_projection.information_loss_inverse == pytest.approx(
        bridge.alpha_uv_inverse - bridge.derive_ir_alpha_inverse(),
        rel=0.0,
        abs=1.0e-12,
    )
    assert uv_projection.boundary_bit_density_inverse == pytest.approx(bridge.alpha_uv_inverse / 26.0, rel=0.0, abs=1.0e-12)
    assert ir_projection.bulk_bit_density_inverse == pytest.approx(
        ir_projection.retained_information_inverse / 4.0,
        rel=0.0,
        abs=1.0e-12,
    )
    assert ir_projection.smeared_bit_density_inverse == pytest.approx(
        ir_projection.information_loss_inverse / 22.0,
        rel=0.0,
        abs=1.0e-12,
    )
    assert ir_projection.retained_information_fraction < 1.0
    assert ir_projection.lost_information_fraction > 0.0


def test_scale_dependent_transport_operator_aligns_with_experiment_without_targeting() -> None:
    bridge = ScaleDependentTransport()
    point = bridge(0.0)
    verification = bridge.verify_experimental_alignment()
    module_verification = verify_scale_dependent_transport_experiment()

    assert point.transported_alpha_inverse == pytest.approx(bridge.derive_ir_alpha_inverse(), rel=0.0, abs=1.0e-12)
    assert verification.point.transported_alpha_inverse == pytest.approx(point.transported_alpha_inverse, rel=0.0, abs=1.0e-12)
    assert verification.target_alpha_inverse == pytest.approx(CODATA_FINE_STRUCTURE_ALPHA_INVERSE)
    assert verification.target_used_in_transport is False
    assert verification.derived_from_geometry_only is True
    assert verification.absolute_error_inverse <= CODATA_ALIGNMENT_ABS_TOLERANCE
    assert verification.aligns_with_experiment is True
    assert module_verification.aligns_with_experiment is True
