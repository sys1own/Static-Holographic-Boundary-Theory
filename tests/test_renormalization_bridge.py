from __future__ import annotations

import pytest

from shbt.core.renormalization_bridge import ScaleDependentTransport
from shbt.core.renormalization_map import BENCHMARK_UV_SCALE_GEV


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
