from __future__ import annotations

from decimal import Decimal

import pytest

from shbt.constants import HOLOGRAPHIC_BITS
from shbt.core.observer_horizon import (
    BENCHMARK_BRANCH,
    audit_observer_holographic_moat,
    calculate_observer_horizon_limit,
    global_coordinate_horizon_radius,
)


def test_observer_horizon_center_recovers_full_bit_budget() -> None:
    horizon = calculate_observer_horizon_limit()

    assert horizon.relative_position == Decimal("0")
    assert horizon.exposed_area_fraction == Decimal("1")
    assert float(horizon.local_available_bits) == pytest.approx(float(HOLOGRAPHIC_BITS), rel=1.0e-12)
    assert horizon.log_horizon_loading_factor > 0


def test_observer_moat_locks_benchmark_branch() -> None:
    moat = audit_observer_holographic_moat()

    assert moat.evaluated_branch == BENCHMARK_BRANCH
    assert moat.inside_published_visible_moat
    assert moat.observer_moat_locked


def test_observer_moat_penalty_grows_toward_horizon() -> None:
    global_radius = global_coordinate_horizon_radius()
    center = audit_observer_holographic_moat(lepton_level=24, quark_level=8, parent_level=312)
    near_horizon = audit_observer_holographic_moat(
        observer_radius_m=global_radius * Decimal("0.9"),
        lepton_level=24,
        quark_level=8,
        parent_level=312,
    )

    assert center.inside_published_visible_moat
    assert near_horizon.inside_published_visible_moat
    assert near_horizon.observer_shifted_defect > center.observer_shifted_defect
    assert not near_horizon.observer_moat_locked
