from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from shbt.sectors import build_gravity_sector_audit


def test_gravity_sector_integrates_observer_tuple_as_markov_collar() -> None:
    audit = build_gravity_sector_audit()

    assert audit.branch == audit.benchmark_branch
    assert audit.observer_is_markov_collar is True
    assert isinstance(audit.observer_tuple, SimpleNamespace)
    assert audit.observer_tuple.position_radius_m == Decimal("0")
    assert audit.observer_tuple.local_horizon_area_m2 == audit.horizon_limit.local_horizon_area_m2
    assert audit.observer_tuple.patch_area_m2 == audit.horizon_limit.local_horizon_area_m2
