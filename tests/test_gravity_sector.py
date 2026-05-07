from __future__ import annotations

from decimal import Decimal

from shbt.core.observer_horizon import global_coordinate_horizon_radius
from shbt.sectors import build_gravity_sector_audit as build_exported_gravity_sector_audit
from shbt.sectors.gravity import (
    build_gravity_sector_audit,
    derive_gravitational_acceleration,
)



def test_gravity_sector_integrates_observer_tuple_as_markov_collar() -> None:
    audit = build_exported_gravity_sector_audit()

    assert audit.branch == (26, 8, 312)
    assert audit.observer_is_markov_collar is True
    assert audit.observer_tuple.position_radius_m == Decimal("0")
    assert audit.observer_tuple.patch_area_m2 == audit.horizon_limit.local_horizon_area_m2
    assert audit.observer_tuple.information_density_bits_per_m2 == audit.rendered_boundary_patch.rendered_information_density_bits_per_m2
    assert audit.observer_tuple.render_fraction == audit.rendered_boundary_patch.rendered_boundary_fraction
    assert audit.observer_markov_collar.factorization_verified is True



def test_gravity_sector_renders_global_boundary_data_onto_local_horizon() -> None:
    global_radius = global_coordinate_horizon_radius()
    audit = build_gravity_sector_audit(observer_radius_m=global_radius / Decimal("2"))

    assert audit.rendering_consistent is True
    assert audit.rendered_boundary_patch.rendered_boundary_bits > 0
    assert audit.rendered_boundary_patch.rendered_boundary_bits < audit.horizon_limit.local_available_bits
    assert abs(
        audit.rendered_boundary_patch.rendered_boundary_fraction
        - (audit.rendered_boundary_patch.rendered_boundary_bits / audit.rendered_boundary_patch.global_bit_budget)
    ) <= Decimal("1e-28")
    assert audit.rendered_boundary_patch.dominant_render_coordinate == audit.observer_markov_collar.collar_sequence[0]
    assert audit.rendered_boundary_patch.rendered_loading_sum_bits_per_m2 == audit.observer_tuple.information_density_bits_per_m2



def test_equivalence_principle_derives_gravity_from_local_information_gradient() -> None:
    global_radius = global_coordinate_horizon_radius()
    audit = build_gravity_sector_audit(observer_radius_m=global_radius / Decimal("2"))

    expected_acceleration = derive_gravitational_acceleration(
        observer_tuple=audit.observer_tuple,
        information_density_gradient_bits_per_m3=audit.local_information_density_gradient_bits_per_m3,
        global_bit_budget=audit.rendered_boundary_patch.global_bit_budget,
    )

    assert audit.local_information_density_gradient_bits_per_m3 < 0
    assert audit.gravitational_acceleration_m_per_s2 > 0
    assert audit.equivalence_principle_verified is True
    assert audit.gravitational_acceleration_m_per_s2 == expected_acceleration
    assert audit.statement.startswith("The observer tuple (P, A(P), rho, R)")
