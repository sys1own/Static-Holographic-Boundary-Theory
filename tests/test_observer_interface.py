from __future__ import annotations

from decimal import Decimal

import pytest

from shbt.core.observer_interface import (
    GET_OBSERVABLE,
    ObservableScale,
    ObserverInterface,
    get_observable,
)


def test_get_observable_manifests_weighted_boundary_sequence_as_bulk_measurement() -> None:
    interface = ObserverInterface(microstate_resolution=32)

    audit = interface.get_observable(
        [1, 6, 1],
        observable_name="mass_gap",
        boundary_address="boundary.mass_gap",
        scale=ObservableScale(zero_point=Decimal("0.1"), step=Decimal("0.05"), units="eV"),
    )

    audit.assert_manifested()

    assert audit.command.verb == GET_OBSERVABLE
    assert audit.noninvertible_map is True
    assert audit.microstate_count == 32
    assert audit.coarse_outcome_count == 3
    assert audit.selected_preimage_size >= 1
    assert audit.pointer_state is not None
    assert audit.pointer_state.crystallizes
    assert audit.manifestation is not None
    assert audit.manifestation.units == "eV"
    assert audit.manifestation.measurement_value in {
        Decimal("0.1"),
        Decimal("0.15"),
        Decimal("0.2"),
    }
    assert audit.statement.startswith("GET_OBSERVABLE closes Saying")


def test_get_observable_is_deterministic_for_fixed_boundary_loading() -> None:
    first = get_observable(
        [3, 1, 2],
        observable_name="temperature",
        boundary_address="boundary.temperature",
        scale=ObservableScale(zero_point=Decimal("2"), step=Decimal("3"), units="K"),
        microstate_resolution=24,
    )
    second = get_observable(
        [3, 1, 2],
        observable_name="temperature",
        boundary_address="boundary.temperature",
        scale=ObservableScale(zero_point=Decimal("2"), step=Decimal("3"), units="K"),
        microstate_resolution=24,
    )

    first.assert_manifested()
    second.assert_manifested()

    assert first.collapse_audit.collapse_index == second.collapse_audit.collapse_index
    assert first.manifestation == second.manifestation
    assert first.microstate_counts == second.microstate_counts


def test_get_observable_uses_uniform_fallback_for_zero_loaded_register() -> None:
    audit = get_observable(
        [0, 0, 0],
        observable_name="spin_projection",
        boundary_address="boundary.spin",
        measurement_values=("down", "flat", "up"),
        microstate_resolution=9,
    )

    audit.assert_manifested()

    assert audit.normalized_bit_loadings == pytest.approx(
        (Decimal("1") / Decimal("3"), Decimal("1") / Decimal("3"), Decimal("1") / Decimal("3"))
    )
    assert audit.microstate_counts == (3, 3, 3)
    assert audit.manifestation is not None
    assert audit.manifestation.measurement_value in {"down", "flat", "up"}
