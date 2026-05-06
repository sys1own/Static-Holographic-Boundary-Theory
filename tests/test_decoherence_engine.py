from __future__ import annotations

from decimal import Decimal

import pytest

from shbt.core.decoherence_engine import (
    BulkWavefunction,
    PointerStateDecoherenceError,
    PointerStateSelector,
)


def test_pointer_state_selector_keeps_only_parity_neutral_states() -> None:
    selector = PointerStateSelector()
    candidates = (
        BulkWavefunction(
            label="neutral_low",
            amplitude=Decimal("0.4"),
            c_vis=Decimal("-1.25"),
            c_dark=Decimal("1.25"),
        ),
        BulkWavefunction(
            label="non_neutral",
            amplitude=Decimal("0.9"),
            c_vis=Decimal("-1.25"),
            c_dark=Decimal("1.00"),
        ),
        BulkWavefunction(
            label="neutral_high",
            amplitude=Decimal("0.7"),
            c_vis=Decimal("-2.5"),
            c_dark=Decimal("2.5"),
        ),
    )

    filtered = selector.filter_bulk_wavefunctions(candidates)
    selected = selector.select_pointer_state(candidates)

    assert tuple(candidate.wavefunction.label for candidate in filtered) == ("neutral_low", "neutral_high")
    assert all(candidate.checksum.passed for candidate in filtered)
    assert selected.wavefunction.label == "neutral_high"
    assert selected.checksum.checksum_equation == "|c_vis + c_dark|"
    assert selected.crystallizes


def test_pointer_state_selector_rejects_non_neutral_classicalization() -> None:
    selector = PointerStateSelector()
    wavefunction = BulkWavefunction(
        label="failed_projection",
        amplitude=Decimal("1"),
        c_vis=Decimal("-1.25"),
        c_dark=Decimal("1.00"),
    )

    with pytest.raises(PointerStateDecoherenceError, match=r"c_vis \+ c_dark = 0"):
        selector.crystallize_classical_observable(wavefunction, Decimal("42"))
