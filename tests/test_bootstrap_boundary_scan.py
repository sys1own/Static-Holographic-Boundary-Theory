from __future__ import annotations

from fractions import Fraction

import pytest

import shbt.core.bootstrap as bootstrap_module


def test_scan_boundary_configurations_defaults_to_benchmark_window() -> None:
    assert bootstrap_module.scan_boundary_configurations() == {
        24: 0.222,
        25: 0.222,
        26: 0.0,
        27: 0.222,
        28: 0.222,
    }


def test_scan_boundary_configurations_consumes_range_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visited_branches: list[tuple[int, int, int]] = []

    def fake_stability_residue(*, parent_level: int, lepton_level: int, quark_level: int) -> Fraction:
        branch = (lepton_level, quark_level, parent_level)
        visited_branches.append(branch)
        return Fraction(0, 1) if branch == (26, 8, 312) else Fraction(1, 1)

    monkeypatch.setattr(bootstrap_module, "_stability_residue", fake_stability_residue)

    assert bootstrap_module.scan_boundary_configurations(
        lepton_levels=range(24, 29),
        quark_levels=range(7, 10),
        parent_levels=range(310, 315),
    ) == {
        24: 0.222,
        25: 0.222,
        26: 0.0,
        27: 0.222,
        28: 0.222,
    }

    expected_branches = {
        (lepton_level, quark_level, parent_level)
        for lepton_level in range(24, 29)
        for quark_level in range(7, 10)
        for parent_level in range(310, 315)
    }
    assert len(visited_branches) == len(expected_branches)
    assert set(visited_branches) == expected_branches


def test_boundary_scan_window_has_unique_global_maximum_at_benchmark_branch() -> None:
    precision_floor = Fraction(1, bootstrap_module.FIXED_POINT_DENOMINATOR)
    branch_residues = {
        (lepton_level, quark_level, parent_level): bootstrap_module._stability_residue(
            parent_level=parent_level,
            lepton_level=lepton_level,
            quark_level=quark_level,
        )
        for lepton_level in range(24, 29)
        for quark_level in range(7, 10)
        for parent_level in range(310, 315)
    }

    best_branch, best_residue = min(
        branch_residues.items(),
        key=lambda item: (item[1], item[0][2], item[0][0], item[0][1]),
    )
    closure_branches = [
        branch
        for branch, stability_residue in branch_residues.items()
        if stability_residue <= precision_floor
    ]

    assert bootstrap_module.PRECISION_GUARD == 128
    assert bootstrap_module._ZERO_ANCHOR_PRECISION_FLOOR == precision_floor
    assert best_branch == (26, 8, 312)
    assert best_residue == Fraction(0, 1)
    assert closure_branches == [(26, 8, 312)]
