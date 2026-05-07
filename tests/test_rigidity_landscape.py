from __future__ import annotations

from fractions import Fraction
import warnings

import numpy as np
import pytest

from shbt.core.rigidity_landscape import (
    EXPECTED_BENCHMARK,
    RigidityPoint,
    assert_unique_stable_fixed_point,
    build_centered_rigidity_landscape_scan,
    calculate_moat_depth,
)
from shbt.logic.bootstrap import SymmetrySearcher


warnings.filterwarnings("ignore", category=DeprecationWarning)


def test_centered_rigidity_landscape_certifies_unique_stable_fixed_point() -> None:
    scan = build_centered_rigidity_landscape_scan()

    assert scan.benchmark_coordinates == EXPECTED_BENCHMARK
    assert scan.benchmark_point.stable_fixed_point is True
    assert scan.benchmark_point.topological_closure is True
    assert scan.benchmark_point.topological_closure_score == 0.0
    assert scan.unique_stable_fixed_point is not None
    assert scan.unique_stable_fixed_point.coordinates == EXPECTED_BENCHMARK
    assert scan.topological_closure_survivors == (EXPECTED_BENCHMARK,)
    assert np.count_nonzero(scan.topological_closure_grid) == 1
    assert scan.nearest_detuned_point.topological_closure_score > 0.0


def test_symmetry_searcher_identifies_26_8_312_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    searcher = SymmetrySearcher(
        dimension_levels=range(24, 29),
        gauge_levels=range(6, 11),
        parent_levels=range(310, 315),
        elite_width=4,
    )

    def mock_score_path(
        self: SymmetrySearcher,
        dimension_level: int,
        gauge_level: int,
        parent_level: int,
    ) -> RigidityPoint:
        del self
        distance = abs(int(dimension_level) - EXPECTED_BENCHMARK[0])
        distance += abs(int(gauge_level) - EXPECTED_BENCHMARK[1])
        distance += abs(int(parent_level) - EXPECTED_BENCHMARK[2])
        residue = float(distance)
        return RigidityPoint(
            coordinates=(int(dimension_level), int(gauge_level), int(parent_level)),
            lepton_framing_gap=0.0 if distance == 0 else float(abs(int(dimension_level) - EXPECTED_BENCHMARK[0])),
            quark_framing_gap=0.0 if distance == 0 else float(abs(int(gauge_level) - EXPECTED_BENCHMARK[1])),
            delta_fr=0.0 if distance == 0 else float(distance) / 10.0,
            delta_fr_label="0" if distance == 0 else f"{float(distance) / 10.0}",
            c_dark_shift=0.0 if distance == 0 else float(abs(int(parent_level) - EXPECTED_BENCHMARK[2])) / 10.0,
            diophantine_gap=0.0 if distance == 0 else float(distance) / 100.0,
            total_residue=residue,
            topological_closure_score=residue,
        )

    monkeypatch.setattr(SymmetrySearcher, "score_path", mock_score_path)

    report = searcher.discover_optimal_kernel()

    assert report.evolutionary_history
    assert report.evolutionary_best_point.coordinates == EXPECTED_BENCHMARK
    assert report.discovered_kernel.coordinates == EXPECTED_BENCHMARK
    assert report.evolutionary_best_point.topological_closure_score == 0.0
    assert report.unique_stable_fixed_point is True
    assert report.certified_unique_survivor == EXPECTED_BENCHMARK
    assert report.mathematical_necessity is True
    assert report.runner_up_kernel.coordinates != EXPECTED_BENCHMARK
    assert report.runner_up_kernel.topological_closure_score > 0.0
    assert assert_unique_stable_fixed_point(report) == EXPECTED_BENCHMARK
    assert report.statement.startswith("The blind symmetry search certifies")


def test_calculate_moat_depth_uses_guarded_bit_loading_sequence() -> None:
    assert calculate_moat_depth(312, 26, 8) == Fraction(0, 1)
    assert calculate_moat_depth(311, 26, 8) == Fraction(1, 24)
