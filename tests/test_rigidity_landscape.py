from __future__ import annotations

import numpy as np

from shbt.core.rigidity_landscape import (
    EXPECTED_BENCHMARK,
    SymmetrySearcher,
    assert_unique_stable_fixed_point,
    build_centered_rigidity_landscape_scan,
)


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


def test_symmetry_searcher_blindly_discovers_and_certifies_benchmark_kernel() -> None:
    searcher = SymmetrySearcher(
        lepton_levels=range(24, 29),
        quark_levels=range(6, 11),
        parent_levels=range(310, 315),
        elite_width=4,
    )

    report = searcher.discover_unique_fixed_point()

    assert report.evolutionary_history
    assert report.evolutionary_best_point.coordinates == EXPECTED_BENCHMARK
    assert report.evolutionary_best_point.topological_closure_score == 0.0
    assert report.unique_stable_fixed_point is True
    assert report.certified_unique_survivor == EXPECTED_BENCHMARK
    assert report.mathematical_necessity is True
    assert report.certified_scan.nearest_detuned_point.coordinates != EXPECTED_BENCHMARK
    assert report.certified_scan.nearest_detuned_point.topological_closure_score > 0.0
    assert assert_unique_stable_fixed_point(report) == EXPECTED_BENCHMARK
    assert report.statement.startswith("The blind symmetry search certifies")
