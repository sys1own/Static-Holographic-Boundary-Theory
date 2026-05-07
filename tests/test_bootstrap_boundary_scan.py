from __future__ import annotations

import pytest

from shbt.core.bootstrap import DEFAULT_BRANCH, discover_kernel_from_bitlogic, scan_boundary_configurations


def test_scan_boundary_configurations_finds_benchmark_as_unique_global_maximum() -> None:
    landscape = scan_boundary_configurations()

    assert landscape.candidate_count == 39 * 20 * 399
    assert landscape.global_maximum.branch == DEFAULT_BRANCH
    assert landscape.global_maximum.topological_closure_penalty == pytest.approx(0.0, rel=0.0, abs=1.0e-15)
    assert landscape.unique_global_maximum is True
    assert landscape.topological_closure_survivors == (DEFAULT_BRANCH,)
    assert landscape.benchmark_is_mathematical_necessity is True
    assert discover_kernel_from_bitlogic() == DEFAULT_BRANCH


def test_scan_boundary_configurations_tracks_entropy_and_prime_index_support() -> None:
    landscape = scan_boundary_configurations(
        lepton_levels=range(24, 29),
        quark_levels=range(6, 11),
        parent_levels=range(310, 315),
    )
    benchmark = landscape.global_maximum
    detuned = next(candidate for candidate in landscape.candidates if candidate.branch == (26, 8, 313))

    assert benchmark.branch == DEFAULT_BRANCH
    assert benchmark.visible_support == 34
    assert benchmark.prime_index_support == (2, 3, 13)
    assert benchmark.information_entropy_bits > 0.0
    assert detuned.topological_closure_penalty > 0.0
    assert detuned.stability_fitness < benchmark.stability_fitness
    assert landscape.discovery_gap > 0.0
