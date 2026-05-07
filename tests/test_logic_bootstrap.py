from __future__ import annotations

from fractions import Fraction

from shbt.logic import evaluate_kernel
from shbt.logic.bootstrap import RadauIIA, build_uniqueness_report


def test_evaluate_kernel_certifies_benchmark_topological_closure() -> None:
    result = evaluate_kernel(26, 8, 312)

    assert result == {"stability_score": Fraction(1, 1), "is_closed": True}


def test_evaluate_kernel_rejects_detuned_kernel() -> None:
    result = evaluate_kernel(26, 8, 313)

    assert result["stability_score"] < Fraction(1, 1)
    assert result["is_closed"] is False


def test_radau_iia_solver_matches_public_helper() -> None:
    solver = RadauIIA(dimension=26, generation=8, nodes=312)

    assert solver.solve() == evaluate_kernel(26, 8, 312)


def test_build_uniqueness_report_selects_global_maximum_stability() -> None:
    scan_results = [
        {"dimension": 26, "generation": 8, "nodes": 312, **evaluate_kernel(26, 8, 312)},
        {"dimension": 26, "generation": 8, "nodes": 313, **evaluate_kernel(26, 8, 313)},
        ((26, 9, 312), evaluate_kernel(26, 9, 312)),
    ]

    report = build_uniqueness_report(scan_results)

    assert report["discovered_kernel"]["coordinates"] == (26, 8, 312)
    assert report["maximum_stability_score"] == Fraction(1, 1)
    assert report["benchmark_uniquely_discovered"] is True
    assert report["degenerate_branch_count"] == 0


def test_build_uniqueness_report_flags_degenerate_branch_within_benchmark_tolerance() -> None:
    benchmark_result = evaluate_kernel(26, 8, 312)
    near_benchmark_score = benchmark_result["stability_score"] - Fraction(1, 10**16)
    scan_results = [
        {"dimension": 26, "generation": 8, "nodes": 312, **benchmark_result},
        {
            "dimension": 25,
            "generation": 8,
            "nodes": 312,
            "stability_score": near_benchmark_score,
            "is_closed": False,
        },
    ]

    report = build_uniqueness_report(scan_results)

    assert report["degenerate_branch_detected"] is True
    assert report["degenerate_branch_count"] == 1
    assert report["degenerate_branches"][0]["coordinates"] == (25, 8, 312)
    assert report["degenerate_branches"][0]["branch_status"] == "Degenerate Branch"
    assert report["benchmark_uniquely_discovered"] is False
    assert report["mathematical_necessity"] is False
