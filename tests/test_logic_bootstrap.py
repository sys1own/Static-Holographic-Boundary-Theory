from __future__ import annotations

from fractions import Fraction

from shbt.logic import evaluate_kernel
from shbt.logic.bootstrap import RadauIIA


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
