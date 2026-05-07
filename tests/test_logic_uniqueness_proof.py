from __future__ import annotations

from fractions import Fraction

import pytest

from shbt.logic import BENCHMARK_PRIME_INDEX, BENCHMARK_REALITY_KERNEL, generate_divergence_report


def test_generate_divergence_report_certifies_benchmark_kernel() -> None:
    report = generate_divergence_report()

    assert report.benchmark_coordinates == BENCHMARK_REALITY_KERNEL
    assert report.evaluated_coordinates == BENCHMARK_REALITY_KERNEL
    assert report.prime_index_coordinate == BENCHMARK_PRIME_INDEX == Fraction(13, 1)
    assert report.moat_divergence == Fraction(0, 1)
    assert report.c_dark_shift == Fraction(0, 1)
    assert report.diophantine_gap == Fraction(0, 1)
    assert report.stability_score == Fraction(1, 1)
    assert report.singular_divergence == Fraction(0, 1)
    assert report.information_retention == Fraction(1, 1)
    assert report.information_loss == Fraction(0, 1)
    assert report.distinction_logic.distinction_jacobian_determinant == Fraction(1, 1)
    assert report.distinction_logic.information_map_invertible is True
    assert report.sustains_reality is True
    assert report.failure_modes == ()


@pytest.mark.parametrize("dimension", [24, 28])
def test_non_prime_non_26d_alternatives_undergo_singular_divergence(dimension: int) -> None:
    report = generate_divergence_report(dimension=dimension, generation=8, nodes=312)

    assert report.evaluated_coordinates == (dimension, 8, 312)
    assert report.prime_index_coordinate == Fraction(dimension, 2)
    assert report.distinction_logic.dimension_locked is False
    assert report.distinction_logic.prime_index_integral is True
    assert report.distinction_logic.prime_index_prime is False
    assert report.prime_index_singularity == Fraction(1, 1)
    assert report.singular_divergence > Fraction(0, 1)
    assert report.information_loss > Fraction(0, 1)
    assert report.distinction_logic.distinction_jacobian_determinant == Fraction(0, 1)
    assert report.non_invertible_information_loss is True
    assert report.sustains_reality is False
    assert any("26D" in failure_mode for failure_mode in report.failure_modes)
    assert any("not prime" in failure_mode for failure_mode in report.failure_modes)


@pytest.mark.parametrize(
    ("dimension", "generation", "nodes"),
    [
        (26, 8, 313),
        (26, 9, 312),
    ],
)
def test_any_drift_from_benchmark_fixed_point_causes_non_invertible_information_loss(
    dimension: int,
    generation: int,
    nodes: int,
) -> None:
    report = generate_divergence_report(dimension=dimension, generation=generation, nodes=nodes)

    assert report.drift_from_fixed_point is True
    assert report.distinction_logic.dimension_locked is True
    assert report.distinction_logic.prime_index_prime is True
    assert report.distinction_logic.topological_closure_locked is False
    assert report.distinction_logic.distinction_jacobian_determinant == Fraction(0, 1)
    assert report.information_loss > Fraction(0, 1)
    assert report.non_invertible_information_loss is True
    assert report.sustains_reality is False
    assert any("Topological closure failed" in failure_mode for failure_mode in report.failure_modes)
