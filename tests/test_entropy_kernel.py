from __future__ import annotations

from decimal import Decimal, localcontext

import pytest

from shbt.constants import AUDIT_TOLERANCE, PLANCK_LENGTH_M
from shbt.core.entropy_kernel import (
    DEFAULT_PRECISION,
    analyze_von_neumann_entropy,
    bekenstein_hawking_bound,
    benchmark_holographic_projection,
)


def test_bekenstein_hawking_bound_reduces_to_area_law_unit_cell() -> None:
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION
        area = Decimal(4) * Decimal(str(PLANCK_LENGTH_M)) * Decimal(str(PLANCK_LENGTH_M))

    bound = bekenstein_hawking_bound(area)

    assert abs(bound.entropy_nats - Decimal(1)) < AUDIT_TOLERANCE
    assert float(bound.entropy_bits) == pytest.approx(1.0 / 0.6931471805599453, rel=0.0, abs=1.0e-12)
    assert float(bound.surface_bit_loading_bits_per_m2) == pytest.approx(
        float(bound.entropy_bits / area),
        rel=0.0,
        abs=1.0e-12,
    )


def test_von_neumann_entropy_distinguishes_pure_and_mixed_qubits() -> None:
    mixed = analyze_von_neumann_entropy([[0.5, 0.0], [0.0, 0.5]])
    pure = analyze_von_neumann_entropy([[1.0, 0.0], [0.0, 0.0]])

    assert mixed.entropy_bits == pytest.approx(1.0, rel=0.0, abs=1.0e-12)
    assert mixed.purity == pytest.approx(0.5, rel=0.0, abs=1.0e-12)
    assert pure.entropy_bits == pytest.approx(0.0, rel=0.0, abs=1.0e-12)
    assert pure.purity == pytest.approx(1.0, rel=0.0, abs=1.0e-12)


def test_benchmark_holographic_projection_locks_surface_loading_to_bulk_density() -> None:
    projection = benchmark_holographic_projection(
        boundary_area_m2=Decimal("10"),
        bulk_volume_m3=Decimal("25"),
        occupied_surface_fraction=Decimal("0.4"),
    )

    assert projection.occupied_surface_bits == projection.available_surface_bits * Decimal("0.4")
    assert projection.radial_information_depth_m == Decimal("2.5")
    assert projection.projection_identity_holds is True
    assert float(projection.bulk_information_density_bits_per_m3 * projection.radial_information_depth_m) == pytest.approx(
        float(projection.surface_bit_loading_bits_per_m2),
        rel=0.0,
        abs=1.0e-12,
    )
