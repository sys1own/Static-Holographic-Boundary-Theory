from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from shbt.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.projector import (
    HolographicCompiler,
    PrimeIndexedInformationLattice,
    PrimeIndexedLatticeSite,
    actualize_boundary_determined_bulk,
    build_benchmark_prime_indexed_lattice,
)


def test_holographic_compiler_builds_prime_indexed_benchmark_lattice() -> None:
    lattice = build_benchmark_prime_indexed_lattice()

    assert lattice.branch == (LEPTON_LEVEL, QUARK_LEVEL, PARENT_LEVEL)
    assert lattice.primes == (2, 3, 5, 7, 11)
    assert lattice.labels == (
        "visible_support_density",
        "central_charge_ratio",
        "inverse_pixel_volume",
        "vacuum_pressure",
        "geometric_kappa",
    )
    assert lattice.state_change_count == 4
    assert all(weight > 0 for weight in lattice.normalized_state)
    assert sum(lattice.normalized_state, Decimal("0")) == Decimal("1")


def test_euler_fixed_point_adjunctions_actualize_3_plus_1_bulk_metric() -> None:
    geometry = actualize_boundary_determined_bulk()

    assert len(geometry.adjunctions) == 4
    assert geometry.spacetime_dimension == 4
    assert geometry.spatial_dimension == 3
    assert geometry.metric_rank == 4
    assert geometry.emergent_from_execution is True
    assert geometry.boundary_determines_bulk is True
    assert geometry.execution_residue >= 0
    assert geometry.resolution_scale > 0
    assert geometry.statement.startswith("The 3+1 bulk metric actualizes")
    assert np.allclose(geometry.spacetime_metric.components, geometry.spacetime_metric.components.T)
    assert np.all(np.linalg.eigvalsh(geometry.spacetime_metric.components) > 0)
    assert np.all(np.linalg.eigvalsh(geometry.spatial_metric.components) > 0)
    assert all(adjunction.adjunction_strength > 0 for adjunction in geometry.adjunctions)
    assert all(adjunction.metric_increment.shape == (4, 4) for adjunction in geometry.adjunctions)


def test_boundary_drift_changes_execution_defined_bulk_geometry() -> None:
    compiler = HolographicCompiler()
    baseline_lattice = compiler.build_benchmark_lattice()
    drifted_sites = list(baseline_lattice.sites)
    drifted_sites[0] = replace(
        drifted_sites[0],
        amplitude=drifted_sites[0].amplitude + Decimal("1e-15"),
    )
    drifted_lattice = PrimeIndexedInformationLattice(
        vacuum=baseline_lattice.vacuum,
        sites=tuple(drifted_sites),
    )

    baseline_geometry = compiler.actualize_bulk_geometry(baseline_lattice)
    drifted_geometry = compiler.actualize_bulk_geometry(drifted_lattice)

    assert drifted_geometry.execution_residue != baseline_geometry.execution_residue
    assert not np.array_equal(drifted_geometry.spacetime_metric.components, baseline_geometry.spacetime_metric.components)


def test_prime_indexed_information_lattice_requires_five_sites_for_3_plus_1_actualization() -> None:
    with pytest.raises(ValueError, match="requires at least five prime-indexed lattice sites"):
        PrimeIndexedInformationLattice(
            vacuum=build_benchmark_prime_indexed_lattice().vacuum,
            sites=(
                PrimeIndexedLatticeSite(2, "a", Decimal("1")),
                PrimeIndexedLatticeSite(3, "b", Decimal("1")),
                PrimeIndexedLatticeSite(5, "c", Decimal("1")),
                PrimeIndexedLatticeSite(7, "d", Decimal("1")),
            ),
        )
