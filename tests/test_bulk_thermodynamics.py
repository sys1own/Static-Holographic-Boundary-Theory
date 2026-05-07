from __future__ import annotations

from decimal import Decimal

from shbt.bulk.thermodynamics import (
    DEFAULT_BOUNDARY_DIMENSION,
    DEFAULT_BULK_DIMENSION,
    DEFAULT_ITERATION_COUNT,
    EntropyEngine,
    build_thermodynamic_audit,
)


def test_entropy_engine_thermalizes_boundary_with_monotonic_entropy_growth() -> None:
    audit = EntropyEngine().simulate_time_evolution()

    assert audit.boundary_dimension == DEFAULT_BOUNDARY_DIMENSION
    assert audit.bulk_dimension == DEFAULT_BULK_DIMENSION
    assert len(audit.steps) == DEFAULT_ITERATION_COUNT
    assert audit.monotonic_entropy_growth is True
    assert audit.arrow_of_time_emerges is True
    assert audit.complexity_monotonic is True
    assert audit.second_law_mandatory is True
    assert audit.final_boundary_entropy_bits > Decimal("0")
    assert audit.final_bulk_entropy_bits > Decimal("0")
    assert audit.final_entanglement_complexity > Decimal("0")
    assert audit.final_time > Decimal("0")
    assert audit.statement == (
        "Time emerges as the monotonic thermodynamic residue of entanglement growth "
        "between the 26D boundary and the 4D bulk."
    )


def test_entropy_engine_maps_arrow_of_time_to_complexity_and_second_law_residue() -> None:
    audit = build_thermodynamic_audit(iterations=8)
    first_step = audit.steps[0]
    last_step = audit.steps[-1]

    assert first_step.transport_bit_residue == first_step.arrow_of_time_increment
    assert first_step.second_law_satisfied is True
    assert first_step.complexity_increases is True
    assert first_step.boundary_mode == audit.manifold_slice.dominant_loading_sequence[0]
    assert first_step.mode_entanglement_weight > Decimal("0")
    assert abs(first_step.second_law_residual) <= Decimal("1e-30")
    assert abs(first_step.complexity_time_residual) <= Decimal("1e-30")
    assert last_step.thermalization_fraction > first_step.thermalization_fraction
    projected_bulk_entropy = (
        last_step.cumulative_boundary_entropy_bits * Decimal(DEFAULT_BULK_DIMENSION) / Decimal(DEFAULT_BOUNDARY_DIMENSION)
    )
    assert abs(last_step.cumulative_bulk_entropy_bits - projected_bulk_entropy) <= Decimal("1e-27") * projected_bulk_entropy
