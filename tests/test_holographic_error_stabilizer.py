from __future__ import annotations

import pytest

from shbt.core.holographic_error_stabilizer import (
    BENCHMARK_BRANCH,
    FAILED_BRANCHES,
    HolographicStabilizer,
    build_holographic_error_stabilizer_audit,
    simulate_charge_bit_flip,
    simulate_failed_checksum,
)


def test_benchmark_branch_is_self_correcting_error_stabilizer() -> None:
    audit = build_holographic_error_stabilizer_audit()

    assert audit.branch == BENCHMARK_BRANCH == (26, 8, 312)
    assert audit.charge.passed
    assert audit.time_reversal.passed
    assert audit.momentum.passed
    assert audit.parity.passed
    assert audit.zero_energy_state_locked
    assert audit.equivalence_principle_preserved
    assert audit.self_correcting
    assert audit.recovery.recovery_threshold_ev > 0
    assert audit.recovery.dimensionless_syndrome_threshold > 0
    assert audit.reference_failure.closure_equivalence_verified


def test_holographic_stabilizer_bulk_integrity_locks_benchmark_branch() -> None:
    stabilizer = HolographicStabilizer()
    verification = stabilizer.verify_bulk_integrity()

    assert verification.benchmark_branch == BENCHMARK_BRANCH
    assert verification.charge_checksum_passed
    assert verification.time_reversal_checksum_passed
    assert verification.momentum_checksum_passed
    assert verification.parity_checksum_passed
    assert verification.zero_energy_boundary_locked
    assert verification.equivalence_principle_preserved
    assert verification.torsion_projection_residual == 0
    assert len(verification.checksums) == 3
    assert verification.passed
    assert verification == stabilizer.verify_bulk_checksum()


def test_holographic_stabilizer_can_simulate_boundary_decoherence() -> None:
    verification = HolographicStabilizer(simulate_boundary_decoherence=True).verify_bulk_integrity()

    assert verification.charge_checksum_passed
    assert verification.time_reversal_checksum_passed
    assert verification.momentum_checksum_passed
    assert verification.parity_checksum_passed
    assert verification.torsion_projection_residual > 0
    assert not verification.equivalence_principle_preserved
    assert not verification.passed


def test_charge_bit_flip_reintroduces_torsion_and_breaks_equivalence_principle() -> None:
    simulation = simulate_charge_bit_flip()

    assert simulation.charge_bit_flip
    assert simulation.failed_branch == FAILED_BRANCHES["charge"]
    assert not simulation.recoverable
    assert simulation.torsion_reintroduced
    assert simulation.equivalence_principle_destroyed
    assert simulation.zero_energy_boundary_required


def test_momentum_alias_resolves_to_time_reversal_checksum() -> None:
    simulation = simulate_failed_checksum(law="momentum")

    assert simulation.law == "time_reversal"
    assert simulation.failed_branch == FAILED_BRANCHES["time_reversal"]


@pytest.mark.parametrize("law", ("charge", "time_reversal", "parity"))
def test_failed_checksums_reintroduce_torsion_and_destroy_equivalence_principle(law: str) -> None:
    simulation = simulate_failed_checksum(law=law)

    assert simulation.failed_branch == FAILED_BRANCHES[law]
    assert simulation.syndrome_energy_ev > simulation.recovery_budget_ev
    assert not simulation.recoverable
    assert simulation.failed_closure_tensor.amplitude > 0
    assert simulation.torsion_reintroduced
    assert simulation.effective_anomalous_source_ev2.amplitude > 0
    assert simulation.effective_anomalous_source_si_m2.amplitude > 0
    assert simulation.equivalence_principle_destroyed
