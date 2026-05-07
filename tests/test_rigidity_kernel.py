from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from shbt.constants import HOLOGRAPHIC_BITS, LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.rigidity_kernel import (
    BENCHMARK_PARITY_CHECK_MATRIX,
    BENCHMARK_MOAT_CENTER,
    GlobalGaugeGuardian,
    HolographicErrorCorrection,
    audit_information_conservation,
    build_guardian_report,
    build_perturbed_guardian_vacuum,
    derive_guarded_cosmology_audit,
    derive_guarded_flavor_audit,
    derive_guarded_gravity_audit,
    guardian_audit_of,
    stabilize_boundary,
)
from shbt.main import SECTOR_AUDIT_MODULES, TopologicalVacuum
from shbt.runtime_config import PhysicalSingularityException


@dataclass(frozen=True)
class _DriftPayload:
    branch: tuple[int, int, int]
    vacuum: TopologicalVacuum


def test_runtime_guardian_recenters_all_sector_derivations_and_conserves_bits() -> None:
    noisy_vacuum = build_perturbed_guardian_vacuum()

    derive_guarded_cosmology_audit(noisy_vacuum)
    derive_guarded_flavor_audit(noisy_vacuum)
    derive_guarded_gravity_audit(noisy_vacuum)

    audits = (
        guardian_audit_of(derive_guarded_cosmology_audit),
        guardian_audit_of(derive_guarded_flavor_audit),
        guardian_audit_of(derive_guarded_gravity_audit),
    )

    for audit in audits:
        assert audit.corrected_branch == BENCHMARK_MOAT_CENTER == (LEPTON_LEVEL, QUARK_LEVEL, PARENT_LEVEL)
        assert audit.correction_applied is True
        assert audit.input_bit_count == pytest.approx(HOLOGRAPHIC_BITS)
        assert audit.output_bit_count == pytest.approx(HOLOGRAPHIC_BITS)
        assert audit.information_conserved is True
        assert audit.syndrome is not None

    assert audits[0].result_branch == BENCHMARK_MOAT_CENTER
    assert audits[1].result_branch == BENCHMARK_MOAT_CENTER


def test_holographic_error_correction_uses_benchmark_branch_as_parity_check_matrix() -> None:
    kernel = HolographicErrorCorrection()
    noisy_vacuum = build_perturbed_guardian_vacuum()
    syndrome = kernel.audit_candidate(noisy_vacuum)
    corrected_vacuum = kernel.correct_candidate(noisy_vacuum)

    assert kernel.parity_check_matrix == BENCHMARK_PARITY_CHECK_MATRIX == ((26.0, 0.0, 0.0), (0.0, 8.0, 0.0), (0.0, 0.0, 312.0))
    assert syndrome.correction_applied is True
    assert syndrome.corrected_branch == BENCHMARK_MOAT_CENTER
    assert syndrome.zero_syndrome is False
    assert corrected_vacuum.target_tuple == BENCHMARK_MOAT_CENTER


def test_stabilize_boundary_feedback_loop_recenters_drifted_sector_payloads() -> None:
    @stabilize_boundary(sector="feedback")
    def emit_drifted_payload(vacuum: TopologicalVacuum | None = None) -> _DriftPayload:
        del vacuum
        return _DriftPayload(
            branch=(27, 9, 311),
            vacuum=build_perturbed_guardian_vacuum(),
        )

    payload = emit_drifted_payload()
    audit = guardian_audit_of(emit_drifted_payload)

    assert payload.branch == BENCHMARK_MOAT_CENTER
    assert payload.vacuum.target_tuple == BENCHMARK_MOAT_CENTER
    assert audit.result_branch == BENCHMARK_MOAT_CENTER
    assert audit.correction_applied is True


def test_information_conservation_audit_tracks_guardian_projection() -> None:
    audit = audit_information_conservation()

    assert audit.benchmark_branch == BENCHMARK_MOAT_CENTER == (26, 8, 312)
    assert audit.input_branch != BENCHMARK_MOAT_CENTER
    assert audit.input_bit_count == pytest.approx(HOLOGRAPHIC_BITS)
    assert [sector_audit.sector for sector_audit in audit.sector_audits] == ["cosmology", "flavor", "gravity"]
    assert audit.information_conserved is True
    assert audit.global_gauge_guardian_active is True
    assert audit.self_healing_locked is True
    assert "quantum error-correcting code" in audit.statement


def test_rigidity_guardian_is_wired_into_rigidity_sector_and_report() -> None:
    assert "shbt.core.rigidity_kernel" in SECTOR_AUDIT_MODULES["rigidity"]

    report = build_guardian_report()
    assert "Rigidity Guardian Audit" in report
    assert "Information conserved        : True" in report
    assert "Global Gauge Guardian        : True" in report
    assert "Mathematically locked        : True" in report


def test_global_gauge_guardian_heals_flavor_instability_to_fixed_point() -> None:
    guardian = GlobalGaugeGuardian()
    benchmark = derive_guarded_flavor_audit(TopologicalVacuum())
    unstable = replace(
        benchmark,
        branch=(27, 9, 311),
        beta_phase_lock=float("nan"),
        ratio_fractional_error=float("inf"),
    )

    healed, audit = guardian.stabilize_sector_result(unstable, sector="flavor")

    assert healed.branch == BENCHMARK_MOAT_CENTER
    assert healed.mandatory_residue_verified is True
    assert audit.instability_detected is True
    assert audit.fixed_point_forced is True
    assert audit.locked_to_fixed_point is True
    assert audit.trigger is not None
    assert "Global Gauge Guardian" in audit.statement


def test_global_gauge_guardian_heals_gravity_instability_to_fixed_point() -> None:
    guardian = GlobalGaugeGuardian()
    unstable = replace(
        TopologicalVacuum().verify_bulk_emergence(),
        gmunu_consistency_score=float("nan"),
        bulk_emergent=False,
        parity_bit_density_constraint_satisfied=False,
        torsion_free=False,
        non_singular_bulk=False,
        lambda_aligned=False,
    )

    healed, audit = guardian.stabilize_sector_result(unstable, sector="gravity")

    assert healed.bulk_emergent is True
    assert healed.parity_bit_density_constraint_satisfied is True
    assert healed.torsion_free is True
    assert healed.non_singular_bulk is True
    assert healed.lambda_aligned is True
    assert audit.instability_detected is True
    assert audit.fixed_point_forced is True
    assert audit.locked_to_fixed_point is True


def test_stabilize_boundary_recovers_from_gravity_exception_with_fixed_point() -> None:
    @stabilize_boundary(sector="gravity")
    def emit_unstable_gravity(vacuum: TopologicalVacuum | None = None):
        del vacuum
        raise PhysicalSingularityException("synthetic gravity instability")

    healed = emit_unstable_gravity(build_perturbed_guardian_vacuum())
    audit = guardian_audit_of(emit_unstable_gravity)

    assert healed.bulk_emergent is True
    assert healed.lambda_aligned is True
    assert audit.gauge_guardian is not None
    assert audit.gauge_guardian.recovered_from_exception is True
    assert audit.gauge_guardian.fixed_point_forced is True
    assert audit.gauge_guardian.locked_to_fixed_point is True
