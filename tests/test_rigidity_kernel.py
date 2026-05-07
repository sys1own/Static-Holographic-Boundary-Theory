from __future__ import annotations

from dataclasses import dataclass

import pytest

from shbt.constants import HOLOGRAPHIC_BITS, LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.rigidity_kernel import (
    BENCHMARK_PARITY_CHECK_MATRIX,
    BENCHMARK_MOAT_CENTER,
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
    assert "quantum error-correcting code" in audit.statement


def test_rigidity_guardian_is_wired_into_rigidity_sector_and_report() -> None:
    assert "shbt.core.rigidity_kernel" in SECTOR_AUDIT_MODULES["rigidity"]

    report = build_guardian_report()
    assert "Rigidity Guardian Audit" in report
    assert "Information conserved        : True" in report
