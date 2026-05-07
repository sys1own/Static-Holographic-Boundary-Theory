from __future__ import annotations

import pytest

from shbt.constants import HOLOGRAPHIC_BITS, LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.rigidity_kernel import (
    BENCHMARK_MOAT_CENTER,
    audit_information_conservation,
    build_guardian_report,
    build_perturbed_guardian_vacuum,
    derive_guarded_cosmology_audit,
    derive_guarded_flavor_audit,
    derive_guarded_gravity_audit,
    guardian_audit_of,
)
from shbt.main import SECTOR_AUDIT_MODULES


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
