from __future__ import annotations

from decimal import Decimal

import pytest

from shbt.constants import HOLOGRAPHIC_BITS, LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.main import SECTOR_AUDIT_MODULES, TARGET_AUDIT_SECTORS
from shbt.sectors.complexity_sector import (
    DUPLEX_COMPLEMENT_EFFICIENCY,
    PRIME_SYNC_SPREAD_TOLERANCE,
    SHANNON_BITS_PER_NUCLEOTIDE,
    ComplexitySectorAudit,
    build_complexity_sector_audit,
    build_complexity_sector_report,
)


@pytest.fixture(scope="module")
def complexity_audit() -> ComplexitySectorAudit:
    return build_complexity_sector_audit()


def test_complexity_sector_is_registered_for_targeted_audits() -> None:
    assert "complexity" in TARGET_AUDIT_SECTORS
    assert SECTOR_AUDIT_MODULES["complexity"] == ("shbt.sectors.complexity_sector",)


def test_complexity_sector_maps_branch_locked_mass_ladder_into_prime_windows(
    complexity_audit: ComplexitySectorAudit,
) -> None:
    assert complexity_audit.branch == (LEPTON_LEVEL, QUARK_LEVEL, PARENT_LEVEL)
    assert complexity_audit.flavor_identity.mandatory_residue_verified
    assert complexity_audit.prime_sync.algebraic_rigidity_preserved
    assert complexity_audit.prime_sync.prime_sync_verified
    assert complexity_audit.prime_sync.max_window_coordinate_spread <= PRIME_SYNC_SPREAD_TOLERANCE
    assert [window.generation for window in complexity_audit.prime_sync.windows] == [1, 2]
    assert all(Decimal("0") < window.normalized_window_coordinate < Decimal("1") for window in complexity_audit.prime_sync.windows)
    assert complexity_audit.sector_passed


def test_complexity_sector_shannon_limit_is_bounded_by_holographic_register(
    complexity_audit: ComplexitySectorAudit,
) -> None:
    shannon_limit = complexity_audit.shannon_limit

    assert shannon_limit.holographic_bits == Decimal(str(HOLOGRAPHIC_BITS))
    assert shannon_limit.shannon_bits_per_nucleotide == SHANNON_BITS_PER_NUCLEOTIDE
    assert shannon_limit.duplex_complement_efficiency == DUPLEX_COMPLEMENT_EFFICIENCY
    assert shannon_limit.maximum_logical_storage_efficiency == (
        DUPLEX_COMPLEMENT_EFFICIENCY * shannon_limit.complexity_utilization_fraction
    )
    assert shannon_limit.maximum_logical_storage_bits == (
        shannon_limit.holographic_bits * shannon_limit.maximum_logical_storage_efficiency
    )
    assert (
        shannon_limit.maximum_error_corrected_base_pairs * SHANNON_BITS_PER_NUCLEOTIDE
        == shannon_limit.maximum_logical_storage_bits
    )
    assert shannon_limit.maximum_logical_storage_bits < shannon_limit.holographic_bits
    assert shannon_limit.within_holographic_budget
    assert shannon_limit.holographic_ceiling_verified


def test_complexity_sector_report_exposes_prime_sync_and_shannon_sections() -> None:
    report = build_complexity_sector_report()

    assert "Complexity Sector Audit" in report
    assert "Prime-Sync Biological Error-Correction Map" in report
    assert "Shannon Entropy Limit Audit" in report
    assert "Sector verdict" in report
