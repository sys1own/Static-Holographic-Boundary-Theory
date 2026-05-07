from __future__ import annotations

from decimal import Decimal

import pytest

from shbt.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.sectors.complexity import (
    CODON_LENGTH,
    GOLDEN_ANGLE_TURN_FRACTION,
    BiologicalComplexityAudit,
    build_biological_complexity_audit,
    build_biological_complexity_report,
)


@pytest.fixture(scope="module")
def biological_audit() -> BiologicalComplexityAudit:
    return build_biological_complexity_audit()


def test_biological_complexity_maps_kernel_into_dna_and_phyllotaxis(
    biological_audit: BiologicalComplexityAudit,
) -> None:
    assert biological_audit.branch == (LEPTON_LEVEL, QUARK_LEVEL, PARENT_LEVEL)
    assert biological_audit.dna_error_correction.codon_length == CODON_LENGTH == 3
    assert biological_audit.dna_error_correction.codon_state_count == 64
    assert biological_audit.dna_error_correction.branch_code_volume == 26 * 8 * 312
    assert biological_audit.dna_error_correction.logical_codon_packets == 1014
    assert biological_audit.dna_error_correction.dna_isomorphic is True
    assert biological_audit.fibonacci_phyllotaxis.golden_angle_turn_fraction == GOLDEN_ANGLE_TURN_FRACTION
    assert biological_audit.fibonacci_phyllotaxis.phyllotaxis_locked is True
    assert biological_audit.biological_limits_locked is True
    assert all(isomorphism.locked for isomorphism in biological_audit.isomorphisms)


def test_biological_information_processing_ceiling_is_rigidity_limited(
    biological_audit: BiologicalComplexityAudit,
) -> None:
    processing = biological_audit.information_processing
    shannon_limit = biological_audit.complexity_sector.shannon_limit
    mass_step_ratio = Decimal(str(biological_audit.complexity_sector.flavor_identity.mass_step_ratio))

    assert abs(processing.mass_hierarchy_lock_fraction - (Decimal(1) / mass_step_ratio)) <= Decimal("1e-27")
    assert processing.maximum_biological_processing_ops_per_second < processing.complexity_growth_rate_ops_per_second
    assert processing.biological_logical_storage_bits < shannon_limit.maximum_logical_storage_bits
    assert abs(
        processing.biological_error_corrected_base_pairs * shannon_limit.shannon_bits_per_nucleotide
        - processing.biological_logical_storage_bits
    ) <= Decimal("1e-27") * processing.biological_logical_storage_bits
    assert processing.upper_bound_verified is True


def test_biological_complexity_report_exposes_biological_limit_sections() -> None:
    report = build_biological_complexity_report()

    assert "Biological Complexity Audit" in report
    assert "Kernel-to-Biology Isomorphisms" in report
    assert "DNA Error-Correction Audit" in report
    assert "Fibonacci Phyllotaxis Audit" in report
    assert "Biological Information-Processing Ceiling" in report
    assert "Biological limits locked             : True" in report
