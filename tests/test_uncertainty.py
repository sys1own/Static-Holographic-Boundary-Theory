from __future__ import annotations

from decimal import Decimal

from shbt.constants import AUDIT_TOLERANCE
from shbt.core.uncertainty import StochasticSensitivityEngine, build_stochastic_sensitivity_audit


def test_stochastic_sensitivity_engine_produces_transfer_function_and_confidence_intervals() -> None:
    audit = build_stochastic_sensitivity_audit(sample_count=32, seed=17, precision=160)

    assert audit.sample_count == 32
    assert audit.bit_loading_jitter_scale == Decimal("1e-77")
    assert audit.integer_drift_scale == Decimal("1e-80")
    assert audit.audit_tolerance == AUDIT_TOLERANCE
    assert "alpha_inverse_decimal" in audit.baseline_residues
    assert "GEOMETRIC_KAPPA" in audit.confidence_intervals
    assert "HOLOGRAPHIC_BITS" in audit.confidence_intervals
    assert "BENCHMARK_C_DARK_RESIDUE" in audit.confidence_intervals
    assert any(sample.effective_holographic_bits != audit.baseline_residues["HOLOGRAPHIC_BITS"] for sample in audit.samples)

    transfer = audit.jitter_transfer_function
    assert transfer.drift_scale == Decimal("1e-80")
    assert transfer.alpha_inverse_shifts["parent_level"] > 0
    assert transfer.alpha_inverse_shifts["lepton_level"] < 0
    assert transfer.alpha_inverse_shifts["quark_level"] < 0
    assert transfer.g_effective_shifts["parent_level"] < 0
    assert transfer.g_effective_shifts["lepton_level"] > 0
    assert transfer.g_effective_shifts["quark_level"] > 0

    assert audit.all_tier3_intervals_robust
    assert audit.max_fractional_shift < AUDIT_TOLERANCE


def test_stochastic_sensitivity_engine_is_seed_reproducible() -> None:
    left = StochasticSensitivityEngine(sample_count=12, seed=9, precision=160).audit_confidence_intervals()
    right = StochasticSensitivityEngine(sample_count=12, seed=9, precision=160).audit_confidence_intervals()

    assert left.confidence_intervals["HOLOGRAPHIC_BITS"].median_shift == right.confidence_intervals["HOLOGRAPHIC_BITS"].median_shift
    assert left.confidence_intervals["alpha_inverse_decimal"].upper_shift == right.confidence_intervals["alpha_inverse_decimal"].upper_shift
    assert left.jitter_transfer_function == right.jitter_transfer_function
