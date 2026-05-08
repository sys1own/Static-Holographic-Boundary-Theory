from __future__ import annotations

import pytest

from shbt.core.correspondence_engine import PointerStateDecoherenceError, PointerStateSelector
import shbt.core.noether_bridge as noether_bridge
from shbt.core.holographic_error_stabilizer import BulkChecksumVerification, HolographicStabilizer
from shbt.core.saturation import SaturationAudit


def test_branch_planck_mass_ev_requires_coherent_bulk_checksum() -> None:
    planck_mass = noether_bridge.branch_planck_mass_ev()

    assert planck_mass > 0


def test_branch_planck_mass_ev_calls_holographic_stabilizer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"init": [], "verify": 0}

    class TrackingStabilizer:
        def __init__(self, *, precision: int, simulate_boundary_decoherence: bool) -> None:
            calls["init"].append((precision, simulate_boundary_decoherence))

        def verify_bulk_checksum(self) -> BulkChecksumVerification:
            calls["verify"] += 1
            return BulkChecksumVerification(
                benchmark_branch=(26, 8, 312),
                charge_residual=noether_bridge.Decimal("0"),
                momentum_residual=noether_bridge.Decimal("0"),
                parity_residual=noether_bridge.Decimal("0"),
                charge_checksum_passed=True,
                momentum_checksum_passed=True,
                parity_checksum_passed=True,
                simulated_boundary_decoherence=False,
                detail="bulk checksum locked",
            )

    monkeypatch.setattr(noether_bridge, "HolographicStabilizer", TrackingStabilizer)

    assert noether_bridge.branch_planck_mass_ev() > 0
    assert calls["init"] == [(noether_bridge.DEFAULT_PRECISION, False)]
    assert calls["verify"] == 2


def test_branch_planck_mass_ev_raises_on_simulated_boundary_decoherence() -> None:
    with pytest.raises(noether_bridge.DecoherenceError, match=r"bulk checksum failed"):
        noether_bridge.branch_planck_mass_ev(simulate_boundary_decoherence=True)


def test_branch_planck_mass_ev_raises_on_off_shell_coordinates_after_checksum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def _verified_checksum(self: HolographicStabilizer) -> BulkChecksumVerification:
        calls["count"] += 1
        return BulkChecksumVerification(
            benchmark_branch=(26, 8, 312),
            charge_residual=noether_bridge.Decimal("0"),
            momentum_residual=noether_bridge.Decimal("0"),
            parity_residual=noether_bridge.Decimal("0"),
            charge_checksum_passed=True,
            momentum_checksum_passed=True,
            parity_checksum_passed=True,
            simulated_boundary_decoherence=False,
            detail="bulk checksum locked",
        )

    monkeypatch.setattr(HolographicStabilizer, "verify_bulk_checksum", _verified_checksum)

    with pytest.raises(
        noether_bridge.DecoherenceError,
        match=r"Boundary instability detected: Physical scale emergence inhibited",
    ):
        noether_bridge.branch_planck_mass_ev(lepton_level=27)

    assert calls["count"] == 1


def test_newton_constant_lock_is_gate_kept_by_holographic_stabilizer(monkeypatch: pytest.MonkeyPatch) -> None:
    def _failed_checksum(self: HolographicStabilizer) -> BulkChecksumVerification:
        return BulkChecksumVerification(
            benchmark_branch=(26, 8, 312),
            charge_residual=noether_bridge.Decimal("0"),
            momentum_residual=noether_bridge.Decimal("0"),
            parity_residual=noether_bridge.Decimal("0"),
            charge_checksum_passed=True,
            momentum_checksum_passed=True,
            parity_checksum_passed=True,
            simulated_boundary_decoherence=True,
            detail="simulated boundary decoherence",
        )

    monkeypatch.setattr(HolographicStabilizer, "verify_bulk_checksum", _failed_checksum)

    with pytest.raises(noether_bridge.DecoherenceError, match=r"simulated boundary decoherence"):
        noether_bridge.newton_constant_lock()


def test_newton_constant_lock_raises_on_detuned_branch_coordinates() -> None:
    with pytest.raises(noether_bridge.DecoherenceError, match=r"off-shell branch coordinates"):
        noether_bridge.newton_constant_lock(quark_level=9)


def test_branch_planck_mass_ev_raises_when_pointer_state_selection_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _reject(self: PointerStateSelector, wavefunction, observable):
        del wavefunction, observable
        raise PointerStateDecoherenceError("No pointer state satisfies the boundary parity-neutrality checksum c_vis + c_dark = 0.")

    monkeypatch.setattr(PointerStateSelector, "crystallize_classical_observable", _reject)

    with pytest.raises(noether_bridge.DecoherenceError, match=r"Pointer-state selector rejected discrete mass-scale emergence"):
        noether_bridge.branch_planck_mass_ev()


def test_high_precision_unity_snapshot_raises_when_pointer_state_selection_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _reject(self: PointerStateSelector, wavefunction, observable):
        del wavefunction, observable
        raise PointerStateDecoherenceError("No pointer state satisfies the boundary parity-neutrality checksum c_vis + c_dark = 0.")

    monkeypatch.setattr(PointerStateSelector, "crystallize_classical_observable", _reject)

    with pytest.raises(noether_bridge.DecoherenceError, match=r"Pointer-state selector rejected discrete mass-scale emergence"):
        noether_bridge.high_precision_unity_of_scale_snapshot(kappa_d5=noether_bridge.derive_kappa_d5())


def test_reviewer_trap_separates_benchmark_lock_from_detuned_stress() -> None:
    report = noether_bridge.build_gravity_side_rigidity_report()
    reviewer = report.reviewer_trap

    assert reviewer.q_iso_ev4 == reviewer.sigma_holo_ev4
    assert reviewer.benchmark_sigma_balance_residual_ev4 == noether_bridge.Decimal("0")
    assert reviewer.topological_fixed_point_pressure_balanced
    assert reviewer.benchmark_lock_verified
    assert reviewer.detuned_stress_verified
    assert reviewer.closure_equivalence_verified
    assert reviewer.equivalence_principle_preserved


def test_render_report_marks_benchmark_lock_verified() -> None:
    report = noether_bridge.build_gravity_side_rigidity_report()
    rendered = noether_bridge.render_report(report)

    assert "Benchmark lock                  : VERIFIED" in rendered
    assert "Detuned stress                  : EXPECTED RESIDUE" in rendered
    assert "Equivalence Principle           : VERIFIED" in rendered
    assert "Saturation status              : PASS (within noise floor)" in rendered


def test_saturation_audit_treats_reconstructed_n_as_boundary_condition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    permissive = noether_bridge.SaturationAudit(
        lambda_obs_si_m2=noether_bridge.Decimal("1.0e-52"),
        lambda_obs_ev2=noether_bridge.Decimal("1.0e-66"),
        holographic_bits_from_lambda=noether_bridge.Decimal("3.31e122"),
        configured_holographic_bits=noether_bridge.Decimal("3.10e122"),
        register_noise_floor=noether_bridge.Decimal("3.0e-123"),
        relative_mismatch=noether_bridge.Decimal("1.0e-2"),
    )

    assert noether_bridge.TREAT_N_AS_BOUNDARY_CONDITION is True
    assert permissive.boundary_condition_locked
    assert permissive.is_saturated is False

    monkeypatch.setattr(noether_bridge, "TREAT_N_AS_BOUNDARY_CONDITION", False)

    strict = noether_bridge.SaturationAudit(
        lambda_obs_si_m2=noether_bridge.Decimal("1.0e-52"),
        lambda_obs_ev2=noether_bridge.Decimal("1.0e-66"),
        holographic_bits_from_lambda=noether_bridge.Decimal("3.31e122"),
        configured_holographic_bits=noether_bridge.Decimal("3.10e122"),
        register_noise_floor=noether_bridge.Decimal("3.0e-123"),
        relative_mismatch=noether_bridge.Decimal("1.0e-2"),
    )

    assert strict.boundary_condition_locked is False


def test_saturation_audit_marks_negligible_mismatch_as_saturated() -> None:
    saturated = noether_bridge.SaturationAudit(
        lambda_obs_si_m2=noether_bridge.Decimal("1.0e-52"),
        lambda_obs_ev2=noether_bridge.Decimal("1.0e-66"),
        holographic_bits_from_lambda=noether_bridge.Decimal("3.31e122"),
        configured_holographic_bits=noether_bridge.Decimal("3.31e122"),
        register_noise_floor=noether_bridge.Decimal("3.0e-123"),
        relative_mismatch=noether_bridge.Decimal("1.0e-16"),
    )

    assert saturated.success is True
    assert saturated.passed is True
    assert saturated.is_saturated


def test_saturation_audit_success_respects_noise_floor_boundary() -> None:
    threshold = noether_bridge.Decimal(str(noether_bridge.HOLOGRAPHIC_NOISE_FLOOR))

    sub_threshold = noether_bridge.SaturationAudit(
        lambda_obs_si_m2=noether_bridge.Decimal("1.0e-52"),
        lambda_obs_ev2=noether_bridge.Decimal("1.0e-66"),
        holographic_bits_from_lambda=noether_bridge.Decimal("3.31e122"),
        configured_holographic_bits=noether_bridge.Decimal("3.31e122"),
        register_noise_floor=noether_bridge.Decimal("3.0e-123"),
        relative_mismatch=threshold / noether_bridge.Decimal("10"),
    )
    at_threshold = noether_bridge.SaturationAudit(
        lambda_obs_si_m2=noether_bridge.Decimal("1.0e-52"),
        lambda_obs_ev2=noether_bridge.Decimal("1.0e-66"),
        holographic_bits_from_lambda=noether_bridge.Decimal("3.31e122"),
        configured_holographic_bits=noether_bridge.Decimal("3.31e122"),
        register_noise_floor=noether_bridge.Decimal("3.0e-123"),
        relative_mismatch=threshold,
    )

    assert sub_threshold.success is True
    assert at_threshold.success is False


def test_standalone_saturation_module_uses_noise_floor_success() -> None:
    threshold = noether_bridge.Decimal(str(noether_bridge.HOLOGRAPHIC_NOISE_FLOOR))

    sub_threshold = SaturationAudit(
        lambda_obs_si_m2=noether_bridge.Decimal("1.0e-52"),
        lambda_obs_ev2=noether_bridge.Decimal("1.0e-66"),
        holographic_bits_from_lambda=noether_bridge.Decimal("3.31e122"),
        configured_holographic_bits=noether_bridge.Decimal("3.31e122"),
        register_noise_floor=noether_bridge.Decimal("3.0e-123"),
        relative_mismatch=threshold / noether_bridge.Decimal("10"),
    )
    above_threshold = SaturationAudit(
        lambda_obs_si_m2=noether_bridge.Decimal("1.0e-52"),
        lambda_obs_ev2=noether_bridge.Decimal("1.0e-66"),
        holographic_bits_from_lambda=noether_bridge.Decimal("3.31e122"),
        configured_holographic_bits=noether_bridge.Decimal("3.31e122"),
        register_noise_floor=noether_bridge.Decimal("3.0e-123"),
        relative_mismatch=threshold * noether_bridge.Decimal("10"),
    )

    assert sub_threshold.success is True
    assert sub_threshold.passed is True
    assert sub_threshold.is_saturated is True
    assert above_threshold.success is False
