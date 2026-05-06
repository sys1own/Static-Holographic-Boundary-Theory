from __future__ import annotations

import pytest

from shbt.core.decoherence_engine import PointerStateDecoherenceError, PointerStateSelector
import shbt.core.noether_bridge as noether_bridge
from shbt.core.holographic_error_stabilizer import BulkChecksumVerification, HolographicStabilizer


def test_branch_planck_mass_ev_requires_coherent_bulk_checksum() -> None:
    planck_mass = noether_bridge.branch_planck_mass_ev()

    assert planck_mass > 0


def test_branch_planck_mass_ev_calls_holographic_stabilizer(monkeypatch: pytest.MonkeyPatch) -> None:
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

    assert noether_bridge.branch_planck_mass_ev() > 0
    assert calls["count"] == 1


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

    with pytest.raises(noether_bridge.DecoherenceError, match=r"off-shell branch coordinates"):
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
