from __future__ import annotations

import pytest

import shbt.core.noether_bridge as noether_bridge
from shbt.core.holographic_error_stabilizer import BulkChecksumVerification, HolographicStabilizer


def test_branch_planck_mass_ev_requires_coherent_bulk_checksum() -> None:
    planck_mass = noether_bridge.branch_planck_mass_ev()

    assert planck_mass > 0


def test_branch_planck_mass_ev_raises_on_simulated_boundary_decoherence() -> None:
    with pytest.raises(noether_bridge.DecoherenceError, match=r"bulk checksum failed"):
        noether_bridge.branch_planck_mass_ev(simulate_boundary_decoherence=True)


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
