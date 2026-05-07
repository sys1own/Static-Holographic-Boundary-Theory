from __future__ import annotations

from decimal import Decimal

import shbt.core.engine as engine_module
from shbt.core.evolutionary_engine import EvolutionaryEngine
from shbt.core.holographic_error_stabilizer import BENCHMARK_BRANCH, HolographicStabilizer
from shbt.main import ZERO_PARAMETER_RUNTIME_BOOTSTRAP


def test_single_point_flavor_drift_breaks_entangled_bulk_checksum(monkeypatch) -> None:
    benchmark_vacuum = EvolutionaryEngine.benchmark_vacuum()

    assert benchmark_vacuum.branch == ZERO_PARAMETER_RUNTIME_BOOTSTRAP.kernel.branch == BENCHMARK_BRANCH

    control_audit = HolographicStabilizer().verify_bulk_checksum()

    assert control_audit.passed is True
    assert control_audit.zero_energy_boundary_locked is True

    def drifted_efe_violation_tensor(*, parent_level: int, lepton_level: int, quark_level: int) -> float:
        assert (int(lepton_level), int(quark_level), int(parent_level)) == BENCHMARK_BRANCH
        shifted_lepton_level = Decimal(str(lepton_level)) + Decimal("1e-15")
        return float(abs(shifted_lepton_level - Decimal(str(lepton_level))))

    monkeypatch.setattr(engine_module, "calculate_efe_violation_tensor", drifted_efe_violation_tensor)

    audit = HolographicStabilizer().verify_bulk_checksum()

    assert audit.benchmark_branch == BENCHMARK_BRANCH
    assert audit.time_reversal_residual == Decimal("1e-15")
    assert audit.time_reversal_checksum_passed is False
    assert audit.zero_energy_boundary_locked is False
    assert "time-reversal residual" in audit.detail
    assert audit.passed is False
