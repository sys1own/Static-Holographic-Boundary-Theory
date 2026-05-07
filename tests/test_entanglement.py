from __future__ import annotations

from decimal import Decimal

import shbt.core.engine as engine_module
import shbt.main as main_module
from shbt.core.evolutionary_engine import EvolutionaryEngine
from shbt.core.holographic_error_stabilizer import BENCHMARK_BRANCH, HolographicStabilizer
from shbt.main import ZERO_PARAMETER_RUNTIME_BOOTSTRAP


def test_single_point_failure_decoheres_zero_parameter_rigidity_lock(monkeypatch) -> None:
    geometry_bootstrap = main_module._initialize_zero_parameter_execution_mode()
    benchmark_vacuum = EvolutionaryEngine.benchmark_vacuum()

    assert geometry_bootstrap.kernel.branch == ZERO_PARAMETER_RUNTIME_BOOTSTRAP.kernel.branch == BENCHMARK_BRANCH
    assert benchmark_vacuum.branch == BENCHMARK_BRANCH

    control_rigidity_audit = HolographicStabilizer().verify_bulk_checksum()

    assert control_rigidity_audit.passed is True
    assert control_rigidity_audit.zero_energy_boundary_locked is True

    def drifted_efe_violation_tensor(*, parent_level: int, lepton_level: int, quark_level: int) -> float:
        assert (int(lepton_level), int(quark_level), int(parent_level)) == BENCHMARK_BRANCH
        shifted_lepton_level = Decimal(str(lepton_level)) + Decimal("1e-15")
        return float(abs(shifted_lepton_level - Decimal(str(lepton_level))))

    monkeypatch.setattr(engine_module, "calculate_efe_violation_tensor", drifted_efe_violation_tensor)

    rigidity_audit = HolographicStabilizer().verify_bulk_checksum()

    assert rigidity_audit.benchmark_branch == BENCHMARK_BRANCH
    assert rigidity_audit.time_reversal_residual == Decimal("1e-15")
    assert rigidity_audit.time_reversal_checksum_passed is False
    assert rigidity_audit.zero_energy_boundary_locked is False
    assert "time-reversal residual" in rigidity_audit.detail
    assert rigidity_audit.passed is False
