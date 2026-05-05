from __future__ import annotations

"""Formal verification suite for anomaly closure and precision-floor rigidity.

This module is intended for direct pytest execution in CI/CD pipelines so that
future optimizer or dependency changes cannot silently relax the anomaly moat,
the precision floor, or the Newton-lock benchmark branch.
"""

from dataclasses import replace

import pytest

import pub.noether_bridge as noether_bridge
import pub.tn as tn
from pub.runtime_config import DEFAULT_SOLVER_CONFIG, SolverException


PrecisionUnderflowError = SolverException


EXPECTED_BENCHMARK = (26, 8, 312)
EXPECTED_OFF_SHELL = (27, 8, 312)


def test_reviewer_trap_rejects_off_shell_pmns_extraction() -> None:
    """Anomalous branches must be rejected before physical flavor extraction."""

    off_shell_model = tn.TopologicalModel(
        k_l=EXPECTED_OFF_SHELL[0],
        k_q=EXPECTED_OFF_SHELL[1],
        parent_level=EXPECTED_OFF_SHELL[2],
    )

    assert off_shell_model.target_tuple == EXPECTED_OFF_SHELL
    assert off_shell_model.framing_gap == pytest.approx(2.0 / 9.0, rel=0.0, abs=1.0e-15)

    with pytest.raises(tn.AnomalyClosureError, match=r"framing closure requires Delta_fr=0"):
        tn.derive_pmns(model=off_shell_model)


def test_low_precision_solver_config_raises_repo_underflow_equivalent() -> None:
    """The theorem-level gravity stack must refuse a precision floor below 1/N."""

    assert DEFAULT_SOLVER_CONFIG.unity_noise_floor_target <= 1.0e-123

    with pytest.raises(PrecisionUnderflowError, match=r"arbitrary_precision_dps >= 250"):
        replace(DEFAULT_SOLVER_CONFIG, arbitrary_precision_dps=49)


def test_benchmark_branch_executes_and_preserves_newton_constant_lock() -> None:
    """The published on-shell branch must run cleanly and keep the gravity lock."""

    benchmark_model = tn.TopologicalModel(
        k_l=EXPECTED_BENCHMARK[0],
        k_q=EXPECTED_BENCHMARK[1],
        parent_level=EXPECTED_BENCHMARK[2],
    )

    uniqueness_audit = tn.verify_derived_uniqueness_theorem(model=benchmark_model)
    pmns = tn.derive_pmns(model=benchmark_model)
    gravity_report = noether_bridge.build_gravity_side_rigidity_report()

    assert benchmark_model.target_tuple == EXPECTED_BENCHMARK
    assert benchmark_model.framing_gap == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert uniqueness_audit.framing_gap == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert pmns.theta12_rg_deg == pytest.approx(32.95284245429993, rel=0.0, abs=1.0e-12)
    assert pmns.theta13_rg_deg == pytest.approx(8.615903228755949, rel=0.0, abs=1.0e-12)
    assert pmns.theta23_rg_deg == pytest.approx(43.22191806537939, rel=0.0, abs=1.0e-12)
    assert pmns.delta_cp_rg_deg == pytest.approx(197.10789340689752, rel=0.0, abs=1.0e-12)
    assert gravity_report.unity.passed
    assert gravity_report.newton_lock.g_effective_ev_minus2 > 0
    assert gravity_report.newton_lock.g_topological_ev_minus2 > 0
    assert float(
        gravity_report.newton_lock.topological_from_effective_factor
        * gravity_report.newton_lock.effective_from_topological_factor
    ) == pytest.approx(1.0, rel=0.0, abs=1.0e-15)
