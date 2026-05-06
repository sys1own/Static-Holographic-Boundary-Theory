from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dashboard.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("dashboard", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_derivation_snapshot_exposes_live_ledger() -> None:
    module = _load_script_module()

    snapshot = module.build_derivation_snapshot()

    assert snapshot.precision >= module.DEFAULT_PRECISION
    assert snapshot.alpha_surface_inverse > 0
    assert snapshot.kappa_d5 > 0
    assert snapshot.neutrino_floor_mev > 0
    assert snapshot.epsilon_lambda <= snapshot.decimal_tolerance
    assert snapshot.decimal_passed is True
    assert "Derivation Ledger" in snapshot.ledger_text
    assert "Alpha Surface Inverse" in snapshot.ledger_text
    assert "Unity of Scale Identity" in snapshot.ledger_text


def test_build_detuning_snapshot_detects_anomaly_spike() -> None:
    module = _load_script_module()

    benchmark = module.build_detuning_snapshot()
    detuned = module.build_detuning_snapshot(delta_lepton=1)

    assert benchmark.benchmark_selected is True
    assert benchmark.candidate_branch == module.BENCHMARK_BRANCH
    assert benchmark.rigidity_point.total_residue == 0.0
    assert benchmark.anomaly_audit.framing.delta_fr == 0
    assert benchmark.anomaly_audit.closure_tensor.amplitude == 0
    assert benchmark.anomaly_audit.anomalous_source_si_m2.amplitude == 0

    assert detuned.benchmark_selected is False
    assert detuned.candidate_branch == (27, 8, 312)
    assert detuned.rigidity_point.total_residue > 0.0
    assert detuned.anomaly_audit.framing.delta_fr != 0
    assert detuned.anomaly_audit.closure_tensor.amplitude > 0
    assert detuned.anomaly_audit.anomalous_source_si_m2.amplitude > 0


def test_build_rigidity_scan_contains_benchmark_valley() -> None:
    module = _load_script_module()

    scan = module.build_rigidity_scan(lepton_half_width=1, quark_half_width=1, parent_half_width=1)

    assert scan.benchmark_coordinates == module.BENCHMARK_BRANCH
    assert scan.benchmark_point.total_residue == 0.0
    assert scan.nearest_detuned_point.total_residue > 0.0
