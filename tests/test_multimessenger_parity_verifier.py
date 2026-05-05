from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from shbt.core.topology import calculate_dark_debt
from shbt.paths import resolve_resource_path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "multimessenger_parity_verifier.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("multimessenger_parity_verifier", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_calculate_dark_debt_matches_benchmark_branch() -> None:
    assert calculate_dark_debt() == pytest.approx(5.780852425251275, rel=0.0, abs=1.0e-12)


def test_multimessenger_parity_audit_passes_benchmark() -> None:
    module = _load_script_module()
    audit = module.build_multimessenger_parity_audit()

    assert audit.nufit_anchor.path.is_file()
    assert audit.cmb_anchor.path.is_file()
    assert audit.cmb_anchor.path == resolve_resource_path("data", "cmb_power_spectrum_benchmarks.json")
    assert "NuFIT 5.3" in audit.nufit_anchor.release
    assert audit.dark_debt.benchmark_locked
    assert audit.bao_mapping.within_tolerance
    assert audit.scalar_tilt.within_reference_band
    assert audit.gravitational_wave.below_current_ceiling
    assert audit.gravitational_wave.above_design_floor
    assert audit.executable_proof_pass


def test_multimessenger_report_is_automated_physical_audit() -> None:
    module = _load_script_module()
    report = module.build_multimessenger_report()

    assert "Automated Physical Audit        : PASS" in report
    assert "automated BAO audit = PASS" in report
    assert "data/cmb_power_spectrum_benchmarks.json" in report


def test_main_returns_zero_for_passing_cmb_bao_audit() -> None:
    module = _load_script_module()

    assert module.main([]) == 0
