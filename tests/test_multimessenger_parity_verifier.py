from __future__ import annotations

import importlib.util
import json
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
    assert audit.bao_mapping.peak_positions_locked
    assert audit.bao_mapping.within_tolerance
    assert audit.chi_squared_fit.observable_count == 4
    assert audit.chi_squared_fit.degrees_of_freedom == 4
    assert len(audit.bao_mapping.peak_audits) == 3
    assert len(audit.chi_squared_fit.components) == 4
    expected_chi_squared = sum(float(component.chi_squared_contribution) for component in audit.chi_squared_fit.components)
    assert float(audit.chi_squared_fit.chi_squared) == pytest.approx(expected_chi_squared, rel=0.0, abs=1.0e-12)
    assert float(audit.chi_squared_fit.reduced_chi_squared) == pytest.approx(
        expected_chi_squared / audit.chi_squared_fit.degrees_of_freedom,
        rel=0.0,
        abs=1.0e-12,
    )
    assert audit.bao_mapping.max_peak_position_residual <= audit.bao_mapping.peak_position_tolerance
    assert all(peak_audit.within_tolerance for peak_audit in audit.bao_mapping.peak_audits)
    assert audit.scalar_tilt.within_reference_band
    assert audit.gravitational_wave.below_current_ceiling
    assert audit.gravitational_wave.above_design_floor
    assert audit.executable_proof_pass


def test_multimessenger_report_is_automated_physical_audit() -> None:
    module = _load_script_module()
    report = module.build_multimessenger_report()

    assert "Automated Physical Audit        : PASS" in report
    assert "automated BAO audit = PASS" in report
    assert "Chi-Squared =" in report
    assert "reduced Chi-Squared =" in report
    assert "BAO peak ladder locked = True" in report
    assert "TT peak 1 [m=1]" in report
    assert "data/cmb_power_spectrum_benchmarks.json" in report


def test_cmb_benchmark_bundle_includes_peak_position_ladder() -> None:
    module = _load_script_module()
    _, benchmark = module.load_cmb_benchmark()

    assert benchmark.bao_acoustic_scale_multipole == pytest.approx(301.0, rel=0.0, abs=1.0e-12)
    assert benchmark.bao_peak_position_tolerance == pytest.approx(6.0, rel=0.0, abs=1.0e-12)
    assert len(benchmark.bao_peak_benchmarks) == 3
    assert benchmark.bao_peak_benchmarks[0].label == "TT peak 1"


def test_main_returns_zero_for_passing_cmb_bao_audit() -> None:
    module = _load_script_module()

    assert module.main([]) == 0
    assert module.DEFAULT_MULTIMESSENGER_AUDIT_PATH.is_file()


def test_multimessenger_audit_json_artifact_reports_chi_squared_fit(tmp_path: Path) -> None:
    module = _load_script_module()
    audit = module.build_multimessenger_parity_audit()
    output_path = tmp_path / "multimessenger_audit.json"

    artifact_path = module.write_multimessenger_audit_artifact(audit, output_path=output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact_path == output_path
    assert payload["artifact_path"] == str(output_path)
    assert payload["chi_squared_fit"]["label"] == "Topological Ghost / BAO acoustic peak fit"
    assert payload["chi_squared_fit"]["observable_count"] == 4
    assert payload["chi_squared_fit"]["degrees_of_freedom"] == 4
    assert payload["chi_squared_fit"]["chi_squared"] == pytest.approx(float(audit.chi_squared_fit.chi_squared), rel=0.0, abs=1.0e-12)
    assert payload["chi_squared_fit"]["reduced_chi_squared"] == pytest.approx(
        float(audit.chi_squared_fit.reduced_chi_squared),
        rel=0.0,
        abs=1.0e-12,
    )
    assert payload["chi_squared_fit"]["components"][0]["label"] == "BAO dark-to-baryon ratio"
    assert payload["chi_squared_fit"]["components"][1]["label"] == "TT peak 1 multipole"
