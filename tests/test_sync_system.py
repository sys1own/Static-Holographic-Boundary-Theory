from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from shbt.config_loader import ConfigLoader


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sync_system.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("sync_system", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_residual_payload(module):
    universal_snapshot = module._build_universal_constants_snapshot(ConfigLoader())
    payload = {
        "benchmark_tuple": [
            universal_snapshot.lepton_level,
            universal_snapshot.quark_level,
            universal_snapshot.parent_level,
        ],
        "unity_of_scale_identity": {
            "epsilon_lambda": 1.0e-199,
            "exact_epsilon_lambda": 1.0e-199,
            "register_noise_floor": 3.0e-123,
            "exact_register_noise_floor": 3.0e-123,
        },
        "gauge_residual_bookkeeping": {
            "topological_alpha_inverse": universal_snapshot.topological_alpha_inverse,
            "codata_alpha_inverse": 137.035999084,
            "two_loop_residual_fraction": 1.25e-4,
            "two_loop_residual_percent": 1.25e-2,
            "two_loop_residual_pull": 0.25,
        },
        "informational_costs": {
            "delta_s_red_nat": 0.5,
        },
        "mixing_angle_drifts_deg": {
            "theta12": 0.1,
            "theta13": 0.2,
            "theta23": 0.3,
            "delta_cp": 0.4,
        },
        "mass_scale_two_loop_fraction": 0.05,
    }
    return universal_snapshot, payload


def test_synchronize_system_updates_readme_and_physics_constants(tmp_path) -> None:
    module = _load_script_module()
    universal_snapshot, payload = _build_residual_payload(module)

    residuals_path = tmp_path / "residuals.json"
    residuals_path.write_text(json.dumps(payload), encoding="utf-8")

    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        "# Demo\n\n"
        "## Derivation Ledger\n\n"
        "Old derivation text.\n\n"
        "### Tier Classification\n\n"
        "## Next Section\n",
        encoding="utf-8",
    )

    physics_constants_path = tmp_path / "physics_constants.tex"
    physics_constants_path.write_text(
        "% Test fallback macros\n"
        "\\def\\NeutrinoFloor{OLD}\n"
        "\\providecommand{\\benchmarkLeptonLevel}{0}\n"
        "\\providecommand{\\benchmarkQuarkLevel}{0}\n"
        "\\providecommand{\\benchmarkParentLevel}{0}\n"
        "\\providecommand{\\alphaSurfBenchmarkDecimal}{0}\n",
        encoding="utf-8",
    )

    updated_readme, updated_physics = module.synchronize_system(
        residuals_path=residuals_path,
        readme_path=readme_path,
        physics_constants_path=physics_constants_path,
    )

    assert updated_readme == readme_path.resolve()
    assert updated_physics == physics_constants_path.resolve()

    readme_text = readme_path.read_text(encoding="utf-8")
    assert module.README_LEDGER_TABLE_HEADER in readme_text
    assert module.README_SYNC_START in readme_text
    assert module.README_SYNC_END in readme_text
    assert "Machine-Synced Residual Ledger" in readme_text
    assert "Unity-of-Scale residue" in readme_text
    assert module._format_markdown_float(payload["unity_of_scale_identity"]["epsilon_lambda"]) in readme_text

    physics_text = physics_constants_path.read_text(encoding="utf-8")
    assert "\\def\\NeutrinoFloor{OLD}" not in physics_text
    assert "\\def\\NeutrinoFloor{" in physics_text
    assert f"\\providecommand{{\\benchmarkLeptonLevel}}{{{universal_snapshot.lepton_level}}}" in physics_text
    assert f"\\providecommand{{\\benchmarkQuarkLevel}}{{{universal_snapshot.quark_level}}}" in physics_text
    assert f"\\providecommand{{\\benchmarkParentLevel}}{{{universal_snapshot.parent_level}}}" in physics_text
    assert module.PHYSICS_CONSTANTS_SYNC_START in physics_text
    assert module.PHYSICS_CONSTANTS_SYNC_END in physics_text
    assert "\\providecommand{\\unityResidueEpsilonLambda}{" in physics_text
