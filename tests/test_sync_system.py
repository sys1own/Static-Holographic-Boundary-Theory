from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sync_system.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("sync_system", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample_residual_payload() -> dict[str, object]:
    return {
        "benchmark_tuple": [26, 8, 312],
        "unity_of_scale_identity": {
            "epsilon_lambda": 1.0e-199,
            "exact_epsilon_lambda": 1.0e-199,
            "register_noise_floor": 3.0e-123,
            "exact_register_noise_floor": 3.0e-123,
            "passed": True,
        },
        "gauge_residual_bookkeeping": {
            "topological_alpha_inverse": 137.6470588235,
            "codata_alpha_inverse": 137.035999084,
            "two_loop_residual_fraction": 0.0044582737,
            "two_loop_residual_percent": 0.44582737,
            "two_loop_residual_pull": 0.6110597394,
        },
        "informational_costs": {
            "delta_s_red_nat": 0.6931471805599453,
        },
        "mixing_angle_drifts_deg": {
            "theta12": 0.00756822,
            "theta13": 0.00098107,
            "theta23": 0.0013584,
            "delta_cp": -0.022169035713716676,
        },
        "mass_scale_two_loop_fraction": 0.0007007607443640616,
    }


def test_synchronize_system_requires_residual_artifact(tmp_path: Path) -> None:
    module = _load_script_module()

    with pytest.raises(FileNotFoundError, match=r"results/residuals\.json first"):
        module.synchronize_system(
            residuals_path=tmp_path / "missing.json",
            readme_path=tmp_path / "README.md",
            physics_constants_path=tmp_path / "physics_constants.tex",
        )


def test_synchronize_system_updates_readme_and_physics_constants_idempotently(tmp_path: Path) -> None:
    module = _load_script_module()
    residuals_path = tmp_path / "residuals.json"
    readme_path = tmp_path / "README.md"
    physics_constants_path = tmp_path / "physics_constants.tex"

    residuals_path.write_text(json.dumps(_sample_residual_payload()), encoding="utf-8")
    readme_path.write_text(
        "\n".join(
            (
                "# Demo",
                "",
                "## Derivation Ledger",
                "",
                "Ledger intro.",
                "",
                "### Tier Classification",
                "",
                "- tier text",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    physics_constants_path.write_text(
        "\n".join(
            (
                r"\providecommand{\alphaSurfBenchmarkDecimal}{0}",
                r"\providecommand{\alphaSurfBenchmarkRounded}{0}",
                r"\providecommand{\leptonThetaTwelveBetaTwoLoop}{0}",
                r"\providecommand{\leptonThetaThirteenBetaTwoLoop}{0}",
                r"\providecommand{\leptonThetaTwentyThreeBetaTwoLoop}{0}",
                r"\providecommand{\leptonDeltaBetaTwoLoop}{0}",
                r"\providecommand{\quarkThetaTwelveBetaTwoLoop}{0}",
                r"\providecommand{\quarkThetaThirteenBetaTwoLoop}{0}",
                r"\providecommand{\quarkThetaTwentyThreeBetaTwoLoop}{0}",
                r"\providecommand{\quarkDeltaBetaTwoLoop}{0}",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    updated_readme, updated_physics_constants = module.synchronize_system(
        residuals_path=residuals_path,
        readme_path=readme_path,
        physics_constants_path=physics_constants_path,
    )

    assert updated_readme == readme_path
    assert updated_physics_constants == physics_constants_path

    readme_text = readme_path.read_text(encoding="utf-8")
    physics_text = physics_constants_path.read_text(encoding="utf-8")
    snapshot = module.build_sync_snapshot(_sample_residual_payload())

    assert "### Machine-Synced Residual Ledger" in readme_text
    assert module.README_SYNC_START in readme_text
    assert "unity_of_scale_identity.epsilon_lambda" in readme_text
    assert "informational_costs.delta_s_red_nat" in readme_text
    assert "derive_transport_curvature_audit()" in readme_text
    assert readme_text.count("### Machine-Synced Residual Ledger") == 1

    assert rf"\providecommand{{\alphaSurfBenchmarkDecimal}}{{{module._format_latex_float(snapshot.gauge_topological_alpha_inverse)}}}" in physics_text
    assert rf"\providecommand{{\leptonThetaTwelveBetaTwoLoop}}{{{module._format_latex_float(snapshot.lepton_theta12_two_loop_deg)}}}" in physics_text
    assert rf"\providecommand{{\quarkThetaThirteenBetaTwoLoop}}{{{module._format_latex_float(snapshot.quark_theta13_two_loop_deg)}}}" in physics_text
    assert module.PHYSICS_CONSTANTS_SYNC_START in physics_text
    assert r"\providecommand{\unityResidueEpsilonLambda}{1\times10^{-199}}" in physics_text
    assert r"\providecommand{\deltaSRedNat}{0.6931471806}" in physics_text

    module.synchronize_system(
        residuals_path=residuals_path,
        readme_path=readme_path,
        physics_constants_path=physics_constants_path,
    )

    readme_text_second_pass = readme_path.read_text(encoding="utf-8")
    physics_text_second_pass = physics_constants_path.read_text(encoding="utf-8")
    assert readme_text_second_pass.count("### Machine-Synced Residual Ledger") == 1
    assert readme_text_second_pass.count(module.README_SYNC_START) == 1
    assert physics_text_second_pass.count(module.PHYSICS_CONSTANTS_SYNC_START) == 1
