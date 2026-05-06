from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sync_system.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("sync_system", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample_residual_payload(
    *,
    benchmark_tuple: tuple[int, int, int] = (26, 8, 312),
    topological_alpha_inverse: float = 137.6470588235,
) -> dict[str, object]:
    return {
        "benchmark_tuple": list(benchmark_tuple),
        "unity_of_scale_identity": {
            "epsilon_lambda": 1.0e-199,
            "exact_epsilon_lambda": 1.0e-199,
            "register_noise_floor": 3.0e-123,
            "exact_register_noise_floor": 3.0e-123,
            "passed": True,
        },
        "gauge_residual_bookkeeping": {
            "topological_alpha_inverse": topological_alpha_inverse,
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


def _sample_universal_constants() -> dict[str, object]:
    return {
        "tier_1": {
            "lepton_level": {"value": 30, "classification": "Topological Necessity"},
            "quark_level": {"value": 10, "classification": "Topological Necessity"},
            "parent_level": {"value": 360, "classification": "Topological Necessity"},
            "g_sm": {"value": 18, "classification": "Topological Necessity"},
        },
        "tier_2": {
            "planck_mass_ev": {"value": 1.23e28, "classification": "Empirical Matching Ansatz"},
            "planck_length_m": {"value": 1.616255e-35, "classification": "Empirical Matching Ansatz"},
            "light_speed_m_per_s": {"value": 299792458.0, "classification": "Empirical Matching Ansatz"},
            "mpc_in_meters": {"value": 3.085677581491367e22, "classification": "Empirical Matching Ansatz"},
            "planck2018_h0_km_s_mpc": {"value": 70.0, "classification": "Empirical Matching Ansatz"},
            "planck2018_h0_sigma_km_s_mpc": {"value": 0.4, "classification": "Empirical Matching Ansatz"},
            "planck2018_omega_lambda": {"value": 0.7, "classification": "Empirical Matching Ansatz"},
            "planck2018_omega_lambda_sigma": {"value": 0.006, "classification": "Empirical Matching Ansatz"},
            "planck2018_lambda_si_m2": {"value": 9.99e-53, "classification": "Empirical Matching Ansatz"},
            "planck2018_lambda_fractional_sigma": {"value": 0.0123, "classification": "Empirical Matching Ansatz"},
            "planck2018_alpha_em_inv_mz": {"value": 128.1, "classification": "Empirical Matching Ansatz"},
            "planck2018_sin2_theta_w_mz": {"value": 0.2315, "classification": "Empirical Matching Ansatz"},
            "planck2018_alpha_s_mz": {"value": 0.1181, "classification": "Empirical Matching Ansatz"},
            "codata_fine_structure_alpha_inverse": {"value": 140.0, "classification": "Empirical Matching Ansatz"},
            "hbar_ev_seconds": {"value": 6.7e-16, "classification": "Empirical Matching Ansatz"},
        },
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
    universal_constants_path = tmp_path / "universal_constants.yaml"
    benchmark_tuple = (30, 10, 360)
    topological_alpha_inverse = 162.0

    residuals_path.write_text(
        json.dumps(
            _sample_residual_payload(
                benchmark_tuple=benchmark_tuple,
                topological_alpha_inverse=topological_alpha_inverse,
            )
        ),
        encoding="utf-8",
    )
    universal_constants_path.write_text(yaml.safe_dump(_sample_universal_constants(), sort_keys=False), encoding="utf-8")
    readme_path.write_text(
        "\n".join(
            (
                "# Demo",
                "",
                "## Derivation Ledger",
                "",
                "Ledger intro.",
                "",
                "| Observable | Derived From | Predicted Value | CODATA / anchor |",
                "| :--- | :--- | :--- | :--- |",
                "| stale | stale | stale | stale |",
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
                r"\def\FineStructureInverse{0}",
                r"\def\PredictedAlphaInverse{0}",
                r"\newcommand{\mZeroBenchmarkMeV}{0}",
                r"\NeutrinoFloor{0}",
                r"\providecommand{\PredictedMassRatio}{0}",
                r"\providecommand{\PredictedNeutrinoFloorMeV}{0}",
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
        universal_constants_path=universal_constants_path,
    )

    assert updated_readme == readme_path
    assert updated_physics_constants == physics_constants_path

    readme_text = readme_path.read_text(encoding="utf-8")
    physics_text = physics_constants_path.read_text(encoding="utf-8")
    universal_snapshot = module._build_universal_constants_snapshot(
        module.ConfigLoader(universal_constants_path=universal_constants_path)
    )
    snapshot = module.build_sync_snapshot_with_universal_constants(
        _sample_residual_payload(
            benchmark_tuple=benchmark_tuple,
            topological_alpha_inverse=topological_alpha_inverse,
        ),
        universal_snapshot=universal_snapshot,
    )

    assert module.README_LEDGER_TABLE_HEADER in readme_text
    assert "| stale | stale | stale | stale |" not in readme_text
    assert "### Machine-Synced Residual Ledger" in readme_text
    assert module.README_SYNC_START in readme_text
    assert "unity_of_scale_identity.epsilon_lambda" in readme_text
    assert "informational_costs.delta_s_red_nat" in readme_text
    assert "derive_transport_curvature_audit()" in readme_text
    assert readme_text.count(module.README_LEDGER_TABLE_HEADER) == 1
    assert readme_text.count("### Machine-Synced Residual Ledger") == 1
    assert "| $\\alpha^{-1}$ | $18 \\times 360 / 40$ | $\\approx 162$ | $137.036$ (Two-Loop Residual) |" in readme_text
    assert "c_q=80/13, c_{\\ell}=45/16, V_{\\rm px}=1/4" in readme_text
    assert f"$\\approx {module._format_markdown_float(snapshot.proton_electron_mass_ratio, digits=2)}$" in readme_text
    assert f"$\\approx {module._format_markdown_float(snapshot.neutrino_floor_mev, digits=2)}$ meV" in readme_text

    assert rf"\providecommand{{\alphaSurfBenchmarkDecimal}}{{{module._format_latex_float(universal_snapshot.topological_alpha_inverse)}}}" in physics_text
    assert rf"\providecommand{{\alphaSurfBenchmarkExact}}{{\dfrac{{{universal_snapshot.topological_alpha_inverse_numerator}}}{{{universal_snapshot.topological_alpha_inverse_denominator}}}}}" in physics_text
    assert rf"\providecommand{{\benchmarkLeptonLevel}}{{{universal_snapshot.lepton_level}}}" in physics_text
    assert rf"\providecommand{{\benchmarkQuarkLevel}}{{{universal_snapshot.quark_level}}}" in physics_text
    assert rf"\providecommand{{\benchmarkParentLevel}}{{{universal_snapshot.parent_level}}}" in physics_text
    assert rf"\providecommand{{\benchmarkVisibleBranch}}{{({universal_snapshot.lepton_level},{universal_snapshot.quark_level},{universal_snapshot.parent_level})}}" in physics_text
    assert rf"\providecommand{{\benchmarkPlanckMassEv}}{{{module._format_latex_float(universal_snapshot.planck_mass_ev)}}}" in physics_text
    assert rf"\newcommand{{\mZeroBenchmarkMeV}}{{{module._format_latex_float(snapshot.neutrino_floor_mev)}}}" in physics_text
    assert rf"\NeutrinoFloor{{{module._format_latex_float(snapshot.neutrino_floor_mev)}}}" in physics_text
    assert rf"\def\FineStructureInverse{{{module._format_latex_float(universal_snapshot.topological_alpha_inverse)}}}" in physics_text
    assert rf"\def\PredictedAlphaInverse{{{module._format_latex_float(universal_snapshot.topological_alpha_inverse)}}}" in physics_text
    assert rf"\providecommand{{\PredictedMassRatio}}{{{module._format_latex_float(snapshot.proton_electron_mass_ratio)}}}" in physics_text
    assert rf"\providecommand{{\PredictedNeutrinoFloorMeV}}{{{module._format_latex_float(snapshot.neutrino_floor_mev)}}}" in physics_text
    assert rf"\providecommand{{\leptonThetaTwelveBetaTwoLoop}}{{{module._format_latex_float(snapshot.lepton_theta12_two_loop_deg)}}}" in physics_text
    assert rf"\providecommand{{\quarkThetaThirteenBetaTwoLoop}}{{{module._format_latex_float(snapshot.quark_theta13_two_loop_deg)}}}" in physics_text
    assert module.PHYSICS_CONSTANTS_SYNC_START in physics_text
    assert r"\providecommand{\unityResidueEpsilonLambda}{1\times10^{-199}}" in physics_text
    assert r"\providecommand{\deltaSRedNat}{0.6931471806}" in physics_text

    module.synchronize_system(
        residuals_path=residuals_path,
        readme_path=readme_path,
        physics_constants_path=physics_constants_path,
        universal_constants_path=universal_constants_path,
    )

    readme_text_second_pass = readme_path.read_text(encoding="utf-8")
    physics_text_second_pass = physics_constants_path.read_text(encoding="utf-8")
    assert readme_text_second_pass.count(module.README_LEDGER_TABLE_HEADER) == 1
    assert readme_text_second_pass.count("### Machine-Synced Residual Ledger") == 1
    assert readme_text_second_pass.count(module.README_SYNC_START) == 1
    assert physics_text_second_pass.count(module.PHYSICS_CONSTANTS_SYNC_START) == 1


def test_synchronize_system_resolves_repo_root_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    repo_root = tmp_path / "repo"
    results_dir = repo_root / "results"
    papers_dir = repo_root / "papers"
    config_dir = repo_root / "config" / "physics_profiles"
    outside_dir = tmp_path / "outside"

    results_dir.mkdir(parents=True)
    papers_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    outside_dir.mkdir()

    (results_dir / "residuals.json").write_text(
        json.dumps(_sample_residual_payload(benchmark_tuple=(30, 10, 360), topological_alpha_inverse=162.0)),
        encoding="utf-8",
    )
    (repo_root / "README.md").write_text(
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
    (papers_dir / "physics_constants.tex").write_text(
        "\n".join(
            (
                r"\def\FineStructureInverse{0}",
                r"\NeutrinoFloor{0}",
                r"\providecommand{\PredictedMassRatio}{0}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (config_dir / "standard_model.yaml").write_text(
        yaml.safe_dump(_sample_universal_constants(), sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(module.ProjectPaths, "ROOT", repo_root)
    monkeypatch.setattr(module.ProjectPaths, "RESULTS", results_dir)
    monkeypatch.setattr(module.ProjectPaths, "PAPERS", papers_dir)
    monkeypatch.setattr(module.ProjectPaths, "CONFIG", repo_root / "config")
    monkeypatch.chdir(outside_dir)

    updated_readme, updated_physics_constants = module.synchronize_system(
        residuals_path=Path("results/residuals.json"),
        readme_path=Path("README.md"),
        physics_constants_path=Path("papers/physics_constants.tex"),
        physics_profile_path=Path("config/physics_profiles/standard_model.yaml"),
    )

    assert updated_readme == repo_root / "README.md"
    assert updated_physics_constants == papers_dir / "physics_constants.tex"
    assert module.README_SYNC_START in updated_readme.read_text(encoding="utf-8")
    physics_text = updated_physics_constants.read_text(encoding="utf-8")
    assert module.PHYSICS_CONSTANTS_SYNC_START in physics_text
    assert "\\NeutrinoFloor{" in physics_text


def test_build_sync_snapshot_uses_universe_factory_mu_derivation(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    universal_snapshot = module._build_universal_constants_snapshot(module.ConfigLoader())
    calls: list[tuple[int, tuple[int, int, int]]] = []

    def _fake_derive_proton_ratio(cls, *, precision: int, vacuum):
        calls.append((precision, vacuum.branch))

        class _Derivation:
            mu_audit = module.Decimal("1999.5")

        return _Derivation()

    monkeypatch.setattr(module.UniverseFactory, "derive_proton_ratio", classmethod(_fake_derive_proton_ratio))

    snapshot = module.build_sync_snapshot_with_universal_constants(
        _sample_residual_payload(),
        universal_snapshot=universal_snapshot,
    )

    assert calls == [
        (
            module.DEFAULT_PRECISION,
            (
                universal_snapshot.lepton_level,
                universal_snapshot.quark_level,
                universal_snapshot.parent_level,
            ),
        )
    ]
    assert snapshot.proton_electron_mass_ratio == 1999.5



def test_build_sync_snapshot_rejects_universal_constant_mismatch(tmp_path: Path) -> None:
    module = _load_script_module()
    universal_constants_path = tmp_path / "universal_constants.yaml"
    universal_constants_path.write_text(yaml.safe_dump(_sample_universal_constants(), sort_keys=False), encoding="utf-8")
    universal_snapshot = module._build_universal_constants_snapshot(
        module.ConfigLoader(universal_constants_path=universal_constants_path)
    )

    with pytest.raises(ValueError, match=r"does not match the configured universal constants"):
        module.build_sync_snapshot_with_universal_constants(
            _sample_residual_payload(),
            universal_snapshot=universal_snapshot,
        )


def test_resolve_sync_path_anchors_repo_relative_targets() -> None:
    module = _load_script_module()

    assert module._resolve_sync_path(Path("results/residuals.json"), base_dir=module.ProjectPaths.RESULTS) == (
        module.ProjectPaths.RESULTS / "residuals.json"
    ).resolve()
    assert module._resolve_sync_path(Path("residuals.json"), base_dir=module.ProjectPaths.RESULTS) == (
        module.ProjectPaths.RESULTS / "residuals.json"
    ).resolve()
    assert module._resolve_sync_path(Path("papers/physics_constants.tex"), base_dir=module.ProjectPaths.PAPERS) == (
        module.ProjectPaths.PAPERS / "physics_constants.tex"
    ).resolve()
    assert module._resolve_sync_path(Path("physics_constants.tex"), base_dir=module.ProjectPaths.PAPERS) == (
        module.ProjectPaths.PAPERS / "physics_constants.tex"
    ).resolve()
    assert module._resolve_sync_path(Path("config/physics_profiles/standard_model.yaml"), base_dir=module.ProjectPaths.CONFIG) == (
        module.ProjectPaths.CONFIG / "physics_profiles" / "standard_model.yaml"
    ).resolve()
