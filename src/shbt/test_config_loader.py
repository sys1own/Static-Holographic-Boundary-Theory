from __future__ import annotations

from pathlib import Path

import yaml

from shbt.config_loader import ConfigLoader


def _classified(value: object, classification: str) -> dict[str, object]:
    return {"value": value, "classification": classification}


def test_universal_constants_override_benchmark_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    universal_constants_path = tmp_path / "data" / "universal_constants.yaml"
    universal_constants_path.parent.mkdir()

    benchmark_config = {
        "model": {
            "parent_level": _classified(312, "Topological Necessity"),
            "lepton_fixed_point_index": _classified(6, "Topological Necessity"),
            "quark_fixed_point_index": _classified(13, "Topological Necessity"),
            "geometric_kappa": _classified(0.9887710512663789, "Empirical Matching Ansatz"),
            "dirac_higgs_benchmark_mass_gev": _classified(1.0e15, "Empirical Matching Ansatz"),
        },
        "physical_constants": {
            "planck_mass_ev": _classified(1.22089e28, "Empirical Matching Ansatz"),
            "planck_length_m": _classified(1.616255e-35, "Empirical Matching Ansatz"),
            "light_speed_m_per_s": _classified(299792458.0, "Empirical Matching Ansatz"),
            "mpc_in_meters": _classified(3.085677581491367e22, "Empirical Matching Ansatz"),
            "planck2018_h0_km_s_mpc": _classified(67.36, "Empirical Matching Ansatz"),
            "planck2018_h0_sigma_km_s_mpc": _classified(0.54, "Empirical Matching Ansatz"),
            "planck2018_omega_lambda": _classified(0.6847, "Empirical Matching Ansatz"),
            "planck2018_omega_lambda_sigma": _classified(0.0073, "Empirical Matching Ansatz"),
            "planck2018_alpha_em_inv_mz": _classified(127.955, "Empirical Matching Ansatz"),
            "planck2018_sin2_theta_w_mz": _classified(0.23122, "Empirical Matching Ansatz"),
            "planck2018_alpha_s_mz": _classified(0.1179, "Empirical Matching Ansatz"),
        },
    }
    config_path = config_dir / "benchmark_v1.yaml"
    config_path.write_text(yaml.safe_dump(benchmark_config, sort_keys=False), encoding="utf-8")

    universal_constants = {
        "tier_1": {
            "lepton_level": _classified(30, "Topological Necessity"),
            "quark_level": _classified(10, "Topological Necessity"),
            "parent_level": _classified(360, "Topological Necessity"),
            "g_sm": _classified(18, "Topological Necessity"),
        },
        "tier_2": {
            "planck_mass_ev": _classified(1.23e28, "Empirical Matching Ansatz"),
            "planck_length_m": _classified(1.61e-35, "Empirical Matching Ansatz"),
            "light_speed_m_per_s": _classified(299792458.0, "Empirical Matching Ansatz"),
            "mpc_in_meters": _classified(3.085677581491367e22, "Empirical Matching Ansatz"),
            "planck2018_h0_km_s_mpc": _classified(70.0, "Empirical Matching Ansatz"),
            "planck2018_h0_sigma_km_s_mpc": _classified(0.4, "Empirical Matching Ansatz"),
            "planck2018_omega_lambda": _classified(0.7, "Empirical Matching Ansatz"),
            "planck2018_omega_lambda_sigma": _classified(0.006, "Empirical Matching Ansatz"),
            "planck2018_lambda_si_m2": _classified(9.99e-53, "Empirical Matching Ansatz"),
            "planck2018_lambda_fractional_sigma": _classified(0.0123, "Empirical Matching Ansatz"),
            "planck2018_alpha_em_inv_mz": _classified(128.1, "Empirical Matching Ansatz"),
            "planck2018_sin2_theta_w_mz": _classified(0.2315, "Empirical Matching Ansatz"),
            "planck2018_alpha_s_mz": _classified(0.1181, "Empirical Matching Ansatz"),
            "codata_fine_structure_alpha_inverse": _classified(140.0, "Empirical Matching Ansatz"),
            "hbar_ev_seconds": _classified(6.7e-16, "Empirical Matching Ansatz"),
        },
    }
    universal_constants_path.write_text(yaml.safe_dump(universal_constants, sort_keys=False), encoding="utf-8")

    loader = ConfigLoader(
        config_dir=config_dir,
        universal_constants_path=universal_constants_path,
    )

    normalized_universal_constants = loader.load_universal_constants()
    physics_constants = loader.load_physics_constants()
    classifications = loader.load_parameter_classifications()

    assert normalized_universal_constants["tier_1"]["g_sm"] == 18
    assert physics_constants["model"]["parent_level"] == 360
    assert physics_constants["model"]["lepton_fixed_point_index"] == 6
    assert physics_constants["model"]["quark_fixed_point_index"] == 12
    assert physics_constants["model"]["g_sm"] == 18
    assert physics_constants["physical_constants"]["planck_mass_ev"] == 1.23e28
    assert physics_constants["physical_constants"]["planck2018_lambda_si_m2"] == 9.99e-53
    assert physics_constants["physical_constants"]["codata_fine_structure_alpha_inverse"] == 140.0
    assert physics_constants["physical_constants"]["hbar_ev_seconds"] == 6.7e-16
    assert classifications["model.g_sm"] == "Topological Necessity"
    assert classifications["physical_constants.codata_fine_structure_alpha_inverse"] == "Empirical Matching Ansatz"
