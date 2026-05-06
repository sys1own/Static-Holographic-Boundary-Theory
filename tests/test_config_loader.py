from __future__ import annotations

from pathlib import Path

import yaml

from shbt.config_loader import COUNTER_UNIVERSAL_CLASSIFICATION, ConfigLoader


def _classified(value: object, classification: str) -> dict[str, object]:
    return {"value": value, "classification": classification}


def test_physics_profile_overrides_benchmark_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    physics_profile_path = config_dir / "physics_profiles" / "standard_model.yaml"
    physics_profile_path.parent.mkdir()

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

    physics_profile = {
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
    physics_profile_path.write_text(yaml.safe_dump(physics_profile, sort_keys=False), encoding="utf-8")

    loader = ConfigLoader(
        config_dir=config_dir,
        physics_profile_path=physics_profile_path,
    )

    normalized_universal_constants = loader.load_physics_profile()
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


def test_injected_counter_universal_parameters_override_profile_values(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    benchmark_config = {
        "model": {
            "parent_level": _classified(312, "Topological Necessity"),
            "lepton_fixed_point_index": _classified(6, "Topological Necessity"),
            "quark_fixed_point_index": _classified(13, "Topological Necessity"),
            "g_sm": _classified(15, "Topological Necessity"),
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
            "codata_fine_structure_alpha_inverse": _classified(137.035999084, "Empirical Matching Ansatz"),
            "hbar_ev_seconds": _classified(6.582119569e-16, "Empirical Matching Ansatz"),
        },
    }
    (config_dir / "benchmark_v1.yaml").write_text(yaml.safe_dump(benchmark_config, sort_keys=False), encoding="utf-8")

    injected_profile = {
        "tier_1": {
            "lepton_level": 30,
            "quark_level": 10,
            "parent_level": 360,
            "g_sm": 18,
        },
        "tier_2": {
            "planck_mass_ev": 1.23e28,
            "planck_length_m": 1.61e-35,
            "light_speed_m_per_s": 299792458.0,
            "mpc_in_meters": 3.085677581491367e22,
            "planck2018_h0_km_s_mpc": 70.0,
            "planck2018_h0_sigma_km_s_mpc": 0.4,
            "planck2018_omega_lambda": 0.7,
            "planck2018_omega_lambda_sigma": 0.006,
            "codata_fine_structure_alpha_inverse": 140.0,
            "hbar_ev_seconds": 6.7e-16,
        },
    }
    loader = ConfigLoader(
        config_dir=config_dir,
        physics_profile=injected_profile,
        physics_parameter_overrides={
            "model": {"g_sm": 21},
            "physical_constants": {"codata_fine_structure_alpha_inverse": 141.5},
        },
    )

    physics_constants = loader.load_physics_constants()
    classifications = loader.load_parameter_classifications()

    assert physics_constants["model"]["parent_level"] == 360
    assert physics_constants["model"]["g_sm"] == 21
    assert physics_constants["physical_constants"]["codata_fine_structure_alpha_inverse"] == 141.5
    assert classifications["model.g_sm"] == COUNTER_UNIVERSAL_CLASSIFICATION
    assert classifications["physical_constants.codata_fine_structure_alpha_inverse"] == COUNTER_UNIVERSAL_CLASSIFICATION


def test_default_compute_cluster_profile_exposes_dask_mpi_and_zarr_defaults() -> None:
    loader = ConfigLoader()

    compute_config = loader.load_compute_cluster_config()

    assert compute_config["cluster"]["default_backend"] == "dask"
    assert compute_config["dask"]["workers"]["threads_per_worker"] >= 1
    assert compute_config["mpi"]["launcher"] == "mpirun"
    assert compute_config["storage"]["tensor_format"] == "zarr"


def test_compute_cluster_profile_uses_project_paths_with_custom_config_dir(tmp_path: Path) -> None:
    custom_config_dir = tmp_path / "isolated-config"
    custom_config_dir.mkdir()

    loader = ConfigLoader(config_dir=custom_config_dir)

    compute_config = loader.load_compute_cluster_config()

    assert compute_config["cluster"]["name"] == "shbt-benchmark-hpc"
    assert compute_config["cluster"]["default_backend"] == "dask"
    assert compute_config["mpi"]["ranks"] == 32
