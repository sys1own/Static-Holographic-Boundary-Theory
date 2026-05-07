from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from shbt.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.config_loader import ConfigLoader, DEFAULT_PHYSICS_PROFILE_RELATIVE_PATH
from shbt.core import differential_geometry, numerics
from shbt.core import holographic_error_stabilizer as stabilizer_module
from shbt.core.evolutionary_engine import EvolutionaryEngine


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "derive_universe.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("derive_universe", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_numerics_module_is_compatibility_surface_for_differential_geometry() -> None:
    assert numerics.require_real_array is differential_geometry.require_real_array
    assert numerics.require_real_scalar is differential_geometry.require_real_scalar
    assert numerics.MetricTensor is differential_geometry.MetricTensor


def test_evolutionary_engine_exposes_programmatic_derivation_api() -> None:
    assert hasattr(EvolutionaryEngine, "generate_ledger")
    ledger = EvolutionaryEngine.generate_ledger(kind="universe")

    assert "Derivation Ledger" in ledger
    assert "alpha_surf^-1 = G_SM * K/(k_l + k_q) = 2340/17" in ledger


def test_derive_universe_script_remains_cli_wrapper_over_engine() -> None:
    module = _load_script_module()

    assert module.DEFAULT_PRECISION >= 50
    assert module.EvolutionaryEngine is EvolutionaryEngine


def test_evolutionary_engine_residue_dictionary_exposes_branch_aliases() -> None:
    residues = EvolutionaryEngine.build_residue_dictionary()

    assert residues["k_l"] == LEPTON_LEVEL
    assert residues["k_q"] == QUARK_LEVEL
    assert residues["K"] == PARENT_LEVEL


def test_evolutionary_engine_derivations_run_boundary_stabilizer(monkeypatch) -> None:
    calls: list[int] = []

    monkeypatch.setattr(stabilizer_module, "_verify_boundary_lock", lambda precision=stabilizer_module.DEFAULT_PRECISION: calls.append(int(precision)))

    residues = EvolutionaryEngine.build_residue_dictionary(precision=60)

    assert residues["K"] == PARENT_LEVEL
    assert len(calls) >= 2
    assert all(call >= 60 for call in calls)


def test_evolutionary_engine_exposes_quantified_residual_payload(monkeypatch) -> None:
    base_payload = {
        "transport_residuals": {"theta12": 0.125},
        "artifact_filename": "residuals.json",
    }
    fake_main = ModuleType("shbt.main")
    fake_main.build_quantified_two_loop_residuals = lambda: dict(base_payload)
    monkeypatch.setitem(sys.modules, "shbt.main", fake_main)

    fake_ledger = SimpleNamespace(
        vacuum=SimpleNamespace(lepton_level=26, quark_level=8, parent_level=312),
        alpha_surface=SimpleNamespace(
            alpha_inverse_decimal=137.64705882352942,
            codata_alpha_inverse=137.035999084,
        ),
        mass_bridge=SimpleNamespace(neutrino_floor_ev=0.0008, neutrino_floor_mev=0.8),
        unity_of_scale=SimpleNamespace(
            epsilon_lambda=1.2e-122,
            decimal_tolerance=1.0e-50,
            register_noise_floor=1.0e-80,
        ),
    )
    fake_lambda_surface = SimpleNamespace(
        lambda_holo_si_m2=1.0e-52,
        anchor_lambda_si_m2=1.1e-52,
    )
    monkeypatch.setattr(
        EvolutionaryEngine,
        "calculate_physical_ledger",
        classmethod(lambda cls, *, precision=50: fake_ledger),
    )
    monkeypatch.setattr(
        EvolutionaryEngine,
        "derive_lambda_surface",
        classmethod(lambda cls, *, precision=50: fake_lambda_surface),
    )

    actual_payload = EvolutionaryEngine.generate_residual_payload()

    assert {"transport_residuals": base_payload["transport_residuals"]}.items() <= actual_payload.items()
    assert actual_payload["artifact"] == "Quantified Two-Loop Residuals"
    assert "derivation_residues" in actual_payload
    assert actual_payload["benchmark_tuple"] == [26, 8, 312]


def test_config_loader_defaults_to_standard_model_profile() -> None:
    loader = ConfigLoader()

    assert loader.physics_profile_path == DEFAULT_PHYSICS_PROFILE_RELATIVE_PATH
    profile = loader.load_physics_profile()

    assert "tier_1" in profile
    assert "tier_2" in profile
    assert loader.load_universal_constants() == profile
