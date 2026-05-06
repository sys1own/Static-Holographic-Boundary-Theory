from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import shbt.main as tn
from shbt.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.config_loader import ConfigLoader, DEFAULT_PHYSICS_PROFILE_RELATIVE_PATH
from shbt.core import differential_geometry, numerics
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


def test_evolutionary_engine_exposes_quantified_residual_payload(monkeypatch) -> None:
    expected_payload = {"artifact": "Quantified Two-Loop Residuals"}
    captured: dict[str, object] = {}

    def _fake_build_quantified_two_loop_residuals(*, model=None):
        captured["model"] = model
        return expected_payload

    monkeypatch.setattr(tn, "build_quantified_two_loop_residuals", _fake_build_quantified_two_loop_residuals)

    assert EvolutionaryEngine.generate_residual_payload() == expected_payload
    assert captured["model"] == EvolutionaryEngine.benchmark_vacuum()


def test_derive_universe_script_remains_cli_wrapper_over_engine() -> None:
    module = _load_script_module()

    assert module.DEFAULT_PRECISION >= 50
    assert module.EvolutionaryEngine is EvolutionaryEngine


def test_evolutionary_engine_residue_dictionary_exposes_branch_aliases() -> None:
    residues = EvolutionaryEngine.build_residue_dictionary()

    assert residues["k_l"] == LEPTON_LEVEL
    assert residues["k_q"] == QUARK_LEVEL
    assert residues["K"] == PARENT_LEVEL


def test_config_loader_defaults_to_standard_model_profile() -> None:
    loader = ConfigLoader()

    assert loader.physics_profile_path == DEFAULT_PHYSICS_PROFILE_RELATIVE_PATH
    profile = loader.load_physics_profile()

    assert "tier_1" in profile
    assert "tier_2" in profile
    assert loader.load_universal_constants() == profile
