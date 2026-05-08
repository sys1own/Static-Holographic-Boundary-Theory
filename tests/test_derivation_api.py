from __future__ import annotations

import contextlib
import importlib.util
import io
import runpy
import sys
from fractions import Fraction
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import shbt
import shbt.evolutionary_engine as evolutionary_engine
from shbt.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core import derivation_api
from shbt.core import EvolutionaryEngine as core_package_evolutionary_engine
from shbt.core import evolutionary_engine as core_evolutionary_engine
from shbt.core.derivation_api import DEFAULT_PRECISION, TopologicalVacuum, UniverseFactory


ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, relative_path: str):
    script_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_universe_factory_generates_universe_ledger() -> None:
    ledger = UniverseFactory.generate_ledger(kind="universe")

    assert ledger.startswith("Derivation Ledger")
    assert "Alpha Surface Inverse" in ledger
    assert "Mu Audit" in ledger
    assert "Unity of Scale Identity" in ledger


def test_calculate_physical_ledger_uses_benchmark_topological_vacuum() -> None:
    physical_ledger = UniverseFactory.calculate_physical_ledger()

    assert physical_ledger.vacuum.branch == (26, 8, 312)
    assert physical_ledger.alpha_surface.alpha_inverse_fraction == Fraction(2340, 17)
    assert physical_ledger.proton_ratio.relative_error <= physical_ledger.proton_ratio.tolerance
    assert physical_ledger.mass_bridge.neutrino_floor_mev > 0


def test_import_shbt_exposes_universe_factory() -> None:
    assert shbt.UniverseFactory is UniverseFactory
    assert shbt.DEFAULT_PRECISION == DEFAULT_PRECISION


def test_evolutionary_engine_aliases_universe_factory() -> None:
    assert shbt.EvolutionaryEngine is UniverseFactory
    assert core_package_evolutionary_engine is UniverseFactory
    assert evolutionary_engine.EvolutionaryEngine is UniverseFactory
    assert evolutionary_engine.UniverseFactory is UniverseFactory
    assert core_evolutionary_engine.EvolutionaryEngine is UniverseFactory


def test_universe_factory_can_derive_proton_ratio_for_nonbenchmark_vacuum() -> None:
    derivation = UniverseFactory.derive_proton_ratio(
        vacuum=TopologicalVacuum(lepton_level=30, quark_level=10, parent_level=360, generation_count=18)
    )

    assert derivation.mu_audit > 0


def test_universe_factory_runtime_kernel_recenters_branch_sensitive_derivations() -> None:
    derivation = UniverseFactory.derive_alpha_surface(
        vacuum=TopologicalVacuum(lepton_level=30, quark_level=10, parent_level=360, generation_count=18)
    )

    assert derivation.visible_support == LEPTON_LEVEL + QUARK_LEVEL
    assert derivation.level_density_ratio == Fraction(PARENT_LEVEL, LEPTON_LEVEL + QUARK_LEVEL)


def test_universe_factory_generates_lambda_ledger() -> None:
    ledger = UniverseFactory.generate_ledger(kind="lambda")

    assert ledger.startswith("Lambda Ledger")
    assert "Holographic Surface Tension" in ledger
    assert "Checked-In Benchmark Payloads" in ledger


def test_universe_factory_generate_ledger_accepts_compatibility_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def _fake_build_derivation_ledger(cls, *, precision: int = DEFAULT_PRECISION) -> str:
        calls.append(precision)
        return f"derivation:{precision}"

    monkeypatch.setattr(derivation_api.UniverseFactory, "build_derivation_ledger", classmethod(_fake_build_derivation_ledger))

    ledger = UniverseFactory.generate_ledger(kind="universe", precision=64, legacy_mode=True)

    assert ledger == "derivation:64"
    assert calls == [64]


def test_universe_factory_generate_residual_payload_accepts_compatibility_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_main = ModuleType("shbt.main")
    fake_main.build_quantified_two_loop_residuals = lambda: {"transport_residuals": {"theta12": 0.125}}
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
    fake_lambda_surface = SimpleNamespace(lambda_holo_si_m2=1.0e-52, anchor_lambda_si_m2=1.1e-52)

    monkeypatch.setattr(
        derivation_api.UniverseFactory,
        "calculate_physical_ledger",
        classmethod(lambda cls, *, precision=50: fake_ledger),
    )
    monkeypatch.setattr(
        derivation_api.UniverseFactory,
        "derive_lambda_surface",
        classmethod(lambda cls, *, precision=50: fake_lambda_surface),
    )

    payload = UniverseFactory.generate_residual_payload(precision=64, legacy_mode=True)

    assert payload["artifact"] == "Quantified Two-Loop Residuals"
    assert payload["benchmark_tuple"] == [26, 8, 312]
    assert payload["derivation_residues"]["precision"] == 64


def test_universe_factory_rejects_unknown_ledger_kind() -> None:
    with pytest.raises(ValueError, match=r"Unknown ledger kind"):
        UniverseFactory.generate_ledger(kind="unknown")


def test_refactored_derive_universe_script_delegates_to_universe_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    def _fake_generate_ledger(cls, *, kind: str = "universe", precision: int = DEFAULT_PRECISION) -> str:
        calls.append((kind, precision))
        return f"{kind}:{precision}"

    monkeypatch.setattr(derivation_api.UniverseFactory, "generate_ledger", classmethod(_fake_generate_ledger))
    script_path = ROOT / "scripts" / "derive_universe.py"
    monkeypatch.setattr(sys, "argv", [str(script_path), "--precision", "80"])
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(script_path), run_name="__main__")

    assert exc_info.value.code == 0
    assert calls == [("universe", 80)]
    assert buffer.getvalue().strip() == "universe:80"


def test_refactored_derive_lambda_script_delegates_to_universe_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    def _fake_generate_ledger(cls, *, kind: str = "universe", precision: int = DEFAULT_PRECISION) -> str:
        calls.append((kind, precision))
        return f"{kind}:{precision}"

    monkeypatch.setattr(derivation_api.UniverseFactory, "generate_ledger", classmethod(_fake_generate_ledger))
    script_path = ROOT / "scripts" / "derive_lambda.py"
    monkeypatch.setattr(sys, "argv", [str(script_path), "--precision", "80"])
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(script_path), run_name="__main__")

    assert exc_info.value.code == 0
    assert calls == [("lambda", 80)]
    assert buffer.getvalue().strip() == "lambda:80"


def test_derive_proton_ratio_uses_universe_factory_kappa(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    original = derivation_api.UniverseFactory.derive_kappa_d5

    def _wrapped(cls, *, precision: int = DEFAULT_PRECISION):
        calls.append(precision)
        return original(precision=precision)

    monkeypatch.setattr(derivation_api.UniverseFactory, "derive_kappa_d5", classmethod(_wrapped))

    module = _load_script_module("derive_proton_ratio", "scripts/derive_proton_ratio.py")
    derivation = module.derive_proton_ratio()

    assert calls
    assert derivation.relative_error <= derivation.tolerance
