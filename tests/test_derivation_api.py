from __future__ import annotations

import contextlib
import importlib.util
import io
import runpy
import sys
from fractions import Fraction
from pathlib import Path

import pytest

import shbt
from shbt.core import derivation_api
from shbt.core.derivation_api import DEFAULT_PRECISION, UniverseFactory


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


def test_universe_factory_generates_lambda_ledger() -> None:
    ledger = UniverseFactory.generate_ledger(kind="lambda")

    assert ledger.startswith("Lambda Ledger")
    assert "Holographic Surface Tension" in ledger
    assert "Checked-In Benchmark Payloads" in ledger


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
