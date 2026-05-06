from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import shbt.main as tn
from shbt.config_loader import ConfigLoader


ROOT = Path(__file__).resolve().parents[1]
SYNC_SYSTEM_SCRIPT_PATH = ROOT / "scripts" / "sync_system.py"


def _load_sync_system_module():
    spec = importlib.util.spec_from_file_location("sync_system", SYNC_SYSTEM_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_quantified_two_loop_residuals_round_trip_into_sync_system(monkeypatch: pytest.MonkeyPatch) -> None:
    sync_module = _load_sync_system_module()
    universal_snapshot = sync_module._build_universal_constants_snapshot(ConfigLoader())
    benchmark_tuple = (
        universal_snapshot.lepton_level,
        universal_snapshot.quark_level,
        universal_snapshot.parent_level,
    )

    resolved_model = SimpleNamespace(
        target_tuple=benchmark_tuple,
        lepton_level=benchmark_tuple[0],
        quark_level=benchmark_tuple[1],
        parent_level=benchmark_tuple[2],
        scale_ratio=1.0,
        bit_count=1.0,
        kappa_geometric=0.1,
    )
    pmns = SimpleNamespace(
        level=benchmark_tuple[0],
        parent_level=benchmark_tuple[2],
        scale_ratio=1.0,
        bit_count=1.0,
        kappa_geometric=0.1,
    )
    ckm = SimpleNamespace(
        level=benchmark_tuple[1],
        parent_level=benchmark_tuple[2],
        scale_ratio=1.0,
        bit_count=1.0,
        kappa_geometric=0.1,
        gut_threshold_residue=0.0,
    )
    audit = SimpleNamespace(
        redundancy_entropy_cost_nat=0.5,
        support_deficit=0,
        required_inverted_rank=0,
        modularity_limit_rank=0,
        relaxed_inverted_gap=0.0,
    )
    transport_curvature = SimpleNamespace(mass_shift_fraction=0.05)
    unity_payload = {
        "epsilon_lambda": 1.0e-199,
        "exact_epsilon_lambda": 1.0e-199,
        "numerical_residual": 1.0e-199,
        "register_noise_floor": 3.0e-123,
        "exact_register_noise_floor": 3.0e-123,
        "passed": True,
    }
    gauge_payload = {
        "topological_alpha_inverse": universal_snapshot.topological_alpha_inverse,
        "codata_alpha_inverse": 137.035999084,
        "two_loop_residual_fraction": 1.25e-4,
        "two_loop_residual_percent": 1.25e-2,
        "two_loop_residual_pull": 0.25,
    }
    transport_residuals = {
        "theta12": {"fractional_residual": 1.0e-4, "signed_two_loop_shift": 0.1},
        "theta13": {"fractional_residual": 2.0e-4, "signed_two_loop_shift": 0.2},
        "theta23": {"fractional_residual": 3.0e-4, "signed_two_loop_shift": 0.3},
        "delta_cp": {"fractional_residual": 4.0e-4, "signed_two_loop_shift": 0.4},
    }

    monkeypatch.setattr(tn, "_coerce_topological_model", lambda **_kwargs: resolved_model)
    monkeypatch.setattr(tn, "verify_unity_of_scale", lambda *, model: unity_payload)
    monkeypatch.setattr(
        tn,
        "verify_gauge_holography",
        lambda *, model: SimpleNamespace(residual_bookkeeping=gauge_payload),
    )
    monkeypatch.setattr(
        tn,
        "_build_transport_observable_residual_summary",
        lambda *args, **kwargs: transport_residuals,
    )
    monkeypatch.setattr(
        sync_module,
        "derive_transport_curvature_audit",
        lambda **_kwargs: SimpleNamespace(quark_theta_two_loop=(0.01, 0.02, 0.03), quark_delta_two_loop=0.04),
    )
    monkeypatch.setattr(sync_module, "_derive_proton_electron_mass_ratio", lambda *_args, **_kwargs: 1836.152673)
    monkeypatch.setattr(sync_module, "_derive_neutrino_floor_mev", lambda *_args, **_kwargs: 2.83)

    payload = tn.build_quantified_two_loop_residuals(
        pmns=pmns,
        ckm=ckm,
        audit=audit,
        model=resolved_model,
        transport_curvature=transport_curvature,
    )
    snapshot = sync_module.build_sync_snapshot_with_universal_constants(
        payload,
        universal_snapshot=universal_snapshot,
    )

    assert payload["benchmark_tuple"] == list(benchmark_tuple)
    assert payload["unity_of_scale_identity"] == unity_payload
    assert snapshot.benchmark_tuple == benchmark_tuple
    assert snapshot.epsilon_lambda == pytest.approx(unity_payload["epsilon_lambda"])
    assert snapshot.delta_s_red_nat == pytest.approx(audit.redundancy_entropy_cost_nat)
    assert snapshot.mass_scale_two_loop_fraction == pytest.approx(transport_curvature.mass_shift_fraction)
    assert snapshot.lepton_theta12_two_loop_deg == pytest.approx(transport_residuals["theta12"]["signed_two_loop_shift"])
    assert snapshot.gauge_two_loop_residual_fraction == pytest.approx(gauge_payload["two_loop_residual_fraction"])


def test_load_checked_in_benchmark_diagnostics_uses_project_root_and_prefers_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "results"
    final_dir = results_dir / "final"
    final_dir.mkdir(parents=True)

    root_payload = {"source": "root"}
    final_payload = {"source": "final"}
    (results_dir / tn.BENCHMARK_DIAGNOSTICS_FILENAME).write_text(json.dumps(root_payload), encoding="utf-8")
    (final_dir / tn.BENCHMARK_DIAGNOSTICS_FILENAME).write_text(json.dumps(final_payload), encoding="utf-8")

    monkeypatch.setattr(tn.ProjectPaths, "ROOT", tmp_path)

    assert tn._load_checked_in_benchmark_diagnostics() == final_payload

    (final_dir / tn.BENCHMARK_DIAGNOSTICS_FILENAME).unlink()

    assert tn._load_checked_in_benchmark_diagnostics() == root_payload
