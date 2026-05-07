from __future__ import annotations

import importlib.util
from decimal import Decimal
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from mpmath import mp
import numpy as np
import pytest

import shbt.main as tn
from shbt.config_loader import ConfigLoader
from shbt.constants import AUDIT_TOLERANCE, BENCHMARK_MANIFEST_FILENAME
from shbt.export import export_transport_covariance_diagnostics, read_zarr_artifact, write_json_artifact
from shbt.reporting_engine import write_report_tensor_sidecar


ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT_PATH = ROOT / "scripts" / "export.py"
SYNC_SYSTEM_SCRIPT_PATH = ROOT / "scripts" / "sync_system.py"


def _load_sync_system_module():
    spec = importlib.util.spec_from_file_location("sync_system", SYNC_SYSTEM_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_export_module():
    spec = importlib.util.spec_from_file_location("export_script", EXPORT_SCRIPT_PATH)
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


def test_export_script_writes_residual_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_export_module()
    assert module.EvolutionaryEngine is module.UniverseFactory

    derivation_calls: dict[str, int] = {}
    base_payload = {
        "artifact": "Quantified Two-Loop Residuals",
        "benchmark_tuple": [26, 8, 312],
        "unity_of_scale_identity": {
            "epsilon_lambda": 1.0e-199,
            "exact_epsilon_lambda": 1.0e-199,
            "numerical_residual": 1.0e-199,
            "register_noise_floor": 3.0e-123,
            "exact_register_noise_floor": 3.0e-123,
            "passed": True,
        },
        "gauge_residual_bookkeeping": {
            "topological_alpha_inverse": 137.14285714285714,
            "codata_alpha_inverse": 137.035999084,
            "two_loop_residual_fraction": 1.25e-4,
            "two_loop_residual_percent": 1.25e-2,
            "two_loop_residual_pull": 0.25,
        },
        "informational_costs": {"delta_s_red_nat": 0.5},
        "mixing_angle_drifts_deg": {
            "theta12": 0.1,
            "theta13": 0.2,
            "theta23": 0.3,
            "delta_cp": 0.4,
        },
        "mass_scale_two_loop_fraction": 0.05,
    }

    fake_ledger = SimpleNamespace(
        vacuum=SimpleNamespace(lepton_level=26, quark_level=8, parent_level=312),
        alpha_surface=SimpleNamespace(
            alpha_inverse_decimal=Decimal("137.14285714285714"),
            codata_alpha_inverse=Decimal("137.035999084"),
        ),
        mass_bridge=SimpleNamespace(
            neutrino_floor_ev=Decimal("0.00283"),
            neutrino_floor_mev=Decimal("2.83"),
        ),
        unity_of_scale=SimpleNamespace(
            epsilon_lambda=Decimal("1e-199"),
            decimal_tolerance=Decimal("1e-180"),
            register_noise_floor=Decimal("3e-123"),
        ),
    )
    fake_lambda_surface = SimpleNamespace(
        lambda_holo_si_m2=Decimal("1.1056e-52"),
        anchor_lambda_si_m2=Decimal("1.1056e-52"),
    )

    class FakeUniverseFactory:
        @classmethod
        def calculate_physical_ledger(cls, *, precision: int):
            derivation_calls["ledger_precision"] = precision
            return fake_ledger

        @classmethod
        def derive_lambda_surface(cls, *, precision: int):
            derivation_calls["lambda_precision"] = precision
            return fake_lambda_surface

    monkeypatch.setattr(module, "EvolutionaryEngine", FakeUniverseFactory)
    monkeypatch.setattr(module, "build_quantified_two_loop_residuals", lambda: dict(base_payload))

    output_path = tmp_path / "results" / "residuals.json"

    assert module.main(["--output-path", str(output_path), "--precision", "64"]) == 0

    actual_content = json.loads(output_path.read_text(encoding="utf-8"))
    expected_payload = {
        key: base_payload[key]
        for key in (
            "artifact",
            "benchmark_tuple",
            "unity_of_scale_identity",
            "gauge_residual_bookkeeping",
            "informational_costs",
            "mixing_angle_drifts_deg",
            "mass_scale_two_loop_fraction",
        )
    }

    assert derivation_calls == {"ledger_precision": 64, "lambda_precision": 64}
    for key, value in expected_payload.items():
        assert actual_content.get(key) == value
    assert actual_content["derivation_residues"]["benchmark_tuple"] == [26, 8, 312]
    assert actual_content["derivation_residues"]["alpha_inverse_decimal"] == str(
        fake_ledger.alpha_surface.alpha_inverse_decimal
    )
    assert actual_content["derivation_residues"]["m_nu_mev"] == str(fake_ledger.mass_bridge.neutrino_floor_mev)
    assert actual_content["derivation_residues"]["lambda_holo_si_m2"] == str(fake_lambda_surface.lambda_holo_si_m2)


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


def test_transport_covariance_export_writes_zarr_sidecar_for_multidimensional_tensors(tmp_path: Path) -> None:
    output_path = tmp_path / "transport_covariance.json"
    payload = {
        "entropy_map": (
            (Decimal("0.10"), Decimal("0.20")),
            (Decimal("0.30"), Decimal("0.40")),
        ),
        "residual_rank": 2,
    }

    artifact_path = export_transport_covariance_diagnostics(output_path, payload)
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    sidecar_path = output_path.with_suffix(".zarr")
    tensors = read_zarr_artifact(sidecar_path)

    assert artifact_path == output_path
    assert manifest["tensor_artifact"] == sidecar_path.as_posix()
    assert tensors["entropy_map"].shape == (2, 2)
    assert tensors["entropy_map"][0, 0] == "0.10"
    assert tensors["entropy_map"][1, 1] == "0.40"


def test_write_json_artifact_preserves_decimal_and_mpmath_values(tmp_path: Path) -> None:
    output_path = tmp_path / "precision_payload.json"
    decimal_value = Decimal("0.000000000000000000123456789")

    with mp.workdps(50):
        real_value = mp.mpf("1.234567890123456789123456789")
        complex_value = mp.mpc("2.5", "-3.75")
        expected_real = str(real_value)
        expected_complex = {"real": str(complex_value.real), "imag": str(complex_value.imag)}

        write_json_artifact(
            output_path,
            {
                "decimal_value": decimal_value,
                "mpmath_real": real_value,
                "mpmath_complex": complex_value,
            },
        )

    manifest = json.loads(output_path.read_text(encoding="utf-8"))

    assert manifest["decimal_value"] == str(decimal_value)
    assert manifest["mpmath_real"] == expected_real
    assert manifest["mpmath_complex"] == expected_complex


def test_write_json_artifact_injects_manifest_precision_floor(tmp_path: Path) -> None:
    injected_manifest_path = tmp_path / "generated" / BENCHMARK_MANIFEST_FILENAME
    preserved_manifest_path = tmp_path / "preserved" / BENCHMARK_MANIFEST_FILENAME

    write_json_artifact(injected_manifest_path, {"benchmark_tuple": [26, 8, 312]})
    write_json_artifact(
        preserved_manifest_path,
        {
            "benchmark_tuple": [26, 8, 312],
            "holographic_precision_floor": "1e-60",
        },
    )

    injected_manifest = json.loads(injected_manifest_path.read_text(encoding="utf-8"))
    preserved_manifest = json.loads(preserved_manifest_path.read_text(encoding="utf-8"))

    assert injected_manifest["benchmark_tuple"] == [26, 8, 312]
    assert injected_manifest["holographic_precision_floor"] == str(AUDIT_TOLERANCE)
    assert preserved_manifest["benchmark_tuple"] == [26, 8, 312]
    assert preserved_manifest["holographic_precision_floor"] == "1e-60"


def test_reporting_engine_writes_zarr_sidecar_for_report_tensors(tmp_path: Path) -> None:
    report_path = tmp_path / "audit_statement.txt"
    report_path.write_text("Benchmark Consistency Statement\n", encoding="utf-8")

    sidecar_path = write_report_tensor_sidecar(
        report_path,
        {
            "entropy_map": np.asarray(((1.0, 2.0), (3.0, 4.0))),
            "scalar_summary": 1.0,
        },
    )
    tensors = read_zarr_artifact(report_path.with_suffix(".zarr"))

    assert sidecar_path == report_path.with_suffix(".zarr")
    assert tensors["entropy_map"].shape == (2, 2)
    assert float(tensors["entropy_map"][1, 0]) == pytest.approx(3.0, rel=0.0, abs=1.0e-12)
