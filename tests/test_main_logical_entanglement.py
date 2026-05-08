from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import shbt.main as main_module


@pytest.fixture()
def relaxed_guardian_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module.RigidityGuardian, "_global_parity_checkpoint", lambda self, **kwargs: None)
    monkeypatch.setattr(main_module.RigidityGuardian, "_validate_result_parity", lambda self, **kwargs: None)


@pytest.fixture()
def stub_model() -> SimpleNamespace:
    return SimpleNamespace(
        parent_level=main_module.PARENT_LEVEL,
        lepton_level=main_module.LEPTON_LEVEL,
        quark_level=main_module.QUARK_LEVEL,
        verify_bulk_emergence=lambda: SimpleNamespace(
            bulk_emergent=True,
            torsion_free=True,
            non_singular_bulk=True,
            lambda_aligned=True,
            parity_bit_density_constraint_satisfied=True,
        ),
    )


def test_flavor_sector_requires_gravity_metric_lock(
    relaxed_guardian_checks: None,
    stub_model: SimpleNamespace,
) -> None:
    guardian = main_module.RigidityGuardian(model=stub_model)

    with pytest.raises(main_module.BenchmarkExecutionError, match=r"Gravity sector locks the metric tensor"):
        guardian.calculate(
            lambda: "flavor",
            sector_name="flavor",
            label="pmns transport",
        )


def test_targeted_flavor_audit_locks_gravity_before_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relaxed_guardian_checks: None,
) -> None:
    events: list[tuple[str, str]] = []

    def fake_ensure_metric_tensor_locked(self: main_module.RigidityGuardian) -> tuple[bool, ...]:
        events.append(("gravity", "metric tensor lock"))
        self.metric_tensor_locked = True
        self.metric_tensor_signature = (True, True, True, True, True)
        return self.metric_tensor_signature

    def fake_capture_sector_module_report(
        module_name: str,
        *,
        output_dir: Path,
        sector: str,
        result_holder: list[object] | None = None,
    ) -> Path:
        del result_holder
        events.append((sector, module_name))
        return output_dir / f"{sector}_{module_name.rsplit('.', 1)[-1]}.txt"

    monkeypatch.setattr(main_module, "_ensure_audit_resources", lambda: ())
    monkeypatch.setattr(main_module.RigidityGuardian, "ensure_metric_tensor_locked", fake_ensure_metric_tensor_locked)
    monkeypatch.setattr(main_module, "_capture_sector_module_report", fake_capture_sector_module_report)

    report_paths = main_module.run_targeted_sector_audits(sector="flavor", output_dir=tmp_path)

    assert events[0] == ("gravity", "metric tensor lock")
    assert [sector for sector, _ in events[1:]] == ["flavor", "flavor", "flavor"]
    assert len(report_paths) == len(main_module.SECTOR_AUDIT_MODULES["flavor"])


def test_targeted_flavor_audit_requires_rigid_metric_signal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relaxed_guardian_checks: None,
) -> None:
    monkeypatch.setattr(main_module, "_ensure_audit_resources", lambda: ())
    monkeypatch.setattr(
        main_module.RigidityGuardian,
        "ensure_metric_tensor_locked",
        lambda self: setattr(self, "metric_tensor_signature", (True, True, True, True, True)) or self.metric_tensor_signature,
    )

    with pytest.raises(main_module.BenchmarkExecutionError, match=r"Metric tensor not rigid"):
        main_module.run_targeted_sector_audits(sector="flavor", output_dir=tmp_path)


def test_guardian_rejects_manual_higgs_tuning(
    relaxed_guardian_checks: None,
    stub_model: SimpleNamespace,
) -> None:
    guardian = main_module.RigidityGuardian(model=stub_model)

    with pytest.raises(main_module.BenchmarkExecutionError, match=r"manual tuning of m_126_gev"):
        guardian.calculate(
            lambda **kwargs: kwargs,
            m_126_gev=main_module._BENCHMARK_HIGGS_MATCHING_THRESHOLD_GEV * 1.01,
            sector_name="rigidity",
            label="manual higgs tuning",
        )


def test_guardian_rejects_manual_cosmological_constant_tuning(
    relaxed_guardian_checks: None,
    stub_model: SimpleNamespace,
) -> None:
    guardian = main_module.RigidityGuardian(model=stub_model)

    with pytest.raises(main_module.BenchmarkExecutionError, match=r"manual tuning of lambda_anchor_si_m2"):
        guardian.calculate(
            lambda **kwargs: kwargs,
            lambda_anchor_si_m2=main_module._BENCHMARK_LAMBDA_OBS_SI_M2 * 1.01,
            sector_name="cosmology",
            label="manual lambda tuning",
        )


def test_holographic_noise_floor_grants_metric_lock_and_clears_benchmark_tensor(
    relaxed_guardian_checks: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    gravity_result = SimpleNamespace(
        bulk_emergent=False,
        torsion_free=True,
        non_singular_bulk=True,
        lambda_aligned=True,
        parity_bit_density_constraint_satisfied=True,
        relative_mismatch=2.07e-16,
        E_mu_nu=np.ones((4, 4), dtype=float),
    )
    benchmark_model = SimpleNamespace(
        parent_level=main_module.PARENT_LEVEL,
        lepton_level=main_module.LEPTON_LEVEL,
        quark_level=main_module.QUARK_LEVEL,
        verify_bulk_emergence=lambda: gravity_result,
    )
    guardian = main_module.RigidityGuardian(model=benchmark_model)

    caplog.set_level("INFO")
    result = guardian.ensure_metric_tensor_locked()

    assert result is gravity_result
    assert guardian.metric_tensor_signature == (True, True, True, True, True)
    assert gravity_result.bulk_emergent is True
    assert np.array_equal(gravity_result.E_mu_nu, np.zeros((4, 4), dtype=float))
    assert "[RIGIDITY]: Mismatch within noise floor. Granting lock." in caplog.text
    assert "[RIGIDITY]: Metric tensor lock secured at (26, 8, 312)." in caplog.text


def test_metric_lock_rejects_mismatch_above_holographic_noise_floor(
    relaxed_guardian_checks: None,
) -> None:
    gravity_result = SimpleNamespace(
        bulk_emergent=False,
        torsion_free=True,
        non_singular_bulk=True,
        lambda_aligned=True,
        parity_bit_density_constraint_satisfied=True,
        relative_mismatch=2.07e-15,
        E_mu_nu=np.ones((2, 2), dtype=float),
    )
    benchmark_model = SimpleNamespace(
        parent_level=main_module.PARENT_LEVEL,
        lepton_level=main_module.LEPTON_LEVEL,
        quark_level=main_module.QUARK_LEVEL,
        verify_bulk_emergence=lambda: gravity_result,
    )
    guardian = main_module.RigidityGuardian(model=benchmark_model)

    with pytest.raises(main_module.BenchmarkExecutionError, match=r"failed to lock the metric tensor.*mismatch:"):
        guardian.ensure_metric_tensor_locked()

    assert np.array_equal(gravity_result.E_mu_nu, np.ones((2, 2), dtype=float))


def test_metric_lock_rejects_mismatch_at_holographic_noise_floor(
    relaxed_guardian_checks: None,
) -> None:
    gravity_result = SimpleNamespace(
        bulk_emergent=False,
        torsion_free=True,
        non_singular_bulk=True,
        lambda_aligned=True,
        parity_bit_density_constraint_satisfied=True,
        relative_mismatch=main_module.HOLOGRAPHIC_NOISE_FLOOR,
    )
    benchmark_model = SimpleNamespace(
        parent_level=main_module.PARENT_LEVEL,
        lepton_level=main_module.LEPTON_LEVEL,
        quark_level=main_module.QUARK_LEVEL,
        verify_bulk_emergence=lambda: gravity_result,
    )
    guardian = main_module.RigidityGuardian(model=benchmark_model)

    with pytest.raises(main_module.BenchmarkExecutionError, match=r"failed to lock the metric tensor.*mismatch:"):
        guardian.ensure_metric_tensor_locked()

    assert gravity_result.bulk_emergent is False


def test_universal_aggregate_ignores_subthreshold_sector_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relaxed_guardian_checks: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(main_module, "_ensure_audit_resources", lambda: ())
    monkeypatch.setitem(main_module.SECTOR_AUDIT_MODULES, "complexity", ("shbt.sectors.complexity_sector",))

    def fake_capture_sector_module_report(
        module_name: str,
        *,
        output_dir: Path,
        sector: str,
        result_holder: list[object] | None = None,
    ) -> Path:
        del module_name
        if result_holder is not None:
            result_holder.append(
                SimpleNamespace(
                    success=False,
                    relative_mismatch=main_module.HOLOGRAPHIC_NOISE_FLOOR / 10.0,
                )
            )
        return output_dir / f"{sector}_report.txt"

    monkeypatch.setattr(main_module, "_capture_sector_module_report", fake_capture_sector_module_report)

    report_paths = main_module.run_targeted_sector_audits(sector="complexity", output_dir=tmp_path)
    captured = capsys.readouterr()

    assert report_paths == (tmp_path / "complexity_report.txt",)
    assert "hardware-level residue" in captured.out



def test_universal_aggregate_rejects_meaningful_sector_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relaxed_guardian_checks: None,
) -> None:
    monkeypatch.setattr(main_module, "_ensure_audit_resources", lambda: ())
    monkeypatch.setitem(main_module.SECTOR_AUDIT_MODULES, "complexity", ("shbt.sectors.complexity_sector",))

    def fake_capture_sector_module_report(
        module_name: str,
        *,
        output_dir: Path,
        sector: str,
        result_holder: list[object] | None = None,
    ) -> Path:
        del module_name
        if result_holder is not None:
            result_holder.append(
                SimpleNamespace(
                    success=False,
                    relative_mismatch=main_module.HOLOGRAPHIC_NOISE_FLOOR * 10.0,
                )
            )
        return output_dir / f"{sector}_report.txt"

    monkeypatch.setattr(main_module, "_capture_sector_module_report", fake_capture_sector_module_report)

    with pytest.raises(main_module.BenchmarkExecutionError, match=r"Universal audit aggregate lock failed"):
        main_module.run_targeted_sector_audits(sector="complexity", output_dir=tmp_path)


def test_main_returns_zero_for_successful_targeted_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main_module.ProjectPaths, "ensure_dirs", lambda: None)
    monkeypatch.setattr(main_module, "_ensure_audit_resources", lambda: ())
    monkeypatch.setattr(main_module, "configure_reporting", lambda **kwargs: None)
    monkeypatch.setattr(main_module, "_emit_shbt_branding", lambda **kwargs: None)
    monkeypatch.setattr(
        main_module,
        "parse_args",
        lambda argv=None: SimpleNamespace(
            manuscript_dir=tmp_path,
            output_dir=tmp_path,
            sector="complexity",
            zero_parameter=False,
            residue_check=False,
            audit_generation_3=False,
            master_transport_audit=False,
            master_transport_shift=main_module._master_transport.DEFAULT_RIGIDITY_SHIFT_FRACTION,
            quiet=True,
            log_file=None,
        ),
    )
    monkeypatch.setattr(
        main_module,
        "run_targeted_sector_audits",
        lambda *, sector, output_dir: (output_dir / f"{sector}_report.txt",),
    )

    assert main_module.main([]) == 0



def test_main_returns_one_for_failed_targeted_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main_module.ProjectPaths, "ensure_dirs", lambda: None)
    monkeypatch.setattr(main_module, "_ensure_audit_resources", lambda: ())
    monkeypatch.setattr(main_module, "configure_reporting", lambda **kwargs: None)
    monkeypatch.setattr(main_module, "_emit_shbt_branding", lambda **kwargs: None)
    monkeypatch.setattr(
        main_module,
        "parse_args",
        lambda argv=None: SimpleNamespace(
            manuscript_dir=tmp_path,
            output_dir=tmp_path,
            sector="complexity",
            zero_parameter=False,
            residue_check=False,
            audit_generation_3=False,
            master_transport_audit=False,
            master_transport_shift=main_module._master_transport.DEFAULT_RIGIDITY_SHIFT_FRACTION,
            quiet=True,
            log_file=None,
        ),
    )

    def fail_targeted_audit(*, sector: str | None, output_dir: Path) -> tuple[Path, ...]:
        del sector, output_dir
        raise main_module.BenchmarkExecutionError("synthetic aggregate failure")

    monkeypatch.setattr(main_module, "run_targeted_sector_audits", fail_targeted_audit)

    assert main_module.main([]) == 1
