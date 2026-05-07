from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import shbt.core.bootstrap as bootstrap_module
import shbt.main as main_module


def test_parse_args_accepts_zero_parameter_flag() -> None:
    args = main_module.parse_args(["--zero-parameter"])

    assert args.zero_parameter is True


def test_main_initializes_zero_parameter_mode_before_sector_audits(monkeypatch, tmp_path: Path) -> None:
    events: list[tuple[object, ...]] = []

    def fake_initialize_from_geometry(
        *,
        lepton_level: int,
        quark_level: int,
        parent_level: int,
        generation_count: int,
        vacuum_pressure: float,
    ) -> SimpleNamespace:
        events.append((
            "bootstrap",
            lepton_level,
            quark_level,
            parent_level,
            generation_count,
            vacuum_pressure,
        ))
        return SimpleNamespace(
            kernel=SimpleNamespace(branch=(lepton_level, quark_level, parent_level)),
            stable_eigenvalue=main_module.ZERO_PARAMETER_STABLE_EIGENVALUE,
        )

    def fake_run_targeted_sector_audits(*, sector: str | None, output_dir: Path) -> tuple[Path, ...]:
        events.append(("sector", sector, output_dir))
        return ()

    monkeypatch.setattr(main_module.ProjectPaths, "ensure_dirs", classmethod(lambda cls: None))
    monkeypatch.setattr(main_module, "_ensure_audit_resources", lambda: ())
    monkeypatch.setattr(main_module, "configure_reporting", lambda **kwargs: None)
    monkeypatch.setattr(main_module, "_emit_shbt_branding", lambda **kwargs: None)
    monkeypatch.setattr(bootstrap_module, "initialize_from_geometry", fake_initialize_from_geometry)
    monkeypatch.setattr(main_module, "run_targeted_sector_audits", fake_run_targeted_sector_audits)

    main_module.main([
        "--zero-parameter",
        "--sector",
        "rigidity",
        "--output-dir",
        str(tmp_path),
        "--quiet",
    ])

    assert events[0] == (
        "bootstrap",
        main_module.LEPTON_LEVEL,
        main_module.QUARK_LEVEL,
        main_module.PARENT_LEVEL,
        main_module.G_SM,
        main_module.BENCHMARK_VACUUM_PRESSURE,
    )
    assert events[1] == ("sector", "rigidity", tmp_path)
