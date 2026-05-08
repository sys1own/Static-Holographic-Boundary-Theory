from __future__ import annotations

from pathlib import Path
import tomllib


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_uv_lockfile_support_is_declared_in_pyproject() -> None:
    pyproject = tomllib.loads((ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["tool"]["uv"]["managed"] is True
    assert pyproject["tool"]["uv"]["default-groups"] == ["dev"]
    assert "pytest==9.0.2" in pyproject["dependency-groups"]["dev"]
    assert (ROOT_DIR / "requirements.lock").is_file()


def test_legacy_data_directory_is_removed_and_assets_are_migrated() -> None:
    assert not (ROOT_DIR / "data").exists()
    assert (ROOT_DIR / "config" / "physics_profiles" / "nufit_5_3.json").is_file()
    assert (ROOT_DIR / "config" / "physics_profiles" / "cmb_power_spectrum_benchmarks.json").is_file()
    assert (ROOT_DIR / "config" / "physics_profiles" / "external_triggers" / "README.md").is_file()
    assert (ROOT_DIR / "config" / "physics_profile_hashes.json").is_file()
    assert (ROOT_DIR / "src" / "shbt" / "live_h0_bridge.py").is_file()


def test_justfile_exposes_locked_audit_manuscript_and_integrity_recipes() -> None:
    justfile = (ROOT_DIR / "Justfile").read_text(encoding="utf-8")

    assert "audit:" in justfile
    assert "manuscript:" in justfile
    assert "verify-integrity:" in justfile
    assert justfile.count("python scripts/verify_dependency_lock.py") == 3
