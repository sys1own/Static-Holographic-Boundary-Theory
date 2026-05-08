from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / "scripts" / "verify_config_integrity.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("verify_config_integrity", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_repository_config_hash_manifest_passes() -> None:
    module = _load_script_module()

    verified, audits = module.verify_config_integrity()

    assert verified is True
    assert audits
    assert all(audit.matches for audit in audits)


def test_verify_config_integrity_detects_hash_drift(tmp_path: Path) -> None:
    module = _load_script_module()
    config_file = tmp_path / "physics_profiles" / "demo.yaml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("value: 26\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"physics_profiles/demo.yaml": "0" * 64}), encoding="utf-8")

    verified, audits = module.verify_config_integrity(manifest_path=manifest_path, root_dir=tmp_path)

    assert verified is False
    assert len(audits) == 1
    assert audits[0].relative_path == "physics_profiles/demo.yaml"
    assert audits[0].file_exists is True
    assert audits[0].matches is False


def test_update_config_integrity_manifest_refreshes_hashes(tmp_path: Path) -> None:
    module = _load_script_module()
    config_file = tmp_path / "physics_profiles" / "demo.yaml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("value: 26\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"physics_profiles/demo.yaml": "0" * 64}), encoding="utf-8")

    updated_manifest = module.update_config_integrity_manifest(manifest_path=manifest_path, root_dir=tmp_path)
    expected_sha256 = module.compute_sha256(config_file)

    assert updated_manifest == {"physics_profiles/demo.yaml": expected_sha256}
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == updated_manifest

    verified, audits = module.verify_config_integrity(manifest_path=manifest_path, root_dir=tmp_path)

    assert verified is True
    assert len(audits) == 1
    assert audits[0].matches is True


def test_main_update_flag_rewrites_manifest(tmp_path: Path) -> None:
    module = _load_script_module()
    config_file = tmp_path / "physics_profiles" / "demo.yaml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("value: 26\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"physics_profiles/demo.yaml": "0" * 64}), encoding="utf-8")

    exit_code = module.main(["--update", "--manifest-path", str(manifest_path), "--root-dir", str(tmp_path)])

    assert exit_code == 0
    expected_sha256 = module.compute_sha256(config_file)
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {"physics_profiles/demo.yaml": expected_sha256}
