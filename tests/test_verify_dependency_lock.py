from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / "scripts" / "verify_dependency_lock.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("verify_dependency_lock", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dependency_lock_matches_current_environment() -> None:
    module = _load_script_module()

    verified, drift = module.verify_dependency_lock()

    assert verified is True
    assert drift == ()


def test_dependency_lock_detects_version_drift(tmp_path: Path) -> None:
    module = _load_script_module()
    lock_path = tmp_path / "requirements.lock"
    lock_path.write_text("pytest==0.0.0\n", encoding="utf-8")

    verified, drift = module.verify_dependency_lock(lock_path)

    assert verified is False
    assert len(drift) == 1
    assert drift[0].startswith("pytest expected 0.0.0")
