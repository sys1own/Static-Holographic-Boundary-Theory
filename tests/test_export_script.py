from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export.py"


def _load_script_module():
    script_dir = str(SCRIPT_PATH.parent)
    inserted_path = False
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        inserted_path = True

    try:
        spec = importlib.util.spec_from_file_location("export_script", SCRIPT_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted_path:
            sys.path.pop(0)


def test_export_script_writes_quantified_two_loop_residuals(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    expected_payload = {
        "artifact": "Quantified Two-Loop Residuals",
        "benchmark_tuple": [26, 8, 312],
    }
    output_path = tmp_path / "results" / "residuals.json"

    monkeypatch.setattr(
        module.EvolutionaryEngine,
        "generate_residual_payload",
        staticmethod(lambda: expected_payload),
    )

    assert module.main(["--output-path", str(output_path)]) == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == expected_payload
