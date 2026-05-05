from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "map_rigidity_landscape.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("map_rigidity_landscape", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_rigidity_landscape_scan_isolates_benchmark() -> None:
    module = _load_script_module()

    scan = module.build_centered_rigidity_landscape_scan(
        lepton_half_width=2,
        quark_half_width=2,
        parent_half_width=2,
    )

    assert scan.benchmark_coordinates == (26, 8, 312)
    assert scan.benchmark_point.total_residue == 0.0
    assert scan.nearest_detuned_point.coordinates != scan.benchmark_coordinates
    assert scan.nearest_detuned_point.total_residue > 0.0
    assert np.count_nonzero(scan.total_residue_grid == 0.0) == 1


def test_render_and_export_rigidity_landscape_artifacts(tmp_path: Path) -> None:
    module = _load_script_module()
    output_dir = tmp_path / "results"
    scan = module.build_centered_rigidity_landscape_scan(
        lepton_half_width=1,
        quark_half_width=1,
        parent_half_width=1,
    )

    figure_path = module.render_rigidity_landscape_plot(scan, output_dir / "rigidity.png", dpi=120)
    data_path = module.write_rigidity_landscape_json(scan, output_dir / "rigidity.json")

    assert figure_path.is_file()
    assert data_path.is_file()

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    assert payload["benchmark_coordinates"] == [26, 8, 312]
    assert payload["benchmark_point"]["total_residue"] == 0.0
    assert payload["nearest_detuned_point"]["total_residue"] > 0.0
    assert len(payload["points"]) == 27
