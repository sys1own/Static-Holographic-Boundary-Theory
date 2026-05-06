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


def test_build_centered_rigidity_landscape_scan_maps_5x5_benchmark_plane() -> None:
    module = _load_script_module()

    scan = module.build_centered_rigidity_landscape_scan()

    assert scan.benchmark_coordinates == (26, 8, 312)
    assert scan.lepton_levels == (24, 25, 26, 27, 28)
    assert scan.quark_levels == (6, 7, 8, 9, 10)
    assert scan.parent_levels == (312,)
    assert scan.delta_fr_grid.shape == (5, 5, 1)
    assert scan.benchmark_point.delta_fr == 0.0
    assert scan.benchmark_point.total_residue == 0.0
    assert scan.nearest_detuned_point.coordinates != scan.benchmark_coordinates
    assert scan.nearest_detuned_point.delta_fr > 0.0
    assert np.count_nonzero(scan.delta_fr_grid == 0.0) == 1
    assert np.count_nonzero(scan.delta_fr_grid[:, :, 0] > 0.0) == 24


def test_main_writes_rigidity_moat_artifacts(tmp_path: Path) -> None:
    module = _load_script_module()
    output_dir = tmp_path / "results"

    exit_code = module.main(["--output-dir", str(output_dir), "--dpi", "120"])

    assert exit_code == 0
    figure_path = output_dir / "rigidity_moat.png"
    data_path = output_dir / "rigidity_moat.json"
    assert figure_path.is_file()
    assert data_path.is_file()

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    assert payload["benchmark_coordinates"] == [26, 8, 312]
    assert payload["benchmark_plane_parent_level"] == 312
    assert payload["grid_shape"] == [5, 5, 1]
    assert payload["benchmark_point"]["delta_fr"] == 0.0
    assert payload["nearest_detuned_point"]["delta_fr"] > 0.0
    assert len(payload["points"]) == 25
    assert len(payload["delta_fr_grid_at_benchmark_parent"]) == 5
    assert len(payload["delta_fr_grid_at_benchmark_parent"][0]) == 5
