from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "docs" / "viz" / "hologram_renderer" / "render_hologram.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("hologram_renderer", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_hologram_scene_tracks_benchmark_collapse_under_femtodetuning() -> None:
    module = _load_script_module()

    scene = module.build_hologram_scene(
        detuning_steps=(0.0, 1.0e-15),
        lepton_levels=(24, 26),
        quark_levels=(8,),
        parent_levels=(312,),
    )

    locked_benchmark = scene.frames[0].point_for_branch((26, 8, 312))
    detuned_benchmark = scene.frames[-1].point_for_branch((26, 8, 312))
    off_shell = scene.frames[0].point_for_branch((24, 8, 312))

    assert scene.benchmark_branch == (26, 8, 312)
    assert locked_benchmark.axiom_ix_closure is True
    assert locked_benchmark.m_pi_state == pytest.approx(1.0, rel=0.0, abs=1.0e-12)
    assert locked_benchmark.moat_divergence == pytest.approx(0.0, rel=0.0, abs=1.0e-18)
    assert detuned_benchmark.moat_divergence > 0.0
    assert detuned_benchmark.m_pi_state < 1.0e-4
    assert off_shell.axiom_ix_closure is False
    assert off_shell.prime_indexed is False
    assert off_shell.moat_divergence > 0.0
    assert off_shell.m_pi_state < locked_benchmark.m_pi_state


def test_build_hologram_figure_exposes_two_scenes_and_detuning_slider() -> None:
    module = _load_script_module()
    scene = module.build_hologram_scene(
        detuning_steps=(0.0, 1.0e-15),
        lepton_levels=(26,),
        quark_levels=(8,),
        parent_levels=(312,),
    )

    figure = module.build_hologram_figure(scene)

    assert len(figure.data) == 4
    assert len(figure.frames) == 2
    assert figure.layout.scene.xaxis.title.text == "k_q"
    assert figure.layout.scene2.zaxis.title.text == "log10(1 + Δ_moat / 10^-18)"
    assert figure.layout.sliders[0].steps[-1].label == "1.0e-15"


def test_write_hologram_html_creates_interactive_artifact(tmp_path: Path) -> None:
    module = _load_script_module()
    scene = module.build_hologram_scene(
        detuning_steps=(0.0, 1.0e-15),
        lepton_levels=(26,),
        quark_levels=(8,),
        parent_levels=(312,),
    )

    output_path = module.write_hologram_html(tmp_path / "renderer.html", scene=scene)
    output_text = output_path.read_text(encoding="utf-8")

    assert output_path.exists()
    assert "Prime-Lattice Hologram Renderer" in output_text
    assert "Plotly.newPlot" in output_text
