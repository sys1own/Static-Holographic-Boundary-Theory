from __future__ import annotations

"""Interactive 4D hologram renderer for the SHBT prime-lattice."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.ontic_cascade import OnticAxioms, evaluate_ontic_cascade


DEFAULT_OUTPUT_PATH = Path("docs/viz/hologram_renderer/index.html")
DEFAULT_INCLUDE_PLOTLYJS = "cdn"
DEFAULT_DETUNING_STEPS = (0.0, 1.0e-18, 1.0e-17, 1.0e-16, 5.0e-16, 1.0e-15)
DEFAULT_LEPTON_HALF_WIDTH = 2
DEFAULT_QUARK_HALF_WIDTH = 2
DEFAULT_PARENT_HALF_WIDTH = 2
BENCHMARK_BRANCH = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))
BENCHMARK_BIT_LOADING_DENSITY = float(PARENT_LEVEL / (LEPTON_LEVEL + QUARK_LEVEL))
MOAT_VISUAL_FLOOR = 1.0e-18
MANIFOLD_COLLAPSE_SCALE = 1.0e-16


@dataclass(frozen=True)
class PrimeLatticePoint:
    branch: tuple[int, int, int]
    detuning: float
    dimension_index: float
    prime_indexed: bool
    axiom_ix_closure: bool
    parent_offset: float
    bit_loading_density: float
    moat_divergence: float
    moat_height: float
    closure_tensor_amplitude: float
    collapse_factor: float
    m_pi_state: float

    @property
    def branch_label(self) -> str:
        return f"({self.branch[0]}, {self.branch[1]}, {self.branch[2]})"


@dataclass(frozen=True)
class HologramFrame:
    detuning: float
    points: tuple[PrimeLatticePoint, ...]

    def point_for_branch(self, branch: tuple[int, int, int]) -> PrimeLatticePoint:
        return next(point for point in self.points if point.branch == branch)


@dataclass(frozen=True)
class HologramScene:
    benchmark_branch: tuple[int, int, int]
    lepton_levels: tuple[int, ...]
    quark_levels: tuple[int, ...]
    parent_levels: tuple[int, ...]
    detuning_steps: tuple[float, ...]
    frames: tuple[HologramFrame, ...]
    max_moat_height: float
    max_m_pi_state: float


def _require_plotly() -> tuple[Any, Any]:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Plotly is required to render the prime-lattice hologram. "
            "Install it, then rerun the renderer."
        ) from error
    return go, make_subplots


def _centered_levels(center: int, half_width: int, *, step: int = 1) -> tuple[int, ...]:
    resolved_center = int(center)
    resolved_half_width = int(half_width)
    resolved_step = int(step)
    if resolved_half_width < 0:
        raise ValueError("Half-width values must be non-negative.")
    if resolved_step <= 0:
        raise ValueError("Step values must be positive.")
    start = max(resolved_step, resolved_center - resolved_half_width * resolved_step)
    stop = resolved_center + resolved_half_width * resolved_step
    return tuple(range(start, stop + 1, resolved_step))


def _coerce_levels(levels: Sequence[int], *, include: int | None = None) -> tuple[int, ...]:
    resolved = {int(level) for level in levels if int(level) > 0}
    if include is not None:
        resolved.add(int(include))
    if not resolved:
        raise ValueError("Renderer axes require at least one positive level.")
    return tuple(sorted(resolved))


def _coerce_detuning_steps(detuning_steps: Sequence[float]) -> tuple[float, ...]:
    resolved = {float(step) for step in detuning_steps if float(step) >= 0.0}
    resolved.add(0.0)
    if not resolved:
        raise ValueError("Detuning steps must contain at least one non-negative value.")
    return tuple(sorted(resolved))


def _distance_to_integer(value: float) -> float:
    residual = math.fmod(math.fmod(float(value), 1.0) + 1.0, 1.0)
    return float(min(residual, 1.0 - residual))


def _continuous_framing_gap(parent_level: float, lepton_level: float, quark_level: float) -> float:
    lepton_gap = _distance_to_integer(parent_level / (2.0 * lepton_level))
    quark_gap = _distance_to_integer(parent_level / (3.0 * quark_level))
    return float(max(lepton_gap, quark_gap))


def _detuned_coordinates(branch: tuple[int, int, int], detuning: float) -> tuple[float, float, float]:
    return (
        float(branch[0]) + float(detuning),
        float(branch[1]) + float(detuning),
        float(branch[2]) + float(detuning),
    )


def _moat_height(moat_divergence: float) -> float:
    if moat_divergence <= 0.0:
        return 0.0
    return float(math.log10(1.0 + moat_divergence / MOAT_VISUAL_FLOOR))


def _collapse_factor(moat_divergence: float) -> float:
    exponent = min(float(moat_divergence) / MANIFOLD_COLLAPSE_SCALE, 700.0)
    return float(math.exp(-exponent))


def _build_point(branch: tuple[int, int, int], *, detuning: float) -> PrimeLatticePoint:
    base_cascade = evaluate_ontic_cascade(
        OnticAxioms(
            lepton_level=int(branch[0]),
            quark_level=int(branch[1]),
            parent_level=int(branch[2]),
        )
    )
    detuned_lepton, detuned_quark, detuned_parent = _detuned_coordinates(branch, detuning)
    bit_loading_density = detuned_parent / (detuned_lepton + detuned_quark)
    continuous_moat = _continuous_framing_gap(detuned_parent, detuned_lepton, detuned_quark)
    base_moat = float(base_cascade.axiom_ix.closure_tensor_amplitude)
    moat_divergence = float(max(base_moat, continuous_moat))
    collapse_factor = _collapse_factor(moat_divergence)
    dimension_index = detuned_lepton / 2.0
    m_pi_state = float((bit_loading_density / BENCHMARK_BIT_LOADING_DENSITY) * collapse_factor)
    return PrimeLatticePoint(
        branch=branch,
        detuning=float(detuning),
        dimension_index=float(dimension_index),
        prime_indexed=bool(base_cascade.prime_indexed),
        axiom_ix_closure=bool(base_cascade.axiom_ix.topological_closure),
        parent_offset=float(branch[2] - PARENT_LEVEL),
        bit_loading_density=float(bit_loading_density),
        moat_divergence=moat_divergence,
        moat_height=_moat_height(moat_divergence),
        closure_tensor_amplitude=moat_divergence,
        collapse_factor=collapse_factor,
        m_pi_state=m_pi_state,
    )


def build_hologram_scene(
    *,
    detuning_steps: Sequence[float] = DEFAULT_DETUNING_STEPS,
    lepton_levels: Sequence[int] | None = None,
    quark_levels: Sequence[int] | None = None,
    parent_levels: Sequence[int] | None = None,
    lepton_half_width: int = DEFAULT_LEPTON_HALF_WIDTH,
    quark_half_width: int = DEFAULT_QUARK_HALF_WIDTH,
    parent_half_width: int = DEFAULT_PARENT_HALF_WIDTH,
) -> HologramScene:
    resolved_lepton_levels = _coerce_levels(
        _centered_levels(int(LEPTON_LEVEL), int(lepton_half_width), step=2) if lepton_levels is None else lepton_levels,
        include=int(LEPTON_LEVEL),
    )
    resolved_quark_levels = _coerce_levels(
        _centered_levels(int(QUARK_LEVEL), int(quark_half_width), step=1) if quark_levels is None else quark_levels,
        include=int(QUARK_LEVEL),
    )
    resolved_parent_levels = _coerce_levels(
        _centered_levels(int(PARENT_LEVEL), int(parent_half_width), step=1) if parent_levels is None else parent_levels,
        include=int(PARENT_LEVEL),
    )
    resolved_detuning_steps = _coerce_detuning_steps(detuning_steps)

    branches = tuple(
        (int(lepton_level), int(quark_level), int(parent_level))
        for lepton_level in resolved_lepton_levels
        for quark_level in resolved_quark_levels
        for parent_level in resolved_parent_levels
    )
    frames = tuple(
        HologramFrame(
            detuning=float(detuning),
            points=tuple(_build_point(branch, detuning=float(detuning)) for branch in branches),
        )
        for detuning in resolved_detuning_steps
    )
    max_moat_height = max(point.moat_height for frame in frames for point in frame.points)
    max_m_pi_state = max(point.m_pi_state for frame in frames for point in frame.points)
    return HologramScene(
        benchmark_branch=BENCHMARK_BRANCH,
        lepton_levels=resolved_lepton_levels,
        quark_levels=resolved_quark_levels,
        parent_levels=resolved_parent_levels,
        detuning_steps=resolved_detuning_steps,
        frames=frames,
        max_moat_height=float(max_moat_height),
        max_m_pi_state=float(max_m_pi_state),
    )


def _marker_sizes(points: Sequence[PrimeLatticePoint]) -> list[float]:
    return [6.0 + 10.0 * (point.bit_loading_density / BENCHMARK_BIT_LOADING_DENSITY) for point in points]


def _custom_data(points: Sequence[PrimeLatticePoint]) -> list[list[object]]:
    return [
        [
            point.branch_label,
            point.bit_loading_density,
            point.moat_divergence,
            point.collapse_factor,
            "closed" if point.axiom_ix_closure else "open",
            "prime" if point.prime_indexed else "non-prime",
            point.m_pi_state,
        ]
        for point in points
    ]


def _prime_lattice_trace(go: Any, frame: HologramFrame, *, showscale: bool) -> Any:
    points = frame.points
    return go.Scatter3d(
        x=[float(point.branch[1]) for point in points],
        y=[float(point.dimension_index) for point in points],
        z=[float(point.parent_offset) for point in points],
        mode="markers",
        marker={
            "size": _marker_sizes(points),
            "color": [float(point.m_pi_state) for point in points],
            "colorscale": [
                [0.0, "#0f172a"],
                [0.25, "#1d4ed8"],
                [0.5, "#0ea5e9"],
                [0.75, "#22c55e"],
                [1.0, "#f59e0b"],
            ],
            "cmin": 0.0,
            "cmax": max(1.0, max(point.m_pi_state for point in points)),
            "colorbar": {"title": "M_pi"},
            "showscale": showscale,
            "line": {"color": "white", "width": 0.4},
            "opacity": 0.9,
        },
        customdata=_custom_data(points),
        hovertemplate=(
            "branch=%{customdata[0]}<br>"
            "prime index=%{y:.6f}<br>"
            "k_q=%{x:.0f}<br>"
            "K-312=%{z:.0f}<br>"
            "M_pi=%{customdata[6]:.6e}<br>"
            "ρ_bit=%{customdata[1]:.6f}<br>"
            "Δ_moat=%{customdata[2]:.6e}<br>"
            "collapse=%{customdata[3]:.6e}<br>"
            "Axiom IX=%{customdata[4]}<br>"
            "index class=%{customdata[5]}"
            "<extra></extra>"
        ),
        name="Prime lattice",
    )


def _benchmark_lattice_trace(go: Any, frame: HologramFrame) -> Any:
    point = frame.point_for_branch(BENCHMARK_BRANCH)
    return go.Scatter3d(
        x=[float(point.branch[1])],
        y=[float(point.dimension_index)],
        z=[float(point.parent_offset)],
        mode="markers+text",
        marker={
            "size": 14,
            "color": "#f8fafc",
            "symbol": "diamond",
            "line": {"color": "#f59e0b", "width": 3},
        },
        text=["benchmark"],
        textposition="top center",
        hovertemplate=(
            f"benchmark={point.branch_label}<br>"
            f"M_pi={point.m_pi_state:.6e}<br>"
            f"ρ_bit={point.bit_loading_density:.6f}<br>"
            f"Δ_moat={point.moat_divergence:.6e}<br>"
            f"collapse={point.collapse_factor:.6e}"
            "<extra></extra>"
        ),
        name="Benchmark",
    )


def _moat_trace(go: Any, frame: HologramFrame, *, showscale: bool) -> Any:
    points = frame.points
    return go.Scatter3d(
        x=[float(point.branch[1]) for point in points],
        y=[float(point.dimension_index) for point in points],
        z=[float(point.moat_height) for point in points],
        mode="markers",
        marker={
            "size": _marker_sizes(points),
            "color": [float(point.m_pi_state) for point in points],
            "colorscale": [
                [0.0, "#111827"],
                [0.25, "#312e81"],
                [0.5, "#2563eb"],
                [0.75, "#10b981"],
                [1.0, "#fbbf24"],
            ],
            "cmin": 0.0,
            "cmax": max(1.0, max(point.m_pi_state for point in points)),
            "colorbar": {"title": "M_pi"},
            "showscale": showscale,
            "line": {"color": "white", "width": 0.4},
            "opacity": 0.9,
        },
        customdata=_custom_data(points),
        hovertemplate=(
            "branch=%{customdata[0]}<br>"
            "prime index=%{y:.6f}<br>"
            "k_q=%{x:.0f}<br>"
            "log moat=%{z:.6f}<br>"
            "M_pi=%{customdata[6]:.6e}<br>"
            "ρ_bit=%{customdata[1]:.6f}<br>"
            "Δ_moat=%{customdata[2]:.6e}<br>"
            "closure=%{customdata[2]:.6e}<br>"
            "Axiom IX=%{customdata[4]}<br>"
            "index class=%{customdata[5]}"
            "<extra></extra>"
        ),
        name="Noether Bridge moat",
    )


def _benchmark_moat_trace(go: Any, frame: HologramFrame) -> Any:
    point = frame.point_for_branch(BENCHMARK_BRANCH)
    return go.Scatter3d(
        x=[float(point.branch[1])],
        y=[float(point.dimension_index)],
        z=[float(point.moat_height)],
        mode="markers+text",
        marker={
            "size": 14,
            "color": "#f8fafc",
            "symbol": "diamond",
            "line": {"color": "#dc2626", "width": 3},
        },
        text=["benchmark"],
        textposition="top center",
        hovertemplate=(
            f"benchmark={point.branch_label}<br>"
            f"log moat={point.moat_height:.6f}<br>"
            f"Δ_moat={point.moat_divergence:.6e}<br>"
            f"M_pi={point.m_pi_state:.6e}"
            "<extra></extra>"
        ),
        name="Benchmark moat",
    )


def build_hologram_figure(scene: HologramScene | None = None) -> Any:
    go, make_subplots = _require_plotly()
    resolved_scene = build_hologram_scene() if scene is None else scene
    initial_index = 0

    def _title(frame: HologramFrame) -> str:
        benchmark_point = frame.point_for_branch(BENCHMARK_BRANCH)
        return (
            "Prime-Lattice Hologram Renderer"
            "<br><sup>"
            f"detuning ε = {frame.detuning:.1e} • "
            f"benchmark M_pi = {benchmark_point.m_pi_state:.3e} • "
            f"benchmark Δ_moat = {benchmark_point.moat_divergence:.3e}"
            "</sup>"
        )

    def _frame_data(frame: HologramFrame) -> list[Any]:
        return [
            _prime_lattice_trace(go, frame, showscale=False),
            _benchmark_lattice_trace(go, frame),
            _moat_trace(go, frame, showscale=True),
            _benchmark_moat_trace(go, frame),
        ]

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Prime lattice occupancy", "Noether Bridge moat response"),
        horizontal_spacing=0.04,
    )
    for trace in _frame_data(resolved_scene.frames[initial_index]):
        figure.add_trace(trace, row=1, col=1 if len(figure.data) < 2 else 2)

    frames = [
        go.Frame(
            name=f"detune-{index}",
            data=_frame_data(frame),
            layout={"title": {"text": _title(frame)}},
        )
        for index, frame in enumerate(resolved_scene.frames)
    ]
    figure.frames = frames
    figure.update_layout(
        title={"text": _title(resolved_scene.frames[initial_index])},
        scene={
            "xaxis": {"title": "k_q"},
            "yaxis": {"title": "prime index k_l / 2"},
            "zaxis": {"title": "parent offset K - 312"},
            "camera": {"eye": {"x": 1.6, "y": 1.4, "z": 1.1}},
        },
        scene2={
            "xaxis": {"title": "k_q"},
            "yaxis": {"title": "prime index k_l / 2"},
            "zaxis": {
                "title": "log10(1 + Δ_moat / 10^-18)",
                "range": [0.0, resolved_scene.max_moat_height * 1.05],
            },
            "camera": {"eye": {"x": 1.55, "y": 1.45, "z": 1.15}},
        },
        margin={"l": 0, "r": 0, "t": 88, "b": 0},
        legend={"orientation": "h", "y": 1.04, "x": 0.0},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.1,
                "buttons": [
                    {
                        "label": "Animate detuning",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": False,
                                "frame": {"duration": 500, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": initial_index,
                "currentvalue": {"prefix": "Detuning ε: "},
                "steps": [
                    {
                        "label": f"{frame.detuning:.1e}",
                        "method": "animate",
                        "args": [
                            [f"detune-{index}"],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for index, frame in enumerate(resolved_scene.frames)
                ],
            }
        ],
    )
    return figure


def write_hologram_html(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    scene: HologramScene | None = None,
    include_plotlyjs: str = DEFAULT_INCLUDE_PLOTLYJS,
) -> Path:
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = build_hologram_figure(scene=scene)
    figure.write_html(str(resolved_output_path), include_plotlyjs=include_plotlyjs, full_html=True)
    return resolved_output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--include-plotlyjs", default=DEFAULT_INCLUDE_PLOTLYJS)
    parser.add_argument("--lepton-half-width", type=int, default=DEFAULT_LEPTON_HALF_WIDTH)
    parser.add_argument("--quark-half-width", type=int, default=DEFAULT_QUARK_HALF_WIDTH)
    parser.add_argument("--parent-half-width", type=int, default=DEFAULT_PARENT_HALF_WIDTH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scene = build_hologram_scene(
        lepton_half_width=args.lepton_half_width,
        quark_half_width=args.quark_half_width,
        parent_half_width=args.parent_half_width,
    )
    output_path = write_hologram_html(
        output_path=args.output,
        scene=scene,
        include_plotlyjs=str(args.include_plotlyjs),
    )
    print(f"Wrote prime-lattice hologram to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
