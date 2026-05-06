from __future__ import annotations

"""Map the 5x5 rigidity moat around the anomaly-free SHBT benchmark."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

from matplotlib import colors as mcolors
import numpy as np

if __package__ in (None, ""):
    import sys

    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.constants import (
    BENCHMARK_C_DARK_RESIDUE,
    LEPTON_LEVEL,
    PARENT_LEVEL,
    QUARK_LEVEL,
    SU2_DIMENSION,
    SU2_DUAL_COXETER,
    SU3_DIMENSION,
    SU3_DUAL_COXETER,
)
from shbt.core.noether_bridge import framing_defect
from shbt.plotting_runtime import plt


EXPECTED_BENCHMARK = (26, 8, 312)
DEFAULT_OUTPUT_DIR = Path("results")
DEFAULT_FIGURE_FILENAME = "rigidity_moat.png"
DEFAULT_DATA_FILENAME = "rigidity_moat.json"
DEFAULT_DPI = 220
DEFAULT_LEPTON_HALF_WIDTH = 2
DEFAULT_QUARK_HALF_WIDTH = 2
DEFAULT_PARENT_HALF_WIDTH = 0
ZERO_TOLERANCE = 1.0e-15
MIN_COLOR_CEILING = 1.0e-6


@dataclass(frozen=True)
class RigidityPoint:
    coordinates: tuple[int, int, int]
    lepton_framing_gap: float
    quark_framing_gap: float
    delta_fr: float
    delta_fr_label: str
    c_dark_shift: float
    diophantine_gap: float
    total_residue: float


@dataclass(frozen=True)
class RigidityLandscapeScan:
    benchmark_coordinates: tuple[int, int, int]
    lepton_levels: tuple[int, ...]
    quark_levels: tuple[int, ...]
    parent_levels: tuple[int, ...]
    total_residue_grid: np.ndarray
    delta_fr_grid: np.ndarray
    c_dark_shift_grid: np.ndarray
    diophantine_gap_grid: np.ndarray
    benchmark_index: tuple[int, int, int]
    benchmark_point: RigidityPoint
    nearest_detuned_point: RigidityPoint
    maximum_residue_point: RigidityPoint
    points: tuple[RigidityPoint, ...]
    color_floor: float


def _benchmark_coordinates() -> tuple[int, int, int]:
    benchmark = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))
    assert benchmark == EXPECTED_BENCHMARK, (
        f"The rigidity map is locked to the published branch {EXPECTED_BENCHMARK}, received {benchmark}."
    )
    return benchmark


def _sanitize_small(value: float) -> float:
    numeric_value = float(value)
    if math.isclose(numeric_value, 0.0, rel_tol=0.0, abs_tol=ZERO_TOLERANCE):
        return 0.0
    return numeric_value


def _centered_levels(center: int, half_width: int) -> tuple[int, ...]:
    resolved_center = int(center)
    resolved_half_width = int(half_width)
    if resolved_half_width < 0:
        raise ValueError("Half-width arguments must be non-negative integers.")
    lower_bound = max(1, resolved_center - resolved_half_width)
    upper_bound = resolved_center + resolved_half_width
    return tuple(range(lower_bound, upper_bound + 1))


def _coerce_levels(levels: Sequence[int]) -> tuple[int, ...]:
    resolved_levels = tuple(sorted({int(level) for level in levels if int(level) > 0}))
    if not resolved_levels:
        raise ValueError("Rigidity scans require at least one positive level in each axis.")
    return resolved_levels


def _wzw_central_charge(level: int, dimension: int, dual_coxeter: int) -> float:
    resolved_level = float(level)
    denominator = resolved_level + float(dual_coxeter)
    if denominator <= 0.0:
        raise ValueError("WZW central charge requires k + h^∨ > 0.")
    return float(resolved_level * float(dimension) / denominator)


def _c_dark_residue(parent_level: int, lepton_level: int, quark_level: int) -> float:
    return float(
        _wzw_central_charge(parent_level, SU3_DIMENSION, SU3_DUAL_COXETER)
        + _wzw_central_charge(parent_level, SU2_DIMENSION, SU2_DUAL_COXETER)
        - _wzw_central_charge(quark_level, SU3_DIMENSION, SU3_DUAL_COXETER)
        - _wzw_central_charge(lepton_level, SU2_DIMENSION, SU2_DUAL_COXETER)
    )


def _diophantine_gap(parent_level: int, lepton_level: int, quark_level: int) -> float:
    minimal_parent_level = math.lcm(2 * int(lepton_level), 3 * int(quark_level))
    return _sanitize_small(abs(int(parent_level) - minimal_parent_level) / float(minimal_parent_level))


def build_rigidity_point(lepton_level: int, quark_level: int, parent_level: int) -> RigidityPoint:
    defect = framing_defect(int(parent_level), int(lepton_level), int(quark_level))
    lepton_gap = _sanitize_small(float(defect.lepton_gap))
    quark_gap = _sanitize_small(float(defect.quark_gap))
    delta_fr = _sanitize_small(float(defect.delta_fr))
    c_dark_shift = _sanitize_small(abs(_c_dark_residue(parent_level, lepton_level, quark_level) - BENCHMARK_C_DARK_RESIDUE))
    diophantine_gap = _diophantine_gap(parent_level, lepton_level, quark_level)
    return RigidityPoint(
        coordinates=(int(lepton_level), int(quark_level), int(parent_level)),
        lepton_framing_gap=lepton_gap,
        quark_framing_gap=quark_gap,
        delta_fr=delta_fr,
        delta_fr_label=str(defect.delta_fr),
        c_dark_shift=c_dark_shift,
        diophantine_gap=diophantine_gap,
        total_residue=delta_fr,
    )


def build_rigidity_landscape_scan(
    *,
    lepton_levels: Sequence[int],
    quark_levels: Sequence[int],
    parent_levels: Sequence[int],
) -> RigidityLandscapeScan:
    benchmark_kl, benchmark_kq, benchmark_parent = _benchmark_coordinates()
    resolved_lepton_levels = _coerce_levels(lepton_levels)
    resolved_quark_levels = _coerce_levels(quark_levels)
    resolved_parent_levels = _coerce_levels(parent_levels)

    if benchmark_kl not in resolved_lepton_levels or benchmark_kq not in resolved_quark_levels or benchmark_parent not in resolved_parent_levels:
        raise ValueError("The rigidity scan ranges must include the published benchmark (26, 8, 312).")

    benchmark_index = (
        resolved_lepton_levels.index(benchmark_kl),
        resolved_quark_levels.index(benchmark_kq),
        resolved_parent_levels.index(benchmark_parent),
    )

    grid_shape = (len(resolved_lepton_levels), len(resolved_quark_levels), len(resolved_parent_levels))
    total_residue_grid = np.zeros(grid_shape, dtype=float)
    delta_fr_grid = np.zeros(grid_shape, dtype=float)
    c_dark_shift_grid = np.zeros(grid_shape, dtype=float)
    diophantine_gap_grid = np.zeros(grid_shape, dtype=float)
    points: list[RigidityPoint] = []

    for lepton_index, lepton_level in enumerate(resolved_lepton_levels):
        for quark_index, quark_level in enumerate(resolved_quark_levels):
            for parent_index, parent_level in enumerate(resolved_parent_levels):
                point = build_rigidity_point(lepton_level, quark_level, parent_level)
                points.append(point)
                total_residue_grid[lepton_index, quark_index, parent_index] = point.total_residue
                delta_fr_grid[lepton_index, quark_index, parent_index] = point.delta_fr
                c_dark_shift_grid[lepton_index, quark_index, parent_index] = point.c_dark_shift
                diophantine_gap_grid[lepton_index, quark_index, parent_index] = point.diophantine_gap

    benchmark_point = next(point for point in points if point.coordinates == (benchmark_kl, benchmark_kq, benchmark_parent))
    assert benchmark_point.delta_fr == 0.0, (
        "The rigidity moat must retain a zero-residue benchmark valley at the published branch."
    )

    nearest_detuned_point = min(
        (point for point in points if point.coordinates != (benchmark_kl, benchmark_kq, benchmark_parent)),
        key=lambda point: (point.delta_fr, point.coordinates[2], point.coordinates[0], point.coordinates[1]),
    )
    maximum_residue_point = max(
        points,
        key=lambda point: (point.delta_fr, point.coordinates[2], point.coordinates[0], point.coordinates[1]),
    )

    positive_residues = delta_fr_grid[delta_fr_grid > 0.0]
    color_floor = float(np.min(positive_residues)) if positive_residues.size else MIN_COLOR_CEILING

    return RigidityLandscapeScan(
        benchmark_coordinates=(benchmark_kl, benchmark_kq, benchmark_parent),
        lepton_levels=resolved_lepton_levels,
        quark_levels=resolved_quark_levels,
        parent_levels=resolved_parent_levels,
        total_residue_grid=np.asarray(total_residue_grid, dtype=float),
        delta_fr_grid=np.asarray(delta_fr_grid, dtype=float),
        c_dark_shift_grid=np.asarray(c_dark_shift_grid, dtype=float),
        diophantine_gap_grid=np.asarray(diophantine_gap_grid, dtype=float),
        benchmark_index=benchmark_index,
        benchmark_point=benchmark_point,
        nearest_detuned_point=nearest_detuned_point,
        maximum_residue_point=maximum_residue_point,
        points=tuple(points),
        color_floor=color_floor,
    )


def build_centered_rigidity_landscape_scan(
    *,
    lepton_half_width: int = DEFAULT_LEPTON_HALF_WIDTH,
    quark_half_width: int = DEFAULT_QUARK_HALF_WIDTH,
    parent_half_width: int = DEFAULT_PARENT_HALF_WIDTH,
) -> RigidityLandscapeScan:
    benchmark_kl, benchmark_kq, benchmark_parent = _benchmark_coordinates()
    return build_rigidity_landscape_scan(
        lepton_levels=_centered_levels(benchmark_kl, lepton_half_width),
        quark_levels=_centered_levels(benchmark_kq, quark_half_width),
        parent_levels=_centered_levels(benchmark_parent, parent_half_width),
    )


def _benchmark_plane(scan: RigidityLandscapeScan) -> np.ndarray:
    return np.asarray(scan.delta_fr_grid[:, :, scan.benchmark_index[2]], dtype=float)


def _benchmark_plane_neighbor(scan: RigidityLandscapeScan) -> RigidityPoint:
    benchmark_kl, benchmark_kq, benchmark_parent = scan.benchmark_coordinates
    return min(
        (
            point
            for point in scan.points
            if point.coordinates[2] == benchmark_parent and point.coordinates[:2] != (benchmark_kl, benchmark_kq)
        ),
        key=lambda point: (point.delta_fr, point.coordinates[0], point.coordinates[1]),
    )


def render_rigidity_landscape_plot(
    scan: RigidityLandscapeScan,
    output_path: Path,
    *,
    dpi: int = DEFAULT_DPI,
) -> Path:
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    benchmark_kl, benchmark_kq, benchmark_parent = scan.benchmark_coordinates
    benchmark_plane_neighbor = _benchmark_plane_neighbor(scan)
    benchmark_plane = _benchmark_plane(scan)

    quark_mesh, lepton_mesh = np.meshgrid(scan.quark_levels, scan.lepton_levels)
    maximum_residue = float(np.max(benchmark_plane))
    color_ceiling = max(maximum_residue, MIN_COLOR_CEILING)
    norm = mcolors.Normalize(vmin=0.0, vmax=color_ceiling)
    base_offset = -max(color_ceiling * 0.16, 0.02)

    figure = plt.figure(figsize=(11.2, 7.2), constrained_layout=True)
    try:
        axis = figure.add_subplot(111, projection="3d")

        surface = axis.plot_surface(
            quark_mesh,
            lepton_mesh,
            benchmark_plane,
            cmap="viridis",
            norm=norm,
            edgecolor="white",
            linewidth=0.75,
            antialiased=True,
            alpha=0.95,
        )
        axis.contourf(
            quark_mesh,
            lepton_mesh,
            benchmark_plane,
            zdir="z",
            offset=base_offset,
            levels=np.linspace(0.0, color_ceiling, 12),
            cmap="viridis",
            norm=norm,
            alpha=0.82,
        )

        axis.scatter(
            [benchmark_kq],
            [benchmark_kl],
            [scan.benchmark_point.delta_fr],
            marker="*",
            s=280,
            color="#ef4444",
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
        )
        axis.scatter(
            [benchmark_plane_neighbor.coordinates[1]],
            [benchmark_plane_neighbor.coordinates[0]],
            [benchmark_plane_neighbor.delta_fr],
            marker="o",
            s=90,
            facecolors="none",
            edgecolors="#0f172a",
            linewidths=1.4,
            zorder=6,
        )

        for point in scan.points:
            if point.coordinates[2] != benchmark_parent:
                continue
            label_color = "#111827" if point.delta_fr <= color_ceiling * 0.55 else "white"
            axis.text(
                point.coordinates[1],
                point.coordinates[0],
                point.delta_fr + color_ceiling * 0.015,
                point.delta_fr_label,
                ha="center",
                va="bottom",
                fontsize=7.2,
                color=label_color,
            )

        axis.text(
            benchmark_kq + 0.10,
            benchmark_kl + 0.10,
            color_ceiling * 0.045,
            "benchmark\n(26, 8, 312)\nE = 0",
            fontsize=9,
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "#6b7280", "alpha": 0.94, "boxstyle": "round,pad=0.24"},
        )
        axis.text(
            benchmark_plane_neighbor.coordinates[1] + 0.08,
            benchmark_plane_neighbor.coordinates[0] + 0.08,
            benchmark_plane_neighbor.delta_fr + color_ceiling * 0.05,
            "nearest detuned cell\n"
            rf"{benchmark_plane_neighbor.coordinates}\n"
            rf"E = {benchmark_plane_neighbor.delta_fr_label}",
            fontsize=8.5,
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "#6b7280", "alpha": 0.94, "boxstyle": "round,pad=0.24"},
        )
        axis.text2D(
            0.02,
            0.98,
            "\n".join(
                (
                    "E = |Delta_fr| = max(|K/(2 k_l) - Z|, |K/(3 k_q) - Z|)",
                    rf"5x5 benchmark-plane scan at fixed K={benchmark_parent}",
                    rf"24/24 detuned cells reopen the framing anomaly; nearest moat wall carries E={benchmark_plane_neighbor.delta_fr_label}.",
                )
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.1,
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "#6b7280", "alpha": 0.96, "boxstyle": "round,pad=0.32"},
        )

        axis.set_title(
            r"Rigidity moat: the $(26, 8, 312)$ branch is a singular stability valley",
            fontsize=14,
            fontweight="bold",
            pad=18,
        )
        axis.set_xlabel(r"$k_q$")
        axis.set_ylabel(r"$k_\ell$")
        axis.set_zlabel(r"Framing anomaly residue $\mathcal{E}$")
        axis.set_xticks(scan.quark_levels)
        axis.set_yticks(scan.lepton_levels)
        axis.set_zlim(base_offset, color_ceiling * 1.10)
        axis.view_init(elev=33, azim=-58)

        colorbar = figure.colorbar(surface, ax=axis, pad=0.08, fraction=0.05, shrink=0.82)
        colorbar.set_label(r"Framing anomaly residue $\mathcal{E}$")

        figure.savefig(resolved_output_path, dpi=int(dpi), format="png")
    finally:
        plt.close(figure)
    return resolved_output_path


def _point_payload(point: RigidityPoint) -> dict[str, object]:
    return {
        "coordinates": list(point.coordinates),
        "lepton_framing_gap": float(point.lepton_framing_gap),
        "quark_framing_gap": float(point.quark_framing_gap),
        "delta_fr": float(point.delta_fr),
        "delta_fr_label": point.delta_fr_label,
        "c_dark_shift": float(point.c_dark_shift),
        "diophantine_gap": float(point.diophantine_gap),
        "total_residue": float(point.total_residue),
    }


def write_rigidity_landscape_json(scan: RigidityLandscapeScan, output_path: Path) -> Path:
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_coordinates": list(scan.benchmark_coordinates),
        "benchmark_plane_parent_level": int(scan.benchmark_coordinates[2]),
        "grid_shape": [len(scan.lepton_levels), len(scan.quark_levels), len(scan.parent_levels)],
        "lepton_levels": list(scan.lepton_levels),
        "quark_levels": list(scan.quark_levels),
        "parent_levels": list(scan.parent_levels),
        "framing_residue_definition": {
            "mathcal_E": "max(|K/(2 k_l) - Z|, |K/(3 k_q) - Z|)",
            "reported_plane": "benchmark parent plane K=312",
        },
        "benchmark_point": _point_payload(scan.benchmark_point),
        "nearest_detuned_point": _point_payload(scan.nearest_detuned_point),
        "maximum_residue_point": _point_payload(scan.maximum_residue_point),
        "delta_fr_grid_at_benchmark_parent": _benchmark_plane(scan).tolist(),
        "points": [_point_payload(point) for point in scan.points],
    }
    resolved_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return resolved_output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-name", default=DEFAULT_FIGURE_FILENAME)
    parser.add_argument("--data-name", default=DEFAULT_DATA_FILENAME)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--lepton-half-width", type=int, default=DEFAULT_LEPTON_HALF_WIDTH)
    parser.add_argument("--quark-half-width", type=int, default=DEFAULT_QUARK_HALF_WIDTH)
    parser.add_argument("--parent-half-width", type=int, default=DEFAULT_PARENT_HALF_WIDTH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scan = build_centered_rigidity_landscape_scan(
        lepton_half_width=args.lepton_half_width,
        quark_half_width=args.quark_half_width,
        parent_half_width=args.parent_half_width,
    )

    figure_path = render_rigidity_landscape_plot(scan, args.output_dir / str(args.figure_name), dpi=args.dpi)
    data_path = write_rigidity_landscape_json(scan, args.output_dir / str(args.data_name))

    print(f"benchmark branch                : {scan.benchmark_coordinates}")
    print(
        "scan dimensions                 : "
        f"{len(scan.lepton_levels)} x {len(scan.quark_levels)} x {len(scan.parent_levels)}"
    )
    print(f"benchmark framing residue      : {scan.benchmark_point.delta_fr:.6e}")
    print(f"nearest detuned cell            : {scan.nearest_detuned_point.coordinates}")
    print(f"nearest detuned framing residue : {scan.nearest_detuned_point.delta_fr_label}")
    print(f"maximum scanned residue         : {scan.maximum_residue_point.delta_fr:.6e}")
    print(f"wrote figure                    : {figure_path.as_posix()}")
    print(f"wrote data                      : {data_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
