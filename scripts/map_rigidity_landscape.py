from __future__ import annotations

"""Map the 3D rigidity landscape around the anomaly-free SHBT benchmark."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

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
    SO10_DIMENSION,
    SO10_DUAL_COXETER,
    SU2_DIMENSION,
    SU2_DUAL_COXETER,
    SU3_DIMENSION,
    SU3_DUAL_COXETER,
)
from shbt.core.noether_bridge import framing_defect
from shbt.plotting_runtime import plt


EXPECTED_BENCHMARK = (26, 8, 312)
DEFAULT_OUTPUT_DIR = Path("results")
DEFAULT_FIGURE_FILENAME = "rigidity_landscape.png"
DEFAULT_DATA_FILENAME = "rigidity_landscape.json"
DEFAULT_DPI = 220
DEFAULT_LEPTON_HALF_WIDTH = 5
DEFAULT_QUARK_HALF_WIDTH = 5
DEFAULT_PARENT_HALF_WIDTH = 6
ZERO_TOLERANCE = 1.0e-15
MIN_COLOR_FLOOR = 1.0e-12


@dataclass(frozen=True)
class RigidityPoint:
    coordinates: tuple[int, int, int]
    lepton_framing_gap: float
    quark_framing_gap: float
    delta_fr: float
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
    total_residue = _sanitize_small(
        math.sqrt(delta_fr * delta_fr + c_dark_shift * c_dark_shift + diophantine_gap * diophantine_gap)
    )
    return RigidityPoint(
        coordinates=(int(lepton_level), int(quark_level), int(parent_level)),
        lepton_framing_gap=lepton_gap,
        quark_framing_gap=quark_gap,
        delta_fr=delta_fr,
        c_dark_shift=c_dark_shift,
        diophantine_gap=diophantine_gap,
        total_residue=total_residue,
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
    assert benchmark_point.total_residue == 0.0, (
        "The rigidity plot must retain a zero-residue benchmark valley at the published branch."
    )

    nearest_detuned_point = min(
        (point for point in points if point.coordinates != (benchmark_kl, benchmark_kq, benchmark_parent)),
        key=lambda point: (point.total_residue, point.coordinates[2], point.coordinates[0], point.coordinates[1]),
    )
    maximum_residue_point = max(
        points,
        key=lambda point: (point.total_residue, point.coordinates[2], point.coordinates[0], point.coordinates[1]),
    )
    positive_residues = total_residue_grid[total_residue_grid > 0.0]
    color_floor = max(float(np.min(positive_residues)) * 1.0e-3, MIN_COLOR_FLOOR) if positive_residues.size else MIN_COLOR_FLOOR

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
        color_floor=float(color_floor),
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


def _log_residue_values(values: np.ndarray, color_floor: float) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(values, dtype=float), float(color_floor)))


def _parent_axis_neighbor(scan: RigidityLandscapeScan) -> RigidityPoint:
    benchmark_kl, benchmark_kq, benchmark_parent = scan.benchmark_coordinates
    return min(
        (
            point
            for point in scan.points
            if point.coordinates[:2] == (benchmark_kl, benchmark_kq) and point.coordinates[2] != benchmark_parent
        ),
        key=lambda point: (point.total_residue, point.coordinates[2]),
    )


def _visible_plane_neighbor(scan: RigidityLandscapeScan) -> RigidityPoint:
    benchmark_kl, benchmark_kq, benchmark_parent = scan.benchmark_coordinates
    return min(
        (
            point
            for point in scan.points
            if point.coordinates[2] == benchmark_parent and point.coordinates[:2] != (benchmark_kl, benchmark_kq)
        ),
        key=lambda point: (point.total_residue, point.coordinates[0], point.coordinates[1]),
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
    benchmark_lepton_index, benchmark_quark_index, benchmark_parent_index = scan.benchmark_index
    parent_axis_neighbor = _parent_axis_neighbor(scan)
    visible_plane_neighbor = _visible_plane_neighbor(scan)

    log_residue_grid = _log_residue_values(scan.total_residue_grid, scan.color_floor)
    vmin = float(np.min(log_residue_grid))
    vmax = float(np.max(log_residue_grid))
    colormap = "viridis"

    lepton_mesh, quark_mesh, parent_mesh = np.meshgrid(
        scan.lepton_levels,
        scan.quark_levels,
        scan.parent_levels,
        indexing="ij",
    )

    parent_profile = np.maximum(scan.total_residue_grid[benchmark_lepton_index, benchmark_quark_index, :], scan.color_floor)
    parent_profile_neighbor_y = max(parent_axis_neighbor.total_residue, scan.color_floor)
    visible_plane_log = log_residue_grid[:, :, benchmark_parent_index]

    figure = plt.figure(figsize=(17.5, 5.8), constrained_layout=True)
    try:
        grid = figure.add_gridspec(1, 3, width_ratios=(1.65, 1.0, 1.05))
        ax_3d = figure.add_subplot(grid[0, 0], projection="3d")
        ax_parent = figure.add_subplot(grid[0, 1])
        ax_plane = figure.add_subplot(grid[0, 2])

        scatter = ax_3d.scatter(
            quark_mesh.ravel(),
            lepton_mesh.ravel(),
            parent_mesh.ravel(),
            c=log_residue_grid.ravel(),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            s=42,
            alpha=0.90,
            linewidths=0.0,
            depthshade=False,
        )
        ax_3d.scatter(
            [benchmark_kq],
            [benchmark_kl],
            [benchmark_parent],
            marker="*",
            s=230,
            color="#ef4444",
            edgecolors="white",
            linewidths=1.0,
            zorder=4,
        )
        ax_3d.scatter(
            [scan.nearest_detuned_point.coordinates[1]],
            [scan.nearest_detuned_point.coordinates[0]],
            [scan.nearest_detuned_point.coordinates[2]],
            marker="o",
            s=88,
            facecolors="none",
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
        )
        ax_3d.text(
            benchmark_kq + 0.25,
            benchmark_kl + 0.20,
            benchmark_parent + 0.35,
            "benchmark\n(26, 8, 312)",
            color="#111827",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#6b7280", "alpha": 0.92, "boxstyle": "round,pad=0.25"},
        )
        ax_3d.text(
            scan.nearest_detuned_point.coordinates[1] + 0.20,
            scan.nearest_detuned_point.coordinates[0] + 0.20,
            scan.nearest_detuned_point.coordinates[2] + 0.35,
            "nearest moat wall\n"
            rf"{scan.nearest_detuned_point.coordinates}\n"
            rf"$\mathcal{{R}}_{{\rm rigid}}={scan.nearest_detuned_point.total_residue:.2e}$",
            color="#f9fafb",
            fontsize=8.5,
            bbox={"facecolor": "#111827", "edgecolor": "#f9fafb", "alpha": 0.90, "boxstyle": "round,pad=0.25"},
        )
        ax_3d.view_init(elev=24, azim=45)
        ax_3d.set_xlabel(r"$k_q$")
        ax_3d.set_ylabel(r"$k_\ell$")
        ax_3d.set_zlabel(r"$K$")
        ax_3d.set_title(r"3D rigidity lattice: $\log_{10}\mathcal{R}_{\rm rigid}$")

        colorbar = figure.colorbar(scatter, ax=ax_3d, pad=0.06, fraction=0.04, shrink=0.88)
        colorbar.set_label(r"$\log_{10}\mathcal{R}_{\rm rigid}$")

        ax_parent.plot(scan.parent_levels, parent_profile, color="#1d4ed8", lw=2.3, marker="o", markersize=4.6)
        ax_parent.scatter(
            [benchmark_parent],
            [scan.color_floor],
            marker="*",
            s=200,
            color="#ef4444",
            edgecolors="white",
            linewidths=1.0,
            zorder=4,
        )
        ax_parent.scatter(
            [parent_axis_neighbor.coordinates[2]],
            [parent_profile_neighbor_y],
            marker="o",
            s=72,
            facecolors="none",
            edgecolors="#0f172a",
            linewidths=1.2,
            zorder=5,
        )
        ax_parent.axvline(benchmark_parent, color="#6b7280", lw=1.0, linestyle="--", alpha=0.8)
        ax_parent.set_yscale("log")
        ax_parent.set_xlabel(r"$K$ at fixed $(k_\ell, k_q) = (26, 8)$")
        ax_parent.set_ylabel(r"$\mathcal{R}_{\rm rigid}$")
        ax_parent.set_title(r"Parent-axis cut: deep stability valley")
        ax_parent.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.5)
        ax_parent.text(
            0.03,
            0.97,
            "\n".join(
                (
                    r"$\mathcal{R}_{\rm rigid}=\sqrt{\Delta_{\rm fr}^2 + (\Delta c_{\rm dark})^2 + (\Delta K_{\rm min})^2}$",
                    r"$\Delta K_{\rm min}=|K-\mathrm{lcm}(2k_\ell,3k_q)| / \mathrm{lcm}(2k_\ell,3k_q)$",
                    rf"nearest detuned parent: $K={parent_axis_neighbor.coordinates[2]}$, $\mathcal{{R}}_{{\rm rigid}}={parent_axis_neighbor.total_residue:.2e}$",
                )
            ),
            transform=ax_parent.transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "#6b7280", "alpha": 0.95, "boxstyle": "round,pad=0.30"},
        )

        plane_image = ax_plane.imshow(
            visible_plane_log,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            extent=(
                scan.quark_levels[0] - 0.5,
                scan.quark_levels[-1] + 0.5,
                scan.lepton_levels[0] - 0.5,
                scan.lepton_levels[-1] + 0.5,
            ),
        )
        ax_plane.scatter(
            [benchmark_kq],
            [benchmark_kl],
            marker="*",
            s=220,
            color="#ef4444",
            edgecolors="white",
            linewidths=1.0,
            zorder=4,
        )
        ax_plane.scatter(
            [visible_plane_neighbor.coordinates[1]],
            [visible_plane_neighbor.coordinates[0]],
            marker="o",
            s=72,
            facecolors="none",
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
        )
        ax_plane.annotate(
            rf"visible moat wall\n{visible_plane_neighbor.coordinates[:2]} at $K=312$\n$\mathcal{{R}}_{{\rm rigid}}={visible_plane_neighbor.total_residue:.2e}$",
            xy=(visible_plane_neighbor.coordinates[1], visible_plane_neighbor.coordinates[0]),
            xycoords="data",
            xytext=(0.97, 0.05),
            textcoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=8.6,
            color="#f9fafb",
            bbox={"facecolor": "#111827", "edgecolor": "#f9fafb", "alpha": 0.90, "boxstyle": "round,pad=0.28"},
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#f9fafb", "shrinkA": 4.0, "shrinkB": 5.0},
        )
        ax_plane.set_title(r"Visible moat at fixed $K = 312$")
        ax_plane.set_xlabel(r"$k_q$")
        ax_plane.set_ylabel(r"$k_\ell$")
        ax_plane.set_xticks(scan.quark_levels)
        ax_plane.set_yticks(scan.lepton_levels)
        ax_plane.grid(False)

        plane_colorbar = figure.colorbar(plane_image, ax=ax_plane, pad=0.03, fraction=0.05, shrink=0.88)
        plane_colorbar.set_label(r"$\log_{10}\mathcal{R}_{\rm rigid}$")

        figure.suptitle(
            r"Holographic Moat rigidity plot: the $(26, 8, 312)$ branch is an isolated stability valley",
            fontsize=14,
            fontweight="bold",
        )
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
        "c_dark_shift": float(point.c_dark_shift),
        "diophantine_gap": float(point.diophantine_gap),
        "total_residue": float(point.total_residue),
    }


def write_rigidity_landscape_json(scan: RigidityLandscapeScan, output_path: Path) -> Path:
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_coordinates": list(scan.benchmark_coordinates),
        "grid_shape": [len(scan.lepton_levels), len(scan.quark_levels), len(scan.parent_levels)],
        "lepton_levels": list(scan.lepton_levels),
        "quark_levels": list(scan.quark_levels),
        "parent_levels": list(scan.parent_levels),
        "rigidity_residue_definition": {
            "delta_fr": "max(|K/(2 k_l) - Z|, |K/(3 k_q) - Z|)",
            "delta_c_dark": "|c_dark(k_l, k_q, K) - c_dark(26, 8, 312)|",
            "delta_k_min": "|K - lcm(2 k_l, 3 k_q)| / lcm(2 k_l, 3 k_q)",
            "total": "sqrt(delta_fr^2 + delta_c_dark^2 + delta_k_min^2)",
        },
        "benchmark_point": _point_payload(scan.benchmark_point),
        "nearest_detuned_point": _point_payload(scan.nearest_detuned_point),
        "maximum_residue_point": _point_payload(scan.maximum_residue_point),
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
    print(f"benchmark rigidity residue      : {scan.benchmark_point.total_residue:.6e}")
    print(f"nearest detuned cell            : {scan.nearest_detuned_point.coordinates}")
    print(f"nearest detuned residue         : {scan.nearest_detuned_point.total_residue:.6e}")
    print(f"maximum scanned residue         : {scan.maximum_residue_point.total_residue:.6e}")
    print(f"wrote figure                    : {figure_path.as_posix()}")
    print(f"wrote data                      : {data_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
