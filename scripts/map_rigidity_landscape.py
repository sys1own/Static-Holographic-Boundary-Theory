from __future__ import annotations

"""Map the 3D rigidity moat around the anomaly-free SHBT benchmark."""

import argparse
import json
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

from shbt.core.rigidity_landscape import (
    DEFAULT_LEPTON_HALF_WIDTH,
    DEFAULT_PARENT_HALF_WIDTH,
    DEFAULT_QUARK_HALF_WIDTH,
    EXPECTED_BENCHMARK,
    MIN_COLOR_CEILING,
    RigidityLandscapeScan,
    RigidityPoint,
    SymmetrySearcher,
    assert_unique_stable_fixed_point,
    build_centered_rigidity_landscape_scan,
    build_rigidity_landscape_scan,
    build_rigidity_point,
)
from shbt.plotting_runtime import plt


DEFAULT_OUTPUT_DIR = Path("results")
DEFAULT_FIGURE_FILENAME = "rigidity_moat.png"
DEFAULT_DATA_FILENAME = "rigidity_moat.json"
DEFAULT_DPI = 220


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
    nearest_detuned_point = scan.nearest_detuned_point
    maximum_residue = float(np.max(scan.delta_fr_grid))
    color_ceiling = max(maximum_residue, MIN_COLOR_CEILING)
    norm = mcolors.Normalize(vmin=0.0, vmax=color_ceiling)

    quark_coordinates = np.array([point.coordinates[1] for point in scan.points], dtype=float)
    lepton_coordinates = np.array([point.coordinates[0] for point in scan.points], dtype=float)
    parent_coordinates = np.array([point.coordinates[2] for point in scan.points], dtype=float)
    residue_values = np.array([point.delta_fr for point in scan.points], dtype=float)

    figure = plt.figure(figsize=(12.8, 7.2), constrained_layout=True)
    try:
        axis = figure.add_subplot(111, projection="3d")

        scatter = axis.scatter(
            quark_coordinates,
            lepton_coordinates,
            parent_coordinates,
            c=residue_values,
            cmap="viridis",
            norm=norm,
            s=58,
            alpha=0.92,
            linewidths=0.0,
            depthshade=False,
        )

        axis.scatter(
            [benchmark_kq],
            [benchmark_kl],
            [benchmark_parent],
            marker="*",
            s=280,
            color="#ef4444",
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
        )
        axis.scatter(
            [nearest_detuned_point.coordinates[1]],
            [nearest_detuned_point.coordinates[0]],
            [nearest_detuned_point.coordinates[2]],
            marker="o",
            s=90,
            facecolors="none",
            edgecolors="#0f172a",
            linewidths=1.4,
            zorder=6,
        )

        axis.text(
            benchmark_kq + 0.10,
            benchmark_kl + 0.10,
            benchmark_parent + 0.18,
            "benchmark\n(26, 8, 312)\nE = 0",
            fontsize=9,
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "#6b7280", "alpha": 0.94, "boxstyle": "round,pad=0.24"},
        )
        axis.text(
            nearest_detuned_point.coordinates[1] + 0.08,
            nearest_detuned_point.coordinates[0] + 0.08,
            nearest_detuned_point.coordinates[2] + 0.18,
            "nearest detuned cell\n"
            rf"{nearest_detuned_point.coordinates}\n"
            rf"E = {nearest_detuned_point.delta_fr_label}",
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
                    rf"5x5x5 benchmark-centered lattice scan around (26, 8, 312)",
                    rf"{len(scan.points) - 1}/{len(scan.points) - 1} detuned cells reopen the framing anomaly; nearest moat wall carries E={nearest_detuned_point.delta_fr_label}.",
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
        axis.set_zlabel(r"$K$")
        axis.set_xticks(scan.quark_levels)
        axis.set_yticks(scan.lepton_levels)
        axis.set_zticks(scan.parent_levels)
        axis.view_init(elev=24, azim=45)

        colorbar = figure.colorbar(scatter, ax=axis, pad=0.08, fraction=0.05, shrink=0.82)
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
            "reported_lattice": "benchmark-centered 3D moat lattice",
        },
        "benchmark_point": _point_payload(scan.benchmark_point),
        "nearest_detuned_point": _point_payload(scan.nearest_detuned_point),
        "maximum_residue_point": _point_payload(scan.maximum_residue_point),
        "delta_fr_grid": scan.delta_fr_grid.tolist(),
        "topological_closure_score_grid": scan.topological_closure_score_grid.tolist(),
        "topological_closure_survivors": [list(point.coordinates) for point in scan.stable_fixed_points],
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
    discovery = SymmetrySearcher(
        lepton_levels=scan.lepton_levels,
        quark_levels=scan.quark_levels,
        parent_levels=scan.parent_levels,
    ).discover_unique_fixed_point()
    certified_survivor = assert_unique_stable_fixed_point(discovery)

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
    print(f"discovered fixed point          : {discovery.evolutionary_best_point.coordinates}")
    print(f"certified unique survivor       : {certified_survivor}")
    print(f"maximum scanned residue         : {scan.maximum_residue_point.delta_fr:.6e}")
    print(f"wrote figure                    : {figure_path.as_posix()}")
    print(f"wrote data                      : {data_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
