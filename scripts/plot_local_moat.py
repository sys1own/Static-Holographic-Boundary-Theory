from __future__ import annotations

"""Visualize the local moat around the anomaly-free SHBT benchmark branch."""

import argparse
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pub.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
    from pub.noether_bridge import (
        anomalous_source_tensor,
        bulk_closure_tensor,
        framing_defect,
        load_c_dark_completion_fraction,
        newton_constant_lock,
        saturation_audit,
    )
    from pub.plotting_runtime import managed_figure, plt
else:
    from .constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
    from .noether_bridge import (
        anomalous_source_tensor,
        bulk_closure_tensor,
        framing_defect,
        load_c_dark_completion_fraction,
        newton_constant_lock,
        saturation_audit,
    )
    from .plotting_runtime import managed_figure, plt


EXPECTED_BENCHMARK = (26, 8, 312)
LEPTON_SCAN = tuple(range(24, 29))
QUARK_SCAN = tuple(range(6, 11))
DEFAULT_OUTPUT_DIR = Path("results")
DEFAULT_OUTPUT_FILENAME = "local_moat_topology.png"
DEFAULT_DPI = 200


@dataclass(frozen=True)
class OneStepMoatWitness:
    coordinates: tuple[int, int]
    delta_fr: Fraction
    anomalous_source_m2: float


@dataclass(frozen=True)
class LocalMoatScan:
    parent_level: int
    lepton_levels: tuple[int, ...]
    quark_levels: tuple[int, ...]
    delta_fr_grid: np.ndarray
    delta_fr_labels: tuple[tuple[str, ...], ...]
    anomalous_source_grid_m2: np.ndarray
    benchmark_coordinates: tuple[int, int, int]
    benchmark_index: tuple[int, int]
    one_step_witness: OneStepMoatWitness
    active_source_count: int



def _benchmark_coordinates() -> tuple[int, int, int]:
    benchmark = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))
    assert benchmark == EXPECTED_BENCHMARK, (
        f"The local moat plotter is locked to the published branch {EXPECTED_BENCHMARK}, received {benchmark}."
    )
    return benchmark



def build_local_moat_scan(
    *,
    parent_level: int = PARENT_LEVEL,
    lepton_levels: Sequence[int] = LEPTON_SCAN,
    quark_levels: Sequence[int] = QUARK_SCAN,
) -> LocalMoatScan:
    benchmark_kl, benchmark_kq, benchmark_parent = _benchmark_coordinates()
    resolved_parent_level = int(parent_level)
    assert resolved_parent_level == benchmark_parent, (
        f"The reviewer-trap moat is defined on K={benchmark_parent}, received K={resolved_parent_level}."
    )

    resolved_lepton_levels = tuple(int(level) for level in lepton_levels)
    resolved_quark_levels = tuple(int(level) for level in quark_levels)
    benchmark_index = (
        resolved_lepton_levels.index(benchmark_kl),
        resolved_quark_levels.index(benchmark_kq),
    )

    c_dark_fraction = load_c_dark_completion_fraction()
    newton_lock = newton_constant_lock(c_dark_fraction=c_dark_fraction)
    saturation = saturation_audit()
    q_iso_ev4 = saturation.lambda_obs_ev2 / newton_lock.eight_pi_g_effective_ev_minus2

    delta_fr_grid = np.zeros((len(resolved_lepton_levels), len(resolved_quark_levels)), dtype=float)
    anomalous_source_grid_m2 = np.zeros_like(delta_fr_grid)
    delta_fr_labels: list[tuple[str, ...]] = []
    one_step_candidates: list[OneStepMoatWitness] = []

    for lepton_index, lepton_level in enumerate(resolved_lepton_levels):
        row_labels: list[str] = []
        for quark_index, quark_level in enumerate(resolved_quark_levels):
            defect = framing_defect(resolved_parent_level, lepton_level, quark_level)
            closure_tensor = bulk_closure_tensor(defect.delta_fr, q_iso_ev4)
            _, anomalous_source_si_m2 = anomalous_source_tensor(closure_tensor, newton_lock)

            delta_fr_grid[lepton_index, quark_index] = float(defect.delta_fr)
            anomalous_source_grid_m2[lepton_index, quark_index] = float(abs(anomalous_source_si_m2.amplitude))
            row_labels.append(str(defect.delta_fr))

            if max(abs(lepton_level - benchmark_kl), abs(quark_level - benchmark_kq)) == 1:
                one_step_candidates.append(
                    OneStepMoatWitness(
                        coordinates=(lepton_level, quark_level),
                        delta_fr=defect.delta_fr,
                        anomalous_source_m2=float(abs(anomalous_source_si_m2.amplitude)),
                    )
                )
        delta_fr_labels.append(tuple(row_labels))

    benchmark_delta = framing_defect(resolved_parent_level, benchmark_kl, benchmark_kq).delta_fr
    assert benchmark_delta == 0, (
        f"The anomaly-free benchmark cell must satisfy Delta_fr=0, received {benchmark_delta}."
    )
    assert anomalous_source_grid_m2[benchmark_index[0], benchmark_index[1]] == 0.0, (
        "The reviewer-trap source term must vanish exactly on the benchmark cell."
    )

    one_step_witness = min(
        one_step_candidates,
        key=lambda witness: (float(witness.delta_fr), witness.coordinates[0], witness.coordinates[1]),
    )
    active_source_count = int(np.count_nonzero(anomalous_source_grid_m2 > 0.0))

    return LocalMoatScan(
        parent_level=resolved_parent_level,
        lepton_levels=resolved_lepton_levels,
        quark_levels=resolved_quark_levels,
        delta_fr_grid=np.asarray(delta_fr_grid, dtype=float),
        delta_fr_labels=tuple(delta_fr_labels),
        anomalous_source_grid_m2=np.asarray(anomalous_source_grid_m2, dtype=float),
        benchmark_coordinates=(benchmark_kl, benchmark_kq, benchmark_parent),
        benchmark_index=benchmark_index,
        one_step_witness=one_step_witness,
        active_source_count=active_source_count,
    )



def render_local_moat_plot(scan: LocalMoatScan, output_path: Path, *, dpi: int = DEFAULT_DPI) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    benchmark_kl, benchmark_kq, benchmark_parent = scan.benchmark_coordinates
    benchmark_row, benchmark_col = scan.benchmark_index
    source_masked = np.ma.masked_less_equal(scan.anomalous_source_grid_m2, 0.0)
    source_cmap = plt.get_cmap("viridis").copy()
    source_cmap.set_bad(color="#f8fafc")
    source_norm = mcolors.LogNorm(vmin=float(source_masked.min()), vmax=float(source_masked.max()))
    extent = (
        scan.quark_levels[0] - 0.5,
        scan.quark_levels[-1] + 0.5,
        scan.lepton_levels[0] - 0.5,
        scan.lepton_levels[-1] + 0.5,
    )

    with managed_figure(1, 2, figsize=(12.4, 5.8), constrained_layout=True) as (fig, axes):
        ax_left, ax_right = axes

        moat_image = ax_left.imshow(
            scan.delta_fr_grid,
            origin="lower",
            aspect="equal",
            interpolation="nearest",
            cmap="magma_r",
            extent=extent,
        )
        moat_colorbar = fig.colorbar(moat_image, ax=ax_left, pad=0.03, shrink=0.92)
        moat_colorbar.set_label(r"$|\Delta_{\rm fr}|$")

        for lepton_index, lepton_level in enumerate(scan.lepton_levels):
            for quark_index, quark_level in enumerate(scan.quark_levels):
                label = scan.delta_fr_labels[lepton_index][quark_index]
                value = scan.delta_fr_grid[lepton_index, quark_index]
                text_color = "white" if value >= 0.28 else "#111827"
                ax_left.text(
                    quark_level,
                    lepton_level,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                    fontweight="bold" if value == 0.0 else None,
                )

        moat_patch = Rectangle(
            (benchmark_kq - 1.5, benchmark_kl - 1.5),
            3.0,
            3.0,
            fill=False,
            edgecolor="#22d3ee",
            linewidth=1.8,
            linestyle="--",
            zorder=4,
        )
        ax_left.add_patch(moat_patch)
        ax_left.scatter(
            [benchmark_kq],
            [benchmark_kl],
            marker="*",
            s=210,
            color="#2563eb",
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
        )
        ax_left.annotate(
            "Anomaly-free island\n"
            rf"$(k_\ell,k_q,K)=({benchmark_kl},{benchmark_kq},{benchmark_parent})$\n"
            r"$\Delta_{\rm fr}=0$",
            xy=(benchmark_kq, benchmark_kl),
            xycoords="data",
            xytext=(0.03, 0.97),
            textcoords="axes fraction",
            ha="left",
            va="top",
            fontsize=9,
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "#6b7280", "alpha": 0.94, "boxstyle": "round,pad=0.30"},
            arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#6b7280", "shrinkA": 4.0, "shrinkB": 5.0},
        )
        ax_left.annotate(
            "Local moat\n"
            rf"one-step minimum = {scan.one_step_witness.delta_fr}\n"
            rf"at $(k_\ell,k_q)=({scan.one_step_witness.coordinates[0]},{scan.one_step_witness.coordinates[1]})$",
            xy=(scan.one_step_witness.coordinates[1], scan.one_step_witness.coordinates[0]),
            xycoords="data",
            xytext=(0.97, 0.10),
            textcoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=9,
            color="#f9fafb",
            bbox={"facecolor": "#111827", "edgecolor": "#f9fafb", "alpha": 0.90, "boxstyle": "round,pad=0.30"},
            arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#f9fafb", "shrinkA": 4.0, "shrinkB": 5.0},
        )
        ax_left.set_title(r"Local moat topology: framing anomaly $|\Delta_{\rm fr}|$")
        ax_left.set_xlabel(r"$k_q$")
        ax_left.set_ylabel(r"$k_\ell$")
        ax_left.set_xticks(scan.quark_levels)
        ax_left.set_yticks(scan.lepton_levels)
        ax_left.set_facecolor("#f8fafc")
        ax_left.grid(False)

        source_image = ax_right.imshow(
            source_masked,
            origin="lower",
            aspect="equal",
            interpolation="nearest",
            cmap=source_cmap,
            norm=source_norm,
            extent=extent,
        )
        source_colorbar = fig.colorbar(source_image, ax=ax_right, pad=0.03, shrink=0.92)
        source_colorbar.set_label(r"$|J_{\mu\nu}^{(a)}|$ amplitude [m$^{-2}$]")

        ax_right.scatter(
            [benchmark_kq],
            [benchmark_kl],
            marker="*",
            s=210,
            color="#2563eb",
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
        )
        ax_right.add_patch(
            Rectangle(
                (benchmark_kq - 0.5, benchmark_kl - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="white",
                linewidth=1.6,
                zorder=4,
            )
        )
        ax_right.annotate(
            "Reviewer Trap\n"
            rf"$|J_{{\mu\nu}}^{{(a)}}|>0$ on {scan.active_source_count}/{scan.delta_fr_grid.size - 1} detuned cells\n"
            r"Equivalence Principle survives only on the island",
            xy=(scan.one_step_witness.coordinates[1], scan.one_step_witness.coordinates[0]),
            xycoords="data",
            xytext=(0.03, 0.97),
            textcoords="axes fraction",
            ha="left",
            va="top",
            fontsize=9,
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "#6b7280", "alpha": 0.94, "boxstyle": "round,pad=0.30"},
            arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#6b7280", "shrinkA": 4.0, "shrinkB": 5.0},
        )
        ax_right.annotate(
            rf"first moat source = {scan.one_step_witness.anomalous_source_m2:.2e} m$^{{-2}}$",
            xy=(scan.one_step_witness.coordinates[1], scan.one_step_witness.coordinates[0]),
            xycoords="data",
            xytext=(0.97, 0.08),
            textcoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=9,
            color="#f9fafb",
            bbox={"facecolor": "#111827", "edgecolor": "#f9fafb", "alpha": 0.90, "boxstyle": "round,pad=0.30"},
            arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#f9fafb", "shrinkA": 4.0, "shrinkB": 5.0},
        )
        ax_right.set_title(r"Equivalence-principle failure: anomalous source $|J_{\mu\nu}^{(a)}|$")
        ax_right.set_xlabel(r"$k_q$")
        ax_right.set_ylabel(r"$k_\ell$")
        ax_right.set_xticks(scan.quark_levels)
        ax_right.set_yticks(scan.lepton_levels)
        ax_right.set_facecolor("#f8fafc")
        ax_right.grid(False)

        fig.suptitle(
            r"Strict isolation of the anomaly-free island and immediate activation of the Reviewer Trap",
            fontsize=13,
            fontweight="bold",
        )
        fig.savefig(output_path, dpi=int(dpi), format="png")
    return output_path



def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parser.parse_args(list(argv) if argv is not None else None)



def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scan = build_local_moat_scan()
    output_path = render_local_moat_plot(scan, args.output_dir / DEFAULT_OUTPUT_FILENAME, dpi=args.dpi)
    print(f"benchmark branch                : {scan.benchmark_coordinates}")
    print(f"benchmark Delta_fr              : {scan.delta_fr_labels[scan.benchmark_index[0]][scan.benchmark_index[1]]}")
    print(
        "one-step moat witness           : "
        f"(k_l,k_q)=({scan.one_step_witness.coordinates[0]},{scan.one_step_witness.coordinates[1]}) "
        f"with Delta_fr={scan.one_step_witness.delta_fr}"
    )
    print(f"reviewer-trap active cells      : {scan.active_source_count}")
    print(f"wrote figure                    : {output_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
