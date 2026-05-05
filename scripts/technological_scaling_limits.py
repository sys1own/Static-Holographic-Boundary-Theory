from __future__ import annotations

"""Evaluate localized technological expansion against the SHBT complexity filter.

The script keeps the published benchmark branch fixed at ``(26, 8, 312)`` and
asks how much localized information loading ``ΔN`` can be injected before one of
two branch-protection mechanisms fails:

1. the localized holographic surface-tension load reaches ``τ = 1``;
2. the stiffness overhead is large enough to push the effective lepton support
   through the first half-integer crossing, reopening the framing anomaly.
"""

import argparse
import math
from dataclasses import dataclass
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

from shbt.audit.stiff_transport_audit import SolverNecessityAudit, build_solver_necessity_audit
from shbt.core import engine as vacuum_engine
from shbt.core.noether_bridge import framing_defect
from shbt.main import TopologicalVacuum, derive_single_bit_rigidity_audit, verify_unitary_bounds
from shbt.paths import ProjectPaths
from shbt.plotting_runtime import managed_figure


BENCHMARK_BRANCH = (26, 8, 312)
DEFAULT_OUTPUT_FILENAME = "technological_scaling_limits.png"
DEFAULT_DPI = 200
DEFAULT_SAMPLE_COUNT = 256
DEFAULT_RK45_TIME_LIMIT_SECONDS = 0.01


@dataclass(frozen=True)
class ScalingPoint:
    expansion_fraction: float
    delta_n_bits: float
    effective_delta_n_bits: float
    tau: float
    shifted_effective_lepton_level: int
    shifted_framing_gap: float
    kernel_panic: bool


@dataclass(frozen=True)
class ComplexityFilterResult:
    benchmark_branch: tuple[int, int, int]
    holographic_bits: float
    visible_density_ratio: float
    surface_alpha_inverse: float
    topological_mass_coordinate_ev: float
    topological_newton_coordinate_ev_minus2: float
    surface_tension_budget_bits: float
    surface_tension_budget_fraction: float
    surface_limit_bits: float
    surface_limit_fraction: float
    overhead_multiplier: float
    transport_stiff: bool
    transport_floor_condition_ratio: float
    transport_warning_count: int
    kernel_panic_bits: float
    kernel_panic_fraction: float
    kernel_panic_tau: float
    kernel_panic_effective_lepton_level: int
    kernel_panic_framing_gap: float
    fermi_limit_bits: float
    fermi_limit_fraction: float
    fermi_limit_radius_fraction: float
    scaling_curve: tuple[ScalingPoint, ...]


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ProjectPaths.ROOT))
    except ValueError:
        return str(path)


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _build_benchmark_vacuum() -> TopologicalVacuum:
    vacuum = TopologicalVacuum(k_l=BENCHMARK_BRANCH[0], k_q=BENCHMARK_BRANCH[1], parent_level=BENCHMARK_BRANCH[2])
    assert vacuum.target_tuple == BENCHMARK_BRANCH, (
        f"The technological scaling filter is locked to the published branch {BENCHMARK_BRANCH}, "
        f"received {vacuum.target_tuple}."
    )
    return vacuum


def _stiff_overhead_multiplier(unitary_audit: object, transport_audit: SolverNecessityAudit) -> tuple[bool, float]:
    transport_stiff = bool(
        transport_audit.pmns_radau.floor_condition_ratio > 1.0
        and transport_audit.pmns_rk45.integration_warning_count > 0
    )
    overhead_multiplier = 1.0 + (unitary_audit.complexity_utilization_fraction if transport_stiff else 0.0)
    return transport_stiff, float(overhead_multiplier)


def _evaluate_scaling_point(
    expansion_fraction: float,
    *,
    vacuum: TopologicalVacuum,
    overhead_multiplier: float,
    surface_tension_budget_bits: float,
) -> ScalingPoint:
    delta_n_bits = float(max(expansion_fraction, 0.0) * vacuum.bit_count)
    effective_delta_n_bits = float(overhead_multiplier * delta_n_bits)
    tau = float(0.0 if surface_tension_budget_bits <= 0.0 else effective_delta_n_bits / surface_tension_budget_bits)

    shifted_lepton_continuous = vacuum.lepton_level + (
        (vacuum.lepton_level + vacuum.quark_level) * effective_delta_n_bits / vacuum.bit_count
    )
    shifted_effective_lepton_level = max(1, _round_half_up(shifted_lepton_continuous))
    shifted_framing_gap = float(
        framing_defect(vacuum.parent_level, shifted_effective_lepton_level, vacuum.quark_level).delta_fr
    )

    return ScalingPoint(
        expansion_fraction=float(expansion_fraction),
        delta_n_bits=delta_n_bits,
        effective_delta_n_bits=effective_delta_n_bits,
        tau=tau,
        shifted_effective_lepton_level=int(shifted_effective_lepton_level),
        shifted_framing_gap=shifted_framing_gap,
        kernel_panic=bool(shifted_framing_gap > 0.0),
    )


def build_complexity_filter(
    *,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    rk45_time_limit_seconds: float = DEFAULT_RK45_TIME_LIMIT_SECONDS,
) -> ComplexityFilterResult:
    vacuum = _build_benchmark_vacuum()
    unitary_audit = verify_unitary_bounds(model=vacuum)
    transport_audit = build_solver_necessity_audit(rk45_time_limit_seconds=max(float(rk45_time_limit_seconds), 0.0))
    transport_stiff, overhead_multiplier = _stiff_overhead_multiplier(unitary_audit, transport_audit)

    visible_density_ratio = vacuum_engine.visible_level_density_ratio(
        parent_level=vacuum.parent_level,
        lepton_level=vacuum.lepton_level,
        quark_level=vacuum.quark_level,
    )
    surface_alpha_inverse = vacuum_engine.surface_tension_gauge_alpha_inverse(
        parent_level=vacuum.parent_level,
        lepton_level=vacuum.lepton_level,
        quark_level=vacuum.quark_level,
    )
    topological_mass_coordinate_ev = vacuum_engine.topological_mass_coordinate_ev(
        bit_count=vacuum.bit_count,
        kappa_geometric=vacuum.kappa_geometric,
    )
    topological_newton_coordinate_ev_minus2 = vacuum_engine.topological_newton_coordinate_ev_minus2()

    surface_tension_budget_bits = float(unitary_audit.clock_skew * vacuum.bit_count)
    surface_tension_budget_fraction = float(surface_tension_budget_bits / vacuum.bit_count)
    surface_limit_bits = float(surface_tension_budget_bits / overhead_multiplier)
    surface_limit_fraction = float(surface_limit_bits / vacuum.bit_count)

    visible_support = float(vacuum.lepton_level + vacuum.quark_level)
    kernel_panic_bits = float(vacuum.bit_count / (2.0 * visible_support * overhead_multiplier))
    kernel_panic_fraction = float(kernel_panic_bits / vacuum.bit_count)
    kernel_panic_bit_shift = max(1, math.ceil(overhead_multiplier * kernel_panic_bits))
    kernel_rigidity = derive_single_bit_rigidity_audit(bit_shift=kernel_panic_bit_shift, model=vacuum)
    kernel_panic_tau = float(0.0 if surface_tension_budget_bits <= 0.0 else (overhead_multiplier * kernel_panic_bits / surface_tension_budget_bits))

    assert kernel_rigidity.shifted_framing_gap > 0.0, "Kernel Panic threshold must reopen the framing anomaly."
    assert kernel_rigidity.shifted_effective_lepton_level > vacuum.lepton_level, (
        "Kernel Panic threshold must detune the effective localized lepton support."
    )

    fermi_limit_bits = float(min(surface_limit_bits, kernel_panic_bits))
    fermi_limit_fraction = float(fermi_limit_bits / vacuum.bit_count)
    fermi_limit_radius_fraction = float(math.sqrt(fermi_limit_fraction))

    resolved_sample_count = max(int(sample_count), 64)
    max_fraction = float(min(0.02, 1.25 * max(surface_limit_fraction, kernel_panic_fraction)))
    expansion_grid = np.linspace(0.0, max_fraction, num=resolved_sample_count)
    scaling_curve = tuple(
        _evaluate_scaling_point(
            float(expansion_fraction),
            vacuum=vacuum,
            overhead_multiplier=overhead_multiplier,
            surface_tension_budget_bits=surface_tension_budget_bits,
        )
        for expansion_fraction in expansion_grid
    )

    return ComplexityFilterResult(
        benchmark_branch=BENCHMARK_BRANCH,
        holographic_bits=float(vacuum.bit_count),
        visible_density_ratio=float(visible_density_ratio),
        surface_alpha_inverse=float(surface_alpha_inverse),
        topological_mass_coordinate_ev=float(topological_mass_coordinate_ev),
        topological_newton_coordinate_ev_minus2=float(topological_newton_coordinate_ev_minus2),
        surface_tension_budget_bits=surface_tension_budget_bits,
        surface_tension_budget_fraction=surface_tension_budget_fraction,
        surface_limit_bits=surface_limit_bits,
        surface_limit_fraction=surface_limit_fraction,
        overhead_multiplier=float(overhead_multiplier),
        transport_stiff=bool(transport_stiff),
        transport_floor_condition_ratio=float(transport_audit.pmns_radau.floor_condition_ratio),
        transport_warning_count=int(transport_audit.pmns_rk45.integration_warning_count),
        kernel_panic_bits=kernel_panic_bits,
        kernel_panic_fraction=kernel_panic_fraction,
        kernel_panic_tau=kernel_panic_tau,
        kernel_panic_effective_lepton_level=int(kernel_rigidity.shifted_effective_lepton_level),
        kernel_panic_framing_gap=float(kernel_rigidity.shifted_framing_gap),
        fermi_limit_bits=fermi_limit_bits,
        fermi_limit_fraction=fermi_limit_fraction,
        fermi_limit_radius_fraction=fermi_limit_radius_fraction,
        scaling_curve=scaling_curve,
    )


def render_scaling_chart(
    result: ComplexityFilterResult,
    output_path: Path,
    *,
    dpi: int = DEFAULT_DPI,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_percent = np.asarray([100.0 * point.expansion_fraction for point in result.scaling_curve], dtype=float)
    tau_values = np.asarray([point.tau for point in result.scaling_curve], dtype=float)
    framing_gap_values = np.asarray([point.shifted_framing_gap for point in result.scaling_curve], dtype=float)

    with managed_figure(2, 1, figsize=(10.5, 8.0), sharex=True, height_ratios=(3.0, 2.0), constrained_layout=True) as (fig, axes):
        ax_top, ax_bottom = axes

        ax_top.plot(x_percent, tau_values, color="#2563eb", lw=2.4, label=r"$\tau(\Delta N)$")
        ax_top.axhline(1.0, color="#dc2626", lw=1.4, ls="--", label=r"surface-tension limit $\tau=1$")
        ax_top.axvline(100.0 * result.kernel_panic_fraction, color="#7c3aed", lw=1.5, ls=":", label="Kernel Panic")
        ax_top.axvline(100.0 * result.fermi_limit_fraction, color="#059669", lw=1.6, ls="-.", label="Fermi Limit")
        ax_top.fill_between(x_percent, tau_values, 1.0, where=tau_values <= 1.0, color="#dbeafe", alpha=0.25)
        ax_top.fill_between(x_percent, tau_values, 1.0, where=tau_values > 1.0, color="#fee2e2", alpha=0.25)
        ax_top.set_ylabel(r"localized surface load $\tau$")
        ax_top.set_ylim(0.0, max(1.15, 1.05 * float(np.max(tau_values))))
        ax_top.grid(alpha=0.24)
        ax_top.legend(loc="upper left", fontsize=9)
        ax_top.annotate(
            (
                rf"$N_{{\rm holo}}={result.holographic_bits:.3e}$\n"
                rf"budget = {100.0 * result.surface_tension_budget_fraction:.3f}\% of $N_{{\rm holo}}$\n"
                rf"stiff overhead = {result.overhead_multiplier:.6f}"
            ),
            xy=(0.98, 0.98),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#94a3b8", "alpha": 0.92, "boxstyle": "round,pad=0.30"},
        )

        ax_bottom.step(x_percent, framing_gap_values, where="post", color="#7c3aed", lw=2.3, label=r"$\Delta_{\rm fr}$")
        ax_bottom.axhline(0.0, color="#111827", lw=1.0)
        ax_bottom.axvline(100.0 * result.kernel_panic_fraction, color="#7c3aed", lw=1.5, ls=":")
        ax_bottom.axvline(100.0 * result.fermi_limit_fraction, color="#059669", lw=1.6, ls="-.")
        ax_bottom.fill_between(
            x_percent,
            0.0,
            framing_gap_values,
            where=framing_gap_values > 0.0,
            step="post",
            color="#ede9fe",
            alpha=0.35,
        )
        ax_bottom.set_xlabel(r"civilization expansion $\Delta N / N_{\rm holo}$ [\%]")
        ax_bottom.set_ylabel(r"framing anomaly $\Delta_{\rm fr}$")
        ax_bottom.grid(alpha=0.24)
        ax_bottom.legend(loc="upper left", fontsize=9)
        ax_bottom.annotate(
            (
                rf"Kernel Panic: $k_\ell \to {result.kernel_panic_effective_lepton_level}$\n"
                rf"$\Delta_{{\rm fr}}={result.kernel_panic_framing_gap:.6f}$ at $\tau={result.kernel_panic_tau:.6f}$"
            ),
            xy=(100.0 * result.kernel_panic_fraction, result.kernel_panic_framing_gap),
            xycoords="data",
            xytext=(0.98, 0.92),
            textcoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#c4b5fd", "alpha": 0.94, "boxstyle": "round,pad=0.30"},
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#7c3aed", "shrinkA": 4.0, "shrinkB": 4.0},
        )

        fig.suptitle(
            "Complexity Filter for localized bulk loading on the anomaly-free SHBT benchmark",
            fontsize=13,
            fontweight="bold",
        )
        fig.savefig(output_path, dpi=int(dpi), format="png")
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ProjectPaths.RESULTS)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument(
        "--rk45-time-limit-seconds",
        type=float,
        default=DEFAULT_RK45_TIME_LIMIT_SECONDS,
        help="Wall-clock ceiling for the explicit RK45 stiffness witness.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ProjectPaths.ensure_dirs()

    result = build_complexity_filter(
        sample_count=max(args.sample_count, 64),
        rk45_time_limit_seconds=float(args.rk45_time_limit_seconds),
    )
    output_path = render_scaling_chart(
        result,
        Path(args.output_dir) / DEFAULT_OUTPUT_FILENAME,
        dpi=max(int(args.dpi), 72),
    )

    print(f"benchmark branch                : {result.benchmark_branch}")
    print(f"visible density ratio           : {result.visible_density_ratio:.12f}")
    print(f"surface alpha inverse           : {result.surface_alpha_inverse:.12f}")
    print(f"topological mass coordinate [eV]: {result.topological_mass_coordinate_ev:.12e}")
    print(f"topological Newton lock [eV^-2] : {result.topological_newton_coordinate_ev_minus2:.12e}")
    print(f"stiff transport active          : {int(result.transport_stiff)}")
    print(f"transport floor ratio           : {result.transport_floor_condition_ratio:.6e}")
    print(f"explicit RK45 warnings          : {result.transport_warning_count}")
    print(f"surface tension budget [bits]   : {result.surface_tension_budget_bits:.6e}")
    print(f"surface-limit threshold [bits]  : {result.surface_limit_bits:.6e}")
    print(f"Kernel Panic threshold [bits]   : {result.kernel_panic_bits:.6e}")
    print(f"Kernel Panic tau                : {result.kernel_panic_tau:.12f}")
    print(f"Kernel Panic shifted k_l        : {result.kernel_panic_effective_lepton_level}")
    print(f"Kernel Panic Delta_fr           : {result.kernel_panic_framing_gap:.12f}")
    print(f"Fermi Limit [bits]              : {result.fermi_limit_bits:.6e}")
    print(f"Fermi Limit / N_holo            : {result.fermi_limit_fraction:.12f}")
    print(f"Fermi radius / R_h              : {result.fermi_limit_radius_fraction:.12f}")
    print(f"wrote figure                    : {_display_path(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
