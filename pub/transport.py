from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .plotting_runtime import managed_figure


def transport_observable_vector(pmns: Any, ckm: Any) -> np.ndarray:
    """Collect the transport observables shared by covariance and sweep audits."""

    return np.array(
        [
            pmns.theta12_rg_deg,
            pmns.theta13_rg_deg,
            pmns.theta23_rg_deg,
            pmns.delta_cp_rg_deg,
            ckm.vus_rg,
            ckm.vcb_rg,
            ckm.vub_rg,
            ckm.gamma_rg_deg,
        ],
        dtype=float,
    )


def transport_observable_delta(
    observable_name: str,
    upper_value: float,
    lower_value: float,
    *,
    angular_observables: frozenset[str] | tuple[str, ...],
    wrapped_angle_difference_deg: Callable[[float, float], float],
) -> float:
    """Return a signed observable displacement, wrapping angles onto the principal branch."""

    if observable_name in angular_observables:
        return wrapped_angle_difference_deg(upper_value, lower_value)
    return float(upper_value - lower_value)


def resolve_ckm_phase_tilt_weight_grid(
    weight_grid: np.ndarray | None,
    benchmark_weight: float,
    *,
    point_count: int = 241,
) -> np.ndarray:
    """Return the audit grid for the CKM threshold-weight profile."""

    if weight_grid is None:
        span = max(0.2, abs(float(benchmark_weight)))
        base_grid = np.linspace(
            float(benchmark_weight) - span,
            float(benchmark_weight) + span,
            int(point_count),
            dtype=float,
        )
        return np.unique(
            np.concatenate((base_grid, np.array([0.0, float(benchmark_weight)], dtype=float)))
        )

    weights = np.asarray(weight_grid, dtype=float)
    if weights.ndim != 1 or weights.size < 1:
        raise ValueError("weight_grid must be a one-dimensional array with at least one point")
    return weights


def derive_ckm_with_threshold_residue(
    derive_ckm: Callable[..., Any],
    /,
    *,
    threshold_residue: float | None,
    **kwargs: Any,
) -> Any:
    """Call a CKM derivation helper across the legacy/new threshold keyword rename."""

    try:
        parameters = inspect.signature(derive_ckm).parameters
    except (TypeError, ValueError):
        parameters = {}

    if "gut_threshold_residue" in parameters:
        return derive_ckm(gut_threshold_residue=threshold_residue, **kwargs)
    if "ckm_phase_tilt_parameter" in parameters:
        return derive_ckm(ckm_phase_tilt_parameter=threshold_residue, **kwargs)
    return derive_ckm(gut_threshold_residue=threshold_residue, **kwargs)


def build_ckm_phase_tilt_profile(
    *,
    reference_pmns: Any,
    weight_grid: np.ndarray | None,
    output_path: Path | None,
    quark_level: int,
    parent_level: int,
    scale_ratio: float,
    benchmark_weight: float,
    ckm_phase_tilt_invariance_tolerance: float,
    derive_ckm: Callable[..., Any],
    derive_pull_table: Callable[[Any, Any], Any],
    plt: Any,
    profile_data_factory: Callable[..., Any],
) -> Any:
    """Evaluate and plot the full CKM threshold-weight profile over the supplied grid."""

    weights = resolve_ckm_phase_tilt_weight_grid(weight_grid, benchmark_weight)
    bare_ckm = derive_ckm_with_threshold_residue(
        derive_ckm,
        level=quark_level,
        parent_level=parent_level,
        scale_ratio=scale_ratio,
        threshold_residue=0.0,
    )
    sampled_ckms = tuple(
        derive_ckm_with_threshold_residue(
            derive_ckm,
            level=quark_level,
            parent_level=parent_level,
            scale_ratio=scale_ratio,
            threshold_residue=float(weight),
        )
        for weight in weights
    )
    sampled_pull_tables = tuple(derive_pull_table(reference_pmns, ckm) for ckm in sampled_ckms)

    chi2_values = np.array([pull_table.predictive_chi2 for pull_table in sampled_pull_tables], dtype=float)
    gamma_values = np.array([ckm.gamma_rg_deg for ckm in sampled_ckms], dtype=float)
    vus_values = np.array([ckm.vus_rg for ckm in sampled_ckms], dtype=float)
    vcb_values = np.array([ckm.vcb_rg for ckm in sampled_ckms], dtype=float)
    vub_values = np.array([ckm.vub_rg for ckm in sampled_ckms], dtype=float)

    best_index = int(np.argmin(chi2_values))
    best_fit_weight = float(weights[best_index])
    best_fit_chi2 = float(chi2_values[best_index])
    delta_chi2_values = chi2_values - best_fit_chi2

    benchmark_ckm = derive_ckm_with_threshold_residue(
        derive_ckm,
        level=quark_level,
        parent_level=parent_level,
        scale_ratio=scale_ratio,
        threshold_residue=benchmark_weight,
    )
    benchmark_pull_table = derive_pull_table(reference_pmns, benchmark_ckm)
    benchmark_delta_chi2 = float(benchmark_pull_table.predictive_chi2 - best_fit_chi2)
    benchmark_gamma_deg = float(benchmark_ckm.gamma_rg_deg)

    evaluated_ckms = (*sampled_ckms, benchmark_ckm)
    max_vus_shift = float(max(abs(ckm.vus_rg - bare_ckm.vus_rg) for ckm in evaluated_ckms))
    max_vcb_shift = float(max(abs(ckm.vcb_rg - bare_ckm.vcb_rg) for ckm in evaluated_ckms))
    max_vub_shift = float(max(abs(ckm.vub_rg - bare_ckm.vub_rg) for ckm in evaluated_ckms))
    if max(max_vus_shift, max_vcb_shift, max_vub_shift) > ckm_phase_tilt_invariance_tolerance:
        raise RuntimeError(
            "GUT-threshold residue audit failed: the matched Wilson coefficient is shifting baseline S-matrix magnitudes beyond tolerance."
        )

    ordered_indices = np.argsort(weights)
    ordered_weights = weights[ordered_indices]
    ordered_delta_chi2 = delta_chi2_values[ordered_indices]

    with managed_figure(figsize=(6.0, 3.8)) as (fig, ax):
        ax.plot(ordered_weights, ordered_delta_chi2, color="#1d4ed8", lw=2.0, label=r"$\Delta\chi^2_{\rm pred}(\mathcal{R}_{\rm GUT})$")
        ax.scatter([benchmark_weight], [benchmark_delta_chi2], color="#2563eb", s=40, zorder=5, label=r"benchmark $\mathcal{R}_{\rm GUT}$")
        ax.scatter([best_fit_weight], [0.0], color="#991b1b", s=35, zorder=5, label="off-shell minimum")
        ax.axhline(0.0, color="#6b7280", lw=1.0, ls="--", alpha=0.75)
        ax.annotate(
            rf"benchmark $\mathcal{{R}}_{{\rm GUT}}={benchmark_weight:.3f}$",
            xy=(benchmark_weight, benchmark_delta_chi2),
            xytext=(0.03, 0.94),
            textcoords="axes fraction",
            fontsize=9,
            color="#1d4ed8",
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "#2563eb", "alpha": 0.88, "boxstyle": "round,pad=0.25"},
        )
        ax.annotate(
            rf"minimum $\mathcal{{R}}_{{\rm GUT}}={best_fit_weight:.3f}$",
            xy=(best_fit_weight, 0.0),
            xytext=(0.03, 0.84),
            textcoords="axes fraction",
            fontsize=9,
            color="#991b1b",
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "#991b1b", "alpha": 0.88, "boxstyle": "round,pad=0.25"},
        )
        ax.text(
            0.03,
            0.14,
            rf"Wilson-coefficient stability check:\nmax $|\Delta |V_{{us}}||={max_vus_shift:.1e}$\nmax $|\Delta |V_{{cb}}||={max_vcb_shift:.1e}$\nmax $|\Delta |V_{{ub}}||={max_vub_shift:.1e}$",
            transform=ax.transAxes,
            fontsize=8,
            color="#111827",
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "#9ca3af", "alpha": 0.92, "boxstyle": "round,pad=0.25"},
        )
        ax.set_xlabel(r"$\mathcal{R}_{\rm GUT}\equiv \lambda_{12}^{(5)}/|Y_{12}^{(0)}|$")
        ax.set_ylabel(r"$\Delta\chi^2_{\rm pred}$")
        x_min = float(np.min(weights))
        x_max = float(np.max(weights))
        x_margin = max(0.05, 0.03 * max(x_max - x_min, 1.0))
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        y_max = max(1.0, float(np.max(delta_chi2_values)) * 1.05)
        y_min = min(-0.05 * y_max, float(np.min(delta_chi2_values)) - 0.05 * y_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(r"One-loop Wilson-coefficient matching audit")
        ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
        ax.legend(frameon=False, fontsize=8, loc="upper center")
        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, dpi=200)

    return profile_data_factory(
        weight_grid=weights,
        chi2_values=chi2_values,
        delta_chi2_values=delta_chi2_values,
        gamma_values=gamma_values,
        vus_values=vus_values,
        vcb_values=vcb_values,
        vub_values=vub_values,
        best_fit_weight=best_fit_weight,
        best_fit_chi2=best_fit_chi2,
        benchmark_weight=benchmark_weight,
        benchmark_delta_chi2=benchmark_delta_chi2,
        benchmark_gamma_deg=benchmark_gamma_deg,
        max_vus_shift=max_vus_shift,
        max_vcb_shift=max_vcb_shift,
        max_vub_shift=max_vub_shift,
    )


__all__ = [
    "build_ckm_phase_tilt_profile",
    "derive_ckm_with_threshold_residue",
    "resolve_ckm_phase_tilt_weight_grid",
    "transport_observable_delta",
    "transport_observable_vector",
]
