from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import numpy as np

from .constants import SUPPLEMENTARY_IH_SUPPORT_MAP_FILENAME, SUPPLEMENTARY_TOLERANCE_TABLE_FILENAME
from .plotting_runtime import managed_figure
from .runtime_config import DEFAULT_SOLVER_CONFIG, SolverConfig
from .template_utils import render_latex_table


def _coerce_solver_config(config: SolverConfig | tuple[float, float]) -> tuple[SolverConfig, tuple[float, float]]:
    if isinstance(config, SolverConfig):
        return config, (config.rtol, config.atol)

    rtol, atol = config
    return replace(DEFAULT_SOLVER_CONFIG, rtol=float(rtol), atol=float(atol)), (float(rtol), float(atol))


def export_support_overlap_table(
    audit,
    output_dir: Path,
    *,
    support_overlap_result_factory,
    level: int,
) -> None:
    ih_matrix = audit.calculate_support_overlap(level=level)["IH"]
    ih_result = support_overlap_result_factory(matrix=np.asarray(ih_matrix, dtype=complex))
    (output_dir / SUPPLEMENTARY_IH_SUPPORT_MAP_FILENAME).write_text(
        ih_result.to_tex(
            support_deficit=audit.support_deficit,
            required_rank=audit.required_inverted_rank,
            relaxed_gap=audit.relaxed_inverted_gap,
        )
        + "\n",
        encoding="utf-8",
    )


def export_supplementary_tolerance_table(
    output_dir: Path,
    *,
    configs,
    derive_pmns,
    derive_ckm,
    derive_pull_table,
    lepton_intervals,
    quark_intervals,
    ckm_gamma_interval,
) -> None:
    resolved_configs = [_coerce_solver_config(config) for config in configs]

    def solve_for_config(solver_config: SolverConfig):
        pmns = derive_pmns(solver_config=solver_config)
        ckm = derive_ckm(solver_config=solver_config)
        return pmns, ckm, derive_pull_table(pmns, ckm)

    solved = [solve_for_config(solver_config) for solver_config, _ in resolved_configs]

    reference_pmns, reference_ckm, reference_pull = solved[-1]

    def max_sigma_shift(pmns, ckm) -> float:
        observable_rows = (
            (pmns.theta12_rg_deg, reference_pmns.theta12_rg_deg, lepton_intervals["theta12"].sigma),
            (pmns.theta13_rg_deg, reference_pmns.theta13_rg_deg, lepton_intervals["theta13"].sigma),
            (pmns.theta23_rg_deg, reference_pmns.theta23_rg_deg, lepton_intervals["theta23"].sigma),
            (pmns.delta_cp_rg_deg, reference_pmns.delta_cp_rg_deg, lepton_intervals["delta_cp"].sigma),
            (ckm.vus_rg, reference_ckm.vus_rg, quark_intervals["vus"].sigma),
            (ckm.vcb_rg, reference_ckm.vcb_rg, quark_intervals["vcb"].sigma),
            (ckm.vub_rg, reference_ckm.vub_rg, quark_intervals["vub"].sigma),
            (ckm.gamma_rg_deg, reference_ckm.gamma_rg_deg, ckm_gamma_interval.sigma),
        )
        return max(abs(left - right) / sigma for left, right, sigma in observable_rows)

    body_rows = []
    for (_, (rtol, atol)), (pmns, ckm, pull_table) in zip(resolved_configs, solved, strict=True):
        body_rows.append(
            rf"$10^{{{int(math.log10(rtol))}}}$ & $10^{{{int(math.log10(atol))}}}$ & {pull_table.predictive_chi2:.9f} & "
            rf"{abs(pull_table.predictive_chi2 - reference_pull.predictive_chi2):.3e} & {max_sigma_shift(pmns, ckm):.3e} \\")
    table_text = render_latex_table(
        column_spec="ccccc",
        header_rows=(r"rtol & atol & $\chi^2_{\rm pred}$ & $|\Delta\chi^2|$ vs. $10^{-12}$ & max $\sigma$-shift \\",),
        body_rows=tuple(body_rows),
        style="booktabs",
    )
    (output_dir / SUPPLEMENTARY_TOLERANCE_TABLE_FILENAME).write_text(table_text, encoding="utf-8")


def derive_ih_singular_value_spectrum(audit, *, level: int) -> dict[str, np.ndarray | float | int]:
    ih_matrix = np.asarray(audit.calculate_support_overlap(level=level)["IH"], dtype=complex)
    singular_values = np.linalg.svd(ih_matrix, compute_uv=False)
    singular_values = np.asarray(singular_values, dtype=float)
    return {
        "indices": np.arange(1, singular_values.size + 1, dtype=int),
        "singular_values": singular_values,
        "sigma_min": float(np.min(singular_values)),
        "rank": int(np.linalg.matrix_rank(ih_matrix)),
    }


def export_ih_singular_value_spectrum_figure(audit, output_path: Path, *, level: int) -> dict[str, np.ndarray | float | int]:
    spectrum = derive_ih_singular_value_spectrum(audit, level=level)
    singular_values = np.asarray(spectrum["singular_values"], dtype=float)
    indices = np.asarray(spectrum["indices"], dtype=int)
    reference_height = max(float(np.max(singular_values)), 1.0)

    with managed_figure(figsize=(5.6, 3.6)) as (fig, ax):
        ax.bar(indices, singular_values, color=("#991b1b", "#2563eb", "#64748b"), width=0.62)
        for index, value in zip(indices, singular_values, strict=True):
            label = f"{value:.2f}" if value >= 1.0e-3 else f"{value:.2e}"
            ax.text(index, value + 0.04 * reference_height, label, ha="center", va="bottom", fontsize=9)
        ax.annotate(
            rf"$\sigma_{{\min}}={spectrum['sigma_min']:.2e}$",
            xy=(indices[-1], singular_values[-1]),
            xytext=(0.52, 0.82),
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#1d4ed8"},
            fontsize=9,
            color="#1d4ed8",
        )
        ax.set_xlabel(r"singular-value index $i$")
        ax.set_ylabel(r"$\sigma_i(\mathcal{O}_{\rm IH})$")
        ax.set_xticks(indices)
        ax.set_ylim(0.0, 1.25 * reference_height)
        ax.set_title(r"Singular-value spectrum of the IH overlap matrix")
        ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
    return spectrum
