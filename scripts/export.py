from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .audit_generator import (
    derive_ih_singular_value_spectrum,
    export_ih_singular_value_spectrum_figure,
    export_support_overlap_table,
    export_supplementary_tolerance_table,
)
from .plotting_runtime import managed_figure
from .template_utils import render_latex_table


def write_json_artifact(output_path: Path, payload: Mapping[str, object]) -> None:
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_artifact(output_path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_skeletal_latex_table(
    output_path: Path,
    *,
    column_spec: str,
    header_rows: Sequence[str],
    body_rows: Sequence[str],
    footer_rows: Sequence[str] = (),
    opening_lines: Sequence[str] = (),
    closing_lines: Sequence[str] = (),
    style: str | None = None,
) -> None:
    output_path.write_text(
        render_latex_table(
            column_spec=column_spec,
            header_rows=header_rows,
            body_rows=body_rows,
            footer_rows=footer_rows,
            opening_lines=opening_lines,
            closing_lines=closing_lines,
            style=style,
        ),
        encoding="utf-8",
    )


def export_matrix_spectrum_csv(output_path: Path, spectrum: Mapping[str, object]) -> None:
    singular_values = spectrum.get("singular_values", ())
    indices = spectrum.get("indices", ())
    rows = [
        {
            "index": int(index),
            "singular_value": float(value),
        }
        for index, value in zip(indices, singular_values, strict=True)
    ]
    write_csv_artifact(output_path, ("index", "singular_value"), rows)


def export_transport_covariance_diagnostics(output_path: Path, covariance_audit) -> None:
    payload = {
        "covariance_mode": covariance_audit.covariance_mode,
        "attempted_samples": covariance_audit.attempted_samples,
        "accepted_samples": covariance_audit.accepted_samples,
        "failure_count": covariance_audit.failure_count,
        "failure_fraction": covariance_audit.failure_fraction,
        "stability_yield": covariance_audit.stability_yield,
        "singularity_chi2_penalty": covariance_audit.singularity_chi2_penalty,
        "observable_names": list(covariance_audit.observable_names),
        "input_names": list(covariance_audit.input_names),
        "max_abs_skewness": covariance_audit.max_abs_skewness,
    }
    write_json_artifact(output_path, payload)


def export_dm_fingerprint_figure(
    *,
    output_path: Path,
    dm_mass_gev: float,
    rhn_scale_gev: float,
    gauge_sigma_cm2: float,
    beta_squared: float,
) -> None:
    with managed_figure(figsize=(6.4, 4.2)) as (fig, ax):
        neutrino_floor_cm2 = 1.0e-49
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvspan(8.0e-1, 1.0e6, color="#fee2e2", alpha=0.28, label=r"framing-anomalous light-WIMP regime")
        ax.axvline(rhn_scale_gev, color="#64748b", linestyle=":", linewidth=1.2, label=r"structural RHN scale $M_N$")
        ax.axhline(neutrino_floor_cm2, color="#0f766e", linestyle="--", linewidth=1.2, label=r"illustrative neutrino floor")
        ax.scatter(
            [dm_mass_gev],
            [gauge_sigma_cm2],
            marker="*",
            s=160,
            color="#7c3aed",
            edgecolors="black",
            linewidths=0.5,
            label=r"Parity-Bit relic $\sigma_{\chi N}$",
            zorder=5,
        )
        ax.annotate(
            rf"$m_{{\rm DM}}=M_N e^{{-\beta^2}}\approx {dm_mass_gev:.2e}\,\mathrm{{GeV}}$",
            xy=(dm_mass_gev, gauge_sigma_cm2),
            xytext=(-120, 12),
            textcoords="offset points",
            fontsize=8,
            color="#4c1d95",
        )
        ax.annotate(
            rf"$\beta^2\approx {beta_squared:.3f}$",
            xy=(dm_mass_gev, gauge_sigma_cm2),
            xytext=(-16, -28),
            textcoords="offset points",
            fontsize=8,
            color="#4c1d95",
        )
        ax.annotate(
            r"LZ / XENONnT: null signal expected above floor",
            xy=(2.5e1, neutrino_floor_cm2),
            xytext=(0.06, 0.16),
            textcoords="axes fraction",
            fontsize=8,
            color="#0f766e",
            bbox={"facecolor": "white", "edgecolor": "#0f766e", "alpha": 0.85, "boxstyle": "round,pad=0.25"},
        )
        ax.set_xlim(8.0e-1, 3.0e14)
        ax.set_ylim(1.0e-60, 3.0e-42)
        ax.set_xlabel(r"$M_\chi$ [GeV]")
        ax.set_ylabel(r"spin-independent proxy $\sigma_{\chi N}$ [cm$^2$]")
        ax.set_title(r"Parity-Bit dark-matter fingerprint")
        ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
        ax.legend(loc="lower left", fontsize=8, frameon=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)


__all__ = [
    "derive_ih_singular_value_spectrum",
    "export_dm_fingerprint_figure",
    "export_ih_singular_value_spectrum_figure",
    "export_matrix_spectrum_csv",
    "export_support_overlap_table",
    "export_supplementary_tolerance_table",
    "export_transport_covariance_diagnostics",
    "write_csv_artifact",
    "write_json_artifact",
    "write_skeletal_latex_table",
]
