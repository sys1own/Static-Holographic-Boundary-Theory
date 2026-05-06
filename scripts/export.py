from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in (None, ""):
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        shbt_package_dir = candidate / "shbt"
        if shbt_package_dir.is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            if str(shbt_package_dir) not in sys.path:
                sys.path.insert(0, str(shbt_package_dir))
            break

from audit_generator import (
    derive_ih_singular_value_spectrum,
    export_ih_singular_value_spectrum_figure,
    export_support_overlap_table,
    export_supplementary_tolerance_table,
)
from evolutionary_engine import DEFAULT_PRECISION, UniverseFactory
from plotting_runtime import managed_figure
from shbt.main import build_quantified_two_loop_residuals
from shbt.paths import ProjectPaths
from shbt.template_utils import render_latex_table


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


def _default_residuals_path() -> Path:
    ProjectPaths.ensure_dirs()
    return ProjectPaths.RESULTS / "residuals.json"


def build_residual_export_payload(*, precision: int = DEFAULT_PRECISION) -> dict[str, object]:
    resolved_precision = max(int(precision), int(DEFAULT_PRECISION))
    physical_ledger = UniverseFactory.calculate_physical_ledger(precision=resolved_precision)
    lambda_surface = UniverseFactory.derive_lambda_surface(precision=resolved_precision)

    payload = dict(build_quantified_two_loop_residuals())
    payload["derivation_residues"] = {
        "precision": resolved_precision,
        "benchmark_tuple": [
            int(physical_ledger.vacuum.lepton_level),
            int(physical_ledger.vacuum.quark_level),
            int(physical_ledger.vacuum.parent_level),
        ],
        "alpha_inverse_decimal": float(physical_ledger.alpha_surface.alpha_inverse_decimal),
        "codata_alpha_inverse": float(physical_ledger.alpha_surface.codata_alpha_inverse),
        "m_nu_ev": float(physical_ledger.mass_bridge.neutrino_floor_ev),
        "m_nu_mev": float(physical_ledger.mass_bridge.neutrino_floor_mev),
        "lambda_holo_si_m2": float(lambda_surface.lambda_holo_si_m2),
        "lambda_obs_si_m2": float(lambda_surface.anchor_lambda_si_m2),
        "epsilon_lambda": float(physical_ledger.unity_of_scale.epsilon_lambda),
        "decimal_tolerance": float(physical_ledger.unity_of_scale.decimal_tolerance),
        "register_noise_floor": float(physical_ledger.unity_of_scale.register_noise_floor),
    }
    return payload


def export_residual_payload(
    output_path: Path | None = None,
    *,
    precision: int = DEFAULT_PRECISION,
) -> Path:
    resolved_output_path = _default_residuals_path() if output_path is None else Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_residual_export_payload(precision=precision)
    with resolved_output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return resolved_output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the SHBT residual audit payload.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Override the residual audit output path (defaults to results/residuals.json).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=DEFAULT_PRECISION,
        help="Decimal precision passed to UniverseFactory derivations.",
    )
    return parser.parse_args(tuple(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = export_residual_payload(output_path=args.output_path, precision=args.precision)
    print(f"Wrote residual audit payload to {output_path}")
    return 0


__all__ = [
    "build_residual_export_payload",
    "derive_ih_singular_value_spectrum",
    "export_residual_payload",
    "export_dm_fingerprint_figure",
    "export_ih_singular_value_spectrum_figure",
    "export_matrix_spectrum_csv",
    "export_support_overlap_table",
    "export_supplementary_tolerance_table",
    "export_transport_covariance_diagnostics",
    "main",
    "parse_args",
    "write_csv_artifact",
    "write_json_artifact",
    "write_skeletal_latex_table",
]


if __name__ == "__main__":
    raise SystemExit(main())
