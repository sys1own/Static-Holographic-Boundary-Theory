from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _normalize_json_value(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _normalize_csv_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), sort_keys=True)
    if is_dataclass(value):
        return json.dumps(asdict(value), default=_normalize_json_value, sort_keys=True)
    if isinstance(value, Mapping):
        return json.dumps(dict(value), default=_normalize_json_value, sort_keys=True)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return json.dumps(list(value), default=_normalize_json_value)
    return str(value)


def write_json_artifact(output_path: Path, payload: Any) -> Path:
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_normalize_json_value) + "\n",
        encoding="utf-8",
    )
    return resolved_output_path


def write_csv_artifact(output_path: Path, fieldnames: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_fieldnames = tuple(str(fieldname) for fieldname in fieldnames)
    with resolved_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: _normalize_csv_value(row.get(fieldname)) for fieldname in resolved_fieldnames})
    return resolved_output_path


def export_matrix_spectrum_csv(output_path: Path, spectrum_audit: Mapping[str, Any]) -> Path:
    indices = np.asarray(spectrum_audit.get("indices", ()), dtype=int)
    singular_values = np.asarray(spectrum_audit.get("singular_values", ()), dtype=float)
    rank = int(spectrum_audit.get("rank", 0))
    sigma_min = float(spectrum_audit.get("sigma_min", float("nan")))
    rows = (
        {
            "index": int(index),
            "singular_value": float(singular_value),
            "rank": rank,
            "sigma_min": sigma_min,
        }
        for index, singular_value in zip(indices, singular_values, strict=True)
    )
    return write_csv_artifact(Path(output_path), ("index", "singular_value", "rank", "sigma_min"), rows)


def export_dm_fingerprint_figure(
    *,
    output_path: Path,
    dm_mass_gev: float,
    rhn_scale_gev: float,
    gauge_sigma_cm2: float,
    beta_squared: float,
) -> Path:
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = (r"$m_{\rm DM}$ [GeV]", r"$M_N$ [GeV]", r"$\sigma_g$ [cm$^2$]", r"$\beta^2$")
    values = np.asarray((dm_mass_gev, rhn_scale_gev, gauge_sigma_cm2, beta_squared), dtype=float)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    try:
        ax.bar(labels, values, color=("#1d4ed8", "#0f766e", "#7c3aed", "#b45309"), width=0.62)
        ax.set_yscale("log")
        ax.set_ylabel("value")
        ax.set_title("Dark-sector fingerprint summary")
        ax.grid(True, axis="y", which="both", alpha=0.25, linewidth=0.6)
        fig.tight_layout()
        fig.savefig(resolved_output_path, dpi=200)
    finally:
        plt.close(fig)

    return resolved_output_path


def export_transport_covariance_diagnostics(output_path: Path, transport_covariance: Any) -> Path:
    return write_json_artifact(Path(output_path), transport_covariance)


__all__ = [
    "export_dm_fingerprint_figure",
    "export_matrix_spectrum_csv",
    "export_transport_covariance_diagnostics",
    "write_csv_artifact",
    "write_json_artifact",
]
