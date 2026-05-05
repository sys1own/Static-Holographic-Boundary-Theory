from __future__ import annotations

from .template_utils import render_template


def render_support_overlap_result(**context: object) -> str:
    """Render the support-overlap audit table from structured context."""

    return render_template("support_overlap_result.tex.j2", **context)


def render_level_stability_scan(**context: object) -> str:
    """Render the fixed-parent level-stability scan table."""

    return render_template("level_stability_scan.tex.j2", **context)


def render_pull_table(**context: object) -> str:
    """Render the publication-facing pull table."""

    return render_template("pull_table.tex.j2", **context)


def render_kappa_sensitivity_audit(**context: object) -> str:
    """Render the κ-sensitivity audit table."""

    return render_template("kappa_sensitivity_audit.tex.j2", **context)


def render_svd_stability_audit(**context: object) -> str:
    """Render the singular-vector stability audit table."""

    return render_template("svd_stability_audit.tex.j2", **context)


def render_modularity_residual_map(**context: object) -> str:
    """Render the local modularity-residual map."""

    return render_template("modularity_residual_map.tex.j2", **context)


def render_landscape_anomaly_map(**context: object) -> str:
    """Render the low-rank landscape anomaly ranking."""

    return render_template("landscape_anomaly_map.tex.j2", **context)


def render_benchmark_stability_table(**context: object) -> str:
    """Render the publication-facing benchmark stability table."""

    return render_template("benchmark_stability_table.tex.j2", **context)


__all__ = [
    "render_benchmark_stability_table",
    "render_kappa_sensitivity_audit",
    "render_landscape_anomaly_map",
    "render_level_stability_scan",
    "render_modularity_residual_map",
    "render_pull_table",
    "render_svd_stability_audit",
    "render_support_overlap_result",
]
