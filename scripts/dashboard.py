from __future__ import annotations

"""Streamlit dashboard for exploring the SHBT rigidity landscape and live ledgers."""

from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.anomaly_detector import BENCHMARK_BRANCH, CandidateSpec, build_candidate_audit
from shbt.constants import G_SM, LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.evolutionary_engine import DEFAULT_PRECISION, TopologicalVacuum, UniverseFactory, quarter_power_inverse
from shbt.plotting_runtime import plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from shbt.anomaly_detector import CandidateAudit


MetricKey = Literal["total_residue", "delta_fr", "c_dark_shift", "diophantine_gap"]

DEFAULT_SCAN_HALF_WIDTHS = {
    "lepton": 4,
    "quark": 4,
    "parent": 6,
}
METRIC_LABELS: dict[MetricKey, str] = {
    "total_residue": "Total rigidity residue",
    "delta_fr": "Framing defect Δ_fr",
    "c_dark_shift": "c_dark shift",
    "diophantine_gap": "Diophantine gap",
}
DISPLAY_TO_CONSTANT_LABEL = {
    "k_l": "LEPTON_LEVEL",
    "k_q": "QUARK_LEVEL",
    "K": "PARENT_LEVEL",
}
CONSTANT_TO_DISPLAY_LABEL = {constant_name: display_label for display_label, constant_name in DISPLAY_TO_CONSTANT_LABEL.items()}
BRANCH_LEVEL_DEFAULTS = {
    "LEPTON_LEVEL": int(LEPTON_LEVEL),
    "QUARK_LEVEL": int(QUARK_LEVEL),
    "PARENT_LEVEL": int(PARENT_LEVEL),
}
BENCHMARK_LOCKED_LABEL = "Benchmark Locked"
KERNEL_PANIC_LABEL = "Kernel Panic"


@dataclass(frozen=True)
class DerivationSnapshot:
    precision: int
    ledger_text: str
    alpha_surface_inverse: Decimal
    codata_alpha_inverse: Decimal
    proton_electron_mass_ratio: Decimal
    codata_proton_electron_mass_ratio: Decimal
    kappa_d5: Decimal
    neutrino_floor_mev: Decimal
    epsilon_lambda: Decimal
    decimal_tolerance: Decimal
    register_noise_floor: Decimal
    decimal_passed: bool
    residues: dict[str, int | Decimal | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class DetuningSnapshot:
    benchmark_branch: tuple[int, int, int]
    candidate_branch: tuple[int, int, int]
    shift: tuple[int, int, int]
    rigidity_point: Any
    anomaly_audit: "CandidateAudit"

    @property
    def benchmark_selected(self) -> bool:
        return self.candidate_branch == self.benchmark_branch


@dataclass(frozen=True)
class ResidueComparisonRow:
    observable: str
    shbt_residue: Decimal
    anchor_value: Decimal
    anchor_label: str
    delta_to_anchor: Decimal


def _load_peer_script(module_name: str) -> Any:
    script_path = Path(__file__).resolve().with_name(f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load peer script {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _rigidity_module() -> Any:
    return _load_peer_script("map_rigidity_landscape")


def _format_decimal(value: Decimal | float | int, *, digits: int = 6) -> str:
    return f"{float(value):.{digits}e}"


def _format_residue_value(value: Decimal | float | int, *, digits: int = 6) -> str:
    numeric_value = float(value)
    if numeric_value == 0.0:
        return "0"
    if abs(numeric_value) >= 1.0e4 or abs(numeric_value) < 1.0e-3:
        return f"{numeric_value:.{digits}e}"
    return f"{numeric_value:.{digits}f}".rstrip("0").rstrip(".")


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _fraction_to_float(value: Fraction) -> float:
    return value.numerator / value.denominator


def _metric_value(point: Any, metric_key: MetricKey) -> float:
    return float(getattr(point, metric_key))


def _metric_floor(values: np.ndarray) -> float:
    positive = values[values > 0.0]
    if positive.size == 0:
        return 1.0e-16
    return max(float(np.min(positive)) * 1.0e-3, 1.0e-16)


def _candidate_label(snapshot: DetuningSnapshot) -> str:
    k_l, k_q, parent_level = snapshot.candidate_branch
    return f"({k_l}, {k_q}, {parent_level})"


def _shift_label(snapshot: DetuningSnapshot) -> str:
    delta_k_l, delta_k_q, delta_parent = snapshot.shift
    return f"({delta_k_l:+d}, {delta_k_q:+d}, {delta_parent:+d})"


def _kernel_state_label(snapshot: DetuningSnapshot) -> str:
    return BENCHMARK_LOCKED_LABEL if snapshot.benchmark_selected else KERNEL_PANIC_LABEL


def _branch_aliases(
    *,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    parent_level: int = PARENT_LEVEL,
) -> dict[str, int]:
    resolved_levels = {
        "LEPTON_LEVEL": int(lepton_level),
        "QUARK_LEVEL": int(quark_level),
        "PARENT_LEVEL": int(parent_level),
    }
    return {
        **resolved_levels,
        "k_l": resolved_levels["LEPTON_LEVEL"],
        "k_q": resolved_levels["QUARK_LEVEL"],
        "K": resolved_levels["PARENT_LEVEL"],
    }


def _resolve_selected_branch(
    *,
    absolute_branch: tuple[int, int, int],
    delta_branch: tuple[int, int, int],
) -> tuple[int, int, int]:
    if absolute_branch != BENCHMARK_BRANCH:
        return absolute_branch
    return delta_branch


@lru_cache(maxsize=8)
def build_derivation_snapshot(precision: int = DEFAULT_PRECISION) -> DerivationSnapshot:
    resolved_precision = max(int(precision), DEFAULT_PRECISION)
    physical_ledger = UniverseFactory.calculate_physical_ledger(precision=resolved_precision)
    alpha = physical_ledger.alpha_surface
    proton_ratio = physical_ledger.proton_ratio
    kappa = physical_ledger.kappa
    mass = physical_ledger.mass_bridge
    unity = physical_ledger.unity_of_scale
    ledger_text = UniverseFactory.generate_ledger(kind="derivation", precision=resolved_precision)
    residues = dict(getattr(physical_ledger, "residues", {}))
    residues.update(UniverseFactory.build_residue_dictionary(precision=resolved_precision))
    residues.update(
        _branch_aliases(
            lepton_level=physical_ledger.vacuum.lepton_level,
            quark_level=physical_ledger.vacuum.quark_level,
            parent_level=physical_ledger.vacuum.parent_level,
        )
    )
    return DerivationSnapshot(
        precision=resolved_precision,
        ledger_text=ledger_text,
        alpha_surface_inverse=alpha.alpha_inverse_decimal,
        codata_alpha_inverse=alpha.codata_alpha_inverse,
        proton_electron_mass_ratio=proton_ratio.mu_audit,
        codata_proton_electron_mass_ratio=proton_ratio.codata_audit.mass_ratio,
        kappa_d5=kappa.kappa,
        neutrino_floor_mev=mass.neutrino_floor_mev,
        epsilon_lambda=unity.epsilon_lambda,
        decimal_tolerance=unity.decimal_tolerance,
        register_noise_floor=unity.register_noise_floor,
        decimal_passed=unity.passed,
        residues=residues,
    )


@lru_cache(maxsize=32)
def build_rigidity_scan(
    lepton_half_width: int = DEFAULT_SCAN_HALF_WIDTHS["lepton"],
    quark_half_width: int = DEFAULT_SCAN_HALF_WIDTHS["quark"],
    parent_half_width: int = DEFAULT_SCAN_HALF_WIDTHS["parent"],
) -> Any:
    module = _rigidity_module()
    return module.build_centered_rigidity_landscape_scan(
        lepton_half_width=int(lepton_half_width),
        quark_half_width=int(quark_half_width),
        parent_half_width=int(parent_half_width),
    )


@lru_cache(maxsize=128)
def build_detuning_snapshot_for_branch(
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    parent_level: int = PARENT_LEVEL,
    precision: int = DEFAULT_PRECISION,
) -> DetuningSnapshot:
    candidate_branch = (
        int(lepton_level),
        int(quark_level),
        int(parent_level),
    )
    if min(candidate_branch) <= 0:
        raise ValueError("Detuned coordinates must remain positive integers.")

    module = _rigidity_module()
    rigidity_point = module.build_rigidity_point(*candidate_branch)
    anomaly_audit = build_candidate_audit(CandidateSpec(branch=candidate_branch, label="dashboard", origin="dashboard"), precision=max(int(precision), DEFAULT_PRECISION))
    return DetuningSnapshot(
        benchmark_branch=BENCHMARK_BRANCH,
        candidate_branch=candidate_branch,
        shift=(
            candidate_branch[0] - int(LEPTON_LEVEL),
            candidate_branch[1] - int(QUARK_LEVEL),
            candidate_branch[2] - int(PARENT_LEVEL),
        ),
        rigidity_point=rigidity_point,
        anomaly_audit=anomaly_audit,
    )


@lru_cache(maxsize=128)
def build_detuning_snapshot(
    delta_lepton: int = 0,
    delta_quark: int = 0,
    delta_parent: int = 0,
    precision: int = DEFAULT_PRECISION,
) -> DetuningSnapshot:
    return build_detuning_snapshot_for_branch(
        lepton_level=int(LEPTON_LEVEL) + int(delta_lepton),
        quark_level=int(QUARK_LEVEL) + int(delta_quark),
        parent_level=int(PARENT_LEVEL) + int(delta_parent),
        precision=precision,
    )


def _candidate_vacuum(candidate_branch: tuple[int, int, int]) -> TopologicalVacuum:
    return TopologicalVacuum(
        lepton_level=int(candidate_branch[0]),
        quark_level=int(candidate_branch[1]),
        parent_level=int(candidate_branch[2]),
        generation_count=int(G_SM),
    )


@lru_cache(maxsize=128)
def build_residue_comparison_rows(
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    parent_level: int = PARENT_LEVEL,
    precision: int = DEFAULT_PRECISION,
) -> tuple[ResidueComparisonRow, ...]:
    resolved_precision = max(int(precision), DEFAULT_PRECISION)
    candidate_branch = (int(lepton_level), int(quark_level), int(parent_level))
    vacuum = _candidate_vacuum(candidate_branch)

    alpha = UniverseFactory.derive_alpha_surface(precision=resolved_precision, vacuum=vacuum)
    proton_ratio = UniverseFactory.derive_proton_ratio(precision=resolved_precision, vacuum=vacuum)
    tuned_kappa = UniverseFactory.derive_kappa_d5(precision=resolved_precision, vacuum=vacuum)
    benchmark_mass = UniverseFactory.derive_mass_bridge(precision=resolved_precision)
    tuned_neutrino_floor_mev = (
        tuned_kappa.kappa
        * benchmark_mass.branch_planck_mass_ev
        * quarter_power_inverse(benchmark_mass.holographic_bits, precision=resolved_precision)
        * Decimal(1000)
    )

    return (
        ResidueComparisonRow(
            observable="α_surf^-1",
            shbt_residue=alpha.alpha_inverse_decimal,
            anchor_value=alpha.codata_alpha_inverse,
            anchor_label="CODATA α^-1",
            delta_to_anchor=alpha.alpha_inverse_decimal - alpha.codata_alpha_inverse,
        ),
        ResidueComparisonRow(
            observable="μ = m_p/m_e",
            shbt_residue=proton_ratio.mu_audit,
            anchor_value=proton_ratio.codata_audit.mass_ratio,
            anchor_label="CODATA m_p/m_e",
            delta_to_anchor=proton_ratio.mu_audit - proton_ratio.codata_audit.mass_ratio,
        ),
        ResidueComparisonRow(
            observable="m_ν [meV]",
            shbt_residue=tuned_neutrino_floor_mev,
            anchor_value=benchmark_mass.neutrino_floor_mev,
            anchor_label="Theory-fixed benchmark",
            delta_to_anchor=tuned_neutrino_floor_mev - benchmark_mass.neutrino_floor_mev,
        ),
    )


def build_residue_comparison_table(
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    parent_level: int = PARENT_LEVEL,
    precision: int = DEFAULT_PRECISION,
) -> list[dict[str, str]]:
    rows = build_residue_comparison_rows(
        lepton_level=lepton_level,
        quark_level=quark_level,
        parent_level=parent_level,
        precision=precision,
    )
    return [
        {
            "Observable": row.observable,
            "SHBT residue": _format_residue_value(row.shbt_residue, digits=6),
            "Anchor": f"{_format_residue_value(row.anchor_value, digits=6)} ({row.anchor_label})",
            "Δ(anchor)": _format_residue_value(row.delta_to_anchor, digits=6),
        }
        for row in rows
    ]


def _build_ui_residue_payload(derivation: DerivationSnapshot) -> dict[str, object]:
    residues = dict(getattr(derivation, "residues", {}))
    residues.update(
        {
            "k_l": float(LEPTON_LEVEL),
            "k_q": float(QUARK_LEVEL),
            "K": float(PARENT_LEVEL),
            "LEPTON_LEVEL": float(LEPTON_LEVEL),
            "QUARK_LEVEL": float(QUARK_LEVEL),
            "PARENT_LEVEL": float(PARENT_LEVEL),
        }
    )
    return residues


def render_residue_table(st: Any, residue_table: list[dict[str, str]], derivation: DerivationSnapshot) -> dict[str, object]:
    residues = _build_ui_residue_payload(derivation)
    st.dataframe(pd.DataFrame(residue_table), hide_index=True, use_container_width=True)
    return residues


def build_rigidity_landscape_figure(
    scan: Any,
    detuning: DetuningSnapshot,
    *,
    metric_key: MetricKey = "total_residue",
    log_scale: bool = True,
) -> "Figure":
    values = np.asarray([_metric_value(point, metric_key) for point in scan.points], dtype=float)
    color_values = values
    color_label = METRIC_LABELS[metric_key]
    if log_scale:
        floor = _metric_floor(values)
        color_values = np.log10(np.maximum(values, floor))
        color_label = rf"$\log_{{10}}({color_label})$"

    quark_values = np.asarray([point.coordinates[1] for point in scan.points], dtype=float)
    lepton_values = np.asarray([point.coordinates[0] for point in scan.points], dtype=float)
    parent_values = np.asarray([point.coordinates[2] for point in scan.points], dtype=float)
    candidate_quark = float(detuning.candidate_branch[1])
    candidate_lepton = float(detuning.candidate_branch[0])
    candidate_parent = float(detuning.candidate_branch[2])

    figure = plt.figure(figsize=(9.8, 7.2), constrained_layout=True)
    ax = figure.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        quark_values,
        lepton_values,
        parent_values,
        c=color_values,
        cmap="viridis",
        s=44,
        alpha=0.88,
        linewidths=0.0,
        depthshade=False,
    )
    ax.scatter(
        [QUARK_LEVEL],
        [LEPTON_LEVEL],
        [PARENT_LEVEL],
        marker="*",
        s=260,
        color="#ef4444",
        edgecolors="white",
        linewidths=1.0,
        label="Benchmark",
    )
    if detuning.benchmark_selected:
        ax.text(
            float(QUARK_LEVEL) + 0.2,
            float(LEPTON_LEVEL) + 0.2,
            float(PARENT_LEVEL) + 0.4,
            "benchmark\nlocked",
            fontsize=9,
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "#6b7280", "alpha": 0.9, "boxstyle": "round,pad=0.25"},
        )
    else:
        ax.scatter(
            [candidate_quark],
            [candidate_lepton],
            [candidate_parent],
            marker="o",
            s=120,
            facecolors="none",
            edgecolors="#111827",
            linewidths=1.6,
            label="Detuned branch",
        )
        ax.text(
            candidate_quark + 0.2,
            candidate_lepton + 0.2,
            candidate_parent + 0.4,
            f"{_kernel_state_label(detuning)}\n"
            f"{_candidate_label(detuning)}\n"
            f"Δ={_shift_label(detuning)}\n"
            f"R={detuning.rigidity_point.total_residue:.2e}",
            fontsize=8.5,
            color="#7f1d1d",
            bbox={"facecolor": "#fef2f2", "edgecolor": "#dc2626", "alpha": 0.96, "boxstyle": "round,pad=0.25"},
        )

    ax.set_xlabel(r"$k_q$")
    ax.set_ylabel(r"$k_\ell$")
    ax.set_zlabel(r"$K$")
    ax.set_title("Rigidity Landscape (3D Moat Plot)")
    ax.view_init(elev=24, azim=45)

    quark_min = min(float(scan.quark_levels[0]), candidate_quark) - 0.5
    quark_max = max(float(scan.quark_levels[-1]), candidate_quark) + 0.5
    lepton_min = min(float(scan.lepton_levels[0]), candidate_lepton) - 0.5
    lepton_max = max(float(scan.lepton_levels[-1]), candidate_lepton) + 0.5
    parent_min = min(float(scan.parent_levels[0]), candidate_parent) - 0.5
    parent_max = max(float(scan.parent_levels[-1]), candidate_parent) + 0.5
    ax.set_xlim(quark_min, quark_max)
    ax.set_ylim(lepton_min, lepton_max)
    ax.set_zlim(parent_min, parent_max)

    colorbar = figure.colorbar(scatter, ax=ax, pad=0.08, fraction=0.04, shrink=0.86)
    colorbar.set_label(color_label)
    return figure


def build_detuning_breakdown_figure(detuning: DetuningSnapshot) -> "Figure":
    rigidity = detuning.rigidity_point
    anomaly = detuning.anomaly_audit

    rigidity_labels = ["Δ_fr", "Δc_dark", "ΔK_min", "R_rigid"]
    rigidity_values = [
        float(rigidity.delta_fr),
        float(rigidity.c_dark_shift),
        float(rigidity.diophantine_gap),
        float(rigidity.total_residue),
    ]
    anomaly_labels = ["|E| [eV^4]", "|J| [m^-2]"]
    anomaly_values = [
        float(abs(anomaly.closure_tensor.amplitude)),
        float(abs(anomaly.anomalous_source_si_m2.amplitude)),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.0), constrained_layout=True)
    try:
        axes[0].bar(rigidity_labels, rigidity_values, color=["#2563eb", "#0f766e", "#d97706", "#7c3aed"])
        axes[0].set_title("Rigidity residue components")
        axes[0].set_ylabel("Magnitude")
        axes[0].grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.45)
        for index, value in enumerate(rigidity_values):
            axes[0].text(index, value + max(max(rigidity_values, default=0.0) * 0.03, 1.0e-12), f"{value:.2e}", ha="center", va="bottom", fontsize=8)

        anomaly_floor = _metric_floor(np.asarray(anomaly_values, dtype=float))
        axes[1].bar(
            anomaly_labels,
            [max(value, anomaly_floor) for value in anomaly_values],
            color=["#16a34a", "#065f46"] if detuning.benchmark_selected else ["#dc2626", "#111827"],
        )
        axes[1].set_yscale("log")
        axes[1].set_title("Gravity-side anomaly floor" if detuning.benchmark_selected else "Kernel Panic anomaly spike")
        axes[1].set_ylabel("Magnitude (log scale)")
        axes[1].grid(True, axis="y", which="both", linestyle=":", linewidth=0.7, alpha=0.45)
        for index, value in enumerate(anomaly_values):
            axes[1].text(index, max(value, anomaly_floor) * 1.2, f"{value:.2e}", ha="center", va="bottom", fontsize=8)
        return figure
    except Exception:
        plt.close(figure)
        raise


def _require_streamlit() -> Any:
    try:
        import streamlit as st
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Streamlit is required to run this dashboard. Install it, then run: streamlit run scripts/dashboard.py"
        ) from error
    return st


def _render_detuning_status(st: Any, detuning: DetuningSnapshot) -> None:
    if detuning.benchmark_selected:
        st.success(
            f"{BENCHMARK_LOCKED_LABEL}: benchmark branch {BENCHMARK_BRANCH} keeps Δ_fr closed and both anomaly residues pinned to zero."
        )
        return
    st.error(
        f"{KERNEL_PANIC_LABEL}: any integer nudge away from {BENCHMARK_BRANCH} exits the anomaly-free shell, reopens Δ_fr, and spikes the closure tensor and anomalous source."
    )


def render_dashboard() -> None:
    st = _require_streamlit()
    st.set_page_config(page_title="SHBT Universe Tuner", layout="wide")

    st.title("SHBT Universe Tuner")
    st.caption(
        "Interactive Holographic Moat heatmap from `map_rigidity_landscape.py`: nudge the benchmark integers and watch the anomaly residues jump into Kernel Panic as soon as the branch leaves (26, 8, 312)."
    )

    with st.sidebar:
        st.header("Controls")
        lepton_half_width = st.slider("Lepton scan half-width", min_value=1, max_value=8, value=DEFAULT_SCAN_HALF_WIDTHS["lepton"])
        quark_half_width = st.slider("Quark scan half-width", min_value=1, max_value=8, value=DEFAULT_SCAN_HALF_WIDTHS["quark"])
        parent_half_width = st.slider("Parent scan half-width", min_value=1, max_value=12, value=DEFAULT_SCAN_HALF_WIDTHS["parent"])
        metric_key = st.selectbox(
            "Moat color metric",
            options=list(METRIC_LABELS),
            format_func=lambda key: METRIC_LABELS[key],
            index=0,
        )
        log_scale = st.toggle("Log color scale", value=True)

        st.subheader("Coordinate tuner")
        branch_half_widths = {
            "LEPTON_LEVEL": lepton_half_width,
            "QUARK_LEVEL": quark_half_width,
            "PARENT_LEVEL": parent_half_width,
        }
        selected_levels: dict[str, int] = {}
        for constant_name, default_value in BRANCH_LEVEL_DEFAULTS.items():
            display_label = CONSTANT_TO_DISPLAY_LABEL[constant_name]
            half_width = branch_half_widths[constant_name]
            selected_levels[constant_name] = st.slider(
                display_label,
                min_value=max(1, default_value - half_width),
                max_value=default_value + half_width,
                value=default_value,
            )

        st.subheader("Integer nudge")
        delta_lepton = st.slider("Δk_l", min_value=-lepton_half_width, max_value=lepton_half_width, value=0)
        delta_quark = st.slider("Δk_q", min_value=-quark_half_width, max_value=quark_half_width, value=0)
        delta_parent = st.slider("ΔK", min_value=-parent_half_width, max_value=parent_half_width, value=0)
        precision = st.select_slider("Ledger precision", options=[DEFAULT_PRECISION, 240, 320], value=DEFAULT_PRECISION)

        absolute_branch = (
            int(selected_levels["LEPTON_LEVEL"]),
            int(selected_levels["QUARK_LEVEL"]),
            int(selected_levels["PARENT_LEVEL"]),
        )
        delta_branch = (
            BRANCH_LEVEL_DEFAULTS["LEPTON_LEVEL"] + int(delta_lepton),
            BRANCH_LEVEL_DEFAULTS["QUARK_LEVEL"] + int(delta_quark),
            BRANCH_LEVEL_DEFAULTS["PARENT_LEVEL"] + int(delta_parent),
        )
        selected_branch = _resolve_selected_branch(absolute_branch=absolute_branch, delta_branch=delta_branch)
        st.caption(
            "Candidate branch = "
            f"{selected_branch}; "
            f"benchmark = {BENCHMARK_BRANCH}."
        )

    scan = build_rigidity_scan(lepton_half_width, quark_half_width, parent_half_width)
    if absolute_branch != BENCHMARK_BRANCH:
        detuning = build_detuning_snapshot_for_branch(*absolute_branch, precision=precision)
    else:
        detuning = build_detuning_snapshot(
            delta_lepton=delta_lepton,
            delta_quark=delta_quark,
            delta_parent=delta_parent,
            precision=precision,
        )
    derivation = build_derivation_snapshot(precision=precision)
    residue_table = build_residue_comparison_table(*detuning.candidate_branch, precision=precision)
    ui_residues = _build_ui_residue_payload(derivation)

    _render_detuning_status(st, detuning)

    metric_columns = st.columns(5)
    metric_columns[0].metric("Kernel state", _kernel_state_label(detuning))
    metric_columns[1].metric("Benchmark", f"{BENCHMARK_BRANCH}")
    metric_columns[2].metric("Candidate", _candidate_label(detuning))
    metric_columns[3].metric("Nudge", _shift_label(detuning))
    metric_columns[4].metric("Rigidity residue", _format_decimal(detuning.rigidity_point.total_residue, digits=3))

    anomaly_columns = st.columns(5)
    anomaly_columns[0].metric("Δ_fr", _format_fraction(detuning.anomaly_audit.framing.delta_fr))
    anomaly_columns[1].metric("Closure tensor |E|", _format_decimal(abs(detuning.anomaly_audit.closure_tensor.amplitude), digits=3))
    anomaly_columns[2].metric("Anomalous source |J|", _format_decimal(abs(detuning.anomaly_audit.anomalous_source_si_m2.amplitude), digits=3))
    anomaly_columns[3].metric("WEP status", detuning.anomaly_audit.wep_status)
    anomaly_columns[4].metric("Verdict", detuning.anomaly_audit.verdict)

    plot_column, detail_column = st.columns((1.5, 1.0), gap="large")
    with plot_column:
        figure = build_rigidity_landscape_figure(scan, detuning, metric_key=metric_key, log_scale=log_scale)
        try:
            st.pyplot(figure, clear_figure=True, use_container_width=True)
        finally:
            plt.close(figure)

    with detail_column:
        breakdown = build_detuning_breakdown_figure(detuning)
        try:
            st.pyplot(breakdown, clear_figure=True, use_container_width=True)
        finally:
            plt.close(breakdown)

        st.markdown("**Current detuning audit**")
        st.json(
            {
                "kernel_state": {
                    "label": _kernel_state_label(detuning),
                    "panic": not detuning.benchmark_selected,
                },
                "candidate_branch": list(detuning.candidate_branch),
                "shift": {
                    "delta_k_l": detuning.shift[0],
                    "delta_k_q": detuning.shift[1],
                    "delta_K": detuning.shift[2],
                },
                "residues": ui_residues,
                "rigidity": {
                    "lepton_framing_gap": _fraction_to_float(detuning.anomaly_audit.framing.lepton_gap),
                    "quark_framing_gap": _fraction_to_float(detuning.anomaly_audit.framing.quark_gap),
                    "delta_fr": _fraction_to_float(detuning.anomaly_audit.framing.delta_fr),
                    "c_dark_shift": float(detuning.rigidity_point.c_dark_shift),
                    "diophantine_gap": float(detuning.rigidity_point.diophantine_gap),
                    "total_residue": float(detuning.rigidity_point.total_residue),
                },
                "gravity_side": {
                    "closure_tensor_ev4": float(abs(detuning.anomaly_audit.closure_tensor.amplitude)),
                    "anomalous_source_si_m2": float(abs(detuning.anomaly_audit.anomalous_source_si_m2.amplitude)),
                    "wep_status": detuning.anomaly_audit.wep_status,
                    "verdict": detuning.anomaly_audit.verdict,
                },
            }
        )

    st.divider()
    st.subheader("Live Residue vs Anchor Ledger")
    summary_columns = st.columns(5)
    summary_columns[0].metric("α_surf^-1", f"{float(derivation.alpha_surface_inverse):.9f}")
    summary_columns[1].metric("CODATA α^-1", f"{float(derivation.codata_alpha_inverse):.9f}")
    summary_columns[2].metric("μ audit", f"{float(derivation.proton_electron_mass_ratio):.6f}")
    summary_columns[3].metric("m_ν floor [meV]", f"{float(derivation.neutrino_floor_mev):.9f}")
    summary_columns[4].metric("Decimal closure", "PASS" if derivation.decimal_passed else "FAIL")
    st.caption(
        f"ε_Λ = {_format_decimal(derivation.epsilon_lambda, digits=6)}; Decimal tolerance = {_format_decimal(derivation.decimal_tolerance, digits=6)}"
    )
    render_residue_table(st, residue_table, derivation)
    with st.expander("Raw derivation ledger", expanded=False):
        st.code(derivation.ledger_text, language="text")


def main() -> None:
    render_dashboard()


if __name__ == "__main__":
    main()
