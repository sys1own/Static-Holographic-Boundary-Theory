from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dashboard.py"


class _ContextBlock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _MetricColumn(_ContextBlock):
    def __init__(self) -> None:
        self.metric_calls: list[tuple[str, str]] = []

    def metric(self, label: str, value: str) -> None:
        self.metric_calls.append((label, value))


class _StreamlitStub:
    def __init__(
        self,
        *,
        slider_values: dict[str, int] | None = None,
        toggle_values: dict[str, bool] | None = None,
        select_slider_values: dict[str, int] | None = None,
    ) -> None:
        self.sidebar = _ContextBlock()
        self.slider_values = slider_values or {}
        self.toggle_values = toggle_values or {}
        self.select_slider_values = select_slider_values or {}
        self.page_config: dict[str, str] | None = None
        self.slider_calls: list[dict[str, int]] = []
        self.toggle_calls: list[dict[str, bool]] = []
        self.selectbox_calls: list[dict[str, object]] = []
        self.select_slider_calls: list[dict[str, object]] = []
        self.subheaders: list[str] = []
        self.captions: list[str] = []
        self.dataframes: list[dict[str, object]] = []
        self.plotly_charts: list[dict[str, object]] = []
        self.pyplots: list[dict[str, object]] = []
        self.json_payloads: list[dict[str, object]] = []
        self.status_messages: list[tuple[str, str]] = []
        self.expanders: list[dict[str, object]] = []
        self.code_blocks: list[dict[str, str | None]] = []
        self.column_sets: list[list[_MetricColumn]] = []

    def set_page_config(self, **kwargs: str) -> None:
        self.page_config = kwargs

    def title(self, _: str) -> None:
        return None

    def caption(self, text: str) -> None:
        self.captions.append(text)

    def header(self, _: str) -> None:
        return None

    def slider(self, label: str, *, min_value: int, max_value: int, value: int) -> int:
        self.slider_calls.append({"label": label, "min_value": min_value, "max_value": max_value, "value": value})
        return self.slider_values.get(label, value)

    def selectbox(self, label: str, *, options: list[str], format_func, index: int) -> str:
        selected = options[index]
        self.selectbox_calls.append({"label": label, "options": options, "selected": selected, "formatted": format_func(selected)})
        return selected

    def toggle(self, label: str, value: bool) -> bool:
        self.toggle_calls.append({"label": label, "value": value})
        return self.toggle_values.get(label, value)

    def select_slider(self, label: str, *, options: list[int], value: int) -> int:
        selected = self.select_slider_values.get(label, value)
        self.select_slider_calls.append({"label": label, "options": options, "value": value, "selected": selected})
        return selected

    def success(self, text: str) -> None:
        self.status_messages.append(("success", text))

    def error(self, text: str) -> None:
        self.status_messages.append(("error", text))

    def warning(self, text: str) -> None:
        self.status_messages.append(("warning", text))

    def columns(self, spec, gap: str | None = None):
        count = spec if isinstance(spec, int) else len(spec)
        columns = [_MetricColumn() for _ in range(count)]
        self.column_sets.append(columns)
        return columns

    def pyplot(self, figure, *, clear_figure: bool, use_container_width: bool) -> None:
        self.pyplots.append(
            {
                "figure": figure,
                "clear_figure": clear_figure,
                "use_container_width": use_container_width,
            }
        )

    def plotly_chart(self, figure, *, use_container_width: bool) -> None:
        self.plotly_charts.append(
            {
                "figure": figure,
                "use_container_width": use_container_width,
            }
        )

    def markdown(self, _: str) -> None:
        return None

    def json(self, payload: dict[str, object]) -> None:
        self.json_payloads.append(payload)

    def divider(self) -> None:
        return None

    def subheader(self, text: str) -> None:
        self.subheaders.append(text)

    def dataframe(self, dataframe: pd.DataFrame, *, hide_index: bool, use_container_width: bool) -> None:
        self.dataframes.append(
            {
                "dataframe": dataframe.copy(),
                "hide_index": hide_index,
                "use_container_width": use_container_width,
            }
        )

    def expander(self, label: str, *, expanded: bool):
        self.expanders.append({"label": label, "expanded": expanded})
        return _ContextBlock()

    def code(self, text: str, *, language: str | None = None) -> None:
        self.code_blocks.append({"text": text, "language": language})


def _load_script_module():
    spec = importlib.util.spec_from_file_location("dashboard", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_derivation_snapshot_exposes_live_ledger() -> None:
    module = _load_script_module()

    snapshot = module.build_derivation_snapshot()

    assert snapshot.precision >= module.DEFAULT_PRECISION
    assert snapshot.alpha_surface_inverse > 0
    assert snapshot.kappa_d5 > 0
    assert snapshot.neutrino_floor_mev > 0
    assert snapshot.epsilon_lambda <= snapshot.decimal_tolerance
    assert snapshot.decimal_passed is True
    assert snapshot.residues["k_l"] == module.LEPTON_LEVEL
    assert snapshot.residues["k_q"] == module.QUARK_LEVEL
    assert snapshot.residues["K"] == module.PARENT_LEVEL
    assert snapshot.residues["LEPTON_LEVEL"] == module.LEPTON_LEVEL
    assert snapshot.residues["QUARK_LEVEL"] == module.QUARK_LEVEL
    assert snapshot.residues["PARENT_LEVEL"] == module.PARENT_LEVEL
    assert isinstance(snapshot.residues["k_l"], float)
    assert isinstance(snapshot.residues["k_q"], float)
    assert isinstance(snapshot.residues["K"], float)
    assert "Derivation Ledger" in snapshot.ledger_text
    assert "Alpha Surface Inverse" in snapshot.ledger_text
    assert "Unity of Scale Identity" in snapshot.ledger_text


def test_build_detuning_snapshot_detects_anomaly_spike() -> None:
    module = _load_script_module()

    benchmark = module.build_detuning_snapshot()
    detuned = module.build_detuning_snapshot(delta_lepton=1)

    assert benchmark.benchmark_selected is True
    assert benchmark.candidate_branch == module.BENCHMARK_BRANCH
    assert module._kernel_state_label(benchmark) == module.BENCHMARK_LOCKED_LABEL
    assert benchmark.rigidity_point.total_residue == 0.0
    assert benchmark.anomaly_audit.framing.delta_fr == 0
    assert benchmark.anomaly_audit.closure_tensor.amplitude == 0
    assert benchmark.anomaly_audit.anomalous_source_si_m2.amplitude == 0

    assert detuned.benchmark_selected is False
    assert detuned.candidate_branch == (27, 8, 312)
    assert module._kernel_state_label(detuned) == module.KERNEL_PANIC_LABEL
    assert detuned.rigidity_point.total_residue > 0.0
    assert detuned.anomaly_audit.framing.delta_fr != 0
    assert detuned.anomaly_audit.closure_tensor.amplitude > 0
    assert detuned.anomaly_audit.anomalous_source_si_m2.amplitude > 0


def test_build_rigidity_scan_contains_benchmark_valley() -> None:
    module = _load_script_module()

    scan = module.build_rigidity_scan(lepton_half_width=1, quark_half_width=1, parent_half_width=1)

    assert scan.benchmark_coordinates == module.BENCHMARK_BRANCH
    assert scan.benchmark_point.total_residue == 0.0
    assert scan.nearest_detuned_point.total_residue > 0.0


def test_build_noether_bridge_symmetry_breaking_path_reaches_target() -> None:
    module = _load_script_module()

    path = module.build_noether_bridge_symmetry_breaking_path(28, 7, 313)

    assert [snapshot.candidate_branch for snapshot in path] == [
        module.BENCHMARK_BRANCH,
        (27, 7, 313),
        (28, 7, 313),
    ]
    assert path[0].benchmark_selected is True
    assert path[-1].candidate_branch == (28, 7, 313)


def test_build_noether_bridge_decoherence_figure_uses_framing_residue_on_z_axis() -> None:
    module = _load_script_module()

    scan = module.build_rigidity_scan(lepton_half_width=1, quark_half_width=1, parent_half_width=1)
    detuning = module.build_detuning_snapshot(delta_lepton=1)
    figure = module.build_noether_bridge_decoherence_figure(scan, detuning)
    surface = np.asarray(figure.data[0].z, dtype=float)

    benchmark_row = scan.lepton_levels.index(module.LEPTON_LEVEL)
    benchmark_column = scan.quark_levels.index(module.QUARK_LEVEL)

    assert figure.data[0].type == "surface"
    assert figure.layout.scene.zaxis.title.text == "Framing anomaly residue 𝓔"
    assert surface[benchmark_row, benchmark_column] == 0.0
    assert float(surface.max()) > 0.0
    assert len(figure.frames) == 2


def test_build_residue_comparison_table_exposes_live_anchor_rows() -> None:
    module = _load_script_module()

    table = module.build_residue_comparison_table()

    assert [row["Observable"] for row in table] == ["α_surf^-1", "μ = m_p/m_e", "m_ν [meV]"]
    assert table[0]["Anchor"].endswith("(CODATA α^-1)")
    assert table[1]["Anchor"].endswith("(CODATA m_p/m_e)")
    assert table[2]["Anchor"].endswith("(Theory-fixed benchmark)")
    assert all(row["SHBT residue"] for row in table)
    assert all("Δ(anchor)" in row for row in table)


def test_render_dashboard_exposes_branch_sliders_and_live_residue_table(monkeypatch) -> None:
    module = _load_script_module()
    fake_st = _StreamlitStub(
        slider_values={
            "Lepton scan half-width": 2,
            "Quark scan half-width": 3,
            "Parent scan half-width": 4,
            "Δk_l": 1,
            "Δk_q": -1,
            "ΔK": -1,
        }
    )
    residue_table = [
        {
            "Observable": "α_surf^-1",
            "SHBT residue": "1.370360e+02",
            "Anchor": "1.370360e+02 (CODATA α^-1)",
            "Δ(anchor)": "0",
        },
        {
            "Observable": "m_ν [meV]",
            "SHBT residue": "8.971000",
            "Anchor": "8.971000 (Theory-fixed benchmark)",
            "Δ(anchor)": "0",
        },
    ]
    detuning = module.DetuningSnapshot(
        benchmark_branch=module.BENCHMARK_BRANCH,
        candidate_branch=(27, 7, 311),
        shift=(1, -1, -1),
        rigidity_point=SimpleNamespace(total_residue=1.25e-2, delta_fr=1.0 / 17.0, c_dark_shift=2.0e-3, diophantine_gap=1.0, coordinates=(27, 7, 311)),
        anomaly_audit=SimpleNamespace(
            framing=SimpleNamespace(delta_fr=Fraction(1, 17), lepton_gap=Fraction(1, 17), quark_gap=Fraction(-1, 17)),
            closure_tensor=SimpleNamespace(amplitude=2.5e-4),
            anomalous_source_si_m2=SimpleNamespace(amplitude=3.5e-7),
            wep_status="Violated",
            verdict="Kernel panic",
        ),
    )
    derivation = module.DerivationSnapshot(
        precision=module.DEFAULT_PRECISION,
        ledger_text="Derivation Ledger\nAlpha Surface Inverse\nUnity of Scale Identity",
        alpha_surface_inverse=Decimal("137.035999084"),
        codata_alpha_inverse=Decimal("137.035999084"),
        proton_electron_mass_ratio=Decimal("1836.152673"),
        codata_proton_electron_mass_ratio=Decimal("1836.152673"),
        kappa_d5=Decimal("1"),
        neutrino_floor_mev=Decimal("8.971"),
        epsilon_lambda=Decimal("1e-199"),
        decimal_tolerance=Decimal("1e-120"),
        register_noise_floor=Decimal("1e-123"),
        decimal_passed=True,
    )

    monkeypatch.setattr(module, "_require_streamlit", lambda: fake_st)
    monkeypatch.setattr(module, "build_rigidity_scan", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(module, "build_detuning_snapshot", lambda *args, **kwargs: detuning)
    monkeypatch.setattr(module, "build_derivation_snapshot", lambda *args, **kwargs: derivation)
    monkeypatch.setattr(module, "build_residue_comparison_table", lambda *args, **kwargs: residue_table)
    monkeypatch.setattr(module, "build_noether_bridge_decoherence_figure", lambda *args, **kwargs: {"renderer": "decoherence"})
    monkeypatch.setattr(module, "build_rigidity_landscape_figure", lambda *args, **kwargs: module.plt.figure())
    monkeypatch.setattr(module, "build_detuning_breakdown_figure", lambda *args, **kwargs: module.plt.figure())

    module.render_dashboard()

    slider_calls = {call["label"]: call for call in fake_st.slider_calls}

    assert fake_st.page_config == {"page_title": "SHBT Universe Tuner", "layout": "wide"}
    assert slider_calls["k_l"] == {"label": "k_l", "min_value": 24, "max_value": 28, "value": 26}
    assert slider_calls["k_q"] == {"label": "k_q", "min_value": 5, "max_value": 11, "value": 8}
    assert slider_calls["K"] == {"label": "K", "min_value": 308, "max_value": 316, "value": 312}
    assert slider_calls["Δk_l"] == {"label": "Δk_l", "min_value": -2, "max_value": 2, "value": 0}
    assert slider_calls["Δk_q"] == {"label": "Δk_q", "min_value": -3, "max_value": 3, "value": 0}
    assert slider_calls["ΔK"] == {"label": "ΔK", "min_value": -4, "max_value": 4, "value": 0}
    assert "Noether Bridge Decoherence Renderer" in fake_st.subheaders
    assert "Live Residue vs Anchor Ledger" in fake_st.subheaders
    assert fake_st.status_messages[0][0] == "error"
    assert "Kernel Panic" in fake_st.status_messages[0][1]
    first_metric_row = [metric for column in fake_st.column_sets[0] for metric in column.metric_calls]
    assert ("Kernel state", "Kernel Panic") in first_metric_row
    assert ("Nudge", "(+1, -1, -1)") in first_metric_row
    assert fake_st.dataframes[0]["dataframe"].to_dict("records") == residue_table
    assert fake_st.dataframes[0]["hide_index"] is True
    assert fake_st.dataframes[0]["use_container_width"] is True
    assert fake_st.plotly_charts[0]["figure"] == {"renderer": "decoherence"}
    assert fake_st.plotly_charts[0]["use_container_width"] is True
    assert fake_st.json_payloads[0]["kernel_state"] == {"label": "Kernel Panic", "panic": True}
    assert fake_st.json_payloads[0]["residues"] == {
        "LEPTON_LEVEL": float(module.LEPTON_LEVEL),
        "QUARK_LEVEL": float(module.QUARK_LEVEL),
        "PARENT_LEVEL": float(module.PARENT_LEVEL),
        "k_l": float(module.LEPTON_LEVEL),
        "k_q": float(module.QUARK_LEVEL),
        "K": float(module.PARENT_LEVEL),
    }


def test_render_dashboard_accepts_absolute_coordinate_aliases(monkeypatch) -> None:
    module = _load_script_module()
    fake_st = _StreamlitStub(
        slider_values={
            "Lepton scan half-width": 2,
            "Quark scan half-width": 3,
            "Parent scan half-width": 4,
            "k_l": 27,
            "k_q": 7,
            "K": 311,
        }
    )
    detuning = module.DetuningSnapshot(
        benchmark_branch=module.BENCHMARK_BRANCH,
        candidate_branch=(27, 7, 311),
        shift=(1, -1, -1),
        rigidity_point=SimpleNamespace(total_residue=1.0, delta_fr=1.0, c_dark_shift=1.0, diophantine_gap=1.0),
        anomaly_audit=SimpleNamespace(
            framing=SimpleNamespace(delta_fr=Fraction(1, 17), lepton_gap=Fraction(1, 17), quark_gap=Fraction(-1, 17)),
            closure_tensor=SimpleNamespace(amplitude=1.0),
            anomalous_source_si_m2=SimpleNamespace(amplitude=1.0),
            wep_status="Violated",
            verdict="Kernel panic",
        ),
    )
    call_log: dict[str, tuple[object, ...]] = {}

    monkeypatch.setattr(module, "_require_streamlit", lambda: fake_st)
    monkeypatch.setattr(module, "build_rigidity_scan", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(module, "build_detuning_snapshot", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("delta path should not be used")))
    monkeypatch.setattr(
        module,
        "build_detuning_snapshot_for_branch",
        lambda *args, **kwargs: (call_log.setdefault("branch", args), detuning)[1],
    )
    monkeypatch.setattr(module, "build_derivation_snapshot", lambda *args, **kwargs: module.DerivationSnapshot(
        precision=module.DEFAULT_PRECISION,
        ledger_text="Derivation Ledger",
        alpha_surface_inverse=Decimal("1"),
        codata_alpha_inverse=Decimal("1"),
        proton_electron_mass_ratio=Decimal("1"),
        codata_proton_electron_mass_ratio=Decimal("1"),
        kappa_d5=Decimal("1"),
        neutrino_floor_mev=Decimal("1"),
        epsilon_lambda=Decimal("1e-199"),
        decimal_tolerance=Decimal("1e-120"),
        register_noise_floor=Decimal("1e-123"),
        decimal_passed=True,
        residues={"k_l": module.LEPTON_LEVEL, "k_q": module.QUARK_LEVEL, "K": module.PARENT_LEVEL},
    ))
    monkeypatch.setattr(module, "build_residue_comparison_table", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "build_noether_bridge_decoherence_figure", lambda *args, **kwargs: {"renderer": "decoherence"})
    monkeypatch.setattr(module, "build_rigidity_landscape_figure", lambda *args, **kwargs: module.plt.figure())
    monkeypatch.setattr(module, "build_detuning_breakdown_figure", lambda *args, **kwargs: module.plt.figure())

    module.render_dashboard()

    assert call_log["branch"] == (27, 7, 311)
