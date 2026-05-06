from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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
    assert "Derivation Ledger" in snapshot.ledger_text
    assert "Alpha Surface Inverse" in snapshot.ledger_text
    assert "Unity of Scale Identity" in snapshot.ledger_text


def test_build_detuning_snapshot_detects_anomaly_spike() -> None:
    module = _load_script_module()

    benchmark = module.build_detuning_snapshot()
    detuned = module.build_detuning_snapshot(delta_lepton=1)

    assert benchmark.benchmark_selected is True
    assert benchmark.candidate_branch == module.BENCHMARK_BRANCH
    assert benchmark.rigidity_point.total_residue == 0.0
    assert benchmark.anomaly_audit.framing.delta_fr == 0
    assert benchmark.anomaly_audit.closure_tensor.amplitude == 0
    assert benchmark.anomaly_audit.anomalous_source_si_m2.amplitude == 0

    assert detuned.benchmark_selected is False
    assert detuned.candidate_branch == (27, 8, 312)
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
            "k_l": 26,
            "k_q": 8,
            "K": 312,
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
        candidate_branch=module.BENCHMARK_BRANCH,
        shift=(0, 0, 0),
        rigidity_point=SimpleNamespace(total_residue=0.0, delta_fr=0.0, c_dark_shift=0.0, diophantine_gap=0.0),
        anomaly_audit=SimpleNamespace(
            framing=SimpleNamespace(delta_fr=Fraction(0, 1), lepton_gap=Fraction(0, 1), quark_gap=Fraction(0, 1)),
            closure_tensor=SimpleNamespace(amplitude=0.0),
            anomalous_source_si_m2=SimpleNamespace(amplitude=0.0),
            wep_status="Protected",
            verdict="Benchmark locked",
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
    monkeypatch.setattr(module, "build_detuning_snapshot_for_branch", lambda *args, **kwargs: detuning)
    monkeypatch.setattr(module, "build_derivation_snapshot", lambda *args, **kwargs: derivation)
    monkeypatch.setattr(module, "build_residue_comparison_table", lambda *args, **kwargs: residue_table)
    monkeypatch.setattr(module, "build_rigidity_landscape_figure", lambda *args, **kwargs: module.plt.figure())
    monkeypatch.setattr(module, "build_detuning_breakdown_figure", lambda *args, **kwargs: module.plt.figure())

    module.render_dashboard()

    slider_calls = {call["label"]: call for call in fake_st.slider_calls}

    assert fake_st.page_config == {"page_title": "SHBT Universe Tuner", "layout": "wide"}
    assert slider_calls["k_l"] == {"label": "k_l", "min_value": 24, "max_value": 28, "value": 26}
    assert slider_calls["k_q"] == {"label": "k_q", "min_value": 5, "max_value": 11, "value": 8}
    assert slider_calls["K"] == {"label": "K", "min_value": 308, "max_value": 316, "value": 312}
    assert "Live Residue vs Anchor Ledger" in fake_st.subheaders
    assert fake_st.dataframes[0]["dataframe"].to_dict("records") == residue_table
    assert fake_st.dataframes[0]["hide_index"] is True
    assert fake_st.dataframes[0]["use_container_width"] is True
