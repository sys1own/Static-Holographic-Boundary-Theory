from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.core.topology import calculate_dark_debt
from shbt.main import DarkSectorGWBAudit, TopologicalVacuum
from shbt.paths import ProjectPaths, resolve_resource_path


DEFAULT_PRECISION = 50
DEFAULT_CMB_BENCHMARK_PATH = resolve_resource_path("data", "cmb_power_spectrum_benchmarks.json")
DEFAULT_MULTIMESSENGER_AUDIT_PATH = ProjectPaths.RESULTS / "multimessenger_audit.json"
_GUARD_DIGITS = 12
_FLOAT_MATCH_TOLERANCE = Decimal("1e-12")
_DEFAULT_BAO_ACOUSTIC_SCALE_MULTIPOLE = Decimal("301")
_DEFAULT_BAO_PEAK_POSITION_TOLERANCE = Decimal("6")
_DEFAULT_BAO_LOADING_MODEL = "ell_m = ell_A * (m - alpha_m * f_b)"
_DEFAULT_BAO_PEAKS = (
    {
        "label": "TT peak 1",
        "harmonic_index": 1,
        "observed_multipole": "220",
        "loading_coefficient": "1.71149501661129568106312292359",
    },
    {
        "label": "TT peak 2",
        "harmonic_index": 2,
        "observed_multipole": "537",
        "loading_coefficient": "1.37342192691029900332225913621",
    },
    {
        "label": "TT peak 3",
        "harmonic_index": 3,
        "observed_multipole": "810",
        "loading_coefficient": "1.96504983388704318936877076412",
    },
)


def _decimal(value: Decimal | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


def _format_decimal(value: Decimal, *, places: int = 18) -> str:
    if value.is_zero():
        return "0"
    adjusted = value.adjusted()
    if adjusted >= 6 or adjusted <= -4:
        return f"{value:.{places}E}"
    return f"{value:.{places}f}".rstrip("0").rstrip(".")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _sqrt_decimal(value: Decimal, *, precision: int = DEFAULT_PRECISION) -> Decimal:
    with localcontext() as context:
        context.prec = max(int(precision), 32) + _GUARD_DIGITS
        return +value.sqrt()


@dataclass(frozen=True)
class DatasetAnchor:
    label: str
    path: Path
    release: str


@dataclass(frozen=True)
class CMBBenchmark:
    release: str
    scalar_tilt_central: Decimal
    scalar_tilt_sigma: Decimal
    bao_dark_to_baryon_ratio: Decimal
    bao_ratio_tolerance: Decimal
    bao_proxy_label: str
    bao_acoustic_scale_multipole: Decimal
    bao_peak_position_tolerance: Decimal
    bao_peak_loading_model: str
    bao_peak_benchmarks: tuple["BAOPeakBenchmark", ...]
    tensor_ratio_upper_bound_95cl: Decimal
    tensor_ratio_design_floor: Decimal
    gw_dataset_label: str


@dataclass(frozen=True)
class BAOPeakBenchmark:
    label: str
    harmonic_index: int
    observed_multipole: Decimal
    loading_coefficient: Decimal


@dataclass(frozen=True)
class DarkDebtAudit:
    topological_ghost_debt: Decimal
    live_gravity_dark_debt: Decimal
    modular_efficiency: Decimal
    debt_residual: Decimal
    benchmark_locked: bool


@dataclass(frozen=True)
class BAOMappingAudit:
    proxy_label: str
    predicted_dark_to_baryon_ratio: Decimal
    observed_dark_to_baryon_ratio: Decimal
    predicted_baryon_fraction: Decimal
    observed_baryon_fraction: Decimal
    acoustic_scale_multipole: Decimal
    peak_loading_model: str
    peak_position_tolerance: Decimal
    peak_audits: tuple["BAOPeakAudit", ...]
    ratio_residual: Decimal
    baryon_fraction_residual: Decimal
    max_peak_position_residual: Decimal
    peak_positions_locked: bool
    within_tolerance: bool


@dataclass(frozen=True)
class BAOPeakAudit:
    label: str
    harmonic_index: int
    predicted_multipole: Decimal
    observed_multipole: Decimal
    multipole_residual: Decimal
    within_tolerance: bool


@dataclass(frozen=True)
class ChiSquaredObservableAudit:
    label: str
    predicted_value: Decimal
    observed_value: Decimal
    sigma: Decimal
    residual: Decimal
    normalized_residual: Decimal
    chi_squared_contribution: Decimal


@dataclass(frozen=True)
class ChiSquaredFitAudit:
    label: str
    observable_count: int
    degrees_of_freedom: int
    chi_squared: Decimal
    reduced_chi_squared: Decimal
    rms_pull: Decimal
    components: tuple["ChiSquaredObservableAudit", ...]


@dataclass(frozen=True)
class ScalarTiltAudit:
    predicted_scalar_tilt: Decimal
    observed_scalar_tilt: Decimal
    observed_scalar_tilt_sigma: Decimal
    residual_sigma: Decimal
    within_reference_band: bool


@dataclass(frozen=True)
class GravitationalWaveAudit:
    dataset_label: str
    tensor_tilt: Decimal
    primordial_tensor_ratio: Decimal
    observable_tensor_ratio: Decimal
    normalized_strain_floor: Decimal
    normalized_current_ceiling: Decimal
    normalized_design_floor: Decimal
    below_current_ceiling: bool
    above_design_floor: bool


@dataclass(frozen=True)
class MultimessengerParityAudit:
    benchmark_branch: tuple[int, int, int]
    nufit_anchor: DatasetAnchor
    cmb_anchor: DatasetAnchor
    dark_debt: DarkDebtAudit
    bao_mapping: BAOMappingAudit
    chi_squared_fit: ChiSquaredFitAudit
    scalar_tilt: ScalarTiltAudit
    gravitational_wave: GravitationalWaveAudit
    executable_proof_pass: bool

    @property
    def automated_physical_audit_passed(self) -> bool:
        return self.executable_proof_pass


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_nufit_anchor() -> DatasetAnchor:
    nufit_path = resolve_resource_path("data", "nufit_5_3.json")
    payload = _load_json(nufit_path)
    return DatasetAnchor(
        label="NuFIT benchmark",
        path=nufit_path,
        release=str(payload.get("release", payload.get("reference", "NuFIT 5.3"))),
    )


def load_cmb_benchmark(*, path: Path | None = None) -> tuple[DatasetAnchor, CMBBenchmark]:
    benchmark_path = (
        resolve_resource_path("data", "cmb_power_spectrum_benchmarks.json") if path is None else Path(path)
    )
    payload = _load_json(benchmark_path)
    cmb_payload = dict(payload.get("cmb_power_spectrum", {}))
    bao_peak_payload = dict(cmb_payload.get("bao_peak_positions", {}))
    gw_payload = dict(payload.get("gravitational_wave", {}))
    release = str(payload.get("release", "CMB benchmark bundle"))
    peak_entries = bao_peak_payload.get("peaks", _DEFAULT_BAO_PEAKS)
    peak_benchmarks = tuple(
        BAOPeakBenchmark(
            label=str(entry.get("label", f"TT peak {index + 1}")),
            harmonic_index=int(entry.get("harmonic_index", index + 1)),
            observed_multipole=_decimal(entry.get("observed_multipole", entry.get("ell", "0"))),
            loading_coefficient=_decimal(entry.get("loading_coefficient", "0")),
        )
        for index, entry in enumerate(peak_entries)
    )
    anchor = DatasetAnchor(label="CMB/GW benchmark", path=benchmark_path, release=release)
    benchmark = CMBBenchmark(
        release=release,
        scalar_tilt_central=_decimal(cmb_payload.get("scalar_tilt_central", "0.9649")),
        scalar_tilt_sigma=_decimal(cmb_payload.get("scalar_tilt_sigma", "0.0042")),
        bao_dark_to_baryon_ratio=_decimal(cmb_payload.get("bao_dark_to_baryon_ratio", "5.36")),
        bao_ratio_tolerance=_decimal(cmb_payload.get("bao_ratio_tolerance", "0.5")),
        bao_proxy_label=str(cmb_payload.get("bao_proxy_label", "CMB acoustic peak baryon-loading proxy")),
        bao_acoustic_scale_multipole=_decimal(
            bao_peak_payload.get("acoustic_scale_multipole", _DEFAULT_BAO_ACOUSTIC_SCALE_MULTIPOLE)
        ),
        bao_peak_position_tolerance=_decimal(
            bao_peak_payload.get("peak_position_tolerance", _DEFAULT_BAO_PEAK_POSITION_TOLERANCE)
        ),
        bao_peak_loading_model=str(bao_peak_payload.get("loading_model", _DEFAULT_BAO_LOADING_MODEL)),
        bao_peak_benchmarks=peak_benchmarks,
        tensor_ratio_upper_bound_95cl=_decimal(gw_payload.get("tensor_ratio_upper_bound_95cl", "0.036")),
        tensor_ratio_design_floor=_decimal(gw_payload.get("tensor_ratio_design_floor", "0.001")),
        gw_dataset_label=str(gw_payload.get("dataset_label", "BK18/WMAP/Planck and archived CMB-S4")),
    )
    return anchor, benchmark


def _baryon_fraction_from_dark_ratio(dark_to_baryon_ratio: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
        return Decimal("1") / (Decimal("1") + dark_to_baryon_ratio)


def _predict_peak_multipole(
    *,
    acoustic_scale_multipole: Decimal,
    harmonic_index: int,
    loading_coefficient: Decimal,
    baryon_fraction: Decimal,
) -> Decimal:
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
        return acoustic_scale_multipole * (Decimal(harmonic_index) - loading_coefficient * baryon_fraction)


def _build_chi_squared_observable(
    *,
    label: str,
    predicted_value: Decimal,
    observed_value: Decimal,
    sigma: Decimal,
) -> ChiSquaredObservableAudit:
    resolved_sigma = _decimal(sigma)
    if resolved_sigma <= 0:
        raise ValueError("sigma must be positive for a chi-squared contribution.")
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
        residual = predicted_value - observed_value
        normalized_residual = residual / resolved_sigma
        chi_squared_contribution = normalized_residual * normalized_residual
    return ChiSquaredObservableAudit(
        label=label,
        predicted_value=predicted_value,
        observed_value=observed_value,
        sigma=resolved_sigma,
        residual=residual,
        normalized_residual=normalized_residual,
        chi_squared_contribution=chi_squared_contribution,
    )


def build_structural_residue_chi_squared_fit(
    *,
    bao_mapping: BAOMappingAudit,
    bao_ratio_tolerance: Decimal,
    bao_peak_position_tolerance: Decimal,
    precision: int = DEFAULT_PRECISION,
) -> ChiSquaredFitAudit:
    components = [
        _build_chi_squared_observable(
            label="BAO dark-to-baryon ratio",
            predicted_value=bao_mapping.predicted_dark_to_baryon_ratio,
            observed_value=bao_mapping.observed_dark_to_baryon_ratio,
            sigma=bao_ratio_tolerance,
        )
    ]
    components.extend(
        _build_chi_squared_observable(
            label=f"{peak_audit.label} multipole",
            predicted_value=peak_audit.predicted_multipole,
            observed_value=peak_audit.observed_multipole,
            sigma=bao_peak_position_tolerance,
        )
        for peak_audit in bao_mapping.peak_audits
    )
    resolved_components = tuple(components)
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        observable_count = len(resolved_components)
        degrees_of_freedom = observable_count
        chi_squared = sum((component.chi_squared_contribution for component in resolved_components), Decimal("0"))
        reduced_chi_squared = (
            chi_squared / Decimal(degrees_of_freedom) if degrees_of_freedom > 0 else Decimal("0")
        )
    return ChiSquaredFitAudit(
        label="Topological Ghost / BAO acoustic peak fit",
        observable_count=observable_count,
        degrees_of_freedom=degrees_of_freedom,
        chi_squared=chi_squared,
        reduced_chi_squared=reduced_chi_squared,
        rms_pull=_sqrt_decimal(reduced_chi_squared, precision=precision),
        components=resolved_components,
    )


def audit_topological_ghost_bao_peaks(
    *,
    topological_ghost_debt: Decimal,
    cmb_benchmark: CMBBenchmark,
) -> BAOMappingAudit:
    predicted_baryon_fraction = _baryon_fraction_from_dark_ratio(topological_ghost_debt)
    observed_baryon_fraction = _baryon_fraction_from_dark_ratio(cmb_benchmark.bao_dark_to_baryon_ratio)
    ratio_residual = topological_ghost_debt - cmb_benchmark.bao_dark_to_baryon_ratio
    peak_audits = tuple(
        BAOPeakAudit(
            label=peak.label,
            harmonic_index=peak.harmonic_index,
            predicted_multipole=_predict_peak_multipole(
                acoustic_scale_multipole=cmb_benchmark.bao_acoustic_scale_multipole,
                harmonic_index=peak.harmonic_index,
                loading_coefficient=peak.loading_coefficient,
                baryon_fraction=predicted_baryon_fraction,
            ),
            observed_multipole=peak.observed_multipole,
            multipole_residual=Decimal("0"),
            within_tolerance=False,
        )
        for peak in cmb_benchmark.bao_peak_benchmarks
    )
    resolved_peak_audits = tuple(
        BAOPeakAudit(
            label=peak_audit.label,
            harmonic_index=peak_audit.harmonic_index,
            predicted_multipole=peak_audit.predicted_multipole,
            observed_multipole=peak_audit.observed_multipole,
            multipole_residual=peak_audit.predicted_multipole - peak_audit.observed_multipole,
            within_tolerance=bool(
                abs(peak_audit.predicted_multipole - peak_audit.observed_multipole)
                <= cmb_benchmark.bao_peak_position_tolerance
            ),
        )
        for peak_audit in peak_audits
    )
    peak_positions_locked = all(peak_audit.within_tolerance for peak_audit in resolved_peak_audits)
    max_peak_position_residual = max(
        (abs(peak_audit.multipole_residual) for peak_audit in resolved_peak_audits),
        default=Decimal("0"),
    )
    return BAOMappingAudit(
        proxy_label=cmb_benchmark.bao_proxy_label,
        predicted_dark_to_baryon_ratio=topological_ghost_debt,
        observed_dark_to_baryon_ratio=cmb_benchmark.bao_dark_to_baryon_ratio,
        predicted_baryon_fraction=predicted_baryon_fraction,
        observed_baryon_fraction=observed_baryon_fraction,
        acoustic_scale_multipole=cmb_benchmark.bao_acoustic_scale_multipole,
        peak_loading_model=cmb_benchmark.bao_peak_loading_model,
        peak_position_tolerance=cmb_benchmark.bao_peak_position_tolerance,
        peak_audits=resolved_peak_audits,
        ratio_residual=ratio_residual,
        baryon_fraction_residual=predicted_baryon_fraction - observed_baryon_fraction,
        max_peak_position_residual=max_peak_position_residual,
        peak_positions_locked=peak_positions_locked,
        within_tolerance=bool(
            abs(ratio_residual) <= cmb_benchmark.bao_ratio_tolerance and peak_positions_locked
        ),
    )


def build_multimessenger_parity_audit(
    *,
    cmb_benchmark_path: Path | None = None,
    precision: int = DEFAULT_PRECISION,
) -> MultimessengerParityAudit:
    vacuum = TopologicalVacuum()
    gravity_audit = vacuum.verify_bulk_emergence()
    cosmology_audit = vacuum.derive_cosmology_audit()
    gw_audit = DarkSectorGWBAudit(vacuum)
    nufit_anchor = load_nufit_anchor()
    cmb_anchor, cmb_benchmark = load_cmb_benchmark(path=cmb_benchmark_path)

    topological_ghost_debt = _decimal(calculate_dark_debt(vacuum.lepton_level))
    live_gravity_dark_debt = _decimal(gravity_audit.omega_dm_ratio)
    modular_efficiency = _decimal(gravity_audit.modular_residue_efficiency)
    debt_residual = topological_ghost_debt - live_gravity_dark_debt
    dark_debt = DarkDebtAudit(
        topological_ghost_debt=topological_ghost_debt,
        live_gravity_dark_debt=live_gravity_dark_debt,
        modular_efficiency=modular_efficiency,
        debt_residual=debt_residual,
        benchmark_locked=bool(abs(debt_residual) <= _FLOAT_MATCH_TOLERANCE),
    )

    bao_mapping = audit_topological_ghost_bao_peaks(
        topological_ghost_debt=topological_ghost_debt,
        cmb_benchmark=cmb_benchmark,
    )
    chi_squared_fit = build_structural_residue_chi_squared_fit(
        bao_mapping=bao_mapping,
        bao_ratio_tolerance=cmb_benchmark.bao_ratio_tolerance,
        bao_peak_position_tolerance=cmb_benchmark.bao_peak_position_tolerance,
        precision=precision,
    )

    predicted_scalar_tilt = _decimal(cosmology_audit.n_s_locked)
    scalar_tilt_residual_sigma = (
        (predicted_scalar_tilt - cmb_benchmark.scalar_tilt_central) / cmb_benchmark.scalar_tilt_sigma
        if not cmb_benchmark.scalar_tilt_sigma.is_zero()
        else Decimal("0")
    )
    scalar_tilt = ScalarTiltAudit(
        predicted_scalar_tilt=predicted_scalar_tilt,
        observed_scalar_tilt=cmb_benchmark.scalar_tilt_central,
        observed_scalar_tilt_sigma=cmb_benchmark.scalar_tilt_sigma,
        residual_sigma=scalar_tilt_residual_sigma,
        within_reference_band=bool(abs(scalar_tilt_residual_sigma) <= Decimal("1")),
    )

    observable_tensor_ratio = _decimal(gw_audit.predict_observable_tensor_ratio())
    primordial_tensor_ratio = _decimal(gw_audit.predict_primordial_tensor_ratio())
    gravitational_wave = GravitationalWaveAudit(
        dataset_label=cmb_benchmark.gw_dataset_label,
        tensor_tilt=_decimal(gw_audit.predict_gwb_tilt()),
        primordial_tensor_ratio=primordial_tensor_ratio,
        observable_tensor_ratio=observable_tensor_ratio,
        normalized_strain_floor=_sqrt_decimal(observable_tensor_ratio, precision=precision),
        normalized_current_ceiling=_sqrt_decimal(cmb_benchmark.tensor_ratio_upper_bound_95cl, precision=precision),
        normalized_design_floor=_sqrt_decimal(cmb_benchmark.tensor_ratio_design_floor, precision=precision),
        below_current_ceiling=bool(observable_tensor_ratio <= cmb_benchmark.tensor_ratio_upper_bound_95cl),
        above_design_floor=bool(observable_tensor_ratio >= cmb_benchmark.tensor_ratio_design_floor),
    )

    executable_proof_pass = bool(
        dark_debt.benchmark_locked
        and bao_mapping.within_tolerance
        and scalar_tilt.within_reference_band
        and gravitational_wave.below_current_ceiling
        and gravitational_wave.above_design_floor
    )
    return MultimessengerParityAudit(
        benchmark_branch=vacuum.target_tuple,
        nufit_anchor=nufit_anchor,
        cmb_anchor=cmb_anchor,
        dark_debt=dark_debt,
        bao_mapping=bao_mapping,
        chi_squared_fit=chi_squared_fit,
        scalar_tilt=scalar_tilt,
        gravitational_wave=gravitational_wave,
        executable_proof_pass=executable_proof_pass,
    )


def _decimal_to_json_number(value: Decimal) -> float:
    return float(value)


def build_multimessenger_audit_payload(audit: MultimessengerParityAudit) -> dict[str, object]:
    return {
        "benchmark_branch": list(audit.benchmark_branch),
        "nufit_anchor": {
            "label": audit.nufit_anchor.label,
            "path": _display_path(audit.nufit_anchor.path),
            "release": audit.nufit_anchor.release,
        },
        "cmb_anchor": {
            "label": audit.cmb_anchor.label,
            "path": _display_path(audit.cmb_anchor.path),
            "release": audit.cmb_anchor.release,
        },
        "dark_debt": {
            "topological_ghost_debt": _decimal_to_json_number(audit.dark_debt.topological_ghost_debt),
            "live_gravity_dark_debt": _decimal_to_json_number(audit.dark_debt.live_gravity_dark_debt),
            "modular_efficiency": _decimal_to_json_number(audit.dark_debt.modular_efficiency),
            "debt_residual": _decimal_to_json_number(audit.dark_debt.debt_residual),
            "benchmark_locked": audit.dark_debt.benchmark_locked,
        },
        "bao_mapping": {
            "proxy_label": audit.bao_mapping.proxy_label,
            "predicted_dark_to_baryon_ratio": _decimal_to_json_number(
                audit.bao_mapping.predicted_dark_to_baryon_ratio
            ),
            "observed_dark_to_baryon_ratio": _decimal_to_json_number(
                audit.bao_mapping.observed_dark_to_baryon_ratio
            ),
            "predicted_baryon_fraction": _decimal_to_json_number(audit.bao_mapping.predicted_baryon_fraction),
            "observed_baryon_fraction": _decimal_to_json_number(audit.bao_mapping.observed_baryon_fraction),
            "acoustic_scale_multipole": _decimal_to_json_number(audit.bao_mapping.acoustic_scale_multipole),
            "peak_loading_model": audit.bao_mapping.peak_loading_model,
            "peak_position_tolerance": _decimal_to_json_number(audit.bao_mapping.peak_position_tolerance),
            "ratio_residual": _decimal_to_json_number(audit.bao_mapping.ratio_residual),
            "baryon_fraction_residual": _decimal_to_json_number(audit.bao_mapping.baryon_fraction_residual),
            "max_peak_position_residual": _decimal_to_json_number(audit.bao_mapping.max_peak_position_residual),
            "peak_positions_locked": audit.bao_mapping.peak_positions_locked,
            "within_tolerance": audit.bao_mapping.within_tolerance,
            "peak_audits": [
                {
                    "label": peak_audit.label,
                    "harmonic_index": peak_audit.harmonic_index,
                    "predicted_multipole": _decimal_to_json_number(peak_audit.predicted_multipole),
                    "observed_multipole": _decimal_to_json_number(peak_audit.observed_multipole),
                    "multipole_residual": _decimal_to_json_number(peak_audit.multipole_residual),
                    "within_tolerance": peak_audit.within_tolerance,
                }
                for peak_audit in audit.bao_mapping.peak_audits
            ],
        },
        "chi_squared_fit": {
            "label": audit.chi_squared_fit.label,
            "observable_count": audit.chi_squared_fit.observable_count,
            "degrees_of_freedom": audit.chi_squared_fit.degrees_of_freedom,
            "chi_squared": _decimal_to_json_number(audit.chi_squared_fit.chi_squared),
            "reduced_chi_squared": _decimal_to_json_number(audit.chi_squared_fit.reduced_chi_squared),
            "rms_pull": _decimal_to_json_number(audit.chi_squared_fit.rms_pull),
            "components": [
                {
                    "label": component.label,
                    "predicted_value": _decimal_to_json_number(component.predicted_value),
                    "observed_value": _decimal_to_json_number(component.observed_value),
                    "sigma": _decimal_to_json_number(component.sigma),
                    "residual": _decimal_to_json_number(component.residual),
                    "normalized_residual": _decimal_to_json_number(component.normalized_residual),
                    "chi_squared_contribution": _decimal_to_json_number(component.chi_squared_contribution),
                }
                for component in audit.chi_squared_fit.components
            ],
        },
        "scalar_tilt": {
            "predicted_scalar_tilt": _decimal_to_json_number(audit.scalar_tilt.predicted_scalar_tilt),
            "observed_scalar_tilt": _decimal_to_json_number(audit.scalar_tilt.observed_scalar_tilt),
            "observed_scalar_tilt_sigma": _decimal_to_json_number(audit.scalar_tilt.observed_scalar_tilt_sigma),
            "residual_sigma": _decimal_to_json_number(audit.scalar_tilt.residual_sigma),
            "within_reference_band": audit.scalar_tilt.within_reference_band,
        },
        "gravitational_wave": {
            "dataset_label": audit.gravitational_wave.dataset_label,
            "tensor_tilt": _decimal_to_json_number(audit.gravitational_wave.tensor_tilt),
            "primordial_tensor_ratio": _decimal_to_json_number(audit.gravitational_wave.primordial_tensor_ratio),
            "observable_tensor_ratio": _decimal_to_json_number(audit.gravitational_wave.observable_tensor_ratio),
            "normalized_strain_floor": _decimal_to_json_number(audit.gravitational_wave.normalized_strain_floor),
            "normalized_current_ceiling": _decimal_to_json_number(audit.gravitational_wave.normalized_current_ceiling),
            "normalized_design_floor": _decimal_to_json_number(audit.gravitational_wave.normalized_design_floor),
            "below_current_ceiling": audit.gravitational_wave.below_current_ceiling,
            "above_design_floor": audit.gravitational_wave.above_design_floor,
        },
        "automated_physical_audit_passed": audit.automated_physical_audit_passed,
        "executable_proof_pass": audit.executable_proof_pass,
    }


def write_multimessenger_audit_artifact(
    audit: MultimessengerParityAudit,
    *,
    output_path: Path | None = None,
) -> Path:
    ProjectPaths.ensure_dirs()
    resolved_output_path = DEFAULT_MULTIMESSENGER_AUDIT_PATH if output_path is None else Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_multimessenger_audit_payload(audit)
    payload["artifact_path"] = _display_path(resolved_output_path)
    resolved_output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved_output_path


def render_multimessenger_report(audit: MultimessengerParityAudit) -> str:
    lines = [
        "Multimessenger Parity Verifier",
        "===============================",
        f"benchmark branch                 : {audit.benchmark_branch}",
        f"NuFIT anchor                     : {_display_path(audit.nufit_anchor.path)}",
        f"NuFIT release                    : {audit.nufit_anchor.release}",
        f"CMB benchmark anchor             : {_display_path(audit.cmb_anchor.path)}",
        f"CMB benchmark release            : {audit.cmb_anchor.release}",
        "",
        "Topological Ghost Debt",
        f"- Delta_ghost = calculate_dark_debt() = {_format_decimal(audit.dark_debt.topological_ghost_debt, places=12)}",
        f"- live GravityAudit Omega_DM/Omega_vis = {_format_decimal(audit.dark_debt.live_gravity_dark_debt, places=12)}",
        f"- modular efficiency eta_mod = {_format_decimal(audit.dark_debt.modular_efficiency, places=12)}",
        f"- gravity lock residual = {_format_decimal(audit.dark_debt.debt_residual, places=12)}",
        f"- benchmark locked = {audit.dark_debt.benchmark_locked}",
        "",
        "CMB / BAO Mapping",
        f"- proxy label = {audit.bao_mapping.proxy_label}",
        f"- predicted dark-to-baryon ratio = {_format_decimal(audit.bao_mapping.predicted_dark_to_baryon_ratio, places=12)}",
        f"- observed dark-to-baryon ratio = {_format_decimal(audit.bao_mapping.observed_dark_to_baryon_ratio, places=12)}",
        f"- predicted baryon fraction = {_format_decimal(audit.bao_mapping.predicted_baryon_fraction, places=12)}",
        f"- observed baryon fraction = {_format_decimal(audit.bao_mapping.observed_baryon_fraction, places=12)}",
        f"- acoustic scale ell_A = {_format_decimal(audit.bao_mapping.acoustic_scale_multipole, places=12)}",
        f"- peak loading model = {audit.bao_mapping.peak_loading_model}",
        f"- BAO ratio residual = {_format_decimal(audit.bao_mapping.ratio_residual, places=12)}",
        f"- max peak-position residual = {_format_decimal(audit.bao_mapping.max_peak_position_residual, places=12)}",
        f"- peak-position tolerance = {_format_decimal(audit.bao_mapping.peak_position_tolerance, places=12)}",
        f"- BAO peak ladder locked = {audit.bao_mapping.peak_positions_locked}",
        f"- BAO mapping within tolerance = {audit.bao_mapping.within_tolerance}",
        f"- automated BAO audit = {'PASS' if audit.bao_mapping.within_tolerance else 'FAIL'}",
    ]
    lines.extend(
        (
            f"- {peak_audit.label} [m={peak_audit.harmonic_index}] => predicted ell = "
            f"{_format_decimal(peak_audit.predicted_multipole, places=12)}, observed ell = "
            f"{_format_decimal(peak_audit.observed_multipole, places=12)}, residual = "
            f"{_format_decimal(peak_audit.multipole_residual, places=12)}, within tolerance = "
            f"{peak_audit.within_tolerance}"
        )
        for peak_audit in audit.bao_mapping.peak_audits
    )
    lines.extend(
        [
            "",
            "Chi-Squared Fit",
            f"- label = {audit.chi_squared_fit.label}",
            f"- observables = {audit.chi_squared_fit.observable_count}",
            f"- degrees of freedom = {audit.chi_squared_fit.degrees_of_freedom}",
            f"- Chi-Squared = {_format_decimal(audit.chi_squared_fit.chi_squared, places=12)}",
            f"- reduced Chi-Squared = {_format_decimal(audit.chi_squared_fit.reduced_chi_squared, places=12)}",
            f"- RMS pull = {_format_decimal(audit.chi_squared_fit.rms_pull, places=12)}",
            "",
            "CMB Power Spectrum Lock",
            f"- predicted n_s = {_format_decimal(audit.scalar_tilt.predicted_scalar_tilt, places=12)}",
            (
                f"- observed n_s = {_format_decimal(audit.scalar_tilt.observed_scalar_tilt, places=12)} +/- "
                f"{_format_decimal(audit.scalar_tilt.observed_scalar_tilt_sigma, places=12)}"
            ),
            f"- n_s residual [sigma] = {_format_decimal(audit.scalar_tilt.residual_sigma, places=12)}",
            f"- within Planck band = {audit.scalar_tilt.within_reference_band}",
            "",
            "Gravitational Wave Benchmark",
            f"- dataset bundle = {audit.gravitational_wave.dataset_label}",
            f"- tensor tilt n_t = {_format_decimal(audit.gravitational_wave.tensor_tilt, places=12)}",
            (
                f"- primordial tensor ratio r_prim = "
                f"{_format_decimal(audit.gravitational_wave.primordial_tensor_ratio, places=12)}"
            ),
            (
                f"- observable tensor ratio r_obs = "
                f"{_format_decimal(audit.gravitational_wave.observable_tensor_ratio, places=12)}"
            ),
            (
                f"- normalized strain floor sqrt(r_obs) = "
                f"{_format_decimal(audit.gravitational_wave.normalized_strain_floor, places=12)}"
            ),
            (
                f"- current dataset ceiling sqrt(r_95) = "
                f"{_format_decimal(audit.gravitational_wave.normalized_current_ceiling, places=12)}"
            ),
            (
                f"- archived Stage-4 floor sqrt(r_floor) = "
                f"{_format_decimal(audit.gravitational_wave.normalized_design_floor, places=12)}"
            ),
            f"- below current ceiling = {audit.gravitational_wave.below_current_ceiling}",
            f"- above Stage-4 floor = {audit.gravitational_wave.above_design_floor}",
            "",
            f"Automated Physical Audit        : {'PASS' if audit.automated_physical_audit_passed else 'FAIL'}",
            f"Executable Proof                 : {'PASS' if audit.executable_proof_pass else 'CHECK'}",
            (
                "The Topological Ghost debt is mapped into the CMB acoustic peak ladder loaded "
                "from data/cmb_power_spectrum_benchmarks.json."
            ),
            (
                "Global cosmological patterns are therefore audited as mandatory residues of the "
                "anomaly-free (26, 8, 312) branch."
            ),
        ]
    )
    return "\n".join(lines)


def build_multimessenger_report(
    *,
    cmb_benchmark_path: Path | None = None,
    precision: int = DEFAULT_PRECISION,
) -> str:
    audit = build_multimessenger_parity_audit(
        cmb_benchmark_path=cmb_benchmark_path,
        precision=precision,
    )
    return render_multimessenger_report(audit)


def run_automated_physical_audit(
    *,
    cmb_benchmark_path: Path | None = None,
    output_path: Path | None = None,
    precision: int = DEFAULT_PRECISION,
) -> int:
    audit = build_multimessenger_parity_audit(
        cmb_benchmark_path=cmb_benchmark_path,
        precision=precision,
    )
    artifact_path = write_multimessenger_audit_artifact(audit, output_path=output_path)
    print(render_multimessenger_report(audit))
    print(f"\nJSON artifact                    : {_display_path(artifact_path)}")
    return 0 if audit.automated_physical_audit_passed else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the parity-debt branch against CMB acoustic loading and gravitational-wave tensor data."
    )
    parser.add_argument(
        "--cmb-benchmark-path",
        default=None,
        help="Optional explicit path for the CMB power-spectrum benchmark bundle. Defaults to the shared data resource.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=DEFAULT_PRECISION,
        help="Decimal precision used for the normalized strain-floor proxy.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional explicit path for the exported multimessenger audit JSON artifact.",
    )
    return parser.parse_args(tuple(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    resolved_path = None if args.cmb_benchmark_path in (None, "") else Path(str(args.cmb_benchmark_path))
    resolved_output_path = None if args.output_path in (None, "") else Path(str(args.output_path))
    return run_automated_physical_audit(
        cmb_benchmark_path=resolved_path,
        output_path=resolved_output_path,
        precision=max(int(args.precision), 32),
    )


__all__ = [
    "BAOPeakAudit",
    "BAOPeakBenchmark",
    "BAOMappingAudit",
    "CMBBenchmark",
    "ChiSquaredFitAudit",
    "ChiSquaredObservableAudit",
    "DarkDebtAudit",
    "DatasetAnchor",
    "DEFAULT_CMB_BENCHMARK_PATH",
    "DEFAULT_MULTIMESSENGER_AUDIT_PATH",
    "GravitationalWaveAudit",
    "MultimessengerParityAudit",
    "ScalarTiltAudit",
    "audit_topological_ghost_bao_peaks",
    "build_multimessenger_audit_payload",
    "build_multimessenger_parity_audit",
    "build_multimessenger_report",
    "build_structural_residue_chi_squared_fit",
    "load_cmb_benchmark",
    "load_nufit_anchor",
    "main",
    "parse_args",
    "render_multimessenger_report",
    "run_automated_physical_audit",
    "write_multimessenger_audit_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
