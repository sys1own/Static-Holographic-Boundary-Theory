from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
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

from shbt.core.observer_horizon import (
    ObserverHorizonLimit,
    ObserverMoatAudit,
    audit_observer_holographic_moat,
    calculate_observer_horizon_limit,
    global_coordinate_horizon_radius,
)
from shbt.core.topology import calculate_dark_debt
from shbt.main import DarkSectorGWBAudit, TopologicalVacuum
from shbt.paths import ProjectPaths, resolve_resource_path


DEFAULT_PRECISION = 50
DEFAULT_CMB_BENCHMARK_PATH = resolve_resource_path("data", "cmb_power_spectrum_benchmarks.json")
DEFAULT_MULTIMESSENGER_AUDIT_PATH = ProjectPaths.RESULTS / "multimessenger_audit.json"
DEFAULT_GWOSC_API_BASE_URL = "https://gwosc.org/api/v2"
DEFAULT_GWOSC_TIMEOUT_SECONDS = 2.5
DEFAULT_OBSERVER_RADIUS_FRACTION = Decimal("0")
_GUARD_DIGITS = 12
_FLOAT_MATCH_TOLERANCE = Decimal("1e-12")
_DEFAULT_BAO_ACOUSTIC_SCALE_MULTIPOLE = Decimal("301")
_DEFAULT_BAO_PEAK_POSITION_TOLERANCE = Decimal("6")
_DEFAULT_BAO_LOADING_MODEL = "ell_m = ell_A * (m - alpha_m * f_b)"
_CANONICAL_AXION_MASS_WINDOW_EV = (Decimal("1e-6"), Decimal("1e-2"))
_CANONICAL_WIMP_MASS_WINDOW_GEV = (Decimal("10"), Decimal("1e4"))
_SUPERHEAVY_WIMPZILLA_FLOOR_GEV = Decimal("1e9")
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
    ligo_virgo: "LigoVirgoBenchmark"


@dataclass(frozen=True)
class LigoVirgoBenchmark:
    dataset_label: str
    event_name: str
    event_version: str
    api_base_url: str
    sample_rate_khz: int
    detectors: tuple[str, ...]
    minimum_detector_count: int
    minimum_peak_absolute_strain: Decimal
    mean_peak_over_rms_floor: Decimal
    minimum_duty_cycle_percent: Decimal


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
class DetectorStrainAudit:
    detector: str
    gps_start: int
    sample_rate_khz: int
    duration_seconds: int
    min_strain: Decimal
    max_strain: Decimal
    strain_standard_deviation: Decimal
    duty_cycle_percent: Decimal
    nan_fraction: Decimal
    peak_absolute_strain: Decimal
    peak_over_rms: Decimal
    detail_url: str


@dataclass(frozen=True)
class LigoVirgoStrainAudit:
    dataset_label: str
    event_name: str
    event_version: str
    requested_detectors: tuple[str, ...]
    reported_detectors: tuple[str, ...]
    data_source: str
    available: bool
    fetch_error: str | None
    detector_audits: tuple["DetectorStrainAudit", ...]
    minimum_detector_count: int
    minimum_peak_absolute_strain: Decimal
    mean_peak_over_rms_floor: Decimal
    minimum_duty_cycle_percent: Decimal
    network_peak_absolute_strain: Decimal
    network_rss_peak_strain: Decimal
    mean_peak_over_rms: Decimal
    benchmark_consistent: bool


@dataclass(frozen=True)
class DarkSectorCandidateAudit:
    label: str
    predicted_mass_gev: Decimal
    predicted_mass_ev: Decimal
    canonical_mass_lower_ev: Decimal
    canonical_mass_upper_ev: Decimal
    within_canonical_window: bool
    superheavy_extension_supported: bool
    observer_projected_loading: Decimal
    candidate_supported: bool
    note: str


@dataclass(frozen=True)
class MultimessengerParityAudit:
    benchmark_branch: tuple[int, int, int]
    nufit_anchor: DatasetAnchor
    cmb_anchor: DatasetAnchor
    dark_debt: DarkDebtAudit
    bao_mapping: BAOMappingAudit
    chi_squared_fit: ChiSquaredFitAudit
    scalar_tilt: ScalarTiltAudit
    observer_horizon: ObserverHorizonLimit
    observer_moat: ObserverMoatAudit
    dark_sector_candidates: tuple["DarkSectorCandidateAudit", ...]
    gravitational_wave: GravitationalWaveAudit
    ligo_virgo_strain: LigoVirgoStrainAudit
    executable_proof_pass: bool

    @property
    def automated_physical_audit_passed(self) -> bool:
        return self.executable_proof_pass


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _lookup_first(payload: dict[str, object], *keys: str, default: object | None = None) -> object | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return default


def _decimal_from_payload(payload: dict[str, object], *keys: str, default: Decimal | str = Decimal("0")) -> Decimal:
    value = _lookup_first(payload, *keys, default=default)
    return _decimal(value if value is not None else default)


def _int_from_payload(payload: dict[str, object], *keys: str, default: int = 0) -> int:
    value = _lookup_first(payload, *keys, default=default)
    return int(default if value is None else value)


def _fetch_json(url: str, *, timeout_seconds: float = DEFAULT_GWOSC_TIMEOUT_SECONDS) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": "shbt-multimessenger-verifier/1.0"})
    with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
        return json.loads(response.read().decode("utf-8"))


def _gwosc_event_version_url(benchmark: LigoVirgoBenchmark) -> str:
    encoded_version = urllib.parse.quote(benchmark.event_version, safe="")
    return f"{benchmark.api_base_url}/event-versions/{encoded_version}?format=api"


def _gwosc_event_strain_files_url(benchmark: LigoVirgoBenchmark, *, event_name: str) -> str:
    encoded_event_name = urllib.parse.quote(event_name, safe="")
    query = urllib.parse.urlencode({"format": "api", "sample-rate": benchmark.sample_rate_khz})
    return f"{benchmark.api_base_url}/events/{encoded_event_name}/strain-files?{query}"


def _select_strain_file_entry(
    results: Sequence[dict[str, object]],
    *,
    benchmark: LigoVirgoBenchmark,
    detector: str,
) -> dict[str, object] | None:
    normalized_detector = detector.upper()
    for entry in results:
        entry_detector = str(_lookup_first(entry, "detector", default="")).upper()
        entry_sample_rate = _int_from_payload(entry, "sample_rate_kHz", "sample_rate_khz", default=benchmark.sample_rate_khz)
        if entry_detector == normalized_detector and entry_sample_rate == benchmark.sample_rate_khz:
            return entry
    return None


def _build_empty_ligo_virgo_strain_audit(
    benchmark: LigoVirgoBenchmark,
    *,
    data_source: str,
    fetch_error: str | None,
) -> LigoVirgoStrainAudit:
    return LigoVirgoStrainAudit(
        dataset_label=benchmark.dataset_label,
        event_name=benchmark.event_name,
        event_version=benchmark.event_version,
        requested_detectors=benchmark.detectors,
        reported_detectors=(),
        data_source=data_source,
        available=False,
        fetch_error=fetch_error,
        detector_audits=(),
        minimum_detector_count=benchmark.minimum_detector_count,
        minimum_peak_absolute_strain=benchmark.minimum_peak_absolute_strain,
        mean_peak_over_rms_floor=benchmark.mean_peak_over_rms_floor,
        minimum_duty_cycle_percent=benchmark.minimum_duty_cycle_percent,
        network_peak_absolute_strain=Decimal("0"),
        network_rss_peak_strain=Decimal("0"),
        mean_peak_over_rms=Decimal("0"),
        benchmark_consistent=False,
    )


def build_ligo_virgo_strain_audit(
    benchmark: LigoVirgoBenchmark,
    *,
    fetch_live: bool = False,
    timeout_seconds: float = DEFAULT_GWOSC_TIMEOUT_SECONDS,
    precision: int = DEFAULT_PRECISION,
) -> LigoVirgoStrainAudit:
    if not fetch_live:
        return _build_empty_ligo_virgo_strain_audit(
            benchmark,
            data_source="benchmark-config",
            fetch_error="live GWOSC pull disabled",
        )

    try:
        event_detail = _fetch_json(_gwosc_event_version_url(benchmark), timeout_seconds=timeout_seconds)
        event_name = str(
            _lookup_first(event_detail, "commonname", "event_name", "name", default=benchmark.event_name)
        )
        reported_detectors_payload = _lookup_first(event_detail, "detectors", default=benchmark.detectors)
        reported_detectors = tuple(str(detector) for detector in reported_detectors_payload)
        strain_payload = _fetch_json(
            _gwosc_event_strain_files_url(benchmark, event_name=event_name),
            timeout_seconds=timeout_seconds,
        )
        results = tuple(
            entry for entry in strain_payload.get("results", ()) if isinstance(entry, dict)
        )

        detector_audits: list[DetectorStrainAudit] = []
        for detector in benchmark.detectors:
            selected_entry = _select_strain_file_entry(results, benchmark=benchmark, detector=detector)
            if selected_entry is None:
                continue
            detail_url = str(_lookup_first(selected_entry, "detail_url", "url", default=""))
            if not detail_url:
                continue
            detail_payload = _fetch_json(detail_url, timeout_seconds=timeout_seconds)
            min_strain = _decimal_from_payload(detail_payload, "min_strain", default="0")
            max_strain = _decimal_from_payload(detail_payload, "max_strain", default="0")
            strain_standard_deviation = _decimal_from_payload(
                detail_payload,
                "std_strain",
                "stdev_strain",
                "standard_deviation_strain",
                default="1",
            )
            if strain_standard_deviation <= 0:
                strain_standard_deviation = Decimal("1")
            duty_cycle_percent = _decimal_from_payload(
                detail_payload,
                "duty_cycle",
                "dutycycle_percent",
                default="100",
            )
            if duty_cycle_percent <= 1:
                duty_cycle_percent *= Decimal("100")
            nan_fraction = _decimal_from_payload(detail_payload, "nan_fraction", default="0")
            if nan_fraction <= 0:
                nan_count = _decimal_from_payload(detail_payload, "nans", default="0")
                sample_count = _decimal_from_payload(
                    detail_payload,
                    "num_points",
                    "sample_count",
                    default="0",
                )
                nan_fraction = nan_count / sample_count if sample_count > 0 else Decimal("0")
            peak_absolute_strain = max(abs(min_strain), abs(max_strain))
            peak_over_rms = peak_absolute_strain / strain_standard_deviation
            detector_audits.append(
                DetectorStrainAudit(
                    detector=detector,
                    gps_start=_int_from_payload(detail_payload, "GPSstart", "gps_start", default=0),
                    sample_rate_khz=_int_from_payload(
                        detail_payload,
                        "sample_rate_kHz",
                        "sample_rate_khz",
                        default=benchmark.sample_rate_khz,
                    ),
                    duration_seconds=_int_from_payload(detail_payload, "duration", default=0),
                    min_strain=min_strain,
                    max_strain=max_strain,
                    strain_standard_deviation=strain_standard_deviation,
                    duty_cycle_percent=duty_cycle_percent,
                    nan_fraction=nan_fraction,
                    peak_absolute_strain=peak_absolute_strain,
                    peak_over_rms=peak_over_rms,
                    detail_url=detail_url,
                )
            )

        resolved_detector_audits = tuple(detector_audits)
        available = len(resolved_detector_audits) >= benchmark.minimum_detector_count
        with localcontext() as context:
            context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
            network_peak_absolute_strain = max(
                (detector_audit.peak_absolute_strain for detector_audit in resolved_detector_audits),
                default=Decimal("0"),
            )
            network_rss_peak_strain = _sqrt_decimal(
                sum(
                    (detector_audit.peak_absolute_strain * detector_audit.peak_absolute_strain)
                    for detector_audit in resolved_detector_audits
                ),
                precision=precision,
            )
            mean_peak_over_rms = (
                sum((detector_audit.peak_over_rms for detector_audit in resolved_detector_audits), Decimal("0"))
                / Decimal(len(resolved_detector_audits))
                if resolved_detector_audits
                else Decimal("0")
            )

        benchmark_consistent = bool(
            available
            and network_peak_absolute_strain >= benchmark.minimum_peak_absolute_strain
            and mean_peak_over_rms >= benchmark.mean_peak_over_rms_floor
            and all(
                detector_audit.duty_cycle_percent >= benchmark.minimum_duty_cycle_percent
                for detector_audit in resolved_detector_audits
            )
        )
        return LigoVirgoStrainAudit(
            dataset_label=benchmark.dataset_label,
            event_name=event_name,
            event_version=benchmark.event_version,
            requested_detectors=benchmark.detectors,
            reported_detectors=reported_detectors,
            data_source="gwosc-live",
            available=available,
            fetch_error=None,
            detector_audits=resolved_detector_audits,
            minimum_detector_count=benchmark.minimum_detector_count,
            minimum_peak_absolute_strain=benchmark.minimum_peak_absolute_strain,
            mean_peak_over_rms_floor=benchmark.mean_peak_over_rms_floor,
            minimum_duty_cycle_percent=benchmark.minimum_duty_cycle_percent,
            network_peak_absolute_strain=network_peak_absolute_strain,
            network_rss_peak_strain=network_rss_peak_strain,
            mean_peak_over_rms=mean_peak_over_rms,
            benchmark_consistent=benchmark_consistent,
        )
    except (OSError, urllib.error.URLError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return _build_empty_ligo_virgo_strain_audit(
            benchmark,
            data_source="gwosc-unavailable",
            fetch_error=str(exc),
        )


def build_dark_sector_candidate_audits(
    *,
    predicted_mass_gev: Decimal,
    observer_horizon: ObserverHorizonLimit,
    topological_ghost_debt: Decimal,
) -> tuple[DarkSectorCandidateAudit, ...]:
    predicted_mass_ev = predicted_mass_gev * Decimal("1e9")
    observer_projected_loading = observer_horizon.log_horizon_loading_factor * topological_ghost_debt
    axion_lower_ev, axion_upper_ev = _CANONICAL_AXION_MASS_WINDOW_EV
    wimp_lower_gev, wimp_upper_gev = _CANONICAL_WIMP_MASS_WINDOW_GEV
    wimp_lower_ev = wimp_lower_gev * Decimal("1e9")
    wimp_upper_ev = wimp_upper_gev * Decimal("1e9")
    wimp_superheavy_extension = predicted_mass_gev >= _SUPERHEAVY_WIMPZILLA_FLOOR_GEV
    return (
        DarkSectorCandidateAudit(
            label="Axion",
            predicted_mass_gev=predicted_mass_gev,
            predicted_mass_ev=predicted_mass_ev,
            canonical_mass_lower_ev=axion_lower_ev,
            canonical_mass_upper_ev=axion_upper_ev,
            within_canonical_window=bool(axion_lower_ev <= predicted_mass_ev <= axion_upper_ev),
            superheavy_extension_supported=False,
            observer_projected_loading=observer_projected_loading,
            candidate_supported=False,
            note="benchmark residue is superheavy and therefore not QCD-axion-like",
        ),
        DarkSectorCandidateAudit(
            label="WIMP",
            predicted_mass_gev=predicted_mass_gev,
            predicted_mass_ev=predicted_mass_ev,
            canonical_mass_lower_ev=wimp_lower_ev,
            canonical_mass_upper_ev=wimp_upper_ev,
            within_canonical_window=bool(wimp_lower_ev <= predicted_mass_ev <= wimp_upper_ev),
            superheavy_extension_supported=wimp_superheavy_extension,
            observer_projected_loading=observer_projected_loading,
            candidate_supported=wimp_superheavy_extension,
            note=(
                "benchmark completion lands in a superheavy WIMPzilla-like regime rather than the canonical thermal window"
                if wimp_superheavy_extension
                else "benchmark completion does not enter the canonical thermal WIMP window"
            ),
        ),
    )


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
    ligo_payload = dict(payload.get("ligo_virgo", {}))
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
        ligo_virgo=LigoVirgoBenchmark(
            dataset_label=str(ligo_payload.get("dataset_label", "GWOSC LIGO/Virgo strain benchmark")),
            event_name=str(ligo_payload.get("event_name", "GW170817")),
            event_version=str(ligo_payload.get("event_version", "GW170817-v2")),
            api_base_url=str(ligo_payload.get("gwosc_api_base_url", DEFAULT_GWOSC_API_BASE_URL)),
            sample_rate_khz=int(ligo_payload.get("sample_rate_khz", 4)),
            detectors=tuple(str(detector) for detector in ligo_payload.get("detectors", ("H1", "L1", "V1"))),
            minimum_detector_count=int(ligo_payload.get("minimum_detector_count", 3)),
            minimum_peak_absolute_strain=_decimal(ligo_payload.get("minimum_peak_absolute_strain", "1e-22")),
            mean_peak_over_rms_floor=_decimal(ligo_payload.get("mean_peak_over_rms_floor", "1")),
            minimum_duty_cycle_percent=_decimal(ligo_payload.get("minimum_duty_cycle_percent", "10")),
        ),
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
    observer_radius_fraction: Decimal | float | int | str = DEFAULT_OBSERVER_RADIUS_FRACTION,
    fetch_live_gw_strain: bool = False,
    gwosc_timeout_seconds: float = DEFAULT_GWOSC_TIMEOUT_SECONDS,
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

    resolved_observer_radius_fraction = _decimal(observer_radius_fraction)
    if resolved_observer_radius_fraction < 0 or resolved_observer_radius_fraction >= 1:
        raise ValueError("observer_radius_fraction must lie in the half-open interval [0, 1).")
    global_horizon_radius = global_coordinate_horizon_radius(precision=precision)
    observer_radius_m = global_horizon_radius * resolved_observer_radius_fraction
    observer_horizon = calculate_observer_horizon_limit(
        observer_radius_m=observer_radius_m,
        global_horizon_radius_m=global_horizon_radius,
        precision=precision,
    )
    observer_moat = audit_observer_holographic_moat(
        observer_radius_m=observer_radius_m,
        lepton_level=vacuum.lepton_level,
        quark_level=vacuum.quark_level,
        parent_level=vacuum.parent_level,
        global_horizon_radius_m=global_horizon_radius,
        precision=precision,
    )
    predicted_dark_mass_gev = _decimal(vacuum.derive_planck_scale_audit().derive_gravity_residues()["m_DM_GeV"])
    dark_sector_candidates = build_dark_sector_candidate_audits(
        predicted_mass_gev=predicted_dark_mass_gev,
        observer_horizon=observer_horizon,
        topological_ghost_debt=topological_ghost_debt,
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
    ligo_virgo_strain = build_ligo_virgo_strain_audit(
        cmb_benchmark.ligo_virgo,
        fetch_live=fetch_live_gw_strain,
        timeout_seconds=gwosc_timeout_seconds,
        precision=precision,
    )

    executable_proof_pass = bool(
        dark_debt.benchmark_locked
        and bao_mapping.within_tolerance
        and scalar_tilt.within_reference_band
        and observer_moat.observer_moat_locked
        and gravitational_wave.below_current_ceiling
        and gravitational_wave.above_design_floor
        and (not ligo_virgo_strain.available or ligo_virgo_strain.benchmark_consistent)
    )
    return MultimessengerParityAudit(
        benchmark_branch=vacuum.target_tuple,
        nufit_anchor=nufit_anchor,
        cmb_anchor=cmb_anchor,
        dark_debt=dark_debt,
        bao_mapping=bao_mapping,
        chi_squared_fit=chi_squared_fit,
        scalar_tilt=scalar_tilt,
        observer_horizon=observer_horizon,
        observer_moat=observer_moat,
        dark_sector_candidates=dark_sector_candidates,
        gravitational_wave=gravitational_wave,
        ligo_virgo_strain=ligo_virgo_strain,
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
        "observer_horizon": {
            "global_horizon_radius_m": _decimal_to_json_number(audit.observer_horizon.global_horizon_radius_m),
            "observer_radius_m": _decimal_to_json_number(audit.observer_horizon.observer_radius_m),
            "coordinate_horizon_radius_m": _decimal_to_json_number(
                audit.observer_horizon.coordinate_horizon_radius_m
            ),
            "relative_position": _decimal_to_json_number(audit.observer_horizon.relative_position),
            "remaining_horizon_fraction": _decimal_to_json_number(
                audit.observer_horizon.remaining_horizon_fraction
            ),
            "exposed_area_fraction": _decimal_to_json_number(audit.observer_horizon.exposed_area_fraction),
            "local_horizon_area_m2": _decimal_to_json_number(audit.observer_horizon.local_horizon_area_m2),
            "bekenstein_hawking_entropy_bits": _decimal_to_json_number(
                audit.observer_horizon.bekenstein_hawking_entropy_bits
            ),
            "local_available_bits": _decimal_to_json_number(audit.observer_horizon.local_available_bits),
            "surface_bit_loading_bits_per_m2": _decimal_to_json_number(
                audit.observer_horizon.surface_bit_loading_bits_per_m2
            ),
            "log_horizon_loading_factor": _decimal_to_json_number(
                audit.observer_horizon.log_horizon_loading_factor
            ),
        },
        "observer_moat": {
            "benchmark_branch": list(audit.observer_moat.benchmark_branch),
            "evaluated_branch": list(audit.observer_moat.evaluated_branch),
            "published_visible_moat_radius": audit.observer_moat.published_visible_moat_radius,
            "branch_chebyshev_distance": audit.observer_moat.branch_chebyshev_distance,
            "fixed_parent_locked": audit.observer_moat.fixed_parent_locked,
            "inside_published_visible_moat": audit.observer_moat.inside_published_visible_moat,
            "benchmark_branch_selected": audit.observer_moat.benchmark_branch_selected,
            "framing_defect_fraction": (
                f"{audit.observer_moat.framing_defect_fraction.numerator}/{audit.observer_moat.framing_defect_fraction.denominator}"
            ),
            "observer_relative_position": _decimal_to_json_number(audit.observer_moat.observer_relative_position),
            "remaining_horizon_fraction": _decimal_to_json_number(
                audit.observer_moat.remaining_horizon_fraction
            ),
            "moat_penalty_factor": _decimal_to_json_number(audit.observer_moat.moat_penalty_factor),
            "observer_shifted_defect": _decimal_to_json_number(audit.observer_moat.observer_shifted_defect),
            "observer_moat_locked": audit.observer_moat.observer_moat_locked,
        },
        "dark_sector_candidates": [
            {
                "label": candidate.label,
                "predicted_mass_gev": _decimal_to_json_number(candidate.predicted_mass_gev),
                "predicted_mass_ev": _decimal_to_json_number(candidate.predicted_mass_ev),
                "canonical_mass_lower_ev": _decimal_to_json_number(candidate.canonical_mass_lower_ev),
                "canonical_mass_upper_ev": _decimal_to_json_number(candidate.canonical_mass_upper_ev),
                "within_canonical_window": candidate.within_canonical_window,
                "superheavy_extension_supported": candidate.superheavy_extension_supported,
                "observer_projected_loading": _decimal_to_json_number(candidate.observer_projected_loading),
                "candidate_supported": candidate.candidate_supported,
                "note": candidate.note,
            }
            for candidate in audit.dark_sector_candidates
        ],
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
        "ligo_virgo_strain": {
            "dataset_label": audit.ligo_virgo_strain.dataset_label,
            "event_name": audit.ligo_virgo_strain.event_name,
            "event_version": audit.ligo_virgo_strain.event_version,
            "requested_detectors": list(audit.ligo_virgo_strain.requested_detectors),
            "reported_detectors": list(audit.ligo_virgo_strain.reported_detectors),
            "data_source": audit.ligo_virgo_strain.data_source,
            "available": audit.ligo_virgo_strain.available,
            "fetch_error": audit.ligo_virgo_strain.fetch_error,
            "minimum_detector_count": audit.ligo_virgo_strain.minimum_detector_count,
            "minimum_peak_absolute_strain": _decimal_to_json_number(
                audit.ligo_virgo_strain.minimum_peak_absolute_strain
            ),
            "mean_peak_over_rms_floor": _decimal_to_json_number(
                audit.ligo_virgo_strain.mean_peak_over_rms_floor
            ),
            "minimum_duty_cycle_percent": _decimal_to_json_number(
                audit.ligo_virgo_strain.minimum_duty_cycle_percent
            ),
            "network_peak_absolute_strain": _decimal_to_json_number(
                audit.ligo_virgo_strain.network_peak_absolute_strain
            ),
            "network_rss_peak_strain": _decimal_to_json_number(audit.ligo_virgo_strain.network_rss_peak_strain),
            "mean_peak_over_rms": _decimal_to_json_number(audit.ligo_virgo_strain.mean_peak_over_rms),
            "benchmark_consistent": audit.ligo_virgo_strain.benchmark_consistent,
            "detector_audits": [
                {
                    "detector": detector_audit.detector,
                    "gps_start": detector_audit.gps_start,
                    "sample_rate_khz": detector_audit.sample_rate_khz,
                    "duration_seconds": detector_audit.duration_seconds,
                    "min_strain": _decimal_to_json_number(detector_audit.min_strain),
                    "max_strain": _decimal_to_json_number(detector_audit.max_strain),
                    "strain_standard_deviation": _decimal_to_json_number(
                        detector_audit.strain_standard_deviation
                    ),
                    "duty_cycle_percent": _decimal_to_json_number(detector_audit.duty_cycle_percent),
                    "nan_fraction": _decimal_to_json_number(detector_audit.nan_fraction),
                    "peak_absolute_strain": _decimal_to_json_number(detector_audit.peak_absolute_strain),
                    "peak_over_rms": _decimal_to_json_number(detector_audit.peak_over_rms),
                    "detail_url": detector_audit.detail_url,
                }
                for detector_audit in audit.ligo_virgo_strain.detector_audits
            ],
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
    reported_detectors = ", ".join(audit.ligo_virgo_strain.reported_detectors) or "n/a"
    lines = [
        "Multimessenger Parity Verifier",
        "===============================",
        f"benchmark branch                 : {audit.benchmark_branch}",
        f"NuFIT anchor                     : {_display_path(audit.nufit_anchor.path)}",
        f"NuFIT release                    : {audit.nufit_anchor.release}",
        f"CMB benchmark anchor             : {_display_path(audit.cmb_anchor.path)}",
        f"CMB benchmark release            : {audit.cmb_anchor.release}",
        "",
        "Observer Horizon",
        f"- observer radius fraction = {_format_decimal(audit.observer_horizon.relative_position, places=12)}",
        f"- coordinate horizon radius [m] = {_format_decimal(audit.observer_horizon.coordinate_horizon_radius_m, places=12)}",
        f"- exposed area fraction = {_format_decimal(audit.observer_horizon.exposed_area_fraction, places=12)}",
        f"- local available bits = {_format_decimal(audit.observer_horizon.local_available_bits, places=12)}",
        f"- log-horizon loading factor = {_format_decimal(audit.observer_horizon.log_horizon_loading_factor, places=12)}",
        f"- moat penalty factor = {_format_decimal(audit.observer_moat.moat_penalty_factor, places=12)}",
        f"- observer moat locked = {audit.observer_moat.observer_moat_locked}",
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
            "Dark Sector Candidate Audits",
        ]
    )
    lines.extend(
        (
            f"- {candidate.label} => predicted mass = {_format_decimal(candidate.predicted_mass_gev, places=12)} GeV "
            f"({_format_decimal(candidate.predicted_mass_ev, places=12)} eV), canonical window = "
            f"[{_format_decimal(candidate.canonical_mass_lower_ev, places=12)}, "
            f"{_format_decimal(candidate.canonical_mass_upper_ev, places=12)}] eV, within canonical window = "
            f"{candidate.within_canonical_window}, superheavy extension = {candidate.superheavy_extension_supported}, "
            f"observer-projected loading = {_format_decimal(candidate.observer_projected_loading, places=12)}, "
            f"supported = {candidate.candidate_supported} ({candidate.note})"
        )
        for candidate in audit.dark_sector_candidates
    )
    lines.extend(
        [
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
            "LIGO/Virgo Strain Benchmark",
            f"- dataset label = {audit.ligo_virgo_strain.dataset_label}",
            f"- event version = {audit.ligo_virgo_strain.event_version}",
            f"- data source = {audit.ligo_virgo_strain.data_source}",
            f"- reported detectors = {reported_detectors}",
            f"- live strain available = {audit.ligo_virgo_strain.available}",
            f"- network peak |h| = {_format_decimal(audit.ligo_virgo_strain.network_peak_absolute_strain, places=12)}",
            f"- network RSS peak |h| = {_format_decimal(audit.ligo_virgo_strain.network_rss_peak_strain, places=12)}",
            f"- mean peak/RMS = {_format_decimal(audit.ligo_virgo_strain.mean_peak_over_rms, places=12)}",
            f"- strain benchmark consistent = {audit.ligo_virgo_strain.benchmark_consistent}",
        ]
    )
    if audit.ligo_virgo_strain.fetch_error is not None:
        lines.append(f"- live strain fetch note = {audit.ligo_virgo_strain.fetch_error}")
    lines.extend(
        (
            f"- {detector_audit.detector} => peak |h| = {_format_decimal(detector_audit.peak_absolute_strain, places=12)}, "
            f"std(h) = {_format_decimal(detector_audit.strain_standard_deviation, places=12)}, peak/RMS = "
            f"{_format_decimal(detector_audit.peak_over_rms, places=12)}, duty cycle [%] = "
            f"{_format_decimal(detector_audit.duty_cycle_percent, places=12)}"
        )
        for detector_audit in audit.ligo_virgo_strain.detector_audits
    )
    lines.extend(
        [
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
    observer_radius_fraction: Decimal | float | int | str = DEFAULT_OBSERVER_RADIUS_FRACTION,
    fetch_live_gw_strain: bool = False,
    gwosc_timeout_seconds: float = DEFAULT_GWOSC_TIMEOUT_SECONDS,
    precision: int = DEFAULT_PRECISION,
) -> str:
    audit = build_multimessenger_parity_audit(
        cmb_benchmark_path=cmb_benchmark_path,
        observer_radius_fraction=observer_radius_fraction,
        fetch_live_gw_strain=fetch_live_gw_strain,
        gwosc_timeout_seconds=gwosc_timeout_seconds,
        precision=precision,
    )
    return render_multimessenger_report(audit)


def run_automated_physical_audit(
    *,
    cmb_benchmark_path: Path | None = None,
    output_path: Path | None = None,
    observer_radius_fraction: Decimal | float | int | str = DEFAULT_OBSERVER_RADIUS_FRACTION,
    fetch_live_gw_strain: bool = True,
    gwosc_timeout_seconds: float = DEFAULT_GWOSC_TIMEOUT_SECONDS,
    precision: int = DEFAULT_PRECISION,
) -> int:
    audit = build_multimessenger_parity_audit(
        cmb_benchmark_path=cmb_benchmark_path,
        observer_radius_fraction=observer_radius_fraction,
        fetch_live_gw_strain=fetch_live_gw_strain,
        gwosc_timeout_seconds=gwosc_timeout_seconds,
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
        "--observer-radius-fraction",
        type=float,
        default=float(DEFAULT_OBSERVER_RADIUS_FRACTION),
        help="Observer radius as a fraction of the benchmark coordinate horizon; must satisfy 0 <= f < 1.",
    )
    parser.add_argument(
        "--no-live-gw-strain",
        action="store_true",
        help="Disable the live LIGO/Virgo GW strain pull and keep the audit on the checked-in benchmark bundle only.",
    )
    parser.add_argument(
        "--gwosc-timeout-seconds",
        type=float,
        default=DEFAULT_GWOSC_TIMEOUT_SECONDS,
        help="Timeout applied to the live GWOSC strain pull when it is enabled.",
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
        observer_radius_fraction=args.observer_radius_fraction,
        fetch_live_gw_strain=not bool(args.no_live_gw_strain),
        gwosc_timeout_seconds=args.gwosc_timeout_seconds,
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
    "DarkSectorCandidateAudit",
    "DatasetAnchor",
    "DEFAULT_CMB_BENCHMARK_PATH",
    "DEFAULT_GWOSC_TIMEOUT_SECONDS",
    "DEFAULT_MULTIMESSENGER_AUDIT_PATH",
    "DetectorStrainAudit",
    "GravitationalWaveAudit",
    "LigoVirgoBenchmark",
    "LigoVirgoStrainAudit",
    "MultimessengerParityAudit",
    "ScalarTiltAudit",
    "audit_topological_ghost_bao_peaks",
    "build_dark_sector_candidate_audits",
    "build_ligo_virgo_strain_audit",
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
