from __future__ import annotations

import argparse
import json
import math
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
from shbt.paths import resolve_resource_path


DEFAULT_PRECISION = 50
DEFAULT_CMB_BENCHMARK_FILENAME = "cmb_power_spectrum_benchmarks.json"
DEFAULT_CMB_BENCHMARK_PATH = resolve_resource_path("data", "cmb_power_spectrum_benchmarks.json")
_GUARD_DIGITS = 12
_FLOAT_MATCH_TOLERANCE = Decimal("1e-12")


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
    tensor_ratio_upper_bound_95cl: Decimal
    tensor_ratio_design_floor: Decimal
    gw_dataset_label: str


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
    ratio_residual: Decimal
    baryon_fraction_residual: Decimal
    within_tolerance: bool


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
    benchmark_path = DEFAULT_CMB_BENCHMARK_PATH if path is None else Path(path)
    payload = _load_json(benchmark_path)
    cmb_payload = dict(payload.get("cmb_power_spectrum", {}))
    gw_payload = dict(payload.get("gravitational_wave", {}))
    release = str(payload.get("release", "CMB benchmark bundle"))
    anchor = DatasetAnchor(label="CMB/GW benchmark", path=benchmark_path, release=release)
    benchmark = CMBBenchmark(
        release=release,
        scalar_tilt_central=_decimal(cmb_payload.get("scalar_tilt_central", "0.9649")),
        scalar_tilt_sigma=_decimal(cmb_payload.get("scalar_tilt_sigma", "0.0042")),
        bao_dark_to_baryon_ratio=_decimal(cmb_payload.get("bao_dark_to_baryon_ratio", "5.36")),
        bao_ratio_tolerance=_decimal(cmb_payload.get("bao_ratio_tolerance", "0.5")),
        bao_proxy_label=str(cmb_payload.get("bao_proxy_label", "CMB acoustic peak baryon-loading proxy")),
        tensor_ratio_upper_bound_95cl=_decimal(gw_payload.get("tensor_ratio_upper_bound_95cl", "0.036")),
        tensor_ratio_design_floor=_decimal(gw_payload.get("tensor_ratio_design_floor", "0.001")),
        gw_dataset_label=str(gw_payload.get("dataset_label", "BK18/WMAP/Planck and archived CMB-S4")),
    )
    return anchor, benchmark


def audit_topological_ghost_bao_peaks(
    *,
    topological_ghost_debt: Decimal,
    cmb_benchmark: CMBBenchmark,
) -> BAOMappingAudit:
    predicted_baryon_fraction = Decimal("1") / (Decimal("1") + topological_ghost_debt)
    observed_baryon_fraction = Decimal("1") / (Decimal("1") + cmb_benchmark.bao_dark_to_baryon_ratio)
    ratio_residual = topological_ghost_debt - cmb_benchmark.bao_dark_to_baryon_ratio
    return BAOMappingAudit(
        proxy_label=cmb_benchmark.bao_proxy_label,
        predicted_dark_to_baryon_ratio=topological_ghost_debt,
        observed_dark_to_baryon_ratio=cmb_benchmark.bao_dark_to_baryon_ratio,
        predicted_baryon_fraction=predicted_baryon_fraction,
        observed_baryon_fraction=observed_baryon_fraction,
        ratio_residual=ratio_residual,
        baryon_fraction_residual=predicted_baryon_fraction - observed_baryon_fraction,
        within_tolerance=bool(abs(ratio_residual) <= cmb_benchmark.bao_ratio_tolerance),
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
        scalar_tilt=scalar_tilt,
        gravitational_wave=gravitational_wave,
        executable_proof_pass=executable_proof_pass,
    )


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
        f"- BAO ratio residual = {_format_decimal(audit.bao_mapping.ratio_residual, places=12)}",
        f"- BAO mapping within tolerance = {audit.bao_mapping.within_tolerance}",
        f"- automated BAO audit = {'PASS' if audit.bao_mapping.within_tolerance else 'FAIL'}",
        "",
        "CMB Power Spectrum Lock",
        f"- predicted n_s = {_format_decimal(audit.scalar_tilt.predicted_scalar_tilt, places=12)}",
        f"- observed n_s = {_format_decimal(audit.scalar_tilt.observed_scalar_tilt, places=12)} +/- {_format_decimal(audit.scalar_tilt.observed_scalar_tilt_sigma, places=12)}",
        f"- n_s residual [sigma] = {_format_decimal(audit.scalar_tilt.residual_sigma, places=12)}",
        f"- within Planck band = {audit.scalar_tilt.within_reference_band}",
        "",
        "Gravitational Wave Benchmark",
        f"- dataset bundle = {audit.gravitational_wave.dataset_label}",
        f"- tensor tilt n_t = {_format_decimal(audit.gravitational_wave.tensor_tilt, places=12)}",
        f"- primordial tensor ratio r_prim = {_format_decimal(audit.gravitational_wave.primordial_tensor_ratio, places=12)}",
        f"- observable tensor ratio r_obs = {_format_decimal(audit.gravitational_wave.observable_tensor_ratio, places=12)}",
        f"- normalized strain floor sqrt(r_obs) = {_format_decimal(audit.gravitational_wave.normalized_strain_floor, places=12)}",
        f"- current dataset ceiling sqrt(r_95) = {_format_decimal(audit.gravitational_wave.normalized_current_ceiling, places=12)}",
        f"- archived Stage-4 floor sqrt(r_floor) = {_format_decimal(audit.gravitational_wave.normalized_design_floor, places=12)}",
        f"- below current ceiling = {audit.gravitational_wave.below_current_ceiling}",
        f"- above Stage-4 floor = {audit.gravitational_wave.above_design_floor}",
        "",
        f"Automated Physical Audit        : {'PASS' if audit.automated_physical_audit_passed else 'FAIL'}",
        f"Executable Proof                 : {'PASS' if audit.executable_proof_pass else 'CHECK'}",
        "The Topological Ghost debt is compared directly against the BAO proxy loaded from data/cmb_power_spectrum_benchmarks.json.",
        "Global cosmological patterns are therefore audited as mandatory residues of the anomaly-free (26, 8, 312) branch.",
    ]
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
    precision: int = DEFAULT_PRECISION,
) -> int:
    audit = build_multimessenger_parity_audit(
        cmb_benchmark_path=cmb_benchmark_path,
        precision=precision,
    )
    print(render_multimessenger_report(audit))
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
    return parser.parse_args(tuple(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    resolved_path = None if args.cmb_benchmark_path in (None, "") else Path(str(args.cmb_benchmark_path))
    return run_automated_physical_audit(
        cmb_benchmark_path=resolved_path,
        precision=max(int(args.precision), 32),
    )


__all__ = [
    "BAOMappingAudit",
    "CMBBenchmark",
    "DarkDebtAudit",
    "DatasetAnchor",
    "DEFAULT_CMB_BENCHMARK_PATH",
    "GravitationalWaveAudit",
    "MultimessengerParityAudit",
    "ScalarTiltAudit",
    "audit_topological_ghost_bao_peaks",
    "build_multimessenger_parity_audit",
    "build_multimessenger_report",
    "load_cmb_benchmark",
    "load_nufit_anchor",
    "main",
    "parse_args",
    "render_multimessenger_report",
    "run_automated_physical_audit",
]


if __name__ == "__main__":
    raise SystemExit(main())
