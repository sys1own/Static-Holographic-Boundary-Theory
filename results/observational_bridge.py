from __future__ import annotations

"""Bridge external transient triggers into SHBT holographic-tension audits.

The bridge ingests JSON and CSV trigger drops from ``data/external_triggers/``
and normalizes them into LIGO/JWST observations. High-redshift measurements
(``z > 10`` by default) are compared against the branch-fixed late-time
expansion law anchored by ``EvolutionaryEngine`` and the precision cosmology
loading audit.
"""

import csv
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Literal

if __package__ in (None, ""):
    import sys

    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.constants import LIGHT_SPEED_M_PER_S
from shbt.evolutionary_engine import EvolutionaryEngine
from shbt.paths import resolve_resource_path
from shbt.precision_cosmology_engine import (
    DEFAULT_PRECISION as COSMOLOGY_DEFAULT_PRECISION,
    build_precision_cosmology_audit,
    redshift_dependent_hubble_constant,
)


TriggerKind = Literal["ligo", "jwst", "unknown"]
IngestHook = Callable[[Path], Sequence["ExternalTrigger"]]
ExpansionHook = Callable[["ExternalTrigger", Decimal], Decimal | None]

DEFAULT_PRECISION = max(int(COSMOLOGY_DEFAULT_PRECISION), 50)
DEFAULT_HIGH_REDSHIFT_FLOOR = Decimal("10")
_GUARD_DIGITS = 12
_DEFAULT_VISIBLE_MOAT_RADIUS = Decimal("2")
_DEFAULT_NEAREST_DETUNED_RESIDUE = Decimal("0.04166666666666666666666666667")
_DEFAULT_BENCHMARK_BRANCH = (26, 8, 312)
_LIGHT_SPEED_KM_PER_S = Decimal(str(LIGHT_SPEED_M_PER_S / 1000.0))


@dataclass(frozen=True)
class ExternalTrigger:
    event_id: str
    source: str
    trigger_kind: TriggerKind
    redshift: Decimal
    observed_expansion_rate_km_s_mpc: Decimal | None
    peak_strain: Decimal | None
    galaxy_count: Decimal | None
    luminosity_distance_mpc: Decimal | None
    path: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class HolographicMoatBounds:
    benchmark_branch: tuple[int, int, int]
    published_visible_moat_radius: Decimal
    nearest_detuned_residue: Decimal
    moat_penalty_factor: Decimal
    remaining_horizon_fraction: Decimal
    source_paths: tuple[Path, Path]

    @property
    def maximum_fractional_tension(self) -> Decimal:
        with localcontext() as context:
            context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
            penalty = self.moat_penalty_factor if self.moat_penalty_factor > 0 else Decimal("1")
            return +(
                self.nearest_detuned_residue
                * self.published_visible_moat_radius
                * self.remaining_horizon_fraction
                / penalty
            )

    def equivalent_branch_shift(self, fractional_tension: Decimal) -> Decimal:
        if self.nearest_detuned_residue == 0:
            return Decimal("Infinity")
        with localcontext() as context:
            context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
            return +(fractional_tension / self.nearest_detuned_residue)


@dataclass(frozen=True)
class TransientTensionDiagnostic:
    trigger: ExternalTrigger
    predicted_expansion_rate_km_s_mpc: Decimal
    observed_expansion_rate_km_s_mpc: Decimal | None
    residual_km_s_mpc: Decimal | None
    fractional_tension: Decimal | None
    equivalent_moat_shift: Decimal | None
    included_in_high_redshift_audit: bool
    exceeds_moat_bounds: bool


@dataclass(frozen=True)
class ObservationalTensionAudit:
    trigger_directory: Path
    high_redshift_floor: Decimal
    lambda_holo_si_m2: Decimal
    lambda_anchor_ratio: Decimal
    holographic_bits: Decimal
    predicted_anchor_hubble_km_s_mpc: Decimal
    moat_bounds: HolographicMoatBounds
    triggers: tuple[ExternalTrigger, ...]
    diagnostics: tuple[TransientTensionDiagnostic, ...]
    multi_messenger_sources: tuple[str, ...]

    @property
    def high_redshift_diagnostics(self) -> tuple[TransientTensionDiagnostic, ...]:
        return tuple(diagnostic for diagnostic in self.diagnostics if diagnostic.included_in_high_redshift_audit)

    @property
    def max_fractional_tension(self) -> Decimal:
        shifts = tuple(
            diagnostic.fractional_tension
            for diagnostic in self.high_redshift_diagnostics
            if diagnostic.fractional_tension is not None
        )
        return max(shifts, default=Decimal("0"))

    @property
    def moat_violation_count(self) -> int:
        return sum(int(diagnostic.exceeds_moat_bounds) for diagnostic in self.high_redshift_diagnostics)

    @property
    def multi_messenger(self) -> bool:
        return len(self.multi_messenger_sources) >= 2

    @property
    def anomaly_detected(self) -> bool:
        return self.multi_messenger and self.moat_violation_count > 0


class TensionAnomaly(RuntimeError):
    def __init__(self, audit: ObservationalTensionAudit) -> None:
        self.audit = audit
        super().__init__(
            "Observed multi-messenger data exceed the holographic moat bounds: "
            f"max fractional tension={audit.max_fractional_tension} > "
            f"{audit.moat_bounds.maximum_fractional_tension}."
        )


def _decimal(value: Decimal | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


def _decimal_or_none(value: object | None) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return Decimal(stripped)
    if isinstance(value, bool):
        return None
    return _decimal(value)


def _first_present(record: Mapping[str, Any], *keys: str) -> object | None:
    for key in keys:
        if key not in record:
            continue
        value = record[key]
        if isinstance(value, str):
            value = value.strip()
        if value in (None, ""):
            continue
        return value
    return None


def _normalize_trigger_kind(value: object | None, *, path: Path) -> TriggerKind:
    candidate = str(path.stem if value is None else value).lower()
    if any(token in candidate for token in ("ligo", "virgo", "kagra", "gw", "strain")):
        return "ligo"
    if any(token in candidate for token in ("jwst", "webb", "galaxy", "deep", "nircam", "redshift")):
        return "jwst"
    return "unknown"


def _load_json_records(path: Path) -> tuple[Mapping[str, Any], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        nested = _first_present(payload, "triggers", "events", "observations", "records")
        records = nested if isinstance(nested, list) else [payload]
    else:
        raise ValueError(f"Unsupported trigger payload in {path}: expected object or list.")
    normalized: list[Mapping[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"Unsupported trigger record in {path}: expected mapping entries.")
        normalized.append(record)
    return tuple(normalized)


def _load_csv_records(path: Path) -> tuple[Mapping[str, Any], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if any(value not in (None, "") for value in row.values())]
    return tuple(rows)


def load_holographic_moat_bounds() -> HolographicMoatBounds:
    multimessenger_path = resolve_resource_path("results/multimessenger_audit.json")
    rigidity_path = resolve_resource_path("results/rigidity_moat.json")

    benchmark_branch = _DEFAULT_BENCHMARK_BRANCH
    published_visible_moat_radius = _DEFAULT_VISIBLE_MOAT_RADIUS
    moat_penalty_factor = Decimal("1")
    remaining_horizon_fraction = Decimal("1")
    nearest_detuned_residue = _DEFAULT_NEAREST_DETUNED_RESIDUE

    try:
        multimessenger_payload = json.loads(multimessenger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        multimessenger_payload = {}
    if isinstance(multimessenger_payload, Mapping):
        benchmark_value = multimessenger_payload.get("benchmark_branch")
        if isinstance(benchmark_value, Sequence) and len(benchmark_value) == 3:
            benchmark_branch = tuple(int(entry) for entry in benchmark_value)
        observer_moat = multimessenger_payload.get("observer_moat")
        if isinstance(observer_moat, Mapping):
            published_visible_moat_radius = _decimal_or_none(observer_moat.get("published_visible_moat_radius")) or published_visible_moat_radius
            moat_penalty_factor = _decimal_or_none(observer_moat.get("moat_penalty_factor")) or moat_penalty_factor
            remaining_horizon_fraction = _decimal_or_none(observer_moat.get("remaining_horizon_fraction")) or remaining_horizon_fraction

    try:
        rigidity_payload = json.loads(rigidity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        rigidity_payload = {}
    if isinstance(rigidity_payload, Mapping):
        nearest_detuned_point = rigidity_payload.get("nearest_detuned_point")
        if isinstance(nearest_detuned_point, Mapping):
            nearest_detuned_residue = _decimal_or_none(nearest_detuned_point.get("total_residue")) or nearest_detuned_residue

    return HolographicMoatBounds(
        benchmark_branch=benchmark_branch,
        published_visible_moat_radius=published_visible_moat_radius,
        nearest_detuned_residue=nearest_detuned_residue,
        moat_penalty_factor=moat_penalty_factor,
        remaining_horizon_fraction=remaining_horizon_fraction,
        source_paths=(multimessenger_path, rigidity_path),
    )


class ObservationalBridge:
    def __init__(
        self,
        trigger_directory: Path | str | None = None,
        *,
        precision: int = DEFAULT_PRECISION,
        high_redshift_floor: Decimal | float | int | str = DEFAULT_HIGH_REDSHIFT_FLOOR,
    ) -> None:
        self.trigger_directory = (
            resolve_resource_path("data", "external_triggers") if trigger_directory is None else Path(trigger_directory)
        )
        self.precision = max(int(precision), DEFAULT_PRECISION)
        self.high_redshift_floor = _decimal(high_redshift_floor)
        if self.high_redshift_floor < 0:
            raise ValueError("high_redshift_floor must be non-negative.")
        self._ingest_hooks: dict[str, IngestHook] = {
            ".csv": self._load_csv_trigger_file,
            ".json": self._load_json_trigger_file,
        }
        self._expansion_hooks: dict[str, ExpansionHook] = {
            "jwst": self._jwst_count_hook,
            "ligo": self._ligo_distance_hook,
        }

    def register_ingest_hook(self, suffix: str, hook: IngestHook) -> None:
        normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        self._ingest_hooks[normalized_suffix.lower()] = hook

    def register_expansion_hook(self, trigger_kind: str, hook: ExpansionHook) -> None:
        self._expansion_hooks[trigger_kind.lower()] = hook

    def ingest_external_triggers(self) -> tuple[ExternalTrigger, ...]:
        return self.load_external_triggers()

    def load_external_triggers(self) -> tuple[ExternalTrigger, ...]:
        if not self.trigger_directory.exists() or not self.trigger_directory.is_dir():
            return ()
        triggers: list[ExternalTrigger] = []
        for path in sorted(self.trigger_directory.iterdir(), key=lambda candidate: candidate.name):
            if not path.is_file() or path.name.startswith("."):
                continue
            hook = self._ingest_hooks.get(path.suffix.lower())
            if hook is None:
                continue
            triggers.extend(hook(path))
        return tuple(triggers)

    def _load_json_trigger_file(self, path: Path) -> Sequence[ExternalTrigger]:
        return self._normalize_trigger_records(_load_json_records(path), path=path)

    def _load_csv_trigger_file(self, path: Path) -> Sequence[ExternalTrigger]:
        return self._normalize_trigger_records(_load_csv_records(path), path=path)

    def _normalize_trigger_records(self, records: Sequence[Mapping[str, Any]], *, path: Path) -> tuple[ExternalTrigger, ...]:
        return tuple(self._normalize_trigger_record(record, path=path, index=index) for index, record in enumerate(records))

    def _normalize_trigger_record(self, record: Mapping[str, Any], *, path: Path, index: int) -> ExternalTrigger:
        redshift_value = _first_present(record, "redshift", "z")
        if redshift_value is None:
            raise ValueError(f"Trigger record {index} in {path} is missing a redshift.")
        source_value = _first_present(record, "source", "instrument", "facility", "experiment", "dataset_label")
        trigger_kind = _normalize_trigger_kind(
            _first_present(record, "trigger_kind", "kind", "messenger", "category", "source", "instrument"),
            path=path,
        )
        return ExternalTrigger(
            event_id=str(_first_present(record, "event_id", "event_name", "trigger", "name", "label") or f"{path.stem}:{index}"),
            source=str(source_value or path.stem),
            trigger_kind=trigger_kind,
            redshift=_decimal(redshift_value),
            observed_expansion_rate_km_s_mpc=_decimal_or_none(
                _first_present(
                    record,
                    "observed_expansion_rate_km_s_mpc",
                    "expansion_rate_km_s_mpc",
                    "observed_hubble_km_s_mpc",
                    "hubble_km_s_mpc",
                    "expansion_proxy_km_s_mpc",
                )
            ),
            peak_strain=_decimal_or_none(
                _first_present(record, "peak_strain", "peak_absolute_strain", "strain_peak", "network_peak_absolute_strain")
            ),
            galaxy_count=_decimal_or_none(
                _first_present(record, "galaxy_count", "observed_galaxy_count", "count", "high_redshift_galaxy_count")
            ),
            luminosity_distance_mpc=_decimal_or_none(
                _first_present(record, "luminosity_distance_mpc", "distance_mpc", "source_distance_mpc")
            ),
            path=path,
            payload=dict(record),
        )

    def _ligo_distance_hook(self, trigger: ExternalTrigger, predicted_rate: Decimal) -> Decimal | None:
        del predicted_rate
        if trigger.luminosity_distance_mpc is None or trigger.luminosity_distance_mpc <= 0:
            return None
        if trigger.redshift <= 0:
            return None
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            observed_rate = _LIGHT_SPEED_KM_PER_S * trigger.redshift / trigger.luminosity_distance_mpc
            context.prec = self.precision
            return +observed_rate

    def _jwst_count_hook(self, trigger: ExternalTrigger, predicted_rate: Decimal) -> Decimal | None:
        observed_galaxy_count = trigger.galaxy_count
        reference_galaxy_count = _decimal_or_none(
            _first_present(
                trigger.payload,
                "reference_galaxy_count",
                "expected_galaxy_count",
                "benchmark_galaxy_count",
            )
        )
        reference_rate = _decimal_or_none(
            _first_present(
                trigger.payload,
                "reference_expansion_rate_km_s_mpc",
                "benchmark_expansion_rate_km_s_mpc",
            )
        )
        if observed_galaxy_count is None or observed_galaxy_count <= 0:
            return None
        if reference_galaxy_count is None or reference_galaxy_count <= 0:
            return None
        baseline_rate = predicted_rate if reference_rate is None else reference_rate
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            observed_rate = baseline_rate * reference_galaxy_count / observed_galaxy_count
            context.prec = self.precision
            return +observed_rate

    def _predict_expansion_rate(self, redshift: Decimal, *, predicted_anchor_hubble: Decimal) -> Decimal:
        return redshift_dependent_hubble_constant(
            redshift,
            h_lambda_km_s_mpc=predicted_anchor_hubble,
            precision=self.precision,
        )

    def _resolve_observed_expansion_rate(self, trigger: ExternalTrigger, predicted_rate: Decimal) -> Decimal | None:
        if trigger.observed_expansion_rate_km_s_mpc is not None:
            return trigger.observed_expansion_rate_km_s_mpc
        hook = self._expansion_hooks.get(trigger.trigger_kind)
        if hook is None:
            return None
        return hook(trigger, predicted_rate)

    def calculate_holographic_tension(
        self,
        triggers: Sequence[ExternalTrigger] | None = None,
        *,
        raise_on_anomaly: bool = True,
    ) -> ObservationalTensionAudit:
        resolved_triggers = tuple(self.load_external_triggers() if triggers is None else triggers)
        moat_bounds = load_holographic_moat_bounds()
        lambda_surface = EvolutionaryEngine.derive_lambda_surface(precision=self.precision)
        cosmology_audit = build_precision_cosmology_audit(redshifts=(Decimal("0"),), precision=self.precision)

        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            lambda_anchor_scale = lambda_surface.anchor_ratio.sqrt()
            predicted_anchor_hubble = cosmology_audit.h0_cmb_km_s_mpc * lambda_anchor_scale
            context.prec = self.precision
            predicted_anchor_hubble = +predicted_anchor_hubble

        diagnostics: list[TransientTensionDiagnostic] = []
        for trigger in resolved_triggers:
            predicted_rate = self._predict_expansion_rate(trigger.redshift, predicted_anchor_hubble=predicted_anchor_hubble)
            observed_rate = self._resolve_observed_expansion_rate(trigger, predicted_rate)
            included_in_high_redshift_audit = trigger.redshift > self.high_redshift_floor and observed_rate is not None
            if observed_rate is None:
                diagnostics.append(
                    TransientTensionDiagnostic(
                        trigger=trigger,
                        predicted_expansion_rate_km_s_mpc=predicted_rate,
                        observed_expansion_rate_km_s_mpc=None,
                        residual_km_s_mpc=None,
                        fractional_tension=None,
                        equivalent_moat_shift=None,
                        included_in_high_redshift_audit=False,
                        exceeds_moat_bounds=False,
                    )
                )
                continue
            with localcontext() as context:
                context.prec = self.precision + _GUARD_DIGITS
                residual = observed_rate - predicted_rate
                fractional_tension = abs(residual) if predicted_rate == 0 else abs(residual) / abs(predicted_rate)
                equivalent_moat_shift = moat_bounds.equivalent_branch_shift(fractional_tension)
                exceeds_moat_bounds = (
                    trigger.redshift > self.high_redshift_floor
                    and fractional_tension > moat_bounds.maximum_fractional_tension
                )
                context.prec = self.precision
                residual = +residual
                fractional_tension = +fractional_tension
                equivalent_moat_shift = +equivalent_moat_shift
            diagnostics.append(
                TransientTensionDiagnostic(
                    trigger=trigger,
                    predicted_expansion_rate_km_s_mpc=predicted_rate,
                    observed_expansion_rate_km_s_mpc=observed_rate,
                    residual_km_s_mpc=residual,
                    fractional_tension=fractional_tension,
                    equivalent_moat_shift=equivalent_moat_shift,
                    included_in_high_redshift_audit=included_in_high_redshift_audit,
                    exceeds_moat_bounds=exceeds_moat_bounds,
                )
            )

        multi_messenger_sources = tuple(
            sorted({trigger.trigger_kind for trigger in resolved_triggers if trigger.trigger_kind != "unknown"})
        )
        audit = ObservationalTensionAudit(
            trigger_directory=self.trigger_directory,
            high_redshift_floor=self.high_redshift_floor,
            lambda_holo_si_m2=lambda_surface.lambda_holo_si_m2,
            lambda_anchor_ratio=lambda_surface.anchor_ratio,
            holographic_bits=lambda_surface.holographic_bits,
            predicted_anchor_hubble_km_s_mpc=predicted_anchor_hubble,
            moat_bounds=moat_bounds,
            triggers=resolved_triggers,
            diagnostics=tuple(diagnostics),
            multi_messenger_sources=multi_messenger_sources,
        )
        if raise_on_anomaly and audit.anomaly_detected:
            raise TensionAnomaly(audit)
        return audit


def load_external_triggers(
    trigger_directory: Path | str | None = None,
    *,
    precision: int = DEFAULT_PRECISION,
    high_redshift_floor: Decimal | float | int | str = DEFAULT_HIGH_REDSHIFT_FLOOR,
) -> tuple[ExternalTrigger, ...]:
    return ObservationalBridge(
        trigger_directory=trigger_directory,
        precision=precision,
        high_redshift_floor=high_redshift_floor,
    ).load_external_triggers()


def calculate_holographic_tension(
    trigger_directory: Path | str | None = None,
    *,
    precision: int = DEFAULT_PRECISION,
    high_redshift_floor: Decimal | float | int | str = DEFAULT_HIGH_REDSHIFT_FLOOR,
    raise_on_anomaly: bool = True,
) -> ObservationalTensionAudit:
    return ObservationalBridge(
        trigger_directory=trigger_directory,
        precision=precision,
        high_redshift_floor=high_redshift_floor,
    ).calculate_holographic_tension(raise_on_anomaly=raise_on_anomaly)


__all__ = [
    "DEFAULT_HIGH_REDSHIFT_FLOOR",
    "DEFAULT_PRECISION",
    "ExternalTrigger",
    "HolographicMoatBounds",
    "ObservationalBridge",
    "ObservationalTensionAudit",
    "TensionAnomaly",
    "TransientTensionDiagnostic",
    "calculate_holographic_tension",
    "load_external_triggers",
    "load_holographic_moat_bounds",
]
