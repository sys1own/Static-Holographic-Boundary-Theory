from __future__ import annotations

"""Live H0 bridge for JWST/LIGO tension checks against the SHBT cosmology audit.

The bridge combines two live data paths:

- NASA ADS literature searches for the latest JWST H0-style measurements,
- GWOSC event feeds for recent LIGO/Virgo/KAGRA luminosity-distance proxies.

The resulting drift is compared against the branch-fixed geometric H0 residue
from ``shbt.precision_cosmology_engine``. If the fractional drift exceeds the
operational moat ``1e-15``, a ``SystemDecoherenceWarning`` is emitted.
"""

import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Final, Literal

if __package__ in (None, ""):
    import sys

    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.constants import LIGHT_SPEED_M_PER_S
from shbt.http_bridge import DEFAULT_TIMEOUT_SECONDS, FetchHook
from shbt.precision_cosmology_engine import DEFAULT_PRECISION, PrecisionCosmologyAudit, build_precision_cosmology_audit
from shbt.verification.live_bridge import (
    DEFAULT_ADS_ENDPOINT,
    DEFAULT_GWOSC_API_ROOT,
    DEFAULT_GWOSC_CATALOG,
    ADSLiteratureRecord,
    LIGOStrainObservation,
    LiveVerificationBridge,
    build_ads_search_url,
    build_gwosc_catalog_events_url,
)


ChannelKind = Literal["jwst", "ligo"]

DEFAULT_THEORETICAL_MOAT: Final[Decimal] = Decimal("1e-15")
DEFAULT_JWST_ADS_QUERY: Final[str] = '(JWST OR "James Webb" OR JADES OR CEERS) AND ("Hubble constant" OR H0)'
DEFAULT_H0_ADS_FIELDS: Final[tuple[str, ...]] = (
    "bibcode",
    "title",
    "pubdate",
    "author",
    "citation_count",
    "abstract",
)
DEFAULT_LIGO_PROXY_SAMPLE_SIZE: Final[int] = 8
_GUARD_DIGITS: Final[int] = 12
_HUNDRED: Final[Decimal] = Decimal("100")
_MIN_RESIDUE_SCALE: Final[Decimal] = Decimal("1e-30")
_ZERO: Final[Decimal] = Decimal("0")
_ONE: Final[Decimal] = Decimal("1")
_LIGHT_SPEED_KM_PER_S: Final[Decimal] = Decimal(str(LIGHT_SPEED_M_PER_S / 1000.0))
_H0_PLUS_MINUS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:\bH0\b|\bHubble\s+constant\b)[^0-9=]{0,40}(?:=|of|:)?\s*"
    r"(?P<value>\d{1,3}(?:\.\d+)?)\s*(?:±|\+/-|\\pm)\s*(?P<uncertainty>\d{1,3}(?:\.\d+)?)"
)
_H0_ASYMMETRIC_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:\bH0\b|\bHubble\s+constant\b)[^0-9=]{0,40}(?:=|of|:)?\s*"
    r"(?P<value>\d{1,3}(?:\.\d+)?)\s*\+\s*(?P<upper>\d{1,3}(?:\.\d+)?)\s*/\s*-\s*(?P<lower>\d{1,3}(?:\.\d+)?)"
)
_H0_UNIT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:\bH0\b|\bHubble\s+constant\b)[^0-9=]{0,40}(?:=|of|:)?\s*"
    r"(?P<value>\d{1,3}(?:\.\d+)?)\s*km\s*/?\s*s-?1\s*/?\s*Mpc-?1"
)


class SystemDecoherenceWarning(RuntimeWarning):
    """Raised when the live observational drift exits the operational moat."""


@dataclass(frozen=True)
class LiveH0Measurement:
    channel: ChannelKind
    instrument: str
    observation_id: str
    observation_date: str | None
    observed_h0_km_s_mpc: Decimal
    uncertainty_km_s_mpc: Decimal | None
    provenance: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class H0DriftDiagnostic:
    measurement: LiveH0Measurement
    observed_residue_km_s_mpc: Decimal
    theory_residue_km_s_mpc: Decimal
    absolute_residue_drift_km_s_mpc: Decimal
    fractional_residue_drift: Decimal
    drift_to_moat_ratio: Decimal


@dataclass(frozen=True)
class LiveH0TensionReport:
    cosmology_audit: PrecisionCosmologyAudit
    theoretical_moat: Decimal
    measurements: tuple[LiveH0Measurement, ...]
    diagnostics: tuple[H0DriftDiagnostic, ...]
    mean_fractional_drift: Decimal
    max_fractional_drift: Decimal
    drift_to_moat_ratio: Decimal
    tension_score: Decimal
    decoherence_triggered: bool

    @property
    def theory_h0_cmb_km_s_mpc(self) -> Decimal:
        return self.cosmology_audit.h0_cmb_km_s_mpc

    @property
    def theory_h0_local_km_s_mpc(self) -> Decimal:
        return self.cosmology_audit.h0_local_km_s_mpc

    @property
    def theory_h0_residue_km_s_mpc(self) -> Decimal:
        return self.cosmology_audit.hubble_gradient_km_s_mpc

    @property
    def data_sources(self) -> tuple[str, ...]:
        return tuple(sorted({measurement.provenance for measurement in self.measurements}))

    @property
    def statement(self) -> str:
        state = "DECOHERENCE" if self.decoherence_triggered else "stable"
        return (
            f"Live H0 drift audit is {state}: max fractional drift={self.max_fractional_drift} "
            f"against moat={self.theoretical_moat}, score={self.tension_score}/100."
        )


def _decimal(value: Decimal | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


def _median_decimal(values: Sequence[Decimal]) -> Decimal:
    if not values:
        raise ValueError("values must be non-empty.")
    ordered = tuple(sorted(values))
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
        return +(ordered[midpoint - 1] + ordered[midpoint]) / Decimal("2")


def _median_absolute_deviation(values: Sequence[Decimal], *, center: Decimal) -> Decimal | None:
    if len(values) <= 1:
        return None
    deviations = tuple(abs(value - center) for value in values)
    return _median_decimal(deviations)


def _normalize_ads_text(text: str) -> str:
    normalized = text.replace("−", "-").replace("–", "-").replace("±", "±")
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("{", "").replace("}", "")
    normalized = normalized.replace("H_0", "H0").replace("H 0", "H0")
    normalized = normalized.replace("km s^-1 Mpc^-1", "km s-1 Mpc-1")
    normalized = normalized.replace("km s^{-1} Mpc^{-1}", "km s-1 Mpc-1")
    normalized = normalized.replace("km/s/Mpc", "km s-1 Mpc-1")
    normalized = normalized.replace("km s−1 Mpc−1", "km s-1 Mpc-1")
    return normalized


def _extract_h0_measurement(text: str) -> tuple[Decimal, Decimal | None] | None:
    normalized_text = _normalize_ads_text(text)
    plus_minus_match = _H0_PLUS_MINUS_PATTERN.search(normalized_text)
    if plus_minus_match is not None:
        return _decimal(plus_minus_match.group("value")), _decimal(plus_minus_match.group("uncertainty"))

    asymmetric_match = _H0_ASYMMETRIC_PATTERN.search(normalized_text)
    if asymmetric_match is not None:
        upper = _decimal(asymmetric_match.group("upper"))
        lower = _decimal(asymmetric_match.group("lower"))
        return _decimal(asymmetric_match.group("value")), max(upper, lower)

    unit_match = _H0_UNIT_PATTERN.search(normalized_text)
    if unit_match is not None:
        return _decimal(unit_match.group("value")), None
    return None


def build_live_gwosc_events_url(
    catalog: str = DEFAULT_GWOSC_CATALOG,
    *,
    gwosc_api_root: str = DEFAULT_GWOSC_API_ROOT,
    include_default_parameters: bool = True,
) -> str:
    url = build_gwosc_catalog_events_url(catalog=catalog, endpoint=gwosc_api_root)
    if include_default_parameters and "include-default-parameters=true" not in url:
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}include-default-parameters=true"
    return url


class LiveH0Bridge:
    def __init__(
        self,
        *,
        precision: int = DEFAULT_PRECISION,
        theoretical_moat: Decimal | float | int | str = DEFAULT_THEORETICAL_MOAT,
        ads_endpoint: str = DEFAULT_ADS_ENDPOINT,
        gwosc_api_root: str = DEFAULT_GWOSC_API_ROOT,
    ) -> None:
        self.precision = max(int(precision), DEFAULT_PRECISION)
        self.theoretical_moat = _decimal(theoretical_moat)
        self._verification_bridge = LiveVerificationBridge(
            precision=self.precision,
            ads_endpoint=ads_endpoint,
            gwosc_api_root=gwosc_api_root,
        )

    def fetch_latest_jwst_measurement(
        self,
        *,
        query: str = DEFAULT_JWST_ADS_QUERY,
        rows: int = 10,
        token: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        fetcher: FetchHook | None = None,
    ) -> LiveH0Measurement:
        records = self._verification_bridge.fetch_ads_literature(
            query,
            rows=rows,
            fields=DEFAULT_H0_ADS_FIELDS,
            token=token,
            headers=headers,
            timeout=timeout,
            fetcher=fetcher,
        )
        for record in records:
            measurement = self._measurement_from_ads_record(record, channel="jwst", instrument="JWST")
            if measurement is not None:
                return measurement
        raise ValueError("No JWST H0 measurement could be extracted from the ADS response.")

    def fetch_latest_ligo_measurement(
        self,
        *,
        catalog: str = DEFAULT_GWOSC_CATALOG,
        url: str | None = None,
        sample_size: int = DEFAULT_LIGO_PROXY_SAMPLE_SIZE,
        headers: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        fetcher: FetchHook | None = None,
    ) -> LiveH0Measurement:
        resolved_url = build_live_gwosc_events_url(catalog=catalog, gwosc_api_root=self._verification_bridge.gwosc_api_root) if url is None else url
        observations = self._verification_bridge.fetch_ligo_events(
            url=resolved_url,
            limit=max(int(sample_size), 1),
            headers=headers,
            timeout=timeout,
            fetcher=fetcher,
        )

        h0_values: list[Decimal] = []
        contributing_events: list[LIGOStrainObservation] = []
        for observation in observations:
            proxy = self._ligo_h0_proxy(observation)
            if proxy is None:
                continue
            h0_values.append(proxy)
            contributing_events.append(observation)
            if len(h0_values) >= max(int(sample_size), 1):
                break

        if not h0_values:
            raise ValueError("No LIGO GWOSC event exposed both redshift and luminosity distance for an H0 proxy.")

        center = _median_decimal(h0_values)
        spread = _median_absolute_deviation(h0_values, center=center)
        event_ids = tuple(observation.event_id for observation in contributing_events)
        catalogs = tuple(sorted({observation.catalog for observation in contributing_events if observation.catalog}))
        latest_event = contributing_events[0]
        return LiveH0Measurement(
            channel="ligo",
            instrument="LIGO/Virgo/KAGRA",
            observation_id=latest_event.event_id,
            observation_date=None,
            observed_h0_km_s_mpc=center,
            uncertainty_km_s_mpc=spread,
            provenance="GWOSC luminosity-distance proxy",
            payload={
                "catalogs": catalogs,
                "event_ids": event_ids,
                "sample_size": len(event_ids),
            },
        )

    def calculate_tension(
        self,
        *,
        jwst_query: str = DEFAULT_JWST_ADS_QUERY,
        ads_rows: int = 10,
        ads_token: str | None = None,
        gwosc_catalog: str = DEFAULT_GWOSC_CATALOG,
        ligo_url: str | None = None,
        ligo_sample_size: int = DEFAULT_LIGO_PROXY_SAMPLE_SIZE,
        headers: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        ads_fetcher: FetchHook | None = None,
        ligo_fetcher: FetchHook | None = None,
        warn_on_decoherence: bool = True,
    ) -> LiveH0TensionReport:
        cosmology_audit = build_precision_cosmology_audit(precision=self.precision)
        jwst_measurement = self.fetch_latest_jwst_measurement(
            query=jwst_query,
            rows=ads_rows,
            token=ads_token,
            headers=headers,
            timeout=timeout,
            fetcher=ads_fetcher,
        )
        ligo_measurement = self.fetch_latest_ligo_measurement(
            catalog=gwosc_catalog,
            url=ligo_url,
            sample_size=ligo_sample_size,
            headers=headers,
            timeout=timeout,
            fetcher=ligo_fetcher,
        )
        report = self._build_report(cosmology_audit, (jwst_measurement, ligo_measurement))
        if warn_on_decoherence and report.decoherence_triggered:
            warnings.warn(
                (
                    "Live observational drift exceeds the SHBT operational moat: "
                    f"max drift={report.max_fractional_drift} > moat={report.theoretical_moat}."
                ),
                SystemDecoherenceWarning,
                stacklevel=2,
            )
        return report

    def _measurement_from_ads_record(
        self,
        record: ADSLiteratureRecord,
        *,
        channel: ChannelKind,
        instrument: str,
    ) -> LiveH0Measurement | None:
        text_candidates = [*record.title]
        abstract = record.payload.get("abstract")
        if isinstance(abstract, str) and abstract.strip():
            text_candidates.append(abstract)
        for candidate_text in text_candidates:
            extracted = _extract_h0_measurement(candidate_text)
            if extracted is None:
                continue
            observed_h0, uncertainty = extracted
            return LiveH0Measurement(
                channel=channel,
                instrument=instrument,
                observation_id=record.bibcode,
                observation_date=record.pubdate,
                observed_h0_km_s_mpc=observed_h0,
                uncertainty_km_s_mpc=uncertainty,
                provenance="NASA ADS literature",
                payload=dict(record.payload),
            )
        return None

    def _ligo_h0_proxy(self, observation: LIGOStrainObservation) -> Decimal | None:
        if observation.redshift is None or observation.luminosity_distance_mpc is None:
            return None
        if observation.redshift <= 0 or observation.luminosity_distance_mpc <= 0:
            return None
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            proxy = _LIGHT_SPEED_KM_PER_S * observation.redshift / observation.luminosity_distance_mpc
            context.prec = self.precision
            return +proxy

    def _build_report(
        self,
        cosmology_audit: PrecisionCosmologyAudit,
        measurements: Sequence[LiveH0Measurement],
    ) -> LiveH0TensionReport:
        diagnostics: list[H0DriftDiagnostic] = []
        theory_residue = cosmology_audit.hubble_gradient_km_s_mpc
        normalization = max(abs(theory_residue), _MIN_RESIDUE_SCALE)

        for measurement in measurements:
            with localcontext() as context:
                context.prec = self.precision + _GUARD_DIGITS
                observed_residue = measurement.observed_h0_km_s_mpc - cosmology_audit.h0_cmb_km_s_mpc
                absolute_drift = abs(observed_residue - theory_residue)
                fractional_drift = absolute_drift / normalization
                moat_ratio = fractional_drift / self.theoretical_moat
                context.prec = self.precision
                diagnostics.append(
                    H0DriftDiagnostic(
                        measurement=measurement,
                        observed_residue_km_s_mpc=+observed_residue,
                        theory_residue_km_s_mpc=+theory_residue,
                        absolute_residue_drift_km_s_mpc=+absolute_drift,
                        fractional_residue_drift=+fractional_drift,
                        drift_to_moat_ratio=+moat_ratio,
                    )
                )

        if not diagnostics:
            raise ValueError("At least one live measurement is required to compute tension.")

        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            total_fractional_drift = sum((diagnostic.fractional_residue_drift for diagnostic in diagnostics), _ZERO)
            mean_fractional_drift = total_fractional_drift / Decimal(len(diagnostics))
            max_fractional_drift = max(diagnostic.fractional_residue_drift for diagnostic in diagnostics)
            drift_to_moat_ratio = max_fractional_drift / self.theoretical_moat
            tension_score = _ZERO if max_fractional_drift <= 0 else _HUNDRED * max_fractional_drift / (max_fractional_drift + self.theoretical_moat)
            context.prec = self.precision
            mean_fractional_drift = +mean_fractional_drift
            max_fractional_drift = +max_fractional_drift
            drift_to_moat_ratio = +drift_to_moat_ratio
            tension_score = +tension_score

        return LiveH0TensionReport(
            cosmology_audit=cosmology_audit,
            theoretical_moat=self.theoretical_moat,
            measurements=tuple(measurements),
            diagnostics=tuple(diagnostics),
            mean_fractional_drift=mean_fractional_drift,
            max_fractional_drift=max_fractional_drift,
            drift_to_moat_ratio=drift_to_moat_ratio,
            tension_score=tension_score,
            decoherence_triggered=max_fractional_drift > self.theoretical_moat,
        )


def calculate_live_h0_tension(
    *,
    precision: int = DEFAULT_PRECISION,
    theoretical_moat: Decimal | float | int | str = DEFAULT_THEORETICAL_MOAT,
    jwst_query: str = DEFAULT_JWST_ADS_QUERY,
    ads_rows: int = 10,
    ads_token: str | None = None,
    gwosc_catalog: str = DEFAULT_GWOSC_CATALOG,
    ligo_url: str | None = None,
    ligo_sample_size: int = DEFAULT_LIGO_PROXY_SAMPLE_SIZE,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ads_fetcher: FetchHook | None = None,
    ligo_fetcher: FetchHook | None = None,
    warn_on_decoherence: bool = True,
) -> LiveH0TensionReport:
    return LiveH0Bridge(precision=precision, theoretical_moat=theoretical_moat).calculate_tension(
        jwst_query=jwst_query,
        ads_rows=ads_rows,
        ads_token=ads_token,
        gwosc_catalog=gwosc_catalog,
        ligo_url=ligo_url,
        ligo_sample_size=ligo_sample_size,
        headers=headers,
        timeout=timeout,
        ads_fetcher=ads_fetcher,
        ligo_fetcher=ligo_fetcher,
        warn_on_decoherence=warn_on_decoherence,
    )


__all__ = [
    "DEFAULT_H0_ADS_FIELDS",
    "DEFAULT_JWST_ADS_QUERY",
    "DEFAULT_LIGO_PROXY_SAMPLE_SIZE",
    "DEFAULT_THEORETICAL_MOAT",
    "H0DriftDiagnostic",
    "LiveH0Bridge",
    "LiveH0Measurement",
    "LiveH0TensionReport",
    "SystemDecoherenceWarning",
    "build_live_gwosc_events_url",
    "calculate_live_h0_tension",
    "build_ads_search_url",
]
