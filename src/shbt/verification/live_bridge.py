from __future__ import annotations

"""Native connectors for live astronomy verification against the SHBT kernel.

The bridge keeps the theory side anchored to the benchmark ``(26, 8, 312)``
branch while exposing lightweight stdlib-only fetchers for astronomy services
that commonly surface JWST and gravitational-wave results:

- VizieR catalog queries for JWST formation samples,
- SIMBAD identifier resolution for object normalization,
- NASA ADS search for current literature provenance,
- GWOSC catalog/event feeds for LIGO/Virgo/KAGRA event metadata.

The resulting ``TheoryTensionScore`` turns the fetched observations into a
dimensionless ``chi^2`` audit against the branch's rigidity-moat envelope.
"""

import csv
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from io import StringIO
from typing import Any, Final, Literal
from urllib.parse import quote, urlencode

from shbt.core.derivation_api import DEFAULT_PRECISION, TopologicalVacuum, UniverseFactory
from shbt.core.projector import HolographicCompiler
from shbt.http_bridge import DEFAULT_TIMEOUT_SECONDS, FetchHook, fetch_json, fetch_resource
from shbt.observational_bridge import HolographicMoatBounds, load_holographic_moat_bounds


DEFAULT_VIZIER_ENDPOINT = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
DEFAULT_SIMBAD_ENDPOINT = "https://simbad.cds.unistra.fr/simbad/sim-id"
DEFAULT_ADS_ENDPOINT = "https://api.adsabs.harvard.edu/v1/search/query"
DEFAULT_GWOSC_API_ROOT = "https://gwosc.org/api/v2"
DEFAULT_GWOSC_CATALOG = "GWTC"

DEFAULT_ADS_FIELDS: Final[tuple[str, ...]] = (
    "bibcode",
    "title",
    "pubdate",
    "author",
    "citation_count",
)
DEFAULT_JWST_COLUMNS: Final[tuple[str, ...]] = (
    "ID",
    "z",
    "Mstar",
)
_NUMERIC_PATTERN: Final[re.Pattern[str]] = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_GUARD_DIGITS: Final[int] = 12
_LIGO_STRAIN_FLOOR: Final[Decimal] = Decimal("1e-24")
_LIGO_STRAIN_CEILING: Final[Decimal] = Decimal("1e-20")
_LIGO_SNR_FLOOR: Final[Decimal] = Decimal("4")
_LIGO_SNR_CEILING: Final[Decimal] = Decimal("100")
_MIN_SIGMA: Final[Decimal] = Decimal("1e-6")
_ZERO: Final[Decimal] = Decimal("0")
_ONE: Final[Decimal] = Decimal("1")
_HUNDRED: Final[Decimal] = Decimal("100")

_JWST_ID_KEYS: Final[tuple[str, ...]] = (
    "ID",
    "Name",
    "Object",
    "Target",
    "source",
    "source_id",
)
_JWST_REDSHIFT_KEYS: Final[tuple[str, ...]] = (
    "z",
    "zspec",
    "z_spec",
    "zphot",
    "z_phot",
    "redshift",
)
_JWST_MASS_KEYS: Final[tuple[str, ...]] = (
    "Mstar",
    "stellar_mass",
    "mass",
    "M",
)
_JWST_LOG_MASS_KEYS: Final[tuple[str, ...]] = (
    "logMstar",
    "logM",
    "log_mass",
    "log_stellar_mass",
)
_JWST_DENSITY_KEYS: Final[tuple[str, ...]] = (
    "rho_mass",
    "mass_density",
    "rhoM",
    "density",
)
_JWST_LOG_DENSITY_KEYS: Final[tuple[str, ...]] = (
    "log_rho_mass",
    "log_mass_density",
    "logrho_mass",
)

_LIGO_ID_KEYS: Final[tuple[str, ...]] = (
    "commonName",
    "event_name",
    "name",
    "graceid",
    "superevent_id",
)
_LIGO_CATALOG_KEYS: Final[tuple[str, ...]] = (
    "catalog",
    "catalog.name",
    "catalog.shortName",
)
_LIGO_STRAIN_KEYS: Final[tuple[str, ...]] = (
    "peak_strain",
    "strain",
    "h_peak",
    "hpeak",
)
_LIGO_SNR_KEYS: Final[tuple[str, ...]] = (
    "network_matched_filter_snr",
    "network_snr",
    "snr",
)
_LIGO_REDSHIFT_KEYS: Final[tuple[str, ...]] = (
    "redshift",
    "redshift_mean",
    "z",
)
_LIGO_DISTANCE_KEYS: Final[tuple[str, ...]] = (
    "luminosity_distance_mpc",
    "luminosity_distance",
    "distance_mpc",
    "distance",
)


ComparisonKind = Literal["jwst", "ligo"]


def _decimal(value: Decimal | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


def _decimal_or_none(value: object | None) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return _decimal(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"nan", "null", "none", "n/a", "--"}:
            return None
        match = _NUMERIC_PATTERN.search(stripped.replace(",", ""))
        if match is None:
            return None
        return Decimal(match.group(0))
    return None


def _normalized_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _mapping_get(payload: Mapping[str, Any], key: str) -> Any | None:
    if key in payload:
        return payload[key]
    normalized_key = _normalized_key(key)
    for candidate_key, candidate_value in payload.items():
        if _normalized_key(str(candidate_key)) == normalized_key:
            return candidate_value
    return None


def _lookup_value(payload: Mapping[str, Any], candidate_paths: Sequence[str]) -> Any | None:
    for path in candidate_paths:
        current: Any = payload
        for part in path.split("."):
            if not isinstance(current, Mapping):
                current = None
                break
            current = _mapping_get({str(key): value for key, value in current.items()}, part)
        if current is not None:
            return current
    return None


def _string_or_none(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return None
    candidate = str(value).strip()
    return candidate or None


def _power_of_ten(value: Decimal) -> Decimal:
    return Decimal(str(10.0 ** float(value)))


def _resolve_numeric_field(
    payload: Mapping[str, Any],
    *,
    linear_keys: Sequence[str],
    logarithmic_keys: Sequence[str] = (),
) -> Decimal | None:
    linear_value = _decimal_or_none(_lookup_value(payload, linear_keys))
    if linear_value is not None:
        return linear_value
    logarithmic_value = _decimal_or_none(_lookup_value(payload, logarithmic_keys))
    if logarithmic_value is None:
        return None
    return _power_of_ten(logarithmic_value)


def _resolve_named_numeric_field(
    payload: Mapping[str, Any],
    *,
    linear_keys: Sequence[str],
    sequence_paths: Sequence[str] = ("default_parameters", "parameters"),
    name_keys: Sequence[str] = ("name", "parameter", "parameter_name"),
    value_keys: Sequence[str] = ("best", "value", "median", "mean", "maximum_likelihood", "maximum_likelihood_value"),
) -> Decimal | None:
    direct_value = _resolve_numeric_field(payload, linear_keys=linear_keys)
    if direct_value is not None:
        return direct_value

    normalized_candidates = {_normalized_key(candidate) for candidate in linear_keys}
    for sequence_path in sequence_paths:
        sequence_value = _lookup_value(payload, (sequence_path,))
        if not isinstance(sequence_value, Sequence) or isinstance(sequence_value, (str, bytes)):
            continue
        for entry in sequence_value:
            if not isinstance(entry, Mapping):
                continue
            name = _string_or_none(_lookup_value(entry, name_keys))
            if name is None or _normalized_key(name) not in normalized_candidates:
                continue
            nested_value = _resolve_numeric_field(entry, linear_keys=value_keys)
            if nested_value is not None:
                return nested_value
    return None


def _clip_unit_interval(value: Decimal) -> Decimal:
    if value <= _ZERO:
        return _ZERO
    if value >= _ONE:
        return _ONE
    return value


def _normalize_log10(value: Decimal, *, floor: Decimal, ceiling: Decimal) -> Decimal:
    if value <= 0 or floor <= 0 or ceiling <= floor:
        return _ZERO
    lower = math.log10(float(floor))
    upper = math.log10(float(ceiling))
    normalized = (math.log10(float(value)) - lower) / (upper - lower)
    return _clip_unit_interval(Decimal(str(normalized)))


def _parse_delimited_rows(text: str, *, delimiter: str) -> tuple[dict[str, str], ...]:
    filtered_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("--"):
            continue
        filtered_lines.append(line)
    if not filtered_lines:
        return ()
    reader = csv.DictReader(StringIO("\n".join(filtered_lines)), delimiter=delimiter)
    return tuple(
        {
            str(key): value.strip() if isinstance(value, str) else ""
            for key, value in row.items()
            if key is not None
        }
        for row in reader
    )


def build_vizier_catalog_url(
    catalog: str,
    *,
    columns: Sequence[str] = DEFAULT_JWST_COLUMNS,
    constraints: Mapping[str, str | int | float] | None = None,
    limit: int = 25,
    endpoint: str = DEFAULT_VIZIER_ENDPOINT,
) -> str:
    if not catalog.strip():
        raise ValueError("catalog must be non-empty.")
    if limit <= 0:
        raise ValueError("limit must be positive.")
    parameters: list[tuple[str, str]] = [
        ("-source", catalog),
        ("-out.max", str(int(limit))),
    ]
    if columns:
        parameters.append(("-out", ",".join(str(column) for column in columns)))
    for key, value in (constraints or {}).items():
        parameters.append((str(key), str(value)))
    return f"{endpoint}?{urlencode(parameters)}"


def build_simbad_identifier_url(
    identifier: str,
    *,
    output_format: str = "ASCII",
    endpoint: str = DEFAULT_SIMBAD_ENDPOINT,
) -> str:
    if not identifier.strip():
        raise ValueError("identifier must be non-empty.")
    parameters = urlencode({"Ident": identifier, "output.format": output_format})
    return f"{endpoint}?{parameters}"


def build_ads_search_url(
    query: str,
    *,
    rows: int = 5,
    fields: Sequence[str] = DEFAULT_ADS_FIELDS,
    sort: str = "date desc",
    endpoint: str = DEFAULT_ADS_ENDPOINT,
) -> str:
    if not query.strip():
        raise ValueError("query must be non-empty.")
    if rows <= 0:
        raise ValueError("rows must be positive.")
    parameters = [
        ("q", query),
        ("rows", str(int(rows))),
        ("sort", sort),
        ("fl", ",".join(str(field_name) for field_name in fields)),
    ]
    return f"{endpoint}?{urlencode(parameters)}"


def build_gwosc_catalog_events_url(
    catalog: str = DEFAULT_GWOSC_CATALOG,
    *,
    endpoint: str = DEFAULT_GWOSC_API_ROOT,
) -> str:
    if not catalog.strip():
        raise ValueError("catalog must be non-empty.")
    return f"{endpoint.rstrip('/')}/catalogs/{quote(catalog)}/events?format=api"


@dataclass(frozen=True)
class SimbadObjectRecord:
    identifier: str
    canonical_identifier: str | None
    object_type: str | None
    redshift: Decimal | None
    coordinates: str | None
    raw_text: str


@dataclass(frozen=True)
class ADSLiteratureRecord:
    bibcode: str
    title: tuple[str, ...]
    pubdate: str | None
    authors: tuple[str, ...]
    citation_count: int | None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JWSTFormationObservation:
    identifier: str
    catalog: str
    redshift: Decimal
    stellar_mass: Decimal | None
    formation_density_proxy: Decimal
    canonical_identifier: str | None = None
    object_type: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LIGOStrainObservation:
    event_id: str
    catalog: str | None
    peak_strain: Decimal | None
    network_snr: Decimal | None
    redshift: Decimal | None
    luminosity_distance_mpc: Decimal | None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TheoryKernelProfile:
    benchmark_branch: tuple[int, int, int]
    normalized_state: tuple[Decimal, ...]
    visible_support_density: Decimal
    central_charge_ratio: Decimal
    inverse_pixel_volume: Decimal
    vacuum_pressure: Decimal
    geometric_kappa: Decimal
    jwst_formation_peak: Decimal
    ligo_strain_peak: Decimal
    moat_sigma: Decimal


@dataclass(frozen=True)
class ObservationTensionComponent:
    source_kind: ComparisonKind
    identifier: str
    observed_proxy: Decimal
    predicted_proxy: Decimal
    sigma: Decimal
    normalized_residual: Decimal
    chi_square: Decimal
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TheoryTensionScore:
    benchmark_branch: tuple[int, int, int]
    kernel_profile: TheoryKernelProfile
    moat_bounds: HolographicMoatBounds
    jwst_observations: tuple[JWSTFormationObservation, ...]
    ligo_observations: tuple[LIGOStrainObservation, ...]
    literature: tuple[ADSLiteratureRecord, ...]
    comparisons: tuple[ObservationTensionComponent, ...]
    chi_square: Decimal
    reduced_chi_square: Decimal
    tension_score: Decimal
    tension_label: str
    data_sources: tuple[str, ...]

    @property
    def comparison_count(self) -> int:
        return len(self.comparisons)

    @property
    def multi_messenger(self) -> bool:
        return bool(self.jwst_observations and self.ligo_observations)

    @property
    def statement(self) -> str:
        return (
            f"The {self.benchmark_branch} kernel returns chi^2={self.chi_square} "
            f"and a Tension Score of {self.tension_score}/100 against {self.comparison_count} live comparisons."
        )


class LiveVerificationBridge:
    """Fetch public astronomy payloads and score them against the SHBT moat."""

    def __init__(
        self,
        *,
        precision: int = DEFAULT_PRECISION,
        vizier_endpoint: str = DEFAULT_VIZIER_ENDPOINT,
        simbad_endpoint: str = DEFAULT_SIMBAD_ENDPOINT,
        ads_endpoint: str = DEFAULT_ADS_ENDPOINT,
        gwosc_api_root: str = DEFAULT_GWOSC_API_ROOT,
    ) -> None:
        self.precision = max(int(precision), DEFAULT_PRECISION)
        self.vizier_endpoint = vizier_endpoint
        self.simbad_endpoint = simbad_endpoint
        self.ads_endpoint = ads_endpoint
        self.gwosc_api_root = gwosc_api_root.rstrip("/")

    def build_kernel_profile(
        self,
        *,
        vacuum: TopologicalVacuum | None = None,
        moat_bounds: HolographicMoatBounds | None = None,
    ) -> TheoryKernelProfile:
        resolved_vacuum = UniverseFactory.benchmark_vacuum() if vacuum is None else vacuum
        resolved_moat_bounds = load_holographic_moat_bounds() if moat_bounds is None else moat_bounds
        lattice = HolographicCompiler(precision=self.precision).build_benchmark_lattice(vacuum=resolved_vacuum)
        normalized_state = tuple(+value for value in lattice.normalized_state)
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            jwst_peak = sum(normalized_state[:3], _ZERO)
            ligo_peak = sum(normalized_state[2:], _ZERO)
            moat_sigma = max(resolved_moat_bounds.maximum_fractional_tension, _MIN_SIGMA)
            context.prec = self.precision
            return TheoryKernelProfile(
                benchmark_branch=resolved_vacuum.branch,
                normalized_state=normalized_state,
                visible_support_density=+normalized_state[0],
                central_charge_ratio=+normalized_state[1],
                inverse_pixel_volume=+normalized_state[2],
                vacuum_pressure=+normalized_state[3],
                geometric_kappa=+normalized_state[4],
                jwst_formation_peak=+jwst_peak,
                ligo_strain_peak=+ligo_peak,
                moat_sigma=+moat_sigma,
            )

    def fetch_simbad_object(
        self,
        identifier: str,
        *,
        output_format: str = "ASCII",
        headers: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        fetcher: FetchHook | None = None,
    ) -> SimbadObjectRecord:
        url = build_simbad_identifier_url(
            identifier,
            output_format=output_format,
            endpoint=self.simbad_endpoint,
        )
        raw_text = fetch_resource(url, headers=headers, timeout=timeout, fetcher=fetcher).text()
        canonical_identifier = self._capture_regex(
            raw_text,
            patterns=(
                r"(?im)^Identifier(?:s)?\s*:\s*(.+)$",
                r"(?im)^Main identifier\s*:\s*(.+)$",
            ),
        )
        object_type = self._capture_regex(
            raw_text,
            patterns=(
                r"(?im)^Object type\s*:\s*(.+)$",
                r"(?im)^Type\s*:\s*(.+)$",
            ),
        )
        coordinates = self._capture_regex(
            raw_text,
            patterns=(
                r"(?im)^Coordinates(?:\s*\([^)]*\))?\s*:\s*(.+)$",
            ),
        )
        redshift_value = self._capture_regex(
            raw_text,
            patterns=(
                r"(?im)^Redshift\s*:\s*([-+0-9.eE]+)",
            ),
        )
        return SimbadObjectRecord(
            identifier=identifier,
            canonical_identifier=canonical_identifier,
            object_type=object_type,
            redshift=_decimal_or_none(redshift_value),
            coordinates=coordinates,
            raw_text=raw_text,
        )

    def fetch_vizier_jwst_catalog(
        self,
        catalog: str,
        *,
        columns: Sequence[str] = DEFAULT_JWST_COLUMNS,
        constraints: Mapping[str, str | int | float] | None = None,
        limit: int = 25,
        resolve_with_simbad: bool = False,
        max_simbad_lookups: int = 5,
        headers: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        fetcher: FetchHook | None = None,
        simbad_fetcher: FetchHook | None = None,
    ) -> tuple[JWSTFormationObservation, ...]:
        url = build_vizier_catalog_url(
            catalog,
            columns=columns,
            constraints=constraints,
            limit=limit,
            endpoint=self.vizier_endpoint,
        )
        response = fetch_resource(url, headers=headers, timeout=timeout, fetcher=fetcher)
        rows = _parse_delimited_rows(response.text(), delimiter="\t")

        observations: list[JWSTFormationObservation] = []
        for index, row in enumerate(rows, start=1):
            redshift = _resolve_numeric_field(row, linear_keys=_JWST_REDSHIFT_KEYS)
            if redshift is None:
                continue
            stellar_mass = _resolve_numeric_field(
                row,
                linear_keys=_JWST_MASS_KEYS,
                logarithmic_keys=_JWST_LOG_MASS_KEYS,
            )
            density = _resolve_numeric_field(
                row,
                linear_keys=_JWST_DENSITY_KEYS,
                logarithmic_keys=_JWST_LOG_DENSITY_KEYS,
            )
            if density is None and stellar_mass is not None:
                with localcontext() as context:
                    context.prec = self.precision + _GUARD_DIGITS
                    density = stellar_mass * (_ONE + redshift) ** 3
                    context.prec = self.precision
                    density = +density
            if density is None:
                density = +redshift
            identifier = _string_or_none(_lookup_value(row, _JWST_ID_KEYS)) or f"{catalog}:{index}"
            observations.append(
                JWSTFormationObservation(
                    identifier=identifier,
                    catalog=catalog,
                    redshift=+redshift,
                    stellar_mass=stellar_mass,
                    formation_density_proxy=+density,
                    payload=dict(row),
                )
            )

        if not resolve_with_simbad or not observations:
            return tuple(observations)

        resolved_observations: list[JWSTFormationObservation] = []
        for index, observation in enumerate(observations):
            if index >= max(int(max_simbad_lookups), 0):
                resolved_observations.append(observation)
                continue
            simbad_record = self.fetch_simbad_object(
                observation.identifier,
                headers=headers,
                timeout=timeout,
                fetcher=fetcher if simbad_fetcher is None else simbad_fetcher,
            )
            resolved_observations.append(
                JWSTFormationObservation(
                    identifier=observation.identifier,
                    catalog=observation.catalog,
                    redshift=observation.redshift,
                    stellar_mass=observation.stellar_mass,
                    formation_density_proxy=observation.formation_density_proxy,
                    canonical_identifier=simbad_record.canonical_identifier,
                    object_type=simbad_record.object_type,
                    payload=observation.payload,
                )
            )
        return tuple(resolved_observations)

    def fetch_ads_literature(
        self,
        query: str,
        *,
        rows: int = 5,
        fields: Sequence[str] = DEFAULT_ADS_FIELDS,
        token: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        fetcher: FetchHook | None = None,
    ) -> tuple[ADSLiteratureRecord, ...]:
        resolved_token = token or os.getenv("ADS_API_TOKEN")
        if resolved_token is None and fetcher is None:
            raise ValueError("NASA ADS requests require an API token via the 'token' argument or ADS_API_TOKEN.")

        request_headers = dict(headers or {})
        if resolved_token is not None:
            request_headers.setdefault("Authorization", f"Bearer {resolved_token}")

        url = build_ads_search_url(
            query,
            rows=rows,
            fields=fields,
            endpoint=self.ads_endpoint,
        )
        payload = fetch_json(url, headers=request_headers, timeout=timeout, fetcher=fetcher)
        response_payload = payload.get("response") if isinstance(payload, Mapping) else None
        docs = response_payload.get("docs") if isinstance(response_payload, Mapping) else None
        if not isinstance(docs, Sequence) or isinstance(docs, (str, bytes)):
            return ()

        records: list[ADSLiteratureRecord] = []
        for raw_document in docs:
            if not isinstance(raw_document, Mapping):
                continue
            title_value = raw_document.get("title")
            if isinstance(title_value, str):
                titles = (title_value,)
            elif isinstance(title_value, Sequence) and not isinstance(title_value, (str, bytes)):
                titles = tuple(str(entry) for entry in title_value)
            else:
                titles = ()
            authors_value = raw_document.get("author")
            if isinstance(authors_value, str):
                authors = (authors_value,)
            elif isinstance(authors_value, Sequence) and not isinstance(authors_value, (str, bytes)):
                authors = tuple(str(entry) for entry in authors_value)
            else:
                authors = ()
            citation_count_value = raw_document.get("citation_count")
            citation_count = int(citation_count_value) if isinstance(citation_count_value, (int, float)) else None
            records.append(
                ADSLiteratureRecord(
                    bibcode=str(raw_document.get("bibcode", "")),
                    title=titles,
                    pubdate=_string_or_none(raw_document.get("pubdate")),
                    authors=authors,
                    citation_count=citation_count,
                    payload={str(key): value for key, value in raw_document.items()},
                )
            )
        return tuple(records)

    def fetch_ligo_events(
        self,
        *,
        url: str | None = None,
        limit: int | None = 25,
        headers: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        fetcher: FetchHook | None = None,
    ) -> tuple[LIGOStrainObservation, ...]:
        resolved_url = build_gwosc_catalog_events_url(endpoint=self.gwosc_api_root) if url is None else url
        payload = fetch_json(resolved_url, headers=headers, timeout=timeout, fetcher=fetcher)
        event_mappings = self._coerce_ligo_event_payloads(payload)
        observations: list[LIGOStrainObservation] = []
        for index, (fallback_name, raw_event) in enumerate(event_mappings, start=1):
            event_id = _string_or_none(_lookup_value(raw_event, _LIGO_ID_KEYS)) or fallback_name or f"ligo-{index}"
            peak_strain = _resolve_named_numeric_field(raw_event, linear_keys=_LIGO_STRAIN_KEYS)
            network_snr = _resolve_named_numeric_field(raw_event, linear_keys=_LIGO_SNR_KEYS)
            redshift = _resolve_named_numeric_field(raw_event, linear_keys=_LIGO_REDSHIFT_KEYS)
            luminosity_distance = _resolve_named_numeric_field(raw_event, linear_keys=_LIGO_DISTANCE_KEYS)
            if peak_strain is None and network_snr is None and luminosity_distance is None:
                continue
            observations.append(
                LIGOStrainObservation(
                    event_id=event_id,
                    catalog=self._resolve_ligo_catalog(raw_event),
                    peak_strain=peak_strain,
                    network_snr=network_snr,
                    redshift=redshift,
                    luminosity_distance_mpc=luminosity_distance,
                    payload={str(key): value for key, value in raw_event.items()},
                )
            )
            if limit is not None and len(observations) >= int(limit):
                break
        return tuple(observations)

    def compute_tension_score(
        self,
        jwst_observations: Sequence[JWSTFormationObservation],
        ligo_observations: Sequence[LIGOStrainObservation],
        *,
        literature: Sequence[ADSLiteratureRecord] = (),
        vacuum: TopologicalVacuum | None = None,
        moat_bounds: HolographicMoatBounds | None = None,
    ) -> TheoryTensionScore:
        resolved_moat_bounds = load_holographic_moat_bounds() if moat_bounds is None else moat_bounds
        kernel_profile = self.build_kernel_profile(vacuum=vacuum, moat_bounds=resolved_moat_bounds)
        sigma = max(kernel_profile.moat_sigma, _MIN_SIGMA)
        resolved_jwst = tuple(jwst_observations)
        resolved_ligo = tuple(ligo_observations)
        literature_records = tuple(literature)

        max_formation_density = max(
            (observation.formation_density_proxy for observation in resolved_jwst if observation.formation_density_proxy > 0),
            default=_ONE,
        )

        comparisons: list[ObservationTensionComponent] = []
        for observation in resolved_jwst:
            observed_proxy = _clip_unit_interval(observation.formation_density_proxy / max_formation_density)
            comparisons.append(
                self._build_component(
                    source_kind="jwst",
                    identifier=observation.canonical_identifier or observation.identifier,
                    observed_proxy=observed_proxy,
                    predicted_proxy=kernel_profile.jwst_formation_peak,
                    sigma=sigma,
                    payload=observation.payload,
                )
            )
        for observation in resolved_ligo:
            observed_proxy = self._ligo_proxy(observation)
            if observed_proxy is None:
                continue
            comparisons.append(
                self._build_component(
                    source_kind="ligo",
                    identifier=observation.event_id,
                    observed_proxy=observed_proxy,
                    predicted_proxy=kernel_profile.ligo_strain_peak,
                    sigma=sigma,
                    payload=observation.payload,
                )
            )

        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            chi_square = sum((component.chi_square for component in comparisons), _ZERO)
            degrees_of_freedom = Decimal(max(len(comparisons), 1))
            reduced_chi_square = chi_square / degrees_of_freedom
            tension_score = _ZERO if not comparisons else _HUNDRED * chi_square / (chi_square + degrees_of_freedom)
            context.prec = self.precision
            chi_square = +chi_square
            reduced_chi_square = +reduced_chi_square
            tension_score = +tension_score

        data_sources = tuple(
            sorted(
                {
                    *(observation.catalog for observation in resolved_jwst if observation.catalog),
                    *(observation.catalog for observation in resolved_ligo if observation.catalog),
                    *("NASA ADS" for _ in literature_records),
                }
            )
        )
        return TheoryTensionScore(
            benchmark_branch=kernel_profile.benchmark_branch,
            kernel_profile=kernel_profile,
            moat_bounds=resolved_moat_bounds,
            jwst_observations=resolved_jwst,
            ligo_observations=resolved_ligo,
            literature=literature_records,
            comparisons=tuple(comparisons),
            chi_square=chi_square,
            reduced_chi_square=reduced_chi_square,
            tension_score=tension_score,
            tension_label=self._tension_label(reduced_chi_square),
            data_sources=data_sources,
        )

    def calculate_live_tension_score(
        self,
        jwst_catalog: str,
        *,
        jwst_columns: Sequence[str] = DEFAULT_JWST_COLUMNS,
        jwst_constraints: Mapping[str, str | int | float] | None = None,
        jwst_limit: int = 25,
        resolve_jwst_with_simbad: bool = False,
        max_simbad_lookups: int = 5,
        ligo_url: str | None = None,
        ligo_limit: int | None = 25,
        ads_query: str | None = None,
        ads_rows: int = 5,
        ads_token: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        vizier_fetcher: FetchHook | None = None,
        simbad_fetcher: FetchHook | None = None,
        ligo_fetcher: FetchHook | None = None,
        ads_fetcher: FetchHook | None = None,
        vacuum: TopologicalVacuum | None = None,
    ) -> TheoryTensionScore:
        jwst_observations = self.fetch_vizier_jwst_catalog(
            jwst_catalog,
            columns=jwst_columns,
            constraints=jwst_constraints,
            limit=jwst_limit,
            resolve_with_simbad=resolve_jwst_with_simbad,
            max_simbad_lookups=max_simbad_lookups,
            headers=headers,
            timeout=timeout,
            fetcher=vizier_fetcher,
            simbad_fetcher=simbad_fetcher,
        )
        ligo_observations = self.fetch_ligo_events(
            url=ligo_url,
            limit=ligo_limit,
            headers=headers,
            timeout=timeout,
            fetcher=ligo_fetcher,
        )
        literature: tuple[ADSLiteratureRecord, ...] = ()
        if ads_query is not None:
            literature = self.fetch_ads_literature(
                ads_query,
                rows=ads_rows,
                token=ads_token,
                headers=headers,
                timeout=timeout,
                fetcher=ads_fetcher,
            )
        return self.compute_tension_score(
            jwst_observations,
            ligo_observations,
            literature=literature,
            vacuum=vacuum,
        )

    def _build_component(
        self,
        *,
        source_kind: ComparisonKind,
        identifier: str,
        observed_proxy: Decimal,
        predicted_proxy: Decimal,
        sigma: Decimal,
        payload: Mapping[str, Any],
    ) -> ObservationTensionComponent:
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            normalized_residual = (observed_proxy - predicted_proxy) / sigma
            chi_square = normalized_residual * normalized_residual
            context.prec = self.precision
            return ObservationTensionComponent(
                source_kind=source_kind,
                identifier=identifier,
                observed_proxy=+observed_proxy,
                predicted_proxy=+predicted_proxy,
                sigma=+sigma,
                normalized_residual=+normalized_residual,
                chi_square=+chi_square,
                payload={str(key): value for key, value in payload.items()},
            )

    def _ligo_proxy(self, observation: LIGOStrainObservation) -> Decimal | None:
        if observation.peak_strain is not None and observation.peak_strain > 0:
            return _normalize_log10(
                observation.peak_strain,
                floor=_LIGO_STRAIN_FLOOR,
                ceiling=_LIGO_STRAIN_CEILING,
            )
        if observation.network_snr is not None and observation.network_snr > 0:
            return _normalize_log10(
                observation.network_snr,
                floor=_LIGO_SNR_FLOOR,
                ceiling=_LIGO_SNR_CEILING,
            )
        return None

    def _resolve_ligo_catalog(self, payload: Mapping[str, Any]) -> str | None:
        catalog_mapping = _lookup_value(payload, ("catalog",))
        if isinstance(catalog_mapping, Mapping):
            for key in ("shortName", "name"):
                resolved = _string_or_none(_mapping_get({str(entry): value for entry, value in catalog_mapping.items()}, key))
                if resolved is not None:
                    return resolved
        return _string_or_none(_lookup_value(payload, _LIGO_CATALOG_KEYS))

    def _coerce_ligo_event_payloads(self, payload: Any) -> tuple[tuple[str | None, Mapping[str, Any]], ...]:
        if isinstance(payload, Mapping):
            results = payload.get("results")
            if isinstance(results, Sequence) and not isinstance(results, (str, bytes)):
                return tuple(
                    (None, item)
                    for item in results
                    if isinstance(item, Mapping)
                )
            events = payload.get("events")
            if isinstance(events, Mapping):
                return tuple(
                    (str(name), item)
                    for name, item in events.items()
                    if isinstance(item, Mapping)
                )
            if isinstance(events, Sequence) and not isinstance(events, (str, bytes)):
                return tuple(
                    (None, item)
                    for item in events
                    if isinstance(item, Mapping)
                )
            return ((None, payload),)
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            return tuple((None, item) for item in payload if isinstance(item, Mapping))
        return ()

    def _capture_regex(self, text: str, *, patterns: Sequence[str]) -> str | None:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match is not None:
                return _string_or_none(match.group(1))
        return None

    def _tension_label(self, reduced_chi_square: Decimal) -> str:
        if reduced_chi_square < Decimal("1"):
            return "consistent"
        if reduced_chi_square < Decimal("4"):
            return "watch"
        if reduced_chi_square < Decimal("9"):
            return "tension"
        return "critical"


def calculate_live_tension_score(
    jwst_catalog: str,
    *,
    precision: int = DEFAULT_PRECISION,
    jwst_columns: Sequence[str] = DEFAULT_JWST_COLUMNS,
    jwst_constraints: Mapping[str, str | int | float] | None = None,
    jwst_limit: int = 25,
    resolve_jwst_with_simbad: bool = False,
    max_simbad_lookups: int = 5,
    ligo_url: str | None = None,
    ligo_limit: int | None = 25,
    ads_query: str | None = None,
    ads_rows: int = 5,
    ads_token: str | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    vizier_fetcher: FetchHook | None = None,
    simbad_fetcher: FetchHook | None = None,
    ligo_fetcher: FetchHook | None = None,
    ads_fetcher: FetchHook | None = None,
    vacuum: TopologicalVacuum | None = None,
) -> TheoryTensionScore:
    return LiveVerificationBridge(precision=precision).calculate_live_tension_score(
        jwst_catalog,
        jwst_columns=jwst_columns,
        jwst_constraints=jwst_constraints,
        jwst_limit=jwst_limit,
        resolve_jwst_with_simbad=resolve_jwst_with_simbad,
        max_simbad_lookups=max_simbad_lookups,
        ligo_url=ligo_url,
        ligo_limit=ligo_limit,
        ads_query=ads_query,
        ads_rows=ads_rows,
        ads_token=ads_token,
        headers=headers,
        timeout=timeout,
        vizier_fetcher=vizier_fetcher,
        simbad_fetcher=simbad_fetcher,
        ligo_fetcher=ligo_fetcher,
        ads_fetcher=ads_fetcher,
        vacuum=vacuum,
    )


__all__ = [
    "ADSLiteratureRecord",
    "DEFAULT_ADS_ENDPOINT",
    "DEFAULT_GWOSC_API_ROOT",
    "DEFAULT_GWOSC_CATALOG",
    "DEFAULT_SIMBAD_ENDPOINT",
    "DEFAULT_VIZIER_ENDPOINT",
    "JWSTFormationObservation",
    "LIGOStrainObservation",
    "LiveVerificationBridge",
    "ObservationTensionComponent",
    "SimbadObjectRecord",
    "TheoryKernelProfile",
    "TheoryTensionScore",
    "build_ads_search_url",
    "build_gwosc_catalog_events_url",
    "build_simbad_identifier_url",
    "build_vizier_catalog_url",
    "calculate_live_tension_score",
]
