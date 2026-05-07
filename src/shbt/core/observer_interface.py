from __future__ import annotations

"""Observer-facing UI that turns boundary bit-loadings into bulk observables.

This module closes the loop between the repository's logical ``Saying`` layer
and the manifested ``Getting`` layer. The core rule is deliberately simple:

- a boundary sequence stores weighted microstate loadings,
- ``GET_OBSERVABLE`` retrieves one entropy-limited boundary register, and
- the retrieved microstate crystallizes into a coarse bulk measurement.

The resulting map is intentionally non-invertible. Multiple fine-grained
boundary microstates are packetized into the same coarse observable bin, so a
classical measurement exposes only the manifested bulk value rather than the
full underlying boundary sequence.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal, localcontext
from fractions import Fraction

from shbt.core.correspondence_engine import (
    DEFAULT_PRECISION as CORRESPONDENCE_DEFAULT_PRECISION,
    BulkWavefunction,
    PointerState,
    PointerStateSelector,
)
from shbt.core.measurement_operator import (
    DEFAULT_PRECISION as MEASUREMENT_DEFAULT_PRECISION,
    BoundaryDatum,
    HorizonEntropyBudget,
    MeasurementCollapseAudit,
    MeasurementCommand,
    MeasurementOperator,
)


GET_OBSERVABLE = "GET_OBSERVABLE"
DEFAULT_MICROSTATE_RESOLUTION = 64
DEFAULT_PRECISION = max(int(MEASUREMENT_DEFAULT_PRECISION), int(CORRESPONDENCE_DEFAULT_PRECISION))
_GUARD_DIGITS = 12
_SCALAR_SEQUENCE_TYPES = (str, bytes, bytearray)


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


@dataclass(frozen=True)
class ObservableScale:
    zero_point: Decimal | Fraction | float | int | str = Decimal("0")
    step: Decimal | Fraction | float | int | str = Decimal("1")
    units: str = "arb"
    eigenvalues: tuple[object, ...] | None = None

    def values_for_count(self, count: int) -> tuple[object, ...]:
        if count <= 0:
            raise ValueError("count must be positive.")
        if self.eigenvalues is not None:
            if len(self.eigenvalues) != count:
                raise ValueError("Observable scale eigenvalues must match the bit-loading sequence length.")
            return tuple(self.eigenvalues)
        zero_point = _decimal(self.zero_point)
        step = _decimal(self.step)
        return tuple(zero_point + step * Decimal(index) for index in range(count))


@dataclass(frozen=True)
class BoundaryMicrostate:
    boundary_address: str
    microstate_label: str
    coarse_index: int
    coarse_label: str
    measurement_value: object
    units: str
    raw_bit_loading: Decimal
    normalized_bit_loading: Decimal


@dataclass(frozen=True)
class ObservableManifestation:
    observable_name: str
    boundary_address: str
    microstate_label: str
    coarse_label: str
    coarse_index: int
    measurement_value: object
    units: str


@dataclass(frozen=True)
class ObservableManifestationAudit:
    command: MeasurementCommand
    boundary_datum: BoundaryDatum
    raw_bit_loadings: tuple[Decimal, ...]
    normalized_bit_loadings: tuple[Decimal, ...]
    measurement_values: tuple[object, ...]
    microstate_counts: tuple[int, ...]
    collapse_audit: MeasurementCollapseAudit
    pointer_state: PointerState | None
    manifestation: ObservableManifestation | None

    @property
    def horizon_budget(self) -> HorizonEntropyBudget:
        return self.collapse_audit.horizon_budget

    @property
    def microstate_count(self) -> int:
        return sum(self.microstate_counts)

    @property
    def coarse_outcome_count(self) -> int:
        return len(self.raw_bit_loadings)

    @property
    def noninvertible_map(self) -> bool:
        return self.microstate_count > self.coarse_outcome_count

    @property
    def manifested(self) -> bool:
        return self.manifestation is not None and self.collapse_audit.collapse_completed

    @property
    def selected_preimage_size(self) -> int:
        if self.manifestation is None:
            return 0
        return self.microstate_counts[self.manifestation.coarse_index]

    @property
    def statement(self) -> str:
        return (
            "GET_OBSERVABLE closes Saying into Getting by collapsing a weighted boundary register "
            "into one classical bulk readout."
        )

    def assert_manifested(self) -> None:
        self.collapse_audit.assert_resolved()
        if self.pointer_state is None or not self.pointer_state.crystallizes:
            raise AssertionError("Boundary-to-bulk correspondence did not crystallize a classical observable.")
        if self.manifestation is None:
            raise AssertionError("GET_OBSERVABLE did not return a manifested bulk measurement.")


def _normalize_bit_loadings(
    bit_loading_sequence: Sequence[Decimal | Fraction | float | int | str],
) -> tuple[tuple[Decimal, ...], tuple[Decimal, ...]]:
    if not isinstance(bit_loading_sequence, Sequence) or isinstance(bit_loading_sequence, _SCALAR_SEQUENCE_TYPES):
        raise TypeError("bit_loading_sequence must be a non-scalar sequence of numeric loads.")

    raw_bit_loadings = tuple(_decimal(value) for value in bit_loading_sequence)
    if not raw_bit_loadings:
        raise ValueError("bit_loading_sequence must not be empty.")
    if any(value < 0 for value in raw_bit_loadings):
        raise ValueError("bit-loading entries must be non-negative.")

    total_loading = sum(raw_bit_loadings, start=Decimal("0"))
    if total_loading == 0:
        uniform_weight = Decimal("1") / Decimal(len(raw_bit_loadings))
        return raw_bit_loadings, tuple(uniform_weight for _ in raw_bit_loadings)

    return raw_bit_loadings, tuple(value / total_loading for value in raw_bit_loadings)


def _resolve_measurement_values(
    *,
    count: int,
    scale: ObservableScale,
    measurement_values: Sequence[object] | None,
) -> tuple[object, ...]:
    if measurement_values is None:
        return scale.values_for_count(count)
    if not isinstance(measurement_values, Sequence) or isinstance(measurement_values, _SCALAR_SEQUENCE_TYPES):
        raise TypeError("measurement_values must be a non-scalar sequence when provided.")
    resolved_values = tuple(measurement_values)
    if len(resolved_values) != count:
        raise ValueError("measurement_values must match the bit-loading sequence length.")
    return resolved_values


def _allocate_microstate_counts(
    normalized_bit_loadings: tuple[Decimal, ...],
    *,
    total_microstates: int,
) -> tuple[int, ...]:
    resolved_total_microstates = max(int(total_microstates), len(normalized_bit_loadings))
    positive_indices = [index for index, loading in enumerate(normalized_bit_loadings) if loading > 0]
    if not positive_indices:
        raise ValueError("At least one normalized bit loading must be positive.")

    counts = [1 if index in positive_indices else 0 for index in range(len(normalized_bit_loadings))]
    remaining_microstates = resolved_total_microstates - len(positive_indices)
    if remaining_microstates <= 0:
        return tuple(counts)

    positive_total = sum((normalized_bit_loadings[index] for index in positive_indices), start=Decimal("0"))
    quotas: list[tuple[int, Decimal, Decimal]] = []
    assigned = 0
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
        remaining_decimal = Decimal(remaining_microstates)
        for index in positive_indices:
            scaled_quota = normalized_bit_loadings[index] * remaining_decimal / positive_total
            floor_quota = int(scaled_quota.to_integral_value(rounding=ROUND_FLOOR))
            counts[index] += floor_quota
            assigned += floor_quota
            quotas.append((index, scaled_quota - Decimal(floor_quota), normalized_bit_loadings[index]))

    leftover = remaining_microstates - assigned
    for index, _remainder, _weight in sorted(quotas, key=lambda item: (-item[1], -item[2], item[0]))[:leftover]:
        counts[index] += 1
    return tuple(counts)


def _build_boundary_payload(
    *,
    observable_name: str,
    boundary_address: str,
    raw_bit_loadings: tuple[Decimal, ...],
    normalized_bit_loadings: tuple[Decimal, ...],
    measurement_values: tuple[object, ...],
    microstate_counts: tuple[int, ...],
    units: str,
) -> Mapping[str, BoundaryMicrostate]:
    payload: dict[str, BoundaryMicrostate] = {}
    for index, (raw_loading, normalized_loading, measurement_value, microstate_count) in enumerate(
        zip(raw_bit_loadings, normalized_bit_loadings, measurement_values, microstate_counts, strict=True)
    ):
        coarse_label = f"{observable_name}[{index}]"
        for microstate_index in range(microstate_count):
            microstate_label = f"{coarse_label}::microstate[{microstate_index}]"
            payload[microstate_label] = BoundaryMicrostate(
                boundary_address=boundary_address,
                microstate_label=microstate_label,
                coarse_index=index,
                coarse_label=coarse_label,
                measurement_value=measurement_value,
                units=units,
                raw_bit_loading=raw_loading,
                normalized_bit_loading=normalized_loading,
            )
    if not payload:
        raise ValueError("Boundary payload must contain at least one microstate.")
    return payload


class ObserverInterface:
    """UI of the universe: manifest bulk observables from boundary GETs."""

    def __init__(
        self,
        *,
        observer_radius_m: Decimal | Fraction | float | int | str = Decimal("0"),
        bit_budget: Decimal | Fraction | float | int | str | None = None,
        global_horizon_radius_m: Decimal | Fraction | float | int | str | None = None,
        planck_length_m: Decimal | Fraction | float | int | str | None = None,
        microstate_resolution: int = DEFAULT_MICROSTATE_RESOLUTION,
        precision: int = DEFAULT_PRECISION,
    ) -> None:
        self.precision = max(int(precision), DEFAULT_PRECISION)
        self.microstate_resolution = max(int(microstate_resolution), 1)
        self.measurement_operator = MeasurementOperator(
            observer_radius_m=observer_radius_m,
            bit_budget=bit_budget,
            global_horizon_radius_m=global_horizon_radius_m,
            planck_length_m=planck_length_m,
            precision=self.precision,
        )
        self.pointer_state_selector = PointerStateSelector(precision=self.precision)

    @property
    def observer_radius_m(self) -> Decimal:
        return self.measurement_operator.observer_radius_m

    @property
    def horizon_budget(self) -> HorizonEntropyBudget:
        return self.measurement_operator.horizon_budget

    def move_to(self, observer_radius_m: Decimal | Fraction | float | int | str) -> HorizonEntropyBudget:
        return self.measurement_operator.move_to(observer_radius_m)

    def build_boundary_datum(
        self,
        bit_loading_sequence: Sequence[Decimal | Fraction | float | int | str],
        *,
        observable_name: str,
        boundary_address: str,
        scale: ObservableScale | None = None,
        measurement_values: Sequence[object] | None = None,
        microstate_resolution: int | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> tuple[BoundaryDatum, tuple[Decimal, ...], tuple[Decimal, ...], tuple[object, ...], tuple[int, ...]]:
        raw_bit_loadings, normalized_bit_loadings = _normalize_bit_loadings(bit_loading_sequence)
        resolved_scale = ObservableScale() if scale is None else scale
        resolved_measurement_values = _resolve_measurement_values(
            count=len(raw_bit_loadings),
            scale=resolved_scale,
            measurement_values=measurement_values,
        )
        resolved_microstate_counts = _allocate_microstate_counts(
            normalized_bit_loadings,
            total_microstates=self.microstate_resolution if microstate_resolution is None else microstate_resolution,
        )
        payload = _build_boundary_payload(
            observable_name=str(observable_name),
            boundary_address=str(boundary_address),
            raw_bit_loadings=raw_bit_loadings,
            normalized_bit_loadings=normalized_bit_loadings,
            measurement_values=resolved_measurement_values,
            microstate_counts=resolved_microstate_counts,
            units=str(resolved_scale.units),
        )
        combined_metadata = {
            "observable_name": str(observable_name),
            "units": str(resolved_scale.units),
            "raw_bit_loadings": raw_bit_loadings,
            "normalized_bit_loadings": normalized_bit_loadings,
            "microstate_counts": resolved_microstate_counts,
        }
        if metadata is not None:
            combined_metadata.update(dict(metadata))
        return (
            BoundaryDatum(address=str(boundary_address), payload=payload, metadata=combined_metadata),
            raw_bit_loadings,
            normalized_bit_loadings,
            resolved_measurement_values,
            resolved_microstate_counts,
        )

    def get_observable(
        self,
        bit_loading_sequence: Sequence[Decimal | Fraction | float | int | str],
        *,
        observable_name: str,
        boundary_address: str = "boundary.observable",
        scale: ObservableScale | None = None,
        measurement_values: Sequence[object] | None = None,
        requested_entropy_bits: Decimal | Fraction | float | int | str | None = None,
        microstate_resolution: int | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ObservableManifestationAudit:
        boundary_datum, raw_bit_loadings, normalized_bit_loadings, resolved_measurement_values, resolved_microstate_counts = (
            self.build_boundary_datum(
                bit_loading_sequence,
                observable_name=observable_name,
                boundary_address=boundary_address,
                scale=scale,
                measurement_values=measurement_values,
                microstate_resolution=microstate_resolution,
                metadata=metadata,
            )
        )
        command = MeasurementCommand(
            observable_name=str(observable_name),
            boundary_address=boundary_datum.address,
            requested_entropy_bits=None if requested_entropy_bits is None else _decimal(requested_entropy_bits),
            metadata={} if metadata is None else dict(metadata),
            verb=GET_OBSERVABLE,
        )
        collapse_audit = self.measurement_operator.execute(command, (boundary_datum,))

        pointer_state = None
        manifestation = None
        if collapse_audit.selected_outcome is not None:
            selected_payload = collapse_audit.selected_outcome.payload
            if not isinstance(selected_payload, BoundaryMicrostate):
                raise TypeError("GET_OBSERVABLE expected a BoundaryMicrostate payload.")
            wavefunction = BulkWavefunction(
                label=selected_payload.microstate_label,
                amplitude=selected_payload.normalized_bit_loading,
                c_vis=-selected_payload.normalized_bit_loading,
                c_dark=selected_payload.normalized_bit_loading,
                observable_family=str(observable_name),
            )
            pointer_state = self.pointer_state_selector.evaluate_wavefunction(wavefunction)
            crystallized_observable = self.pointer_state_selector.crystallize_classical_observable(
                wavefunction,
                selected_payload.measurement_value,
            )
            manifestation = ObservableManifestation(
                observable_name=str(observable_name),
                boundary_address=boundary_datum.address,
                microstate_label=selected_payload.microstate_label,
                coarse_label=selected_payload.coarse_label,
                coarse_index=selected_payload.coarse_index,
                measurement_value=crystallized_observable,
                units=selected_payload.units,
            )

        return ObservableManifestationAudit(
            command=command,
            boundary_datum=boundary_datum,
            raw_bit_loadings=raw_bit_loadings,
            normalized_bit_loadings=normalized_bit_loadings,
            measurement_values=resolved_measurement_values,
            microstate_counts=resolved_microstate_counts,
            collapse_audit=collapse_audit,
            pointer_state=pointer_state,
            manifestation=manifestation,
        )

    def collapse_wavefunction(
        self,
        bit_loading_sequence: Sequence[Decimal | Fraction | float | int | str],
        **kwargs: object,
    ) -> ObservableManifestationAudit:
        return self.get_observable(bit_loading_sequence, **kwargs)

    def measure(self, bit_loading_sequence: Sequence[Decimal | Fraction | float | int | str], **kwargs: object) -> ObservableManifestationAudit:
        return self.get_observable(bit_loading_sequence, **kwargs)


def get_observable(
    bit_loading_sequence: Sequence[Decimal | Fraction | float | int | str],
    *,
    observable_name: str,
    boundary_address: str = "boundary.observable",
    scale: ObservableScale | None = None,
    measurement_values: Sequence[object] | None = None,
    requested_entropy_bits: Decimal | Fraction | float | int | str | None = None,
    microstate_resolution: int = DEFAULT_MICROSTATE_RESOLUTION,
    observer_radius_m: Decimal | Fraction | float | int | str = Decimal("0"),
    bit_budget: Decimal | Fraction | float | int | str | None = None,
    global_horizon_radius_m: Decimal | Fraction | float | int | str | None = None,
    planck_length_m: Decimal | Fraction | float | int | str | None = None,
    precision: int = DEFAULT_PRECISION,
    metadata: Mapping[str, object] | None = None,
) -> ObservableManifestationAudit:
    interface = ObserverInterface(
        observer_radius_m=observer_radius_m,
        bit_budget=bit_budget,
        global_horizon_radius_m=global_horizon_radius_m,
        planck_length_m=planck_length_m,
        microstate_resolution=microstate_resolution,
        precision=precision,
    )
    return interface.get_observable(
        bit_loading_sequence,
        observable_name=observable_name,
        boundary_address=boundary_address,
        scale=scale,
        measurement_values=measurement_values,
        requested_entropy_bits=requested_entropy_bits,
        metadata=metadata,
    )


__all__ = [
    "BoundaryMicrostate",
    "DEFAULT_MICROSTATE_RESOLUTION",
    "DEFAULT_PRECISION",
    "GET_OBSERVABLE",
    "ObservableManifestation",
    "ObservableManifestationAudit",
    "ObservableScale",
    "ObserverInterface",
    "get_observable",
]
