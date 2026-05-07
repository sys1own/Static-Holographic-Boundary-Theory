from __future__ import annotations

"""Measurement layer turning boundary statements into bulk ``GET`` events.

This module formalizes the repository's transition from mathematical
``Saying`` to physical ``Getting``:

- the boundary stores a register of candidate data,
- a bulk observer only controls a finite local horizon sub-budget of the
  global holographic register, and
- a measurement is therefore modeled as an entropy-limited ``GET`` command
  that retrieves an addressed boundary datum and collapses its local ensemble
  into a single classical readout.

The local horizon bookkeeping mirrors the observer-horizon logic already used
elsewhere in the codebase, but this module keeps a self-contained fallback so
that explicit bit-budget calculations remain available even when the benchmark
constant layer is not loaded.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import math


DEFAULT_PRECISION = 80
_GUARD_DIGITS = 12
_DECIMAL_PI = Decimal("3.14159265358979323846264338327950288419716939937510")
_DECIMAL_LN2 = Decimal("0.69314718055994530941723212145817656807550013436026")
_MIN_RETRIEVAL_COST_BITS = Decimal("1")
_SCALAR_SEQUENCE_TYPES = (str, bytes, bytearray)


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


def _decimal_ln(value: Decimal) -> Decimal:
    try:
        return value.ln()
    except AttributeError:
        return Decimal(str(math.log(float(value))))


def _decimal_log2_integer(size: int) -> Decimal:
    if size <= 0:
        raise ValueError("size must be positive.")
    if size == 1:
        return Decimal("0")
    return _decimal_ln(Decimal(size)) / _DECIMAL_LN2


def _benchmark_decimal(name: str) -> Decimal:
    try:
        from shbt import constants as _constants
    except Exception as exc:
        raise RuntimeError(
            f"Could not import benchmark constant '{name}'. Pass the corresponding argument explicitly."
        ) from exc
    return _decimal(getattr(_constants, name))


def _normalize_outcomes(payload: object) -> tuple[tuple[str, object], ...]:
    if isinstance(payload, Mapping):
        outcomes = tuple((str(label), value) for label, value in payload.items())
    elif isinstance(payload, Sequence) and not isinstance(payload, _SCALAR_SEQUENCE_TYPES):
        outcomes = tuple((str(index), value) for index, value in enumerate(payload))
    else:
        outcomes = (("resolved", payload),)
    if not outcomes:
        raise ValueError("Boundary ensemble must contain at least one candidate outcome.")
    return outcomes


def _collapse_index(
    *,
    observable_name: str,
    boundary_address: str,
    horizon_budget: "HorizonEntropyBudget",
    outcome_count: int,
) -> int:
    if outcome_count <= 1:
        return 0
    seed = "|".join(
        (
            observable_name,
            boundary_address,
            format(horizon_budget.remaining_horizon_fraction, "f"),
            format(horizon_budget.local_available_bits, "f"),
            str(outcome_count),
        )
    )
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % outcome_count


@dataclass(frozen=True)
class BoundaryDatum:
    address: str
    payload: object
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def candidate_outcomes(self) -> tuple[tuple[str, object], ...]:
        return _normalize_outcomes(self.payload)

    @property
    def ensemble_size(self) -> int:
        return len(self.candidate_outcomes)

    @property
    def ensemble_entropy_bits(self) -> Decimal:
        return _decimal_log2_integer(self.ensemble_size)


@dataclass(frozen=True)
class MeasurementCommand:
    observable_name: str
    boundary_address: str
    requested_entropy_bits: Decimal | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    verb: str = "GET"


@dataclass(frozen=True)
class BoundaryOutcome:
    boundary_address: str
    label: str
    payload: object
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class HorizonEntropyBudget:
    total_bit_budget: Decimal
    global_horizon_radius_m: Decimal
    observer_radius_m: Decimal
    local_horizon_radius_m: Decimal
    remaining_horizon_fraction: Decimal
    exposed_area_fraction: Decimal
    local_horizon_area_m2: Decimal
    local_available_bits: Decimal
    hidden_bits: Decimal
    bekenstein_hawking_entropy_bits: Decimal
    surface_bit_loading_bits_per_m2: Decimal
    log_horizon_loading_factor: Decimal

    @property
    def local_access_fraction(self) -> Decimal:
        return self.local_available_bits / self.total_bit_budget

    @property
    def entropy_limit_bits(self) -> Decimal:
        return min(self.local_available_bits, self.bekenstein_hawking_entropy_bits)

    @property
    def horizon_is_resolved(self) -> bool:
        return self.local_available_bits > _MIN_RETRIEVAL_COST_BITS


@dataclass(frozen=True)
class MeasurementCollapseAudit:
    command: MeasurementCommand
    horizon_budget: HorizonEntropyBudget
    available_boundary_addresses: tuple[str, ...]
    candidate_outcome_labels: tuple[str, ...]
    address_entropy_bits: Decimal
    ensemble_entropy_bits: Decimal
    retrieval_cost_bits: Decimal
    selected_outcome: BoundaryOutcome | None
    collapse_index: int | None
    remaining_local_bits: Decimal
    entropy_budget_residual: Decimal

    @property
    def within_entropy_budget(self) -> bool:
        return self.entropy_budget_residual >= 0

    @property
    def collapse_completed(self) -> bool:
        return self.within_entropy_budget and self.selected_outcome is not None

    @property
    def saturation_fraction(self) -> Decimal:
        if self.horizon_budget.entropy_limit_bits == 0:
            return Decimal("0")
        return self.retrieval_cost_bits / self.horizon_budget.entropy_limit_bits

    @property
    def statement(self) -> str:
        return (
            "Wavefunction collapse is realized as an entropy-limited GET from the boundary; "
            "the observer's local horizon fixes how much of the global bit budget can become classical data."
        )

    def assert_resolved(self) -> None:
        if not self.within_entropy_budget:
            raise AssertionError(
                "Measurement exceeds the observer's local entropy budget: "
                f"cost={self.retrieval_cost_bits:.6e} bits, "
                f"limit={self.horizon_budget.entropy_limit_bits:.6e} bits."
            )
        if self.selected_outcome is None:
            raise AssertionError("Measurement did not return a classical boundary outcome.")


class MeasurementOperator:
    """Entropy-limited operator mediating bulk observers and boundary data.

    The operator interprets a measurement as a ``GET`` command with two coupled
    costs:

    - addressing cost: selecting one datum from the boundary register, and
    - collapse cost: resolving one concrete outcome from the addressed datum's
      local candidate ensemble.

    Both costs are charged against the observer's local horizon capacity, which
    is a finite sub-budget of the total holographic bit count.
    """

    def __init__(
        self,
        *,
        observer_radius_m: Decimal | Fraction | float | int | str = Decimal("0"),
        bit_budget: Decimal | Fraction | float | int | str | None = None,
        global_horizon_radius_m: Decimal | Fraction | float | int | str | None = None,
        planck_length_m: Decimal | Fraction | float | int | str | None = None,
        precision: int = DEFAULT_PRECISION,
    ) -> None:
        self.precision = max(int(precision), DEFAULT_PRECISION)
        self.total_bit_budget = (
            _benchmark_decimal("HOLOGRAPHIC_BITS")
            if bit_budget is None
            else _decimal(bit_budget)
        )
        if self.total_bit_budget <= 0:
            raise ValueError("bit_budget must be positive.")

        self._planck_length_m = None if global_horizon_radius_m is not None else (
            _benchmark_decimal("PLANCK_LENGTH_M")
            if planck_length_m is None
            else _decimal(planck_length_m)
        )
        self.global_horizon_radius_m = self._resolve_global_horizon_radius(global_horizon_radius_m)
        self._horizon_budget = self._resolve_horizon_budget(observer_radius_m)

    @property
    def observer_radius_m(self) -> Decimal:
        return self._horizon_budget.observer_radius_m

    @property
    def horizon_budget(self) -> HorizonEntropyBudget:
        return self._horizon_budget

    def move_to(
        self,
        observer_radius_m: Decimal | Fraction | float | int | str,
    ) -> HorizonEntropyBudget:
        self._horizon_budget = self._resolve_horizon_budget(observer_radius_m)
        return self._horizon_budget

    def measure(
        self,
        boundary_register: Mapping[str, object] | Sequence[BoundaryDatum | tuple[str, object]],
        *,
        observable_name: str,
        boundary_address: str,
        requested_entropy_bits: Decimal | Fraction | float | int | str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> MeasurementCollapseAudit:
        command = MeasurementCommand(
            observable_name=str(observable_name),
            boundary_address=str(boundary_address),
            requested_entropy_bits=None if requested_entropy_bits is None else _decimal(requested_entropy_bits),
            metadata={} if metadata is None else dict(metadata),
        )
        return self.execute(command, boundary_register)

    def collapse(
        self,
        boundary_register: Mapping[str, object] | Sequence[BoundaryDatum | tuple[str, object]],
        *,
        observable_name: str,
        boundary_address: str,
        requested_entropy_bits: Decimal | Fraction | float | int | str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> MeasurementCollapseAudit:
        return self.measure(
            boundary_register,
            observable_name=observable_name,
            boundary_address=boundary_address,
            requested_entropy_bits=requested_entropy_bits,
            metadata=metadata,
        )

    def get(
        self,
        boundary_register: Mapping[str, object] | Sequence[BoundaryDatum | tuple[str, object]],
        *,
        observable_name: str,
        boundary_address: str,
        requested_entropy_bits: Decimal | Fraction | float | int | str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> MeasurementCollapseAudit:
        return self.measure(
            boundary_register,
            observable_name=observable_name,
            boundary_address=boundary_address,
            requested_entropy_bits=requested_entropy_bits,
            metadata=metadata,
        )

    def execute(
        self,
        command: MeasurementCommand,
        boundary_register: Mapping[str, object] | Sequence[BoundaryDatum | tuple[str, object]],
    ) -> MeasurementCollapseAudit:
        register = self._normalize_boundary_register(boundary_register)
        if command.boundary_address not in register:
            available = ", ".join(sorted(register))
            raise KeyError(
                f"Boundary address '{command.boundary_address}' not present in register. Available addresses: {available}."
            )

        datum = register[command.boundary_address]
        candidate_outcomes = datum.candidate_outcomes
        address_entropy_bits = _decimal_log2_integer(len(register))
        ensemble_entropy_bits = datum.ensemble_entropy_bits
        explicit_cost = Decimal("0") if command.requested_entropy_bits is None else _decimal(command.requested_entropy_bits)
        if explicit_cost < 0:
            raise ValueError("requested_entropy_bits must be non-negative.")
        retrieval_cost_bits = max(
            _MIN_RETRIEVAL_COST_BITS,
            address_entropy_bits + ensemble_entropy_bits,
            explicit_cost,
        )

        entropy_budget_residual = self._horizon_budget.entropy_limit_bits - retrieval_cost_bits
        selected_outcome = None
        collapse_index = None
        if entropy_budget_residual >= 0:
            collapse_index = _collapse_index(
                observable_name=command.observable_name,
                boundary_address=command.boundary_address,
                horizon_budget=self._horizon_budget,
                outcome_count=len(candidate_outcomes),
            )
            outcome_label, outcome_payload = candidate_outcomes[collapse_index]
            combined_metadata = dict(datum.metadata)
            combined_metadata.update(dict(command.metadata))
            combined_metadata.update(
                {
                    "verb": command.verb,
                    "observable_name": command.observable_name,
                    "address_entropy_bits": address_entropy_bits,
                    "ensemble_entropy_bits": ensemble_entropy_bits,
                }
            )
            selected_outcome = BoundaryOutcome(
                boundary_address=command.boundary_address,
                label=outcome_label,
                payload=outcome_payload,
                metadata=combined_metadata,
            )

        remaining_local_bits = self._horizon_budget.entropy_limit_bits - retrieval_cost_bits
        return MeasurementCollapseAudit(
            command=command,
            horizon_budget=self._horizon_budget,
            available_boundary_addresses=tuple(sorted(register)),
            candidate_outcome_labels=tuple(label for label, _ in candidate_outcomes),
            address_entropy_bits=+address_entropy_bits,
            ensemble_entropy_bits=+ensemble_entropy_bits,
            retrieval_cost_bits=+retrieval_cost_bits,
            selected_outcome=selected_outcome,
            collapse_index=collapse_index,
            remaining_local_bits=+remaining_local_bits,
            entropy_budget_residual=+entropy_budget_residual,
        )

    def _normalize_boundary_register(
        self,
        boundary_register: Mapping[str, object] | Sequence[BoundaryDatum | tuple[str, object]],
    ) -> dict[str, BoundaryDatum]:
        if isinstance(boundary_register, Mapping):
            normalized = {
                str(address): BoundaryDatum(address=str(address), payload=payload)
                for address, payload in boundary_register.items()
            }
        elif isinstance(boundary_register, Sequence) and not isinstance(boundary_register, _SCALAR_SEQUENCE_TYPES):
            normalized = {}
            for entry in boundary_register:
                if isinstance(entry, BoundaryDatum):
                    datum = entry
                elif isinstance(entry, tuple) and len(entry) == 2:
                    address, payload = entry
                    datum = BoundaryDatum(address=str(address), payload=payload)
                else:
                    raise TypeError(
                        "Sequence boundary registers must contain BoundaryDatum entries or (address, payload) tuples."
                    )
                if datum.address in normalized:
                    raise ValueError(f"Duplicate boundary address '{datum.address}' in measurement register.")
                normalized[datum.address] = datum
        else:
            raise TypeError("boundary_register must be a mapping or a sequence of boundary entries.")

        if not normalized:
            raise ValueError("boundary_register must not be empty.")
        return normalized

    def _resolve_global_horizon_radius(
        self,
        global_horizon_radius_m: Decimal | Fraction | float | int | str | None,
    ) -> Decimal:
        if global_horizon_radius_m is not None:
            resolved_radius = _decimal(global_horizon_radius_m)
            if resolved_radius <= 0:
                raise ValueError("global_horizon_radius_m must be positive.")
            return resolved_radius

        if self._planck_length_m is None or self._planck_length_m <= 0:
            raise ValueError("planck_length_m must be positive when global_horizon_radius_m is omitted.")

        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            horizon_area_m2 = Decimal(4) * self._planck_length_m * self._planck_length_m * self.total_bit_budget
            horizon_radius_m = (horizon_area_m2 / (Decimal(4) * _DECIMAL_PI)).sqrt()
            context.prec = self.precision
            return +horizon_radius_m

    def _resolve_horizon_budget(
        self,
        observer_radius_m: Decimal | Fraction | float | int | str,
    ) -> HorizonEntropyBudget:
        resolved_observer_radius = _decimal(observer_radius_m)
        if resolved_observer_radius < 0:
            raise ValueError("observer_radius_m must be non-negative.")
        if resolved_observer_radius >= self.global_horizon_radius_m:
            raise ValueError("observer_radius_m must lie strictly inside the global horizon.")

        effective_planck_length = self._planck_length_m
        if effective_planck_length is None:
            effective_planck_length = _benchmark_decimal("PLANCK_LENGTH_M")

        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            local_horizon_radius_m = self.global_horizon_radius_m - resolved_observer_radius
            remaining_horizon_fraction = local_horizon_radius_m / self.global_horizon_radius_m
            exposed_area_fraction = remaining_horizon_fraction * remaining_horizon_fraction
            local_horizon_area_m2 = Decimal(4) * _DECIMAL_PI * local_horizon_radius_m * local_horizon_radius_m
            local_available_bits = self.total_bit_budget * exposed_area_fraction
            if local_available_bits <= _MIN_RETRIEVAL_COST_BITS:
                raise ValueError("Observer horizon leaves fewer than one effective holographic bit.")
            bekenstein_hawking_entropy_bits = local_horizon_area_m2 / (
                Decimal(4) * effective_planck_length * effective_planck_length * _DECIMAL_LN2
            )
            hidden_bits = self.total_bit_budget - local_available_bits
            surface_bit_loading = local_available_bits / local_horizon_area_m2
            log_horizon_loading_factor = (Decimal(3) * _decimal_ln(Decimal(10))) / (
                Decimal(2) * _decimal_ln(local_available_bits)
            )
            context.prec = self.precision
            return HorizonEntropyBudget(
                total_bit_budget=+self.total_bit_budget,
                global_horizon_radius_m=+self.global_horizon_radius_m,
                observer_radius_m=+resolved_observer_radius,
                local_horizon_radius_m=+local_horizon_radius_m,
                remaining_horizon_fraction=+remaining_horizon_fraction,
                exposed_area_fraction=+exposed_area_fraction,
                local_horizon_area_m2=+local_horizon_area_m2,
                local_available_bits=+local_available_bits,
                hidden_bits=+hidden_bits,
                bekenstein_hawking_entropy_bits=+bekenstein_hawking_entropy_bits,
                surface_bit_loading_bits_per_m2=+surface_bit_loading,
                log_horizon_loading_factor=+log_horizon_loading_factor,
            )


__all__ = [
    "BoundaryDatum",
    "BoundaryOutcome",
    "DEFAULT_PRECISION",
    "HorizonEntropyBudget",
    "MeasurementCollapseAudit",
    "MeasurementCommand",
    "MeasurementOperator",
]
