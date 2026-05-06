from __future__ import annotations

"""Observer-horizon bookkeeping for local holographic limits.

This module packages a compact observer-facing layer on top of the existing
branch-fixed holographic bookkeeping. It exposes two complementary statements:

- a local observer's coordinate horizon inherits a finite bit budget that is
  reduced by the remaining radial distance to the horizon, and
- the benchmark's published visible-level moat can be dressed by that local
  horizon fraction to quantify how rapidly framing defects are amplified near
  the observer's horizon.
"""

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction

from shbt.constants import HOLOGRAPHIC_BITS, LEPTON_LEVEL, PARENT_LEVEL, PLANCK_LENGTH_M, QUARK_LEVEL
from shbt.core.entropy_kernel import bekenstein_hawking_bound
from shbt.core.noether_bridge import framing_defect


DEFAULT_PRECISION = 80
_GUARD_DIGITS = 12
_DECIMAL_PI = Decimal("3.14159265358979323846264338327950288419716939937510")
PUBLISHED_VISIBLE_MOAT_RADIUS = 2
BENCHMARK_BRANCH = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


def _fraction_to_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


@dataclass(frozen=True)
class ObserverHorizonLimit:
    global_horizon_radius_m: Decimal
    observer_radius_m: Decimal
    coordinate_horizon_radius_m: Decimal
    relative_position: Decimal
    remaining_horizon_fraction: Decimal
    exposed_area_fraction: Decimal
    local_horizon_area_m2: Decimal
    bekenstein_hawking_entropy_bits: Decimal
    local_available_bits: Decimal
    surface_bit_loading_bits_per_m2: Decimal
    log_horizon_loading_factor: Decimal


@dataclass(frozen=True)
class ObserverMoatAudit:
    benchmark_branch: tuple[int, int, int]
    evaluated_branch: tuple[int, int, int]
    published_visible_moat_radius: int
    branch_chebyshev_distance: int
    fixed_parent_locked: bool
    inside_published_visible_moat: bool
    benchmark_branch_selected: bool
    framing_defect_fraction: Fraction
    observer_relative_position: Decimal
    remaining_horizon_fraction: Decimal
    moat_penalty_factor: Decimal
    observer_shifted_defect: Decimal

    @property
    def observer_moat_locked(self) -> bool:
        return self.benchmark_branch_selected and self.framing_defect_fraction == 0


def global_coordinate_horizon_radius(
    *,
    bit_count: Decimal | Fraction | float | int | str = HOLOGRAPHIC_BITS,
    precision: int = DEFAULT_PRECISION,
) -> Decimal:
    resolved_bit_count = _decimal(bit_count)
    if resolved_bit_count <= 0:
        raise ValueError("bit_count must be positive.")

    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        planck_length_m = _decimal(PLANCK_LENGTH_M)
        horizon_area_m2 = Decimal(4) * planck_length_m * planck_length_m * resolved_bit_count
        horizon_radius_m = (horizon_area_m2 / (Decimal(4) * _DECIMAL_PI)).sqrt()
        context.prec = max(int(precision), DEFAULT_PRECISION)
        return +horizon_radius_m


def calculate_observer_horizon_limit(
    observer_radius_m: Decimal | Fraction | float | int | str = Decimal("0"),
    *,
    global_horizon_radius_m: Decimal | Fraction | float | int | str | None = None,
    bit_count: Decimal | Fraction | float | int | str = HOLOGRAPHIC_BITS,
    precision: int = DEFAULT_PRECISION,
) -> ObserverHorizonLimit:
    resolved_bit_count = _decimal(bit_count)
    resolved_global_radius = (
        global_coordinate_horizon_radius(bit_count=resolved_bit_count, precision=precision)
        if global_horizon_radius_m is None
        else _decimal(global_horizon_radius_m)
    )
    resolved_observer_radius = _decimal(observer_radius_m)

    if resolved_global_radius <= 0:
        raise ValueError("global_horizon_radius_m must be positive.")
    if resolved_observer_radius < 0:
        raise ValueError("observer_radius_m must be non-negative.")
    if resolved_observer_radius >= resolved_global_radius:
        raise ValueError("observer_radius_m must lie strictly inside the coordinate horizon.")

    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        coordinate_horizon_radius_m = resolved_global_radius - resolved_observer_radius
        relative_position = resolved_observer_radius / resolved_global_radius
        remaining_horizon_fraction = coordinate_horizon_radius_m / resolved_global_radius
        exposed_area_fraction = remaining_horizon_fraction * remaining_horizon_fraction
        local_horizon_area_m2 = Decimal(4) * _DECIMAL_PI * coordinate_horizon_radius_m * coordinate_horizon_radius_m
        local_available_bits = resolved_bit_count * exposed_area_fraction
        if local_available_bits <= 1:
            raise ValueError("observer horizon leaves fewer than one effective holographic bit.")
        bekenstein_bound = bekenstein_hawking_bound(local_horizon_area_m2, precision=max(int(precision), DEFAULT_PRECISION))
        surface_bit_loading = local_available_bits / local_horizon_area_m2
        log_horizon_loading_factor = (Decimal(3) * Decimal(10).ln()) / (Decimal(2) * local_available_bits.ln())
        context.prec = max(int(precision), DEFAULT_PRECISION)
        return ObserverHorizonLimit(
            global_horizon_radius_m=+resolved_global_radius,
            observer_radius_m=+resolved_observer_radius,
            coordinate_horizon_radius_m=+coordinate_horizon_radius_m,
            relative_position=+relative_position,
            remaining_horizon_fraction=+remaining_horizon_fraction,
            exposed_area_fraction=+exposed_area_fraction,
            local_horizon_area_m2=+local_horizon_area_m2,
            bekenstein_hawking_entropy_bits=+bekenstein_bound.entropy_bits,
            local_available_bits=+local_available_bits,
            surface_bit_loading_bits_per_m2=+surface_bit_loading,
            log_horizon_loading_factor=+log_horizon_loading_factor,
        )


def audit_observer_holographic_moat(
    observer_radius_m: Decimal | Fraction | float | int | str = Decimal("0"),
    *,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    parent_level: int = PARENT_LEVEL,
    global_horizon_radius_m: Decimal | Fraction | float | int | str | None = None,
    bit_count: Decimal | Fraction | float | int | str = HOLOGRAPHIC_BITS,
    precision: int = DEFAULT_PRECISION,
) -> ObserverMoatAudit:
    horizon_limit = calculate_observer_horizon_limit(
        observer_radius_m,
        global_horizon_radius_m=global_horizon_radius_m,
        bit_count=bit_count,
        precision=precision,
    )
    resolved_branch = (int(lepton_level), int(quark_level), int(parent_level))
    benchmark_lepton, benchmark_quark, benchmark_parent = BENCHMARK_BRANCH
    branch_distance = max(abs(resolved_branch[0] - benchmark_lepton), abs(resolved_branch[1] - benchmark_quark))
    fixed_parent_locked = resolved_branch[2] == benchmark_parent
    defect = framing_defect(resolved_branch[2], resolved_branch[0], resolved_branch[1]).delta_fr

    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        moat_penalty_factor = Decimal(1) / max(
            horizon_limit.remaining_horizon_fraction,
            Decimal(1) / _decimal(bit_count),
        )
        observer_shifted_defect = abs(_fraction_to_decimal(defect)) * moat_penalty_factor
        context.prec = max(int(precision), DEFAULT_PRECISION)
        return ObserverMoatAudit(
            benchmark_branch=BENCHMARK_BRANCH,
            evaluated_branch=resolved_branch,
            published_visible_moat_radius=PUBLISHED_VISIBLE_MOAT_RADIUS,
            branch_chebyshev_distance=branch_distance,
            fixed_parent_locked=fixed_parent_locked,
            inside_published_visible_moat=bool(
                fixed_parent_locked and branch_distance <= PUBLISHED_VISIBLE_MOAT_RADIUS
            ),
            benchmark_branch_selected=resolved_branch == BENCHMARK_BRANCH,
            framing_defect_fraction=defect,
            observer_relative_position=+horizon_limit.relative_position,
            remaining_horizon_fraction=+horizon_limit.remaining_horizon_fraction,
            moat_penalty_factor=+moat_penalty_factor,
            observer_shifted_defect=+observer_shifted_defect,
        )


__all__ = [
    "BENCHMARK_BRANCH",
    "DEFAULT_PRECISION",
    "ObserverHorizonLimit",
    "ObserverMoatAudit",
    "PUBLISHED_VISIBLE_MOAT_RADIUS",
    "audit_observer_holographic_moat",
    "calculate_observer_horizon_limit",
    "global_coordinate_horizon_radius",
]
