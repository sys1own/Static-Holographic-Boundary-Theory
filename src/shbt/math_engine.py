from __future__ import annotations

"""Guarded rational and interval arithmetic helpers for the SHBT audit surface.

The verifier keeps exact rational helpers for branch-fixed bookkeeping while
exposing a strict interval wrapper around ``mpmath`` for any transcendental or
residual-sensitive arithmetic. Interval results are exported as closed decimal
bounds ``[low, high]`` so sub-moat residues can be audited without silently
collapsing into point estimates.
"""

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
import re
from typing import Any, TypeAlias

import mpmath
from mpmath.libmp import to_str as _mpf_to_str


PRECISION_GUARD = 128
FIXED_POINT_DENOMINATOR = 2**PRECISION_GUARD
INTERVAL_PRECISION = max(PRECISION_GUARD * 2, 160)
MOAT_ABSOLUTE_BOUND = Decimal("1e-124")
_MICROSCOPIC_THRESHOLD = Fraction(1, 10**PRECISION_GUARD)
_INTERVAL_REPR_PATTERN = re.compile(r"^mpi\('([^']+)', '([^']+)'\)$")

GuardScalar: TypeAlias = Fraction | Decimal | int | float | str


def _coerce_decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        with localcontext() as context:
            context.prec = INTERVAL_PRECISION
            return Decimal(value.numerator) / Decimal(value.denominator)
    if isinstance(value, bool):
        return Decimal(int(value))
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(repr(value))
    if isinstance(value, mpmath.mpf):
        return Decimal(str(value))
    return Decimal(str(value))


@dataclass(frozen=True)
class IntervalBounds:
    low: Decimal
    high: Decimal

    def __post_init__(self) -> None:
        resolved_low = _coerce_decimal(self.low)
        resolved_high = _coerce_decimal(self.high)
        if resolved_low > resolved_high:
            raise ValueError("IntervalBounds requires low <= high.")
        object.__setattr__(self, "low", resolved_low)
        object.__setattr__(self, "high", resolved_high)

    def __iter__(self) -> Iterator[Decimal]:
        yield self.low
        yield self.high

    @property
    def width(self) -> Decimal:
        return self.high - self.low

    @property
    def midpoint(self) -> Decimal:
        with localcontext() as context:
            context.prec = INTERVAL_PRECISION
            return (self.low + self.high) / Decimal("2")

    def as_tuple(self) -> tuple[Decimal, Decimal]:
        return (self.low, self.high)

    def contains(self, value: GuardScalar) -> bool:
        resolved_value = _coerce_decimal(value)
        return self.low <= resolved_value <= self.high

    def contains_zero(self) -> bool:
        return self.contains(0)

    def within_absolute_bound(self, bound: GuardScalar = MOAT_ABSOLUTE_BOUND) -> bool:
        resolved_bound = abs(_coerce_decimal(bound))
        return max(abs(self.low), abs(self.high)) <= resolved_bound


IntervalOperand: TypeAlias = IntervalBounds | GuardScalar | tuple[GuardScalar, GuardScalar]

MOAT_INTERVAL = IntervalBounds(-MOAT_ABSOLUTE_BOUND, MOAT_ABSOLUTE_BOUND)


def to_rational(value: GuardScalar) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        return Fraction(value)
    if isinstance(value, bool):
        return Fraction(int(value), 1)
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction(Decimal(str(value)))
    try:
        return Fraction(value)
    except ValueError:
        return Fraction(Decimal(str(value)))


def _coerce_fraction(value: GuardScalar) -> Fraction:
    return to_rational(value)


@contextmanager
def _interval_precision_guard(dps: int = INTERVAL_PRECISION) -> Iterator[None]:
    target_precision = max(int(dps), PRECISION_GUARD)
    previous_mp_precision = mpmath.mp.dps
    previous_iv_precision = mpmath.iv.dps
    mpmath.mp.dps = max(previous_mp_precision, target_precision)
    mpmath.iv.dps = max(previous_iv_precision, target_precision)
    try:
        yield
    finally:
        mpmath.mp.dps = previous_mp_precision
        mpmath.iv.dps = previous_iv_precision


def _interval_from_mpmath(value: Any) -> IntervalBounds:
    repr_match = _INTERVAL_REPR_PATTERN.fullmatch(repr(value))
    if repr_match is not None:
        return IntervalBounds(Decimal(repr_match.group(1)), Decimal(repr_match.group(2)))
    low, high = value._mpi_
    digits = max(INTERVAL_PRECISION, int(mpmath.mp.dps), int(mpmath.iv.dps))
    return IntervalBounds(Decimal(_mpf_to_str(low, digits)), Decimal(_mpf_to_str(high, digits)))


def _scalar_to_iv_mpf(value: Any) -> Any:
    if hasattr(value, "_mpi_"):
        return value
    if isinstance(value, Fraction):
        return mpmath.iv.mpf(value.numerator) / mpmath.iv.mpf(value.denominator)
    if isinstance(value, bool):
        return mpmath.iv.mpf(int(value))
    if isinstance(value, int):
        return mpmath.iv.mpf(value)
    if isinstance(value, float):
        return mpmath.iv.mpf(repr(value))
    if isinstance(value, Decimal):
        return mpmath.iv.mpf(str(value))
    if isinstance(value, mpmath.mpf):
        return mpmath.iv.mpf(str(value))
    return mpmath.iv.mpf(str(value))


def _coerce_iv_mpf(value: IntervalOperand | Any) -> Any:
    if isinstance(value, IntervalBounds):
        return mpmath.iv.mpf([str(value.low), str(value.high)])
    if hasattr(value, "_mpi_"):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2 and not isinstance(value, (str, bytes)):
        low = _interval_from_mpmath(_scalar_to_iv_mpf(value[0]))
        high = _interval_from_mpmath(_scalar_to_iv_mpf(value[1]))
        resolved_bounds = IntervalBounds(low.low, high.high)
        return mpmath.iv.mpf([str(resolved_bounds.low), str(resolved_bounds.high)])
    return _scalar_to_iv_mpf(value)


def guard_fraction(numerator: GuardScalar, denominator: GuardScalar = 1) -> Fraction:
    resolved_denominator = _coerce_fraction(denominator)
    if resolved_denominator == 0:
        raise ZeroDivisionError("guard_fraction denominator must be non-zero.")
    return _coerce_fraction(numerator) / resolved_denominator


def guard_sum(iterable: Iterable[GuardScalar]) -> Fraction:
    total = Fraction(0, 1)
    for value in iterable:
        total += _coerce_fraction(value)
    return total


def guard_interval(value: IntervalOperand, *, dps: int = INTERVAL_PRECISION) -> IntervalBounds:
    if isinstance(value, IntervalBounds):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2 and not isinstance(value, (str, bytes)):
        return IntervalBounds(_coerce_decimal(value[0]), _coerce_decimal(value[1]))
    with _interval_precision_guard(dps):
        return _interval_from_mpmath(_coerce_iv_mpf(value))


def guard_interval_sum(iterable: Iterable[IntervalOperand], *, dps: int = INTERVAL_PRECISION) -> IntervalBounds:
    with _interval_precision_guard(dps):
        total = mpmath.iv.mpf("0")
        for value in iterable:
            total += _coerce_iv_mpf(value)
        return _interval_from_mpmath(total)


def interval_apply(
    function: str | Callable[..., Any],
    *args: IntervalOperand,
    dps: int = INTERVAL_PRECISION,
    **kwargs: Any,
) -> IntervalBounds:
    with _interval_precision_guard(dps):
        resolved_function = getattr(mpmath.iv, function) if isinstance(function, str) else function
        result = resolved_function(*(_coerce_iv_mpf(argument) for argument in args), **kwargs)
        return _interval_from_mpmath(result)


def interval_add(lhs: IntervalOperand, rhs: IntervalOperand, *, dps: int = INTERVAL_PRECISION) -> IntervalBounds:
    return interval_apply(lambda left, right: left + right, lhs, rhs, dps=dps)


def interval_sub(lhs: IntervalOperand, rhs: IntervalOperand, *, dps: int = INTERVAL_PRECISION) -> IntervalBounds:
    return interval_apply(lambda left, right: left - right, lhs, rhs, dps=dps)


def interval_mul(lhs: IntervalOperand, rhs: IntervalOperand, *, dps: int = INTERVAL_PRECISION) -> IntervalBounds:
    return interval_apply(lambda left, right: left * right, lhs, rhs, dps=dps)


def interval_div(lhs: IntervalOperand, rhs: IntervalOperand, *, dps: int = INTERVAL_PRECISION) -> IntervalBounds:
    return interval_apply(lambda left, right: left / right, lhs, rhs, dps=dps)


def interval_power(
    base: IntervalOperand,
    exponent: IntervalOperand | int,
    *,
    dps: int = INTERVAL_PRECISION,
) -> IntervalBounds:
    with _interval_precision_guard(dps):
        resolved_base = _coerce_iv_mpf(base)
        resolved_exponent = exponent if isinstance(exponent, int) and not isinstance(exponent, bool) else _coerce_iv_mpf(exponent)
        result = resolved_base**resolved_exponent
        return _interval_from_mpmath(result)


def interval_abs(value: IntervalOperand, *, dps: int = INTERVAL_PRECISION) -> IntervalBounds:
    return interval_apply(lambda operand: abs(operand), value, dps=dps)


def residue_interval(lhs: IntervalOperand, rhs: IntervalOperand, *, dps: int = INTERVAL_PRECISION) -> IntervalBounds:
    return interval_sub(lhs, rhs, dps=dps)


def absolute_residue_interval(lhs: IntervalOperand, rhs: IntervalOperand, *, dps: int = INTERVAL_PRECISION) -> IntervalBounds:
    return interval_abs(residue_interval(lhs, rhs, dps=dps), dps=dps)


def moat_contains(value: IntervalOperand, *, bound: GuardScalar = MOAT_ABSOLUTE_BOUND, dps: int = INTERVAL_PRECISION) -> bool:
    return guard_interval(value, dps=dps).within_absolute_bound(bound)


def is_guard_zero(value: GuardScalar | IntervalBounds) -> bool:
    if isinstance(value, IntervalBounds):
        threshold = _coerce_decimal(_MICROSCOPIC_THRESHOLD)
        return value.contains_zero() and value.within_absolute_bound(threshold)
    resolved_value = _coerce_fraction(value)
    return resolved_value == 0 or abs(resolved_value) <= _MICROSCOPIC_THRESHOLD


__all__ = [
    "FIXED_POINT_DENOMINATOR",
    "INTERVAL_PRECISION",
    "IntervalBounds",
    "MOAT_ABSOLUTE_BOUND",
    "MOAT_INTERVAL",
    "PRECISION_GUARD",
    "absolute_residue_interval",
    "guard_fraction",
    "guard_interval",
    "guard_interval_sum",
    "guard_sum",
    "interval_abs",
    "interval_add",
    "interval_apply",
    "interval_div",
    "interval_mul",
    "interval_power",
    "interval_sub",
    "is_guard_zero",
    "moat_contains",
    "residue_interval",
    "to_rational",
]
