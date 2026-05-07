from __future__ import annotations

"""Exact rational helpers for the guarded SHBT arithmetic surface.

The physical-audit stack occasionally needs bit-identical accumulation for
small moat residues and threshold checks. This module keeps that surface tiny
and stdlib-only by normalizing values into ``fractions.Fraction`` instances.
Those results remain compatible with the rest of the verifier stack because
``Fraction`` cleanly converts to ``float`` wherever legacy code still expects
floating-point inputs.
"""

from collections.abc import Iterable
from decimal import Decimal
from fractions import Fraction
from typing import TypeAlias


PRECISION_GUARD = 128
FIXED_POINT_DENOMINATOR = 2**128
_MICROSCOPIC_THRESHOLD = Fraction(1, FIXED_POINT_DENOMINATOR)

GuardScalar: TypeAlias = Fraction | Decimal | int | float | str


def _coerce_fraction(value: GuardScalar) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        return Fraction(value)
    if isinstance(value, bool):
        return Fraction(int(value), 1)
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction.from_float(value)
    return Fraction(value)


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


def is_guard_zero(value: GuardScalar) -> bool:
    resolved_value = _coerce_fraction(value)
    return resolved_value == 0 or abs(resolved_value) <= _MICROSCOPIC_THRESHOLD


__all__ = [
    "FIXED_POINT_DENOMINATOR",
    "PRECISION_GUARD",
    "guard_fraction",
    "guard_sum",
    "is_guard_zero",
]
