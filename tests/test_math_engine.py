from __future__ import annotations

from fractions import Fraction

from shbt.math_engine import FIXED_POINT_DENOMINATOR, PRECISION_GUARD, fixed_point_integer, guard_fraction


def test_guard_fraction_quantizes_onto_128_bit_fixed_point_lattice() -> None:
    guarded = guard_fraction(Fraction(1, 24))

    assert PRECISION_GUARD == 128
    assert fixed_point_integer(guarded) == fixed_point_integer(Fraction(1, 24))
    assert (guarded * FIXED_POINT_DENOMINATOR).denominator == 1


def test_guard_fraction_preserves_exact_zero() -> None:
    guarded = guard_fraction(Fraction(0, 1))

    assert guarded == 0
    assert fixed_point_integer(guarded) == 0
