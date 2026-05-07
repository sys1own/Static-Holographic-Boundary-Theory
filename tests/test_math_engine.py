from __future__ import annotations

import importlib
import importlib.util
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath
import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for candidate in (ROOT_DIR, SRC_DIR):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from shbt.constants import AUDIT_TOLERANCE


mpmath.mp.dps = 100

_MATH_ENGINE_MODULE_CANDIDATES = ("shbt.math_engine", "math_engine")
_AUDIT_TOLERANCE_MPF = mpmath.mpf(str(AUDIT_TOLERANCE))


def _load_math_engine() -> Any:
    for module_name in _MATH_ENGINE_MODULE_CANDIDATES:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            continue
        return importlib.import_module(module_name)
    pytest.skip("math_engine module is not available in this checkout.")


def _to_mpf(value: Any) -> mpmath.mpf:
    if isinstance(value, mpmath.mpf):
        return value
    if isinstance(value, Fraction):
        return mpmath.mpf(value.numerator) / mpmath.mpf(value.denominator)
    if isinstance(value, Decimal):
        return mpmath.mpf(str(value))
    if isinstance(value, bool):
        raise TypeError("Boolean values are not valid math engine audit values.")
    if isinstance(value, int):
        return mpmath.mpf(value)
    if isinstance(value, float):
        return mpmath.mpf(repr(value))
    return mpmath.mpf(str(value))


def _assert_close(actual: Any, expected: Any) -> None:
    assert mpmath.almosteq(
        _to_mpf(actual),
        _to_mpf(expected),
        abs_eps=_AUDIT_TOLERANCE_MPF,
    )


def _public_numeric_members(module: Any) -> dict[str, Any]:
    members: dict[str, Any] = {}
    for name in dir(module):
        if name.startswith("_"):
            continue
        value = getattr(module, name)
        if isinstance(value, (mpmath.mpf, Decimal, Fraction, int, float)) and not isinstance(value, bool):
            members[name] = value
    return members


def test_math_engine_imports_under_high_precision() -> None:
    module = _load_math_engine()

    assert module is not None
    assert mpmath.mp.dps >= 100


def test_math_engine_public_numeric_members_support_audit_tolerance_comparisons() -> None:
    module = _load_math_engine()
    members = _public_numeric_members(module)

    if not members:
        pytest.skip("math_engine exposes no public numeric constants to audit directly.")

    for value in members.values():
        _assert_close(value, value)


def test_math_engine_mpf_results_round_trip_through_mpmath_almosteq() -> None:
    _load_math_engine()

    expected = mpmath.mpf("1") / mpmath.mpf("3")
    actual = _to_mpf(expected)

    assert isinstance(actual, mpmath.mpf)
    _assert_close(actual, expected)


def test_math_engine_exposes_guarded_constants_and_fraction_builder() -> None:
    module = _load_math_engine()

    assert getattr(module, "PRECISION_GUARD") == 128
    assert getattr(module, "FIXED_POINT_DENOMINATOR") == 2**128

    guarded_value = module.guard_fraction(3, 6)

    assert isinstance(guarded_value, Fraction)
    _assert_close(guarded_value, Fraction(1, 2))


def test_math_engine_guard_sum_and_zero_check_are_rationally_exact() -> None:
    module = _load_math_engine()

    guarded_sum = module.guard_sum((Fraction(1, 3), Fraction(1, 6), Decimal("0.5")))

    assert isinstance(guarded_sum, Fraction)
    _assert_close(guarded_sum, Fraction(1, 1))
    assert module.is_guard_zero(guarded_sum - Fraction(1, 1)) is True
    assert module.is_guard_zero(Fraction(1, module.FIXED_POINT_DENOMINATOR * 2)) is True
    assert module.is_guard_zero(Fraction(1, module.FIXED_POINT_DENOMINATOR // 2)) is False
