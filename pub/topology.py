from __future__ import annotations

import math
from fractions import Fraction

from sympy import Matrix, Rational
try:
    from sympy import NonInvertibleMatrixError
except ImportError:
    try:
        from sympy.matrices.exceptions import NonInvertibleMatrixError
    except ImportError:
        # Fallback: catch the generic exception that SymPy raises
        NonInvertibleMatrixError = ValueError

def add_fraction_vectors(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    """Add two exact rational vectors componentwise."""

    return tuple(a + b for a, b in zip(left, right))


def scale_fraction_vector(vector: tuple[Fraction, ...], scale: int | Fraction) -> tuple[Fraction, ...]:
    """Scale an exact rational vector by an integer or rational prefactor."""

    resolved_scale = Fraction(scale)
    return tuple(resolved_scale * component for component in vector)


def fraction_dot(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> Fraction:
    """Return the exact rational dot product of two vectors."""

    return sum((a * b for a, b in zip(left, right)), Fraction(0))


def lcm_int(left: int, right: int) -> int:
    """Return the least common multiple of two integers."""

    return abs(left * right) // math.gcd(left, right)


def _fraction_to_sympy(value: Fraction) -> Rational:
    return Rational(value.numerator, value.denominator)


def _sympy_to_fraction(value: object) -> Fraction:
    rational = Rational(value)
    return Fraction(int(rational.p), int(rational.q))


def solve_fraction_linear_system(
    column_vectors: tuple[tuple[Fraction, ...], ...],
    target: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    """Solve a small exact linear system with SymPy's rational matrix solver."""

    dimension = len(target)
    if len(column_vectors) != dimension:
        raise ValueError(f"Expected {dimension} column vectors, received {len(column_vectors)}")
    if any(len(column) != dimension for column in column_vectors):
        raise ValueError("All column vectors must match the target dimension")

    matrix = Matrix(
        [
            [_fraction_to_sympy(column_vectors[column][row]) for column in range(dimension)]
            for row in range(dimension)
        ]
    )
    rhs = Matrix([_fraction_to_sympy(value) for value in target])
    try:
        solution = matrix.solve(rhs)
    except (NonInvertibleMatrixError, ValueError) as exc:
        raise RuntimeError("Visible Cartan projection basis is singular.") from exc
    return tuple(_sympy_to_fraction(value) for value in solution)


__all__ = [
    "add_fraction_vectors",
    "fraction_dot",
    "lcm_int",
    "scale_fraction_vector",
    "solve_fraction_linear_system",
]
