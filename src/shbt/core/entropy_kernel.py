from __future__ import annotations

"""Entropy primitives for holographic information accounting.

This module packages three closely related statements used throughout the SHBT
bookkeeping:

- the Bekenstein-Hawking area law fixes the maximum entropy carried by a finite
  boundary area,
- the Von Neumann entropy measures the mixed-state information content of a
  quantum density matrix, and
- the bulk information density is the surface bit loading projected through the
  effective radial depth ``V / A`` of the bulk image.
"""

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import numpy.typing as npt

from shbt.constants import HOLOGRAPHIC_BITS, PLANCK_LENGTH_M
from shbt.core.differential_geometry import require_real_array, require_real_scalar


DEFAULT_PRECISION = 120
_LN_2_FLOAT = math.log(2.0)
_MATRIX_ATOL = 1.0e-12


@dataclass(frozen=True)
class BekensteinHawkingBound:
    horizon_area_m2: Decimal
    planck_area_m2: Decimal
    entropy_nats: Decimal
    entropy_bits: Decimal
    surface_bit_loading_bits_per_m2: Decimal


@dataclass(frozen=True)
class VonNeumannEntropyAudit:
    eigenvalues: tuple[float, ...]
    entropy_nats: float
    entropy_bits: float
    purity: float
    trace: float
    positive_semidefinite: bool


@dataclass(frozen=True)
class BulkInformationProjection:
    boundary_area_m2: Decimal
    bulk_volume_m3: Decimal
    available_surface_bits: Decimal
    occupied_surface_fraction: Decimal
    occupied_surface_bits: Decimal
    radial_information_depth_m: Decimal
    surface_bit_loading_bits_per_m2: Decimal
    bulk_information_density_bits_per_m3: Decimal
    projection_identity_residual: Decimal

    @property
    def projection_identity_holds(self) -> bool:
        return abs(self.projection_identity_residual) <= Decimal("1e-30")


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(str(value))


def _decimal_ln(value: Decimal) -> Decimal:
    try:
        return value.ln()
    except AttributeError:
        return Decimal(str(math.log(float(value))))


def bekenstein_hawking_bound(
    horizon_area_m2: Decimal | Fraction | float | int | str,
    *,
    precision: int = DEFAULT_PRECISION,
) -> BekensteinHawkingBound:
    area = _decimal(horizon_area_m2)
    if area <= 0:
        raise ValueError("horizon_area_m2 must be positive.")

    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION)
        planck_area = _decimal(PLANCK_LENGTH_M) * _decimal(PLANCK_LENGTH_M)
        entropy_nats = area / (Decimal(4) * planck_area)
        entropy_bits = entropy_nats / _decimal_ln(Decimal(2))
        surface_bit_loading = entropy_bits / area
    return BekensteinHawkingBound(
        horizon_area_m2=area,
        planck_area_m2=planck_area,
        entropy_nats=entropy_nats,
        entropy_bits=entropy_bits,
        surface_bit_loading_bits_per_m2=surface_bit_loading,
    )


def analyze_von_neumann_entropy(
    density_matrix: npt.ArrayLike,
    *,
    atol: float = _MATRIX_ATOL,
) -> VonNeumannEntropyAudit:
    matrix = np.asarray(density_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"density_matrix must be square, received shape {matrix.shape}.")
    if not np.allclose(matrix, np.conjugate(matrix.T), atol=atol, rtol=0.0):
        raise ValueError("density_matrix must be Hermitian.")

    trace = require_real_scalar(np.trace(matrix), label="density_matrix.trace", atol=atol)
    if not math.isclose(trace, 1.0, rel_tol=0.0, abs_tol=atol):
        raise ValueError(f"density_matrix trace must equal 1 within {atol:.1e}, received {trace:.6g}.")

    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = require_real_array(eigenvalues, label="density_matrix.eigenvalues", atol=atol)
    if np.any(eigenvalues < -atol):
        raise ValueError("density_matrix must be positive semidefinite.")

    clipped = np.clip(eigenvalues, 0.0, None)
    normalization = float(np.sum(clipped))
    if normalization <= 0.0:
        raise ValueError("density_matrix must carry non-zero support.")
    normalized = clipped / normalization
    entropy_nats = float(-np.sum([value * math.log(value) for value in normalized if value > 0.0]))
    entropy_bits = entropy_nats / _LN_2_FLOAT
    purity = float(np.sum(normalized * normalized))
    return VonNeumannEntropyAudit(
        eigenvalues=tuple(float(value) for value in normalized),
        entropy_nats=entropy_nats,
        entropy_bits=entropy_bits,
        purity=purity,
        trace=trace,
        positive_semidefinite=True,
    )


def von_neumann_entropy(density_matrix: npt.ArrayLike, *, atol: float = _MATRIX_ATOL) -> float:
    return analyze_von_neumann_entropy(density_matrix, atol=atol).entropy_bits


def project_surface_bits_to_bulk(
    *,
    boundary_area_m2: Decimal | Fraction | float | int | str,
    bulk_volume_m3: Decimal | Fraction | float | int | str,
    occupied_surface_fraction: Decimal | Fraction | float | int | str = Decimal(1),
    available_surface_bits: Decimal | Fraction | float | int | str | None = None,
    precision: int = DEFAULT_PRECISION,
) -> BulkInformationProjection:
    area = _decimal(boundary_area_m2)
    volume = _decimal(bulk_volume_m3)
    occupancy = _decimal(occupied_surface_fraction)
    if area <= 0:
        raise ValueError("boundary_area_m2 must be positive.")
    if volume <= 0:
        raise ValueError("bulk_volume_m3 must be positive.")
    if occupancy < 0 or occupancy > 1:
        raise ValueError("occupied_surface_fraction must lie in the closed interval [0, 1].")

    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION)
        if available_surface_bits is None:
            surface_bits = bekenstein_hawking_bound(area, precision=context.prec).entropy_bits
        else:
            surface_bits = _decimal(available_surface_bits)
        occupied_bits = surface_bits * occupancy
        radial_depth = volume / area
        surface_loading = occupied_bits / area
        bulk_density = occupied_bits / volume
        projection_residual = surface_loading - (bulk_density * radial_depth)
    return BulkInformationProjection(
        boundary_area_m2=area,
        bulk_volume_m3=volume,
        available_surface_bits=surface_bits,
        occupied_surface_fraction=occupancy,
        occupied_surface_bits=occupied_bits,
        radial_information_depth_m=radial_depth,
        surface_bit_loading_bits_per_m2=surface_loading,
        bulk_information_density_bits_per_m3=bulk_density,
        projection_identity_residual=projection_residual,
    )


def benchmark_holographic_projection(
    *,
    boundary_area_m2: Decimal | Fraction | float | int | str,
    bulk_volume_m3: Decimal | Fraction | float | int | str,
    occupied_surface_fraction: Decimal | Fraction | float | int | str = Decimal(1),
    precision: int = DEFAULT_PRECISION,
) -> BulkInformationProjection:
    return project_surface_bits_to_bulk(
        boundary_area_m2=boundary_area_m2,
        bulk_volume_m3=bulk_volume_m3,
        occupied_surface_fraction=occupied_surface_fraction,
        available_surface_bits=_decimal(HOLOGRAPHIC_BITS),
        precision=precision,
    )


__all__ = [
    "BekensteinHawkingBound",
    "BulkInformationProjection",
    "DEFAULT_PRECISION",
    "VonNeumannEntropyAudit",
    "analyze_von_neumann_entropy",
    "bekenstein_hawking_bound",
    "benchmark_holographic_projection",
    "project_surface_bits_to_bulk",
    "von_neumann_entropy",
]
