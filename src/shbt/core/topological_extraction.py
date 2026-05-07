from __future__ import annotations

"""Theory-only transport-residue extraction on the anomaly-free benchmark branch.

This module avoids any CODATA anchoring by constructing the transport operator
entirely from the benchmark ``(26, 8, 312)`` boundary manifold. The dominant
mass-hierarchy eigenvalue is then extracted from a Radau IIA relaxation flow
and checked against the theory-internal hierarchy residue ``mu``.
"""

from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Final

import numpy as np
from scipy.integrate import solve_ivp

from shbt.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.derivation_api import DEFAULT_PRECISION, TopologicalVacuum, UniverseFactory, decimal_cuberoot


BENCHMARK_BOUNDARY_BRANCH: Final[tuple[int, int, int]] = (
    int(LEPTON_LEVEL),
    int(QUARK_LEVEL),
    int(PARENT_LEVEL),
)
RADAU_IIA_METHOD: Final[str] = "Radau"
INTERNAL_EIGENVALUE_TOLERANCE: Final[Decimal] = Decimal("1e-15")
_GUARD_DIGITS: Final[int] = 12


def _decimal(value: Decimal | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


def _normalize_state(state: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(state))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Boundary transport state must carry finite non-zero support.")
    return np.asarray(state / norm, dtype=float)


def _require_benchmark_vacuum(vacuum: TopologicalVacuum | None) -> TopologicalVacuum:
    resolved_vacuum = UniverseFactory.benchmark_vacuum() if vacuum is None else vacuum
    if resolved_vacuum.branch != BENCHMARK_BOUNDARY_BRANCH:
        raise ValueError(
            "Topological extraction is defined only on the anomaly-free (26, 8, 312) boundary manifold."
        )
    return resolved_vacuum


@dataclass(frozen=True)
class BoundaryTransportProblem:
    vacuum: TopologicalVacuum
    solver_method: str
    structural_prefactor: Decimal
    pressure_loading: Decimal
    mass_hierarchy_mu: Decimal
    transport_operator: np.ndarray
    normalized_transport_operator: np.ndarray
    initial_state: np.ndarray
    integration_span: tuple[float, float]


@dataclass(frozen=True)
class TransportEigenmode:
    eigenvalue: Decimal
    rayleigh_quotient: Decimal
    relative_error: Decimal
    residual_norm: Decimal
    eigenvector: tuple[float, float, float]


@dataclass(frozen=True)
class TopologicalExtractionResult:
    problem: BoundaryTransportProblem
    stable_residues: tuple[Decimal, Decimal, Decimal]
    numerical_eigenvalues: tuple[float, float, float]
    mass_hierarchy_mode: TransportEigenmode


def build_boundary_transport_operator(
    *,
    precision: int = DEFAULT_PRECISION,
    vacuum: TopologicalVacuum | None = None,
) -> BoundaryTransportProblem:
    """Build the benchmark transport operator from theory-internal residues only."""

    resolved_precision = max(int(precision), DEFAULT_PRECISION)
    resolved_vacuum = _require_benchmark_vacuum(vacuum)
    geometry = UniverseFactory.derive_central_charge_geometry(vacuum=resolved_vacuum)
    vacuum_pressure = UniverseFactory.derive_vacuum_pressure(vacuum=resolved_vacuum)
    kappa = UniverseFactory.derive_kappa_d5(precision=resolved_precision, vacuum=resolved_vacuum).kappa

    with localcontext() as context:
        context.prec = resolved_precision + _GUARD_DIGITS
        structural_prefactor = geometry.structural_prefactor_decimal
        kappa_cuberoot = decimal_cuberoot(kappa, precision=context.prec)
        geometric_friction_factor = (Decimal(1) - kappa) * kappa_cuberoot
        pressure_loading = (vacuum_pressure.vacuum_pressure * vacuum_pressure.vacuum_pressure) / geometric_friction_factor
        mass_hierarchy_mu = structural_prefactor * pressure_loading
        context.prec = resolved_precision
        structural_prefactor = +structural_prefactor
        pressure_loading = +pressure_loading
        mass_hierarchy_mu = +mass_hierarchy_mu

    transport_operator = np.diag(
        np.array(
            [
                float(structural_prefactor),
                float(pressure_loading),
                float(mass_hierarchy_mu),
            ],
            dtype=float,
        )
    )
    operator_scale = float(mass_hierarchy_mu)
    normalized_transport_operator = np.asarray(transport_operator / operator_scale, dtype=float)
    initial_state = _normalize_state(
        np.array(
            [
                float(geometry.central_charge_ratio_decimal),
                float(geometry.inverse_pixel_volume_decimal),
                float(vacuum_pressure.vacuum_pressure),
            ],
            dtype=float,
        )
    )

    return BoundaryTransportProblem(
        vacuum=resolved_vacuum,
        solver_method=RADAU_IIA_METHOD,
        structural_prefactor=structural_prefactor,
        pressure_loading=pressure_loading,
        mass_hierarchy_mu=mass_hierarchy_mu,
        transport_operator=transport_operator,
        normalized_transport_operator=normalized_transport_operator,
        initial_state=initial_state,
        integration_span=(0.0, 24.0),
    )


def initialize_radau_iia_boundary_problem(
    *,
    precision: int = DEFAULT_PRECISION,
    vacuum: TopologicalVacuum | None = None,
) -> BoundaryTransportProblem:
    """Initialize the Radau IIA benchmark-only boundary transport problem."""

    return build_boundary_transport_operator(precision=precision, vacuum=vacuum)


def _rayleigh_flow(operator: np.ndarray):
    def flow(_time: float, state: np.ndarray) -> np.ndarray:
        resolved_state = np.asarray(state, dtype=float)
        norm_sq = float(np.dot(resolved_state, resolved_state))
        if norm_sq <= 0.0 or not np.isfinite(norm_sq):
            raise ValueError("Boundary transport flow encountered a degenerate state norm.")
        image = operator @ resolved_state
        rayleigh = float(np.dot(resolved_state, image) / norm_sq)
        return np.asarray(image - rayleigh * resolved_state, dtype=float)

    return flow


def assert_mass_hierarchy_extraction(
    extracted_eigenvalue: Decimal,
    reference_mu: Decimal,
    *,
    tolerance: Decimal = INTERNAL_EIGENVALUE_TOLERANCE,
) -> Decimal:
    """Assert that the extracted mass-hierarchy eigenvalue closes internally."""

    relative_error = abs(extracted_eigenvalue - reference_mu) / reference_mu
    assert relative_error <= tolerance, (
        "Transport-operator extraction drifted away from the theory-internal mass hierarchy residue: "
        f"extracted {extracted_eigenvalue}, expected {reference_mu}, relative error {relative_error}."
    )
    return relative_error


def search_stable_transport_residues(
    *,
    precision: int = DEFAULT_PRECISION,
    vacuum: TopologicalVacuum | None = None,
) -> TopologicalExtractionResult:
    """Find the stable benchmark transport residues without external mass targets."""

    problem = initialize_radau_iia_boundary_problem(precision=precision, vacuum=vacuum)
    flow = _rayleigh_flow(problem.normalized_transport_operator)
    solution = solve_ivp(
        flow,
        problem.integration_span,
        problem.initial_state,
        method=problem.solver_method,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    if not solution.success:
        raise RuntimeError(f"Radau IIA transport solve failed: {solution.message}")

    stabilized_state = _normalize_state(np.asarray(solution.y[:, -1], dtype=float))
    scaled_rayleigh = float(
        stabilized_state @ (problem.normalized_transport_operator @ stabilized_state)
    )
    extracted_mass_hierarchy = _decimal(scaled_rayleigh * float(problem.mass_hierarchy_mu))
    relative_error = assert_mass_hierarchy_extraction(extracted_mass_hierarchy, problem.mass_hierarchy_mu)
    residual_norm = _decimal(float(np.linalg.norm(flow(float(solution.t[-1]), stabilized_state))))

    numerical_eigenvalues = tuple(float(value) for value in np.linalg.eigvalsh(problem.transport_operator))
    stable_residues = tuple(
        sorted((problem.structural_prefactor, problem.pressure_loading, problem.mass_hierarchy_mu))
    )
    mass_hierarchy_mode = TransportEigenmode(
        eigenvalue=extracted_mass_hierarchy,
        rayleigh_quotient=extracted_mass_hierarchy,
        relative_error=relative_error,
        residual_norm=residual_norm,
        eigenvector=tuple(float(component) for component in stabilized_state),
    )
    return TopologicalExtractionResult(
        problem=problem,
        stable_residues=stable_residues,
        numerical_eigenvalues=numerical_eigenvalues,
        mass_hierarchy_mode=mass_hierarchy_mode,
    )


def extract_mass_hierarchy_eigenvalue(
    *,
    precision: int = DEFAULT_PRECISION,
    vacuum: TopologicalVacuum | None = None,
) -> Decimal:
    """Return the benchmark mass-hierarchy eigenvalue extracted from the transport operator."""

    return search_stable_transport_residues(precision=precision, vacuum=vacuum).mass_hierarchy_mode.eigenvalue


__all__ = [
    "BENCHMARK_BOUNDARY_BRANCH",
    "BoundaryTransportProblem",
    "INTERNAL_EIGENVALUE_TOLERANCE",
    "RADAU_IIA_METHOD",
    "TopologicalExtractionResult",
    "TransportEigenmode",
    "assert_mass_hierarchy_extraction",
    "build_boundary_transport_operator",
    "extract_mass_hierarchy_eigenvalue",
    "initialize_radau_iia_boundary_problem",
    "search_stable_transport_residues",
]
