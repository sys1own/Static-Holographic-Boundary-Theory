from __future__ import annotations

import itertools
import math
from fractions import Fraction
from functools import lru_cache

import numpy as np
import numpy.typing as npt

from .runtime_config import DEFAULT_SOLVER_CONFIG, Sector, SolverConfig
from .topology import add_fraction_vectors, fraction_dot, lcm_int, scale_fraction_vector, solve_fraction_linear_system

LOW_SU3_WEIGHTS = ((0, 0), (1, 0), (0, 1))

SO10_RANK = 5
SO10_DIMENSION = 45
SU2_DIMENSION = 3
SU2_DUAL_COXETER = 2
SU3_DIMENSION = 8
SU3_DUAL_COXETER = 3

ComplexMatrix = npt.NDArray[np.complex128]
RealMatrix = npt.NDArray[np.float64]
RealVector = npt.NDArray[np.float64]


def polar_unitary(
    matrix: npt.NDArray[np.complexfloating | np.floating],
    *,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> ComplexMatrix:
    _ = solver_config
    complex_matrix = np.asarray(matrix, dtype=np.complex128)
    u_matrix, _, vh_matrix = np.linalg.svd(complex_matrix)
    return np.asarray(u_matrix @ vh_matrix, dtype=np.complex128)


def rotation_23(phi: float) -> ComplexMatrix:
    cosine = math.cos(phi)
    sine = math.sin(phi)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, sine], [0.0, -sine, cosine]],
        dtype=np.complex128,
    )


def pdg_unitary(theta12_deg: float, theta13_deg: float, theta23_deg: float, delta_deg: float) -> ComplexMatrix:
    theta12 = math.radians(theta12_deg)
    theta13 = math.radians(theta13_deg)
    theta23 = math.radians(theta23_deg)
    delta = math.radians(delta_deg)

    s12, c12 = math.sin(theta12), math.cos(theta12)
    s13, c13 = math.sin(theta13), math.cos(theta13)
    s23, c23 = math.sin(theta23), math.cos(theta23)
    e_minus = np.exp(-1j * delta)
    e_plus = np.exp(1j * delta)

    return np.array(
        [
            [c12 * c13, s12 * c13, s13 * e_minus],
            [-s12 * c23 - c12 * s23 * s13 * e_plus, c12 * c23 - s12 * s23 * s13 * e_plus, s23 * c13],
            [s12 * s23 - c12 * c23 * s13 * e_plus, -c12 * s23 - s12 * c23 * s13 * e_plus, c23 * c13],
        ],
        dtype=np.complex128,
    )


def jarlskog_invariant(unitary: npt.NDArray[np.complexfloating | np.floating]) -> float:
    return float(np.imag(unitary[0, 0] * unitary[1, 1] * np.conjugate(unitary[0, 1]) * np.conjugate(unitary[1, 0])))


def pdg_parameters(
    unitary: npt.NDArray[np.complexfloating | np.floating],
    *,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> tuple[float, float, float, float, float]:
    guard = solver_config.stability_guard
    s13 = guard.clamp_unit_interval(abs(unitary[0, 2]), coordinate="s13")
    c13_squared = guard.clamp_unit_interval(1.0 - s13 * s13, coordinate="c13^2")
    c13 = guard.require_positive(
        math.sqrt(c13_squared),
        coordinate="c13",
        detail="The PDG parameterization is undefined once the 1-3 mixing cosine collapses.",
        floor=guard.zero_magnitude_threshold,
    )
    s12 = guard.clamp_unit_interval(abs(unitary[0, 1]) / c13, coordinate="s12")
    s23 = guard.clamp_unit_interval(abs(unitary[1, 2]) / c13, coordinate="s23")
    c12 = math.sqrt(guard.clamp_unit_interval(1.0 - s12 * s12, coordinate="c12^2"))
    c23 = math.sqrt(guard.clamp_unit_interval(1.0 - s23 * s23, coordinate="c23^2"))

    theta12 = math.degrees(math.asin(s12))
    theta13 = math.degrees(math.asin(s13))
    theta23 = math.degrees(math.asin(s23))

    jarlskog = jarlskog_invariant(unitary)
    denominator = float(
        guard.require_nonzero_magnitude(
            s12 * c12 * s23 * c23 * s13 * c13 * c13,
            coordinate="pdg sin(delta) denominator",
            detail="The Dirac phase becomes a forbidden transition when the PDG area factor vanishes.",
        )
    )
    sin_delta = guard.clamp_signed_unit_interval(jarlskog / denominator, coordinate="sin(delta)")

    denominator_cos = float(
        guard.require_nonzero_magnitude(
            2.0 * s12 * c12 * s23 * c23 * s13,
            coordinate="pdg cos(delta) denominator",
            detail="The Dirac phase becomes a forbidden transition when the cosine projection vanishes.",
        )
    )
    numerator_cos = abs(unitary[1, 0]) ** 2 - s12 * s12 * c23 * c23 - c12 * c12 * s23 * s23 * s13 * s13
    cos_delta = guard.clamp_signed_unit_interval(numerator_cos / denominator_cos, coordinate="cos(delta)")

    delta = math.degrees(math.atan2(sin_delta, cos_delta)) % 360.0
    return theta12, theta13, theta23, delta, jarlskog


@lru_cache(maxsize=128)
def so10_fundamental_weights() -> tuple[tuple[Fraction, ...], ...]:
    return (
        (Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(1), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(1), Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(-1, 2)),
        (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)),
    )


@lru_cache(maxsize=128)
def so10_weyl_vector() -> tuple[Fraction, ...]:
    return (Fraction(4), Fraction(3), Fraction(2), Fraction(1), Fraction(0))


@lru_cache(maxsize=128)
def so10_positive_roots() -> tuple[tuple[Fraction, ...], ...]:
    roots: list[tuple[Fraction, ...]] = []
    for left in range(SO10_RANK):
        for right in range(left + 1, SO10_RANK):
            plus_root = [Fraction(0) for _ in range(SO10_RANK)]
            plus_root[left] += 1
            plus_root[right] += 1
            roots.append(tuple(plus_root))

            minus_root = [Fraction(0) for _ in range(SO10_RANK)]
            minus_root[left] += 1
            minus_root[right] -= 1
            roots.append(tuple(minus_root))
    return tuple(roots)


@lru_cache(maxsize=128)
def so10_simple_roots() -> tuple[tuple[Fraction, ...], ...]:
    return (
        (Fraction(1), Fraction(-1), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(-1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1), Fraction(-1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(-1)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(1)),
    )


@lru_cache(maxsize=128)
def so10_visible_cartan_projection_basis() -> tuple[tuple[Fraction, ...], ...]:
    simple_roots = so10_simple_roots()
    return (
        simple_roots[0],
        simple_roots[1],
        simple_roots[3],
        (Fraction(1, 3), Fraction(1, 3), Fraction(1, 3), Fraction(-1, 2), Fraction(-1, 2)),
        (Fraction(1), Fraction(1), Fraction(1), Fraction(1), Fraction(1)),
    )


@lru_cache(maxsize=128)
def so10_visible_cartan_projection_denominator() -> int:
    denominator = 1
    basis = so10_visible_cartan_projection_basis()
    for weight in so10_fundamental_weights():
        coefficients = solve_fraction_linear_system(basis, weight)
        for coefficient in coefficients[3:]:
            denominator = lcm_int(denominator, coefficient.denominator)
    return denominator


@lru_cache(maxsize=128)
def so10_highest_weight(dynkin_labels: tuple[int, int, int, int, int]) -> tuple[Fraction, ...]:
    if len(dynkin_labels) != SO10_RANK:
        raise ValueError(f"Expected {SO10_RANK} Dynkin labels, received {dynkin_labels}")

    weight = tuple(Fraction(0) for _ in range(SO10_RANK))
    for coefficient, fundamental_weight in zip(dynkin_labels, so10_fundamental_weights()):
        weight = add_fraction_vectors(weight, scale_fraction_vector(fundamental_weight, coefficient))
    return weight


@lru_cache(maxsize=128)
def so10_rep_dimension(dynkin_labels: tuple[int, int, int, int, int]) -> int:
    highest_weight = so10_highest_weight(dynkin_labels)
    rho = so10_weyl_vector()
    dimension = Fraction(1)
    for root in so10_positive_roots():
        dimension *= fraction_dot(add_fraction_vectors(highest_weight, rho), root) / fraction_dot(rho, root)
    if dimension.denominator != 1:
        raise RuntimeError(f"Non-integral $SO(10)$ dimension obtained for {dynkin_labels}: {dimension}")
    return int(dimension)


@lru_cache(maxsize=128)
def so10_rep_quadratic_casimir(dynkin_labels: tuple[int, int, int, int, int]) -> Fraction:
    highest_weight = so10_highest_weight(dynkin_labels)
    rho = so10_weyl_vector()
    return fraction_dot(highest_weight, add_fraction_vectors(highest_weight, scale_fraction_vector(rho, 2))) / 2


@lru_cache(maxsize=128)
def so10_rep_dynkin_index(dynkin_labels: tuple[int, int, int, int, int]) -> Fraction:
    return Fraction(so10_rep_dimension(dynkin_labels), SO10_DIMENSION) * so10_rep_quadratic_casimir(dynkin_labels)


def su2_rep_quadratic_casimir(dimension: int) -> Fraction:
    spin = Fraction(dimension - 1, 2)
    return spin * (spin + 1)


def su2_rep_dynkin_index(dimension: int) -> Fraction:
    return Fraction(dimension, SU2_DIMENSION) * su2_rep_quadratic_casimir(dimension)


def su3_rep_dimension(dynkin_labels: tuple[int, int]) -> int:
    p_label, q_label = dynkin_labels
    return ((p_label + 1) * (q_label + 1) * (p_label + q_label + 2)) // 2


def su3_rep_quadratic_casimir(dynkin_labels: tuple[int, int]) -> Fraction:
    p_label, q_label = dynkin_labels
    numerator = p_label * p_label + q_label * q_label + p_label * q_label + 3 * p_label + 3 * q_label
    return Fraction(numerator, 3)


def su3_rep_dynkin_index(dynkin_labels: tuple[int, int]) -> Fraction:
    return Fraction(su3_rep_dimension(dynkin_labels), SU3_DIMENSION) * su3_rep_quadratic_casimir(dynkin_labels)


def adjoint_quadratic_casimir(dual_coxeter: int) -> Fraction:
    return Fraction(dual_coxeter, 1)


def su2_total_quantum_dimension(level: int) -> float:
    return math.sqrt((level + 2.0) / 2.0) / math.sin(math.pi / (level + 2.0))


def su2_quantum_dimension(level: int, label: int) -> float:
    return math.sin((label + 1.0) * math.pi / (level + 2.0)) / math.sin(math.pi / (level + 2.0))


def su2_conformal_weight(level: int, label: int) -> float:
    return label * (label + 2.0) / (4.0 * (level + 2.0))


def su2_modular_s(level: int) -> RealMatrix:
    prefactor = math.sqrt(2.0 / (level + 2.0))
    size = level + 1
    return np.array(
        [
            [
                prefactor * math.sin((row + 1.0) * (col + 1.0) * math.pi / (level + 2.0))
                for col in range(size)
            ]
            for row in range(size)
        ],
        dtype=np.float64,
    )


def charge_embedding(level: int) -> tuple[int, int, int]:
    return (level - 4, level - 3, level)


def permutation_sign(permutation: tuple[int, int, int]) -> int:
    inversions = 0
    for left in range(len(permutation)):
        for right in range(left + 1, len(permutation)):
            inversions += int(permutation[left] > permutation[right])
    return -1 if inversions % 2 else 1


def su3_conformal_weight(level: int, weight: tuple[int, int]) -> float:
    dynkin_left, dynkin_right = weight
    numerator = dynkin_left * dynkin_left + dynkin_right * dynkin_right + dynkin_left * dynkin_right + 3 * dynkin_left + 3 * dynkin_right
    return numerator / (3.0 * (level + 3.0))


def su3_weight_vector(weight: tuple[int, int]) -> RealVector:
    dynkin_left, dynkin_right = weight
    return np.array(
        [
            (2.0 * dynkin_left + dynkin_right) / 3.0,
            (-dynkin_left + dynkin_right) / 3.0,
            -(dynkin_left + 2.0 * dynkin_right) / 3.0,
        ],
        dtype=np.float64,
    )


def su3_modular_s_entry(level: int, left_weight: tuple[int, int], right_weight: tuple[int, int]) -> complex:
    rho_vector = su3_weight_vector((1, 1))
    left = su3_weight_vector(left_weight) + rho_vector
    right = su3_weight_vector(right_weight) + rho_vector
    prefactor = (1j ** 3) / (math.sqrt(3.0) * (level + 3.0))
    total = 0.0j
    for permutation in itertools.permutations((0, 1, 2)):
        total += permutation_sign(permutation) * np.exp(
            -2j * math.pi * np.dot(left[list(permutation)], right) / (level + 3.0)
        )
    return prefactor * total


def su3_low_weight_block(level: int, weights: tuple[tuple[int, int], ...] = LOW_SU3_WEIGHTS) -> ComplexMatrix:
    return np.array(
        [[su3_modular_s_entry(level, left_weight, right_weight) for right_weight in weights] for left_weight in weights],
        dtype=np.complex128,
    )


def interference_holonomy_phase(
    interference_block: npt.NDArray[np.complexfloating | np.floating],
    framing_phases: npt.NDArray[np.complexfloating | np.floating],
    *,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> float:
    size = min(interference_block.shape[0], interference_block.shape[1], len(framing_phases))
    weighted_phase = 0.0j
    for left in range(size):
        for right in range(left + 1, size):
            weight = abs(
                interference_block[0, left]
                * interference_block[1, right]
                * interference_block[0, right]
                * interference_block[1, left]
            )
            weighted_phase += weight * framing_phases[right] * np.conjugate(framing_phases[left])
    solver_config.stability_guard.require_nonzero_magnitude(
        weighted_phase,
        coordinate="interference holonomy phase",
        detail="An exactly vanishing interference phase signals an unphysical CP branch with no resolved holonomy.",
    )
    return math.degrees(np.angle(weighted_phase)) % 360.0


def predict_delta_cp(
    interference_block: npt.NDArray[np.complexfloating | np.floating],
    framing_phases: npt.NDArray[np.complexfloating | np.floating],
    branch_shift_deg: float = 0.0,
    *,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> float:
    return (interference_holonomy_phase(interference_block, framing_phases, solver_config=solver_config) + branch_shift_deg) % 360.0


def weyl_vector_norm_sq(dimension: int, dual_coxeter: int) -> float:
    return dual_coxeter * dimension / 12.0


def rank_deficit_pressure(parent_level: int, quark_level: int) -> float:
    so10_weyl_norm_sq = weyl_vector_norm_sq(SO10_DIMENSION, 8)
    su3_weyl_norm_sq = weyl_vector_norm_sq(SU3_DIMENSION, SU3_DUAL_COXETER)
    weyl_ratio = math.sqrt(so10_weyl_norm_sq / su3_weyl_norm_sq)
    return weyl_ratio * math.sqrt((quark_level + SU3_DUAL_COXETER) / (parent_level + 8))


class ModularKernel:
    def __init__(
        self,
        level: int,
        sector: Sector | str,
        *,
        solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    ) -> None:
        self.level = level
        self.sector = Sector.coerce(sector)
        self.solver_config = solver_config

    def restricted_block(self) -> ComplexMatrix:
        if self.sector is Sector.LEPTON:
            labels = (0, 1, 2)
            charges = charge_embedding(self.level)
            s_matrix = su2_modular_s(self.level)
            return np.asarray(s_matrix[np.ix_(charges, labels)], dtype=np.complex128)
        if self.sector is Sector.QUARK:
            return su3_low_weight_block(self.level)
        raise ValueError(f"Unsupported sector: {self.sector.value}")

    def framing_phases(self) -> npt.NDArray[np.complex128]:
        if self.sector is Sector.LEPTON:
            labels = (0, 1, 2)
            return np.exp(2j * math.pi * np.array([su2_conformal_weight(self.level, label) for label in labels], dtype=float))
        if self.sector is Sector.QUARK:
            return np.exp(2j * math.pi * np.array([su3_conformal_weight(self.level, weight) for weight in LOW_SU3_WEIGHTS], dtype=float))
        raise ValueError(f"Unsupported sector: {self.sector.value}")

    def t_decorated_unitary(self, seed_matrix: npt.NDArray[np.complexfloating | np.floating]) -> ComplexMatrix:
        return polar_unitary(np.asarray(seed_matrix, dtype=np.complex128) @ np.diag(self.framing_phases()), solver_config=self.solver_config)

    def complex_unitary(
        self,
        seed_matrix: npt.NDArray[np.complexfloating | np.floating],
        interference_block: npt.NDArray[np.complexfloating | np.floating] | None = None,
        branch_shift_deg: float = 0.0,
    ) -> tuple[ComplexMatrix, float]:
        decorated = self.t_decorated_unitary(seed_matrix)
        theta12, theta13, theta23, _, _ = pdg_parameters(decorated, solver_config=self.solver_config)
        block = self.restricted_block() if interference_block is None else np.asarray(interference_block, dtype=np.complex128)
        delta_cp_deg = predict_delta_cp(
            block,
            self.framing_phases(),
            branch_shift_deg=branch_shift_deg,
            solver_config=self.solver_config,
        )
        return pdg_unitary(theta12, theta13, theta23, delta_cp_deg), delta_cp_deg


__all__ = [
    "ComplexMatrix",
    "LOW_SU3_WEIGHTS",
    "ModularKernel",
    "RealMatrix",
    "RealVector",
    "SO10_DIMENSION",
    "SO10_RANK",
    "SU2_DIMENSION",
    "SU3_DIMENSION",
    "adjoint_quadratic_casimir",
    "charge_embedding",
    "interference_holonomy_phase",
    "jarlskog_invariant",
    "pdg_parameters",
    "pdg_unitary",
    "permutation_sign",
    "polar_unitary",
    "predict_delta_cp",
    "rank_deficit_pressure",
    "rotation_23",
    "so10_fundamental_weights",
    "so10_highest_weight",
    "so10_positive_roots",
    "so10_rep_dimension",
    "so10_rep_dynkin_index",
    "so10_rep_quadratic_casimir",
    "so10_simple_roots",
    "so10_visible_cartan_projection_basis",
    "so10_visible_cartan_projection_denominator",
    "so10_weyl_vector",
    "su2_conformal_weight",
    "su2_modular_s",
    "su2_quantum_dimension",
    "su2_rep_dynkin_index",
    "su2_rep_quadratic_casimir",
    "su2_total_quantum_dimension",
    "su3_conformal_weight",
    "su3_low_weight_block",
    "su3_modular_s_entry",
    "su3_rep_dimension",
    "su3_rep_dynkin_index",
    "su3_rep_quadratic_casimir",
    "su3_weight_vector",
    "weyl_vector_norm_sq",
]
