from __future__ import annotations

"""Differential-geometry primitives and real-valued numeric consistency helpers.

This module retains the original array/sanity helpers that lived in
``numerics.py`` and extends them with a light-weight differential-geometry
toolbox: metric tensors, index raising/lowering, Christoffel symbols, curvature
contractions, and a simple Ricci-flow style manifold evolution step.
"""

from dataclasses import dataclass, fields, is_dataclass
import math
from typing import Any, Iterable, TypeVar

import numpy as np
import numpy.typing as npt


T = TypeVar("T")

COMPLEX_CONSISTENCY_ATOL = float(np.finfo(float).eps)
SYMMETRY_ATOL = 1.0e-12


def require_real_array(
    values: npt.ArrayLike,
    *,
    label: str,
    atol: float = COMPLEX_CONSISTENCY_ATOL,
    rtol: float = 0.0,
) -> npt.NDArray[np.float64]:
    array = np.asarray(values)
    if np.isrealobj(array):
        return np.asarray(array, dtype=np.float64)

    imaginary_part = np.abs(np.imag(array))
    if np.allclose(imaginary_part, 0.0, atol=atol, rtol=rtol):
        return np.asarray(np.real(array), dtype=np.float64)

    max_index = np.unravel_index(int(np.argmax(imaginary_part)), imaginary_part.shape)
    max_imaginary = float(imaginary_part[max_index])
    raise ValueError(
        f"{label} carries a non-negligible imaginary component at {max_index}: "
        f"|Im|={max_imaginary:.3e} exceeds {atol:.3e}."
    )


def require_real_scalar(
    value: Any,
    *,
    label: str,
    atol: float = COMPLEX_CONSISTENCY_ATOL,
    rtol: float = 0.0,
) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{label} must be scalar, received shape {array.shape}.")
    return float(require_real_array(array, label=label, atol=atol, rtol=rtol).reshape(()))


def freeze_numpy_arrays(value: T) -> T:
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
        return value

    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            freeze_numpy_arrays(getattr(value, field.name))
        return value

    if isinstance(value, dict):
        for element in value.values():
            freeze_numpy_arrays(element)
        return value

    if isinstance(value, (tuple, list, set, frozenset)):
        for element in value:
            freeze_numpy_arrays(element)
        return value

    return value


def _require_square_matrix(
    values: npt.ArrayLike,
    *,
    label: str,
    symmetry_atol: float = SYMMETRY_ATOL,
) -> npt.NDArray[np.float64]:
    matrix = require_real_array(values, label=label)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{label} must be a square matrix, received shape {matrix.shape}.")
    if not np.allclose(matrix, matrix.T, atol=symmetry_atol, rtol=0.0):
        raise ValueError(f"{label} must be symmetric within {symmetry_atol:.1e}.")
    return np.array(matrix, dtype=np.float64, copy=True)


def _require_vector(values: npt.ArrayLike, *, label: str, dimension: int) -> npt.NDArray[np.float64]:
    vector = require_real_array(values, label=label)
    if vector.ndim != 1 or vector.shape[0] != dimension:
        raise ValueError(f"{label} must be a length-{dimension} vector, received shape {vector.shape}.")
    return np.array(vector, dtype=np.float64, copy=True)


def _freeze_array(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    frozen = np.array(values, dtype=np.float64, copy=True)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True)
class MetricTensor:
    components: npt.ArrayLike
    label: str = "metric_tensor"
    symmetry_atol: float = SYMMETRY_ATOL

    def __post_init__(self) -> None:
        matrix = _require_square_matrix(self.components, label=self.label, symmetry_atol=self.symmetry_atol)
        object.__setattr__(self, "components", _freeze_array(matrix))

    @property
    def dimension(self) -> int:
        return int(self.components.shape[0])

    @property
    def determinant(self) -> float:
        return require_real_scalar(np.linalg.det(self.components), label=f"{self.label}.determinant")

    @property
    def inverse(self) -> npt.NDArray[np.float64]:
        try:
            inverse = np.linalg.inv(self.components)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"{self.label} is singular and cannot be inverted.") from exc
        return _freeze_array(require_real_array(inverse, label=f"{self.label}.inverse"))

    @property
    def eigenvalues(self) -> tuple[float, ...]:
        return tuple(float(value) for value in np.linalg.eigvalsh(self.components))

    @property
    def signature(self) -> tuple[int, int, int]:
        positive = sum(value > self.symmetry_atol for value in self.eigenvalues)
        negative = sum(value < -self.symmetry_atol for value in self.eigenvalues)
        zero = self.dimension - positive - negative
        return positive, negative, zero

    @property
    def positive_definite(self) -> bool:
        positive, negative, zero = self.signature
        return positive == self.dimension and negative == 0 and zero == 0

    def transform_coordinates(
        self,
        jacobian: npt.ArrayLike,
        *,
        label: str = "transformed_metric_tensor",
    ) -> "MetricTensor":
        return coordinate_transform(self, jacobian, label=label)

    def project_to_bulk(
        self,
        block_to_bulk_projector: npt.ArrayLike,
        *,
        stabilizer: Any | None = None,
        precision: int = 200,
        simulate_boundary_decoherence: bool = False,
        label: str = "projected_bulk_metric_tensor",
    ) -> "StabilizedBulkMetricProjection":
        return project_static_block_to_bulk(
            self,
            block_to_bulk_projector,
            stabilizer=stabilizer,
            precision=precision,
            simulate_boundary_decoherence=simulate_boundary_decoherence,
            label=label,
        )


@dataclass(frozen=True)
class ManifoldState:
    metric: MetricTensor
    ricci_tensor: npt.ArrayLike
    ricci_scalar: float
    volume_element: float

    def __post_init__(self) -> None:
        ricci = _require_square_matrix(self.ricci_tensor, label="ricci_tensor")
        if ricci.shape != self.metric.components.shape:
            raise ValueError(
                "ricci_tensor must match the metric dimension, "
                f"received {ricci.shape} for metric dimension {self.metric.dimension}."
            )
        object.__setattr__(self, "ricci_tensor", _freeze_array(ricci))


@dataclass(frozen=True)
class RiemannianManifoldEvolution:
    step_size: float
    states: tuple[ManifoldState, ...]

    @property
    def step_count(self) -> int:
        return max(len(self.states) - 1, 0)

    @property
    def final_state(self) -> ManifoldState:
        return self.states[-1]


@dataclass(frozen=True)
class StabilizedBulkMetricProjection:
    static_metric: MetricTensor
    bulk_metric: MetricTensor
    block_to_bulk_projector: npt.ArrayLike
    benchmark_branch: tuple[int, int, int]
    zero_energy_boundary_locked: bool
    equivalence_principle_preserved: bool
    torsion_projection_residual: float
    stabilizer_detail: str

    def __post_init__(self) -> None:
        projector = require_real_array(self.block_to_bulk_projector, label="block_to_bulk_projector")
        expected_shape = (self.bulk_metric.dimension, self.static_metric.dimension)
        if projector.ndim != 2 or projector.shape != expected_shape:
            raise ValueError(
                "block_to_bulk_projector must have shape "
                f"{expected_shape}, received {projector.shape}."
            )
        if np.linalg.matrix_rank(projector) != self.bulk_metric.dimension:
            raise ValueError(
                "block_to_bulk_projector must have full row rank for a valid bulk projection."
            )
        object.__setattr__(self, "block_to_bulk_projector", _freeze_array(projector))
        object.__setattr__(self, "benchmark_branch", tuple(int(index) for index in self.benchmark_branch))
        object.__setattr__(self, "torsion_projection_residual", float(self.torsion_projection_residual))

    @property
    def static_block_dimension(self) -> int:
        return self.static_metric.dimension

    @property
    def projected_bulk_dimension(self) -> int:
        return self.bulk_metric.dimension

    @property
    def stabilizer_passed(self) -> bool:
        return bool(
            self.zero_energy_boundary_locked
            and self.equivalence_principle_preserved
            and math.isclose(self.torsion_projection_residual, 0.0, rel_tol=0.0, abs_tol=SYMMETRY_ATOL)
        )


def build_metric_tensor(
    values: npt.ArrayLike,
    *,
    label: str = "metric_tensor",
    symmetry_atol: float = SYMMETRY_ATOL,
) -> MetricTensor:
    return MetricTensor(values, label=label, symmetry_atol=symmetry_atol)


def metric_inner_product(
    metric: MetricTensor | npt.ArrayLike,
    left: npt.ArrayLike,
    right: npt.ArrayLike,
) -> float:
    resolved_metric = metric if isinstance(metric, MetricTensor) else build_metric_tensor(metric)
    left_vector = _require_vector(left, label="left_vector", dimension=resolved_metric.dimension)
    right_vector = _require_vector(right, label="right_vector", dimension=resolved_metric.dimension)
    return require_real_scalar(
        left_vector @ resolved_metric.components @ right_vector,
        label=f"{resolved_metric.label}.inner_product",
    )


def line_element(metric: MetricTensor | npt.ArrayLike, displacement: npt.ArrayLike) -> float:
    return metric_inner_product(metric, displacement, displacement)


def lower_index(metric: MetricTensor | npt.ArrayLike, contravariant_vector: npt.ArrayLike) -> npt.NDArray[np.float64]:
    resolved_metric = metric if isinstance(metric, MetricTensor) else build_metric_tensor(metric)
    vector = _require_vector(contravariant_vector, label="contravariant_vector", dimension=resolved_metric.dimension)
    return _freeze_array(resolved_metric.components @ vector)


def raise_index(metric: MetricTensor | npt.ArrayLike, covariant_vector: npt.ArrayLike) -> npt.NDArray[np.float64]:
    resolved_metric = metric if isinstance(metric, MetricTensor) else build_metric_tensor(metric)
    vector = _require_vector(covariant_vector, label="covariant_vector", dimension=resolved_metric.dimension)
    return _freeze_array(resolved_metric.inverse @ vector)


def volume_element(metric: MetricTensor | npt.ArrayLike) -> float:
    resolved_metric = metric if isinstance(metric, MetricTensor) else build_metric_tensor(metric)
    determinant = resolved_metric.determinant
    if determinant == 0.0:
        return 0.0
    return float(np.sqrt(abs(determinant)))


def coordinate_transform(
    metric: MetricTensor | npt.ArrayLike,
    jacobian: npt.ArrayLike,
    *,
    label: str = "transformed_metric_tensor",
    symmetry_atol: float = SYMMETRY_ATOL,
) -> MetricTensor:
    resolved_metric = metric if isinstance(metric, MetricTensor) else build_metric_tensor(metric)
    transformation = require_real_array(jacobian, label="coordinate_jacobian")
    if transformation.ndim != 2 or transformation.shape[1] != resolved_metric.dimension:
        raise ValueError(
            "coordinate_jacobian must have shape (target_dimension, source_dimension), "
            f"received {transformation.shape} for source dimension {resolved_metric.dimension}."
        )
    required_rank = min(transformation.shape)
    if np.linalg.matrix_rank(transformation) != required_rank:
        raise ValueError("coordinate_jacobian must have full row rank.")
    transformed = transformation @ resolved_metric.components @ transformation.T
    return build_metric_tensor(transformed, label=label, symmetry_atol=symmetry_atol)


def _resolve_stabilizer_verification(
    *,
    stabilizer: Any | None,
    precision: int,
    simulate_boundary_decoherence: bool,
) -> tuple[Any, tuple[int, int, int]]:
    from shbt.core.holographic_error_stabilizer import BENCHMARK_BRANCH, HolographicStabilizer

    resolved_stabilizer = (
        HolographicStabilizer(
            precision=max(int(precision), 1),
            simulate_boundary_decoherence=simulate_boundary_decoherence,
        )
        if stabilizer is None
        else stabilizer
    )
    if hasattr(resolved_stabilizer, "verify_bulk_integrity"):
        verification = resolved_stabilizer.verify_bulk_integrity()
    elif hasattr(resolved_stabilizer, "verify_bulk_checksum"):
        verification = resolved_stabilizer.verify_bulk_checksum()
    else:
        raise TypeError(
            "stabilizer must expose `verify_bulk_integrity()` or `verify_bulk_checksum()` for metric projection."
        )

    benchmark_branch = tuple(int(index) for index in getattr(verification, "benchmark_branch", BENCHMARK_BRANCH))
    return verification, benchmark_branch


def project_static_block_to_bulk(
    metric: MetricTensor | npt.ArrayLike,
    block_to_bulk_projector: npt.ArrayLike,
    *,
    stabilizer: Any | None = None,
    precision: int = 200,
    simulate_boundary_decoherence: bool = False,
    label: str = "projected_bulk_metric_tensor",
) -> StabilizedBulkMetricProjection:
    resolved_metric = metric if isinstance(metric, MetricTensor) else build_metric_tensor(metric, label="static_block_metric")
    if resolved_metric.dimension != 4:
        raise ValueError("Static-to-bulk metric projection requires a 4D block metric tensor.")

    projector = require_real_array(block_to_bulk_projector, label="block_to_bulk_projector")
    expected_shape = (3, resolved_metric.dimension)
    if projector.ndim != 2 or projector.shape != expected_shape:
        raise ValueError(
            f"block_to_bulk_projector must have shape {expected_shape}, received {projector.shape}."
        )
    if np.linalg.matrix_rank(projector) != 3:
        raise ValueError("block_to_bulk_projector must have rank 3 to project the static 4D block into the bulk.")

    verification, benchmark_branch = _resolve_stabilizer_verification(
        stabilizer=stabilizer,
        precision=precision,
        simulate_boundary_decoherence=simulate_boundary_decoherence,
    )
    zero_energy_boundary_locked = bool(
        getattr(verification, "zero_energy_boundary_locked", getattr(verification, "passed", False))
    )
    equivalence_principle_preserved = bool(
        getattr(verification, "equivalence_principle_preserved", True)
    )
    torsion_projection_residual = float(getattr(verification, "torsion_projection_residual", 0.0))
    stabilizer_detail = str(getattr(verification, "detail", "bulk integrity locked"))

    if not (
        bool(getattr(verification, "passed", False))
        and zero_energy_boundary_locked
        and equivalence_principle_preserved
        and math.isclose(torsion_projection_residual, 0.0, rel_tol=0.0, abs_tol=SYMMETRY_ATOL)
    ):
        raise ValueError(
            "HolographicStabilizer rejected bulk metric evolution: "
            f"{stabilizer_detail}."
        )

    bulk_metric = coordinate_transform(
        resolved_metric,
        projector,
        label=label,
    )
    if not bulk_metric.positive_definite:
        raise ValueError("Projected bulk metric must remain positive-definite after holographic stabilization.")

    return StabilizedBulkMetricProjection(
        static_metric=resolved_metric,
        bulk_metric=bulk_metric,
        block_to_bulk_projector=projector,
        benchmark_branch=benchmark_branch,
        zero_energy_boundary_locked=zero_energy_boundary_locked,
        equivalence_principle_preserved=equivalence_principle_preserved,
        torsion_projection_residual=torsion_projection_residual,
        stabilizer_detail=stabilizer_detail,
    )


def christoffel_symbols(
    metric: MetricTensor | npt.ArrayLike,
    metric_derivatives: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    resolved_metric = metric if isinstance(metric, MetricTensor) else build_metric_tensor(metric)
    derivatives = require_real_array(metric_derivatives, label="metric_derivatives")
    expected_shape = (resolved_metric.dimension, resolved_metric.dimension, resolved_metric.dimension)
    if derivatives.shape != expected_shape:
        raise ValueError(
            "metric_derivatives must have shape (dimension, dimension, dimension), "
            f"received {derivatives.shape} and expected {expected_shape}."
        )
    if not np.allclose(derivatives, np.swapaxes(derivatives, 1, 2), atol=SYMMETRY_ATOL, rtol=0.0):
        raise ValueError("metric_derivatives must preserve symmetry in the metric indices.")

    inverse_metric = resolved_metric.inverse
    dimension = resolved_metric.dimension
    gamma = np.zeros((dimension, dimension, dimension), dtype=np.float64)
    for upper_index in range(dimension):
        for first_lower_index in range(dimension):
            for second_lower_index in range(dimension):
                total = 0.0
                for contracted_index in range(dimension):
                    total += inverse_metric[upper_index, contracted_index] * (
                        derivatives[first_lower_index, contracted_index, second_lower_index]
                        + derivatives[second_lower_index, contracted_index, first_lower_index]
                        - derivatives[contracted_index, first_lower_index, second_lower_index]
                    )
                gamma[upper_index, first_lower_index, second_lower_index] = 0.5 * total
    return _freeze_array(gamma)


def riemann_curvature_tensor(
    christoffel: npt.ArrayLike,
    christoffel_derivatives: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    gamma = require_real_array(christoffel, label="christoffel")
    if gamma.ndim != 3 or gamma.shape[0] != gamma.shape[1] or gamma.shape[1] != gamma.shape[2]:
        raise ValueError(f"christoffel must have shape (n, n, n), received {gamma.shape}.")

    dimension = int(gamma.shape[0])
    derivatives = require_real_array(christoffel_derivatives, label="christoffel_derivatives")
    expected_shape = (dimension, dimension, dimension, dimension)
    if derivatives.shape != expected_shape:
        raise ValueError(
            "christoffel_derivatives must have shape (n, n, n, n), "
            f"received {derivatives.shape} and expected {expected_shape}."
        )

    riemann = np.zeros((dimension, dimension, dimension, dimension), dtype=np.float64)
    for upper_index in range(dimension):
        for lower_index in range(dimension):
            for first_curvature_index in range(dimension):
                for second_curvature_index in range(dimension):
                    connection_terms = 0.0
                    for contracted_index in range(dimension):
                        connection_terms += (
                            gamma[upper_index, first_curvature_index, contracted_index]
                            * gamma[contracted_index, lower_index, second_curvature_index]
                            - gamma[upper_index, second_curvature_index, contracted_index]
                            * gamma[contracted_index, lower_index, first_curvature_index]
                        )
                    riemann[upper_index, lower_index, first_curvature_index, second_curvature_index] = (
                        derivatives[first_curvature_index, upper_index, lower_index, second_curvature_index]
                        - derivatives[second_curvature_index, upper_index, lower_index, first_curvature_index]
                        + connection_terms
                    )
    return _freeze_array(riemann)


def ricci_tensor(riemann: npt.ArrayLike) -> npt.NDArray[np.float64]:
    curvature = require_real_array(riemann, label="riemann")
    if curvature.ndim != 4 or len({curvature.shape[0], curvature.shape[1], curvature.shape[2], curvature.shape[3]}) != 1:
        raise ValueError(f"riemann must have shape (n, n, n, n), received {curvature.shape}.")
    return _freeze_array(np.einsum("ijil->jl", curvature))


def ricci_scalar(metric: MetricTensor | npt.ArrayLike, ricci: npt.ArrayLike) -> float:
    resolved_metric = metric if isinstance(metric, MetricTensor) else build_metric_tensor(metric)
    ricci_matrix = _require_square_matrix(ricci, label="ricci")
    if ricci_matrix.shape != resolved_metric.components.shape:
        raise ValueError(
            f"ricci must match metric shape {resolved_metric.components.shape}, received {ricci_matrix.shape}."
        )
    return require_real_scalar(
        np.trace(resolved_metric.inverse @ ricci_matrix),
        label=f"{resolved_metric.label}.ricci_scalar",
    )


def build_manifold_state(metric: MetricTensor | npt.ArrayLike, ricci: npt.ArrayLike) -> ManifoldState:
    resolved_metric = metric if isinstance(metric, MetricTensor) else build_metric_tensor(metric)
    resolved_ricci = _require_square_matrix(ricci, label="ricci")
    return ManifoldState(
        metric=resolved_metric,
        ricci_tensor=resolved_ricci,
        ricci_scalar=ricci_scalar(resolved_metric, resolved_ricci),
        volume_element=volume_element(resolved_metric),
    )


def ricci_flow_step(
    metric: MetricTensor | npt.ArrayLike,
    ricci: npt.ArrayLike,
    *,
    step_size: float,
    label: str = "ricci_flow_metric",
) -> MetricTensor:
    resolved_metric = metric if isinstance(metric, MetricTensor) else build_metric_tensor(metric)
    resolved_ricci = _require_square_matrix(ricci, label="ricci")
    if resolved_ricci.shape != resolved_metric.components.shape:
        raise ValueError(
            f"ricci must match metric shape {resolved_metric.components.shape}, received {resolved_ricci.shape}."
        )
    evolved_metric = resolved_metric.components - 2.0 * float(step_size) * resolved_ricci
    return build_metric_tensor(evolved_metric, label=label)


def evolve_riemannian_manifold(
    metric: MetricTensor | npt.ArrayLike,
    ricci_sequence: Iterable[npt.ArrayLike],
    *,
    step_size: float,
    label: str = "riemannian_manifold",
) -> RiemannianManifoldEvolution:
    resolved_metric = metric if isinstance(metric, MetricTensor) else build_metric_tensor(metric, label=label)
    if not resolved_metric.positive_definite:
        raise ValueError("Riemannian manifold evolution requires a positive-definite metric tensor.")

    zero_ricci = np.zeros_like(resolved_metric.components)
    states = [build_manifold_state(resolved_metric, zero_ricci)]
    current_metric = resolved_metric
    for index, ricci in enumerate(ricci_sequence, start=1):
        current_metric = ricci_flow_step(
            current_metric,
            ricci,
            step_size=step_size,
            label=f"{label}[{index}]",
        )
        states.append(build_manifold_state(current_metric, ricci))

    return RiemannianManifoldEvolution(step_size=float(step_size), states=tuple(states))


__all__ = [
    "COMPLEX_CONSISTENCY_ATOL",
    "ManifoldState",
    "MetricTensor",
    "RiemannianManifoldEvolution",
    "SYMMETRY_ATOL",
    "StabilizedBulkMetricProjection",
    "build_manifold_state",
    "build_metric_tensor",
    "christoffel_symbols",
    "coordinate_transform",
    "evolve_riemannian_manifold",
    "freeze_numpy_arrays",
    "line_element",
    "lower_index",
    "metric_inner_product",
    "project_static_block_to_bulk",
    "raise_index",
    "require_real_array",
    "require_real_scalar",
    "ricci_flow_step",
    "ricci_scalar",
    "ricci_tensor",
    "riemann_curvature_tensor",
    "volume_element",
]
