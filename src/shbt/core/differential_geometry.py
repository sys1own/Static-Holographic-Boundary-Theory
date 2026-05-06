from __future__ import annotations

"""Differential-geometry primitives and real-valued numeric consistency helpers.

This module retains the original array/sanity helpers that lived in
``numerics.py`` and extends them with a light-weight differential-geometry
toolbox: metric tensors, index raising/lowering, Christoffel symbols, curvature
contractions, and a simple Ricci-flow style manifold evolution step.
"""

from dataclasses import dataclass, fields, is_dataclass
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
    "build_manifold_state",
    "build_metric_tensor",
    "christoffel_symbols",
    "evolve_riemannian_manifold",
    "freeze_numpy_arrays",
    "line_element",
    "lower_index",
    "metric_inner_product",
    "raise_index",
    "require_real_array",
    "require_real_scalar",
    "ricci_flow_step",
    "ricci_scalar",
    "ricci_tensor",
    "riemann_curvature_tensor",
    "volume_element",
]
