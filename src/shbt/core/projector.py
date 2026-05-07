from __future__ import annotations

"""Boundary-Determined Bulk Actualization for SHBT.

This module now exposes two compatible layers:

- ``HolographicCompiler``: the original prime-lattice compiler that turns the
  benchmark boundary dictionary into a single static ``3+1`` bulk metric.
- ``BulkProjector``: a Radau-IIA-driven bulk projector that ingests solver
  trajectories, converts prime-indexed bit-loading sequences on ``M_pi`` into a
  continuous metric tensor field ``g_{mu nu}(tau)``, and makes the rendered bulk
  a direct consequence of topological closure at the boundary.

In the BDBA reading, the boundary is not a source of constants alone. The
ordered prime lattice carries the information load, the Radau transport
trajectory supplies the execution history, and the closed boundary theorem fixes
which trajectories are allowed to actualize a macroscopic spacetime metric.
"""

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from fractions import Fraction
import math
from typing import Any, Final, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from shbt.core.derivation_api import DEFAULT_PRECISION, TopologicalVacuum, UniverseFactory
from shbt.core.differential_geometry import MetricTensor, build_metric_tensor, require_real_array
from shbt.core.ontic_cascade import OnticCascade, evaluate_ontic_cascade


_GUARD_DIGITS: Final[int] = 16
_SPACETIME_DIMENSION: Final[int] = 4
_SPATIAL_DIMENSION: Final[int] = 3
_BENCHMARK_PRIMES: Final[tuple[int, ...]] = (2, 3, 5, 7, 11)
_METRIC_REGULARIZATION: Final[float] = 1.0e-12


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


def _fraction_to_decimal(value: Fraction, *, precision: int = DEFAULT_PRECISION) -> Decimal:
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
        context.prec = max(int(precision), DEFAULT_PRECISION)
        return +decimal_value


def _is_prime(value: int) -> bool:
    resolved_value = int(value)
    if resolved_value < 2:
        return False
    if resolved_value == 2:
        return True
    if resolved_value % 2 == 0:
        return False
    limit = int(math.isqrt(resolved_value))
    for candidate in range(3, limit + 1, 2):
        if resolved_value % candidate == 0:
            return False
    return True


def _freeze_array(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    frozen = np.array(values, dtype=np.float64, copy=True)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True)
class PrimeIndexedLatticeSite:
    prime: int
    label: str
    amplitude: Decimal | Fraction | float | int | str

    def __post_init__(self) -> None:
        resolved_prime = int(self.prime)
        resolved_amplitude = _decimal(self.amplitude)
        if not _is_prime(resolved_prime):
            raise ValueError(f"Prime-indexed lattice site requires a prime index, received {resolved_prime}.")
        if resolved_amplitude <= 0:
            raise ValueError("Prime-indexed lattice amplitudes must be strictly positive.")
        if not self.label:
            raise ValueError("Prime-indexed lattice sites require a non-empty label.")
        object.__setattr__(self, "prime", resolved_prime)
        object.__setattr__(self, "amplitude", resolved_amplitude)


@dataclass(frozen=True)
class PrimeIndexedInformationLattice:
    vacuum: TopologicalVacuum
    sites: tuple[PrimeIndexedLatticeSite, ...]
    normalized_state: tuple[Decimal, ...] = field(init=False)

    def __post_init__(self) -> None:
        resolved_sites = tuple(self.sites)
        if len(resolved_sites) < _SPACETIME_DIMENSION + 1:
            raise ValueError(
                "Boundary-Determined Bulk Actualization requires at least five prime-indexed lattice sites "
                "to realize a 3+1 bulk geometry."
            )
        primes = tuple(site.prime for site in resolved_sites)
        if any(primes[index] >= primes[index + 1] for index in range(len(primes) - 1)):
            raise ValueError("Prime-indexed lattice sites must be ordered by strictly increasing prime index.")
        total_amplitude = sum((site.amplitude for site in resolved_sites), Decimal("0"))
        if total_amplitude <= 0:
            raise ValueError("Prime-indexed information lattice must carry positive total support.")

        with localcontext() as context:
            context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
            normalized_state = [site.amplitude / total_amplitude for site in resolved_sites]
            running_total = sum(normalized_state[:-1], Decimal("0"))
            normalized_state[-1] = Decimal("1") - running_total

        object.__setattr__(self, "sites", resolved_sites)
        object.__setattr__(self, "normalized_state", tuple(normalized_state))

    @property
    def branch(self) -> tuple[int, int, int]:
        return self.vacuum.branch

    @property
    def primes(self) -> tuple[int, ...]:
        return tuple(site.prime for site in self.sites)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(site.label for site in self.sites)

    @property
    def amplitudes(self) -> tuple[Decimal, ...]:
        return tuple(site.amplitude for site in self.sites)

    @property
    def state_change_count(self) -> int:
        return len(self.sites) - 1


@dataclass(frozen=True)
class EulerFixedPointAdjunction:
    step_index: int
    source_site: PrimeIndexedLatticeSite
    target_site: PrimeIndexedLatticeSite
    source_state: Decimal
    target_state: Decimal
    state_change: Decimal
    logarithmic_prime_ratio: Decimal
    euler_flux: Decimal
    fixed_point_residue: Decimal
    harmonic_bridge: Decimal
    adjunction_strength: Decimal
    execution_vector: tuple[float, float, float, float]
    metric_increment: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_increment", _freeze_array(self.metric_increment))

    @property
    def resolves_topology(self) -> bool:
        return self.adjunction_strength > 0 and self.fixed_point_residue >= 0


@dataclass(frozen=True)
class RadauBoundaryTrace:
    lattice: PrimeIndexedInformationLattice
    solver_method: str
    parameter_grid: npt.ArrayLike
    state_matrix: npt.ArrayLike
    bit_loading_sequences: npt.ArrayLike
    state_chunk_map: tuple[tuple[int, ...], ...]
    closure_audit: OnticCascade
    closure_residue: Decimal
    closure_scale: Decimal

    def __post_init__(self) -> None:
        parameter_grid = require_real_array(self.parameter_grid, label="parameter_grid")
        if parameter_grid.ndim != 1 or parameter_grid.size < 2:
            raise ValueError("parameter_grid must contain at least two ordered Radau samples.")
        if np.any(np.diff(parameter_grid) <= 0.0):
            raise ValueError("parameter_grid must be strictly increasing.")

        state_matrix = require_real_array(self.state_matrix, label="state_matrix")
        if state_matrix.ndim != 2 or state_matrix.shape[1] != parameter_grid.size:
            raise ValueError(
                "state_matrix must have shape (state_dimension, sample_count) with sample_count matching parameter_grid."
            )
        if state_matrix.shape[0] < len(self.lattice.sites):
            raise ValueError(
                "state_matrix must expose at least as many solver channels as prime-indexed lattice sites."
            )

        bit_loading_sequences = require_real_array(self.bit_loading_sequences, label="bit_loading_sequences")
        expected_shape = (len(self.lattice.sites), parameter_grid.size)
        if bit_loading_sequences.shape != expected_shape:
            raise ValueError(
                "bit_loading_sequences must have shape "
                f"{expected_shape}, received {bit_loading_sequences.shape}."
            )
        if np.any(bit_loading_sequences <= 0.0):
            raise ValueError("bit_loading_sequences must remain strictly positive.")
        if not np.allclose(np.sum(bit_loading_sequences, axis=0), 1.0, atol=1.0e-10, rtol=0.0):
            raise ValueError("Each bit-loading slice must normalize to unity across the prime lattice.")

        object.__setattr__(self, "parameter_grid", _freeze_array(parameter_grid))
        object.__setattr__(self, "state_matrix", _freeze_array(state_matrix))
        object.__setattr__(self, "bit_loading_sequences", _freeze_array(bit_loading_sequences))
        object.__setattr__(self, "solver_method", str(self.solver_method))
        object.__setattr__(self, "closure_residue", _decimal(self.closure_residue))
        object.__setattr__(self, "closure_scale", _decimal(self.closure_scale))
        object.__setattr__(self, "state_chunk_map", tuple(tuple(int(index) for index in chunk) for chunk in self.state_chunk_map))

    @property
    def sample_count(self) -> int:
        return int(self.parameter_grid.size)

    @property
    def state_dimension(self) -> int:
        return int(self.state_matrix.shape[0])

    @property
    def site_count(self) -> int:
        return int(self.bit_loading_sequences.shape[0])

    @property
    def time_span(self) -> tuple[float, float]:
        return float(self.parameter_grid[0]), float(self.parameter_grid[-1])

    @property
    def topological_closure_locked(self) -> bool:
        return bool(self.closure_audit.axiom_ix.topological_closure)

    @property
    def final_bit_loading(self) -> npt.NDArray[np.float64]:
        return _freeze_array(self.bit_loading_sequences[:, -1])


@dataclass(frozen=True)
class ContinuousMetricTensorField:
    benchmark_branch: tuple[int, int, int]
    parameter_grid: npt.ArrayLike
    metric_components: npt.ArrayLike
    bit_loading_sequences: npt.ArrayLike
    closure_scale: Decimal
    closure_audit: OnticCascade

    def __post_init__(self) -> None:
        parameter_grid = require_real_array(self.parameter_grid, label="metric_field.parameter_grid")
        metric_components = require_real_array(self.metric_components, label="metric_field.metric_components")
        bit_loading_sequences = require_real_array(self.bit_loading_sequences, label="metric_field.bit_loading_sequences")
        if parameter_grid.ndim != 1 or parameter_grid.size < 2:
            raise ValueError("metric field requires at least two ordered samples.")
        expected_metric_shape = (parameter_grid.size, _SPACETIME_DIMENSION, _SPACETIME_DIMENSION)
        if metric_components.shape != expected_metric_shape:
            raise ValueError(
                "metric_components must have shape "
                f"{expected_metric_shape}, received {metric_components.shape}."
            )
        expected_loading_shape = (len(_BENCHMARK_PRIMES), parameter_grid.size)
        if bit_loading_sequences.shape != expected_loading_shape:
            raise ValueError(
                "bit_loading_sequences must have shape "
                f"{expected_loading_shape}, received {bit_loading_sequences.shape}."
            )
        for index, matrix in enumerate(metric_components):
            if not np.allclose(matrix, matrix.T, atol=1.0e-12, rtol=0.0):
                raise ValueError(f"metric_components[{index}] must remain symmetric.")
        object.__setattr__(self, "benchmark_branch", tuple(int(entry) for entry in self.benchmark_branch))
        object.__setattr__(self, "parameter_grid", _freeze_array(parameter_grid))
        object.__setattr__(self, "metric_components", _freeze_array(metric_components))
        object.__setattr__(self, "bit_loading_sequences", _freeze_array(bit_loading_sequences))
        object.__setattr__(self, "closure_scale", _decimal(self.closure_scale))

    @property
    def sample_count(self) -> int:
        return int(self.parameter_grid.size)

    @property
    def topological_closure_locked(self) -> bool:
        return bool(self.closure_audit.axiom_ix.topological_closure)

    @property
    def final_metric(self) -> MetricTensor:
        return build_metric_tensor(self.metric_components[-1], label="bdba_continuum_spacetime_metric")

    @property
    def initial_metric(self) -> MetricTensor:
        return build_metric_tensor(self.metric_components[0], label="bdba_initial_spacetime_metric")

    def metric_at(self, parameter_value: float) -> MetricTensor:
        resolved_parameter_value = float(parameter_value)
        if resolved_parameter_value <= float(self.parameter_grid[0]):
            interpolated = self.metric_components[0]
        elif resolved_parameter_value >= float(self.parameter_grid[-1]):
            interpolated = self.metric_components[-1]
        else:
            upper_index = int(np.searchsorted(self.parameter_grid, resolved_parameter_value))
            lower_index = upper_index - 1
            lower_parameter = float(self.parameter_grid[lower_index])
            upper_parameter = float(self.parameter_grid[upper_index])
            interpolation_weight = (resolved_parameter_value - lower_parameter) / (upper_parameter - lower_parameter)
            interpolated = (1.0 - interpolation_weight) * self.metric_components[lower_index] + interpolation_weight * self.metric_components[upper_index]
        interpolated = 0.5 * (interpolated + interpolated.T)
        return build_metric_tensor(interpolated, label=f"bdba_metric_tensor@{resolved_parameter_value:.6e}")


@dataclass(frozen=True)
class BoundaryDeterminedBulkGeometry:
    lattice: PrimeIndexedInformationLattice
    adjunctions: tuple[EulerFixedPointAdjunction, ...]
    spacetime_metric: MetricTensor
    spatial_metric: MetricTensor
    execution_residue: Decimal
    resolution_scale: Decimal
    diagonal_completion: tuple[Decimal, Decimal, Decimal, Decimal]
    metric_field: ContinuousMetricTensorField | None = None
    boundary_trace: RadauBoundaryTrace | None = None
    closure_audit: OnticCascade | None = None
    boundary_closure_locked: bool = True

    @property
    def spacetime_dimension(self) -> int:
        return self.spacetime_metric.dimension

    @property
    def spatial_dimension(self) -> int:
        return self.spatial_metric.dimension

    @property
    def metric_rank(self) -> int:
        return int(np.linalg.matrix_rank(self.spacetime_metric.components))

    @property
    def continuum_sample_count(self) -> int:
        return int(self.metric_field.sample_count) if self.metric_field is not None else 0

    @property
    def emergent_from_execution(self) -> bool:
        continuum_resolved = self.metric_field is None or self.continuum_sample_count >= 2
        return bool(
            self.lattice.state_change_count >= _SPACETIME_DIMENSION
            and len(self.adjunctions) == self.lattice.state_change_count
            and self.spacetime_dimension == _SPACETIME_DIMENSION
            and self.spatial_dimension == _SPATIAL_DIMENSION
            and self.metric_rank == _SPACETIME_DIMENSION
            and continuum_resolved
        )

    @property
    def boundary_determines_bulk(self) -> bool:
        return bool(self.emergent_from_execution and all(adjunction.resolves_topology for adjunction in self.adjunctions))

    @property
    def rendered_from_boundary_closure(self) -> bool:
        if self.metric_field is None or self.boundary_trace is None or self.closure_audit is None:
            return bool(self.boundary_determines_bulk)
        return bool(self.boundary_determines_bulk and self.boundary_closure_locked and self.metric_field.topological_closure_locked)

    @property
    def statement(self) -> str:
        if self.metric_field is not None and self.boundary_trace is not None:
            return (
                "The 3+1 bulk metric actualizes by feeding Radau IIA bit-loading trajectories on the "
                "prime-indexed boundary lattice into a boundary-closure projector."
            )
        return (
            "The 3+1 bulk metric actualizes only through ordered Euler fixed-point adjunctions on the "
            "prime-indexed boundary lattice."
        )


class HolographicCompiler:
    """Compile a prime-indexed boundary lattice into an emergent 3+1 bulk metric."""

    def __init__(self, *, precision: int = DEFAULT_PRECISION) -> None:
        self.precision = max(int(precision), DEFAULT_PRECISION)

    def ingest_lattice(
        self,
        lattice: PrimeIndexedInformationLattice | Mapping[int, Decimal | Fraction | float | int | str] | Sequence[PrimeIndexedLatticeSite],
        *,
        vacuum: TopologicalVacuum | None = None,
        labels: Mapping[int, str] | None = None,
    ) -> PrimeIndexedInformationLattice:
        if isinstance(lattice, PrimeIndexedInformationLattice):
            return lattice

        resolved_vacuum = UniverseFactory.benchmark_vacuum() if vacuum is None else vacuum
        if isinstance(lattice, Mapping):
            ordered_primes = tuple(sorted(int(prime) for prime in lattice))
            resolved_labels = {} if labels is None else {int(prime): str(label) for prime, label in labels.items()}
            sites = tuple(
                PrimeIndexedLatticeSite(
                    prime=prime,
                    label=resolved_labels.get(prime, f"M_pi[{prime}]"),
                    amplitude=lattice[prime],
                )
                for prime in ordered_primes
            )
            return PrimeIndexedInformationLattice(vacuum=resolved_vacuum, sites=sites)

        sites = tuple(lattice)
        return PrimeIndexedInformationLattice(vacuum=resolved_vacuum, sites=sites)

    def build_benchmark_lattice(self, *, vacuum: TopologicalVacuum | None = None) -> PrimeIndexedInformationLattice:
        resolved_vacuum = UniverseFactory.benchmark_vacuum() if vacuum is None else vacuum
        geometry = UniverseFactory.derive_central_charge_geometry(vacuum=resolved_vacuum)
        vacuum_pressure = UniverseFactory.derive_vacuum_pressure(vacuum=resolved_vacuum)
        kappa = UniverseFactory.derive_kappa_d5(precision=self.precision, vacuum=resolved_vacuum)

        visible_support_density = Decimal(resolved_vacuum.visible_support) / Decimal(resolved_vacuum.parent_level)
        sites = (
            PrimeIndexedLatticeSite(
                prime=_BENCHMARK_PRIMES[0],
                label="visible_support_density",
                amplitude=visible_support_density,
            ),
            PrimeIndexedLatticeSite(
                prime=_BENCHMARK_PRIMES[1],
                label="central_charge_ratio",
                amplitude=geometry.central_charge_ratio_decimal,
            ),
            PrimeIndexedLatticeSite(
                prime=_BENCHMARK_PRIMES[2],
                label="inverse_pixel_volume",
                amplitude=geometry.inverse_pixel_volume_decimal,
            ),
            PrimeIndexedLatticeSite(
                prime=_BENCHMARK_PRIMES[3],
                label="vacuum_pressure",
                amplitude=vacuum_pressure.vacuum_pressure,
            ),
            PrimeIndexedLatticeSite(
                prime=_BENCHMARK_PRIMES[4],
                label="geometric_kappa",
                amplitude=kappa.kappa,
            ),
        )
        return PrimeIndexedInformationLattice(vacuum=resolved_vacuum, sites=sites)

    def derive_euler_fixed_point_adjunctions(
        self,
        lattice: PrimeIndexedInformationLattice,
    ) -> tuple[EulerFixedPointAdjunction, ...]:
        adjunctions: list[EulerFixedPointAdjunction] = []

        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            for step_index in range(lattice.state_change_count):
                source_site = lattice.sites[step_index]
                target_site = lattice.sites[step_index + 1]
                source_state = lattice.normalized_state[step_index]
                target_state = lattice.normalized_state[step_index + 1]
                state_change = target_state - source_state
                logarithmic_prime_ratio = _decimal_ln(Decimal(target_site.prime) / Decimal(source_site.prime))
                euler_flux = state_change / logarithmic_prime_ratio
                fixed_point_envelope = (source_state + target_state) / Decimal(2)
                fixed_point_residue = abs(euler_flux - fixed_point_envelope)
                harmonic_bridge = Decimal(1) / Decimal(source_site.prime) + Decimal(1) / Decimal(target_site.prime)
                adjunction_strength = abs(euler_flux) + fixed_point_envelope + harmonic_bridge

                execution_vector = (
                    float(Decimal(1) / Decimal(source_site.prime)),
                    float(source_state),
                    float(target_state),
                    float(abs(euler_flux) + Decimal(1) / Decimal(target_site.prime)),
                )
                metric_increment = float(adjunction_strength) * np.outer(execution_vector, execution_vector)

                adjunctions.append(
                    EulerFixedPointAdjunction(
                        step_index=step_index,
                        source_site=source_site,
                        target_site=target_site,
                        source_state=+source_state,
                        target_state=+target_state,
                        state_change=+state_change,
                        logarithmic_prime_ratio=+logarithmic_prime_ratio,
                        euler_flux=+euler_flux,
                        fixed_point_residue=+fixed_point_residue,
                        harmonic_bridge=+harmonic_bridge,
                        adjunction_strength=+adjunction_strength,
                        execution_vector=execution_vector,
                        metric_increment=metric_increment,
                    )
                )

        return tuple(adjunctions)

    def compile(self, lattice: PrimeIndexedInformationLattice | None = None) -> BoundaryDeterminedBulkGeometry:
        resolved_lattice = self.build_benchmark_lattice() if lattice is None else lattice
        adjunctions = self.derive_euler_fixed_point_adjunctions(resolved_lattice)

        raw_metric = np.sum([adjunction.metric_increment for adjunction in adjunctions], axis=0)
        execution_residue = sum((adjunction.fixed_point_residue for adjunction in adjunctions), Decimal("0"))

        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            diagonal_completion = tuple(adjunction.adjunction_strength for adjunction in adjunctions[:_SPACETIME_DIMENSION])
            resolution_scale = Decimal(1) / (
                sum((Decimal(site.prime) for site in resolved_lattice.sites), Decimal("0"))
                + sum((adjunction.adjunction_strength for adjunction in adjunctions), Decimal("0"))
            )

        completion_matrix = float(resolution_scale) * np.diag([float(value) for value in diagonal_completion])
        emergent_metric_matrix = np.asarray(raw_metric + completion_matrix, dtype=float)
        emergent_metric_matrix = 0.5 * (emergent_metric_matrix + emergent_metric_matrix.T)
        trace = float(np.trace(emergent_metric_matrix))
        if not np.isfinite(trace) or trace <= 0.0:
            raise RuntimeError("Emergent metric trace must remain finite and positive.")

        emergent_metric_matrix = emergent_metric_matrix / trace
        eigenspectrum = np.linalg.eigvalsh(emergent_metric_matrix)
        if float(np.min(eigenspectrum)) <= 0.0:
            emergent_metric_matrix += np.eye(_SPACETIME_DIMENSION, dtype=float) * (abs(float(np.min(eigenspectrum))) + _METRIC_REGULARIZATION)
            emergent_metric_matrix = emergent_metric_matrix / float(np.trace(emergent_metric_matrix))
        spacetime_metric = build_metric_tensor(
            emergent_metric_matrix,
            label="bdba_spacetime_metric",
        )
        spatial_metric = build_metric_tensor(
            emergent_metric_matrix[1:, 1:],
            label="bdba_spatial_metric",
        )

        return BoundaryDeterminedBulkGeometry(
            lattice=resolved_lattice,
            adjunctions=adjunctions,
            spacetime_metric=spacetime_metric,
            spatial_metric=spatial_metric,
            execution_residue=+execution_residue,
            resolution_scale=+resolution_scale,
            diagonal_completion=tuple(+value for value in diagonal_completion),
        )

    def actualize_bulk_geometry(self, lattice: PrimeIndexedInformationLattice | None = None) -> BoundaryDeterminedBulkGeometry:
        return self.compile(lattice=lattice)

    def actualize_benchmark_bulk_geometry(self) -> BoundaryDeterminedBulkGeometry:
        return self.compile(self.build_benchmark_lattice())


class BulkProjector(HolographicCompiler):
    """Project Radau-IIA boundary trajectories into a continuous ``3+1`` bulk metric field."""

    def __init__(self, *, precision: int = DEFAULT_PRECISION, require_radau_method: bool = True) -> None:
        super().__init__(precision=precision)
        self.require_radau_method = bool(require_radau_method)
        self._trace_cache: dict[tuple[int, str, PrimeIndexedInformationLattice], RadauBoundaryTrace] = {}

    def _trace_cache_key(
        self,
        solution: Any,
        *,
        lattice: PrimeIndexedInformationLattice,
        method: str,
    ) -> tuple[int, str, PrimeIndexedInformationLattice]:
        return (id(solution), str(method), lattice)

    def _closure_residue(self, closure_audit: OnticCascade) -> Decimal:
        residue = Decimal("0")
        if not closure_audit.axiom_ix.diophantine_pass:
            minimal_parent_level = max(int(closure_audit.axiom_ix.minimal_parent_level), 1)
            residue += abs(Decimal(closure_audit.axioms.parent_level - minimal_parent_level)) / Decimal(minimal_parent_level)
        if not closure_audit.axiom_ix.framing_pass:
            residue += _fraction_to_decimal(closure_audit.axiom_ix.framing_gap, precision=self.precision)
        if not closure_audit.axiom_ix.gko_pass and closure_audit.axiom_ix.gko_c_dark_residue <= 0:
            residue += abs(closure_audit.axiom_ix.gko_c_dark_residue)
        residue += abs(closure_audit.axiom_ix.closure_tensor_amplitude)
        return +residue

    def _resolve_solution_method(self, solution: Any, method: str | None = None) -> str:
        resolved_method = str(getattr(solution, "method", method or "Radau")).strip()
        if not resolved_method:
            resolved_method = "Radau"
        if self.require_radau_method and resolved_method != "Radau":
            raise ValueError(f"BulkProjector requires Radau IIA transport output, received {resolved_method!r}.")
        return resolved_method

    def ingest_solver_output(
        self,
        solution: Any,
        *,
        lattice: PrimeIndexedInformationLattice | None = None,
        vacuum: TopologicalVacuum | None = None,
        labels: Mapping[int, str] | None = None,
        method: str | None = None,
    ) -> RadauBoundaryTrace:
        resolved_method = self._resolve_solution_method(solution, method=method)
        if hasattr(solution, "success") and not bool(solution.success):
            raise ValueError(f"Radau IIA solver output is unsuccessful: {getattr(solution, 'message', '')}")
        if not hasattr(solution, "t") or not hasattr(solution, "y"):
            raise TypeError("BulkProjector requires a solver output exposing 't' and 'y' arrays.")

        resolved_lattice = self.build_benchmark_lattice(vacuum=vacuum) if lattice is None else self.ingest_lattice(lattice, vacuum=vacuum, labels=labels)
        cache_key = self._trace_cache_key(solution, lattice=resolved_lattice, method=resolved_method)
        cached_trace = self._trace_cache.get(cache_key)
        if cached_trace is not None:
            return cached_trace

        parameter_grid = require_real_array(getattr(solution, "t"), label="solution.t")
        state_matrix = require_real_array(getattr(solution, "y"), label="solution.y")
        if state_matrix.ndim != 2:
            raise ValueError("solution.y must be a rank-2 state matrix.")
        if parameter_grid.ndim != 1 or parameter_grid.size < 2:
            raise ValueError("solution.t must contain at least two monotone Radau samples.")
        if state_matrix.shape[1] != parameter_grid.size:
            raise ValueError("solution.y sample count must match solution.t.")
        if state_matrix.shape[0] < len(resolved_lattice.sites):
            raise ValueError(
                "solution.y must expose at least five state channels to reconstruct the 3+1 bulk metric."
            )
        if np.any(~np.isfinite(state_matrix)):
            raise ValueError("solution.y must remain finite across the full Radau trajectory.")

        chunk_indices = tuple(
            tuple(int(index) for index in chunk)
            for chunk in np.array_split(np.arange(state_matrix.shape[0]), len(resolved_lattice.sites))
        )
        reduced_sequences = np.vstack(
            [
                np.mean(np.abs(state_matrix[np.asarray(chunk, dtype=int), :]), axis=0)
                for chunk in chunk_indices
            ]
        )
        base_amplitudes = np.asarray([float(site.amplitude) for site in resolved_lattice.sites], dtype=float).reshape(-1, 1)
        combined_sequences = base_amplitudes * (1.0 + reduced_sequences)
        combined_sequences_sum = np.sum(combined_sequences, axis=0, keepdims=True)
        bit_loading_sequences = combined_sequences / combined_sequences_sum

        closure_audit = evaluate_ontic_cascade(resolved_lattice.vacuum.to_ontic_axioms(), precision=self.precision)
        closure_residue = self._closure_residue(closure_audit)
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            closure_scale = Decimal("1") / (Decimal("1") + closure_residue)

        trace = RadauBoundaryTrace(
            lattice=resolved_lattice,
            solver_method=resolved_method,
            parameter_grid=parameter_grid,
            state_matrix=state_matrix,
            bit_loading_sequences=bit_loading_sequences,
            state_chunk_map=chunk_indices,
            closure_audit=closure_audit,
            closure_residue=+closure_residue,
            closure_scale=+closure_scale,
        )
        self._trace_cache[cache_key] = trace
        return trace

    def _stabilize_metric_matrix(self, metric_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        stabilized = np.asarray(metric_matrix, dtype=float)
        stabilized = 0.5 * (stabilized + stabilized.T)
        eigenspectrum = np.linalg.eigvalsh(stabilized)
        minimum_eigenvalue = float(np.min(eigenspectrum))
        if minimum_eigenvalue <= 0.0:
            stabilized += np.eye(_SPACETIME_DIMENSION, dtype=float) * (abs(minimum_eigenvalue) + _METRIC_REGULARIZATION)
        trace = float(np.trace(stabilized))
        if not np.isfinite(trace) or trace <= 0.0:
            raise RuntimeError("Projected bulk metric trace must remain finite and positive.")
        stabilized = stabilized / trace
        stabilized = 0.5 * (stabilized + stabilized.T)
        return np.asarray(stabilized, dtype=float)

    def _metric_slice_from_load_vector(
        self,
        load_vector: npt.NDArray[np.float64],
        *,
        primes: Sequence[int],
        closure_scale: float,
    ) -> npt.NDArray[np.float64]:
        metric_matrix = np.zeros((_SPACETIME_DIMENSION, _SPACETIME_DIMENSION), dtype=float)
        diagonal_completion = np.ones(_SPACETIME_DIMENSION, dtype=float) * max(float(closure_scale), _METRIC_REGULARIZATION)
        for step_index in range(_SPACETIME_DIMENSION):
            source_state = float(load_vector[step_index])
            target_state = float(load_vector[step_index + 1])
            logarithmic_prime_ratio = math.log(float(primes[step_index + 1]) / float(primes[step_index]))
            euler_flux = (target_state - source_state) / logarithmic_prime_ratio
            fixed_point_envelope = 0.5 * (source_state + target_state)
            harmonic_bridge = 1.0 / float(primes[step_index]) + 1.0 / float(primes[step_index + 1])
            projection_weight = max(float(closure_scale), _METRIC_REGULARIZATION) * (
                abs(euler_flux) + fixed_point_envelope + harmonic_bridge
            )
            basis_vector = np.eye(_SPACETIME_DIMENSION, dtype=float)[step_index]
            mixed_vector = np.array(
                [
                    source_state + (1.0 if step_index == 0 else 0.0),
                    target_state + (1.0 if step_index == 1 else 0.0),
                    float(load_vector[(step_index + 2) % load_vector.size]) + (1.0 if step_index == 2 else 0.0),
                    abs(euler_flux) + float(load_vector[(step_index + 3) % load_vector.size]) + (1.0 if step_index == 3 else 0.0),
                ],
                dtype=float,
            )
            execution_vector = basis_vector + mixed_vector
            metric_matrix += projection_weight * np.outer(execution_vector, execution_vector)
            diagonal_completion[step_index] += projection_weight + source_state + target_state
        metric_matrix += np.diag(diagonal_completion)
        return self._stabilize_metric_matrix(metric_matrix)

    def project_metric_field(self, trace: RadauBoundaryTrace) -> ContinuousMetricTensorField:
        closure_scale = float(trace.closure_scale)
        metric_components = np.stack(
            [
                self._metric_slice_from_load_vector(
                    np.asarray(trace.bit_loading_sequences[:, index], dtype=float),
                    primes=trace.lattice.primes,
                    closure_scale=closure_scale,
                )
                for index in range(trace.sample_count)
            ],
            axis=0,
        )
        return ContinuousMetricTensorField(
            benchmark_branch=trace.lattice.branch,
            parameter_grid=trace.parameter_grid,
            metric_components=metric_components,
            bit_loading_sequences=trace.bit_loading_sequences,
            closure_scale=trace.closure_scale,
            closure_audit=trace.closure_audit,
        )

    def project_bulk(
        self,
        solution: Any,
        *,
        lattice: PrimeIndexedInformationLattice | None = None,
        vacuum: TopologicalVacuum | None = None,
        labels: Mapping[int, str] | None = None,
        method: str | None = None,
        require_boundary_closure: bool = False,
    ) -> BoundaryDeterminedBulkGeometry:
        trace = self.ingest_solver_output(
            solution,
            lattice=lattice,
            vacuum=vacuum,
            labels=labels,
            method=method,
        )
        if require_boundary_closure and not trace.topological_closure_locked:
            raise RuntimeError("Boundary topological closure is required before actualizing the bulk metric.")

        metric_field = self.project_metric_field(trace)
        static_geometry = self.compile(trace.lattice)
        spacetime_metric = metric_field.final_metric
        spatial_metric = build_metric_tensor(
            spacetime_metric.components[1:, 1:],
            label="bdba_continuum_spatial_metric",
        )
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            execution_residue = static_geometry.execution_residue + trace.closure_residue
            resolution_scale = static_geometry.resolution_scale * trace.closure_scale
        diagonal_completion = tuple(_decimal(value) for value in np.diag(spacetime_metric.components))
        return BoundaryDeterminedBulkGeometry(
            lattice=trace.lattice,
            adjunctions=static_geometry.adjunctions,
            spacetime_metric=spacetime_metric,
            spatial_metric=spatial_metric,
            execution_residue=+execution_residue,
            resolution_scale=+resolution_scale,
            diagonal_completion=tuple(+value for value in diagonal_completion),
            metric_field=metric_field,
            boundary_trace=trace,
            closure_audit=trace.closure_audit,
            boundary_closure_locked=trace.topological_closure_locked,
        )

    def actualize_bulk_geometry_from_solver(
        self,
        solution: Any,
        *,
        lattice: PrimeIndexedInformationLattice | None = None,
        vacuum: TopologicalVacuum | None = None,
        labels: Mapping[int, str] | None = None,
        method: str | None = None,
        require_boundary_closure: bool = False,
    ) -> BoundaryDeterminedBulkGeometry:
        return self.project_bulk(
            solution,
            lattice=lattice,
            vacuum=vacuum,
            labels=labels,
            method=method,
            require_boundary_closure=require_boundary_closure,
        )


def build_benchmark_prime_indexed_lattice() -> PrimeIndexedInformationLattice:
    return HolographicCompiler().build_benchmark_lattice()


def actualize_boundary_determined_bulk(
    solution: Any | None = None,
    *,
    lattice: PrimeIndexedInformationLattice | None = None,
    vacuum: TopologicalVacuum | None = None,
    labels: Mapping[int, str] | None = None,
    precision: int = DEFAULT_PRECISION,
    method: str | None = None,
    require_boundary_closure: bool = False,
) -> BoundaryDeterminedBulkGeometry:
    if solution is None:
        return HolographicCompiler(precision=precision).actualize_benchmark_bulk_geometry()
    return BulkProjector(precision=precision).project_bulk(
        solution,
        lattice=lattice,
        vacuum=vacuum,
        labels=labels,
        method=method,
        require_boundary_closure=require_boundary_closure,
    )


def actualize_boundary_determined_bulk_from_radau(
    solution: Any,
    *,
    lattice: PrimeIndexedInformationLattice | None = None,
    vacuum: TopologicalVacuum | None = None,
    labels: Mapping[int, str] | None = None,
    precision: int = DEFAULT_PRECISION,
    method: str | None = None,
    require_boundary_closure: bool = False,
) -> BoundaryDeterminedBulkGeometry:
    return BulkProjector(precision=precision).project_bulk(
        solution,
        lattice=lattice,
        vacuum=vacuum,
        labels=labels,
        method=method,
        require_boundary_closure=require_boundary_closure,
    )


__all__ = [
    "BoundaryDeterminedBulkGeometry",
    "BulkProjector",
    "ContinuousMetricTensorField",
    "EulerFixedPointAdjunction",
    "HolographicCompiler",
    "PrimeIndexedInformationLattice",
    "PrimeIndexedLatticeSite",
    "RadauBoundaryTrace",
    "actualize_boundary_determined_bulk",
    "actualize_boundary_determined_bulk_from_radau",
    "build_benchmark_prime_indexed_lattice",
]
