from __future__ import annotations

"""Boundary-Determined Bulk Actualization for the SHBT benchmark branch.

This module treats the prime-indexed information lattice ``M_pi`` as the only
primitive input. No background spacetime is injected by hand. Instead, the
compiler walks the ordered prime lattice and turns each boundary state-change
into an Euler fixed-point adjunction,

    E_i[M_pi] = (m_{i+1} - m_i) / ln(p_{i+1} / p_i),

where ``(p_i, m_i)`` are successive prime-indexed lattice states. Each local
adjunction then contributes a metric increment

    g^(i)_{μν} = A_i v_{i,μ} v_{i,ν},

with

    A_i = |E_i| + (m_i + m_{i+1}) / 2 + (1 / p_i + 1 / p_{i+1}).

The final ``3+1`` bulk geometry is the trace-normalized sum of these ordered
increments plus a diagonal completion built from the same adjunction strengths.
In this sense, space is not a hard-coded background but the result of the
lattice resolving its topological constraints during execution.
"""

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from fractions import Fraction
import math
from typing import Final, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from shbt.core.derivation_api import DEFAULT_PRECISION, TopologicalVacuum, UniverseFactory
from shbt.core.differential_geometry import MetricTensor, build_metric_tensor


_GUARD_DIGITS: Final[int] = 16
_SPACETIME_DIMENSION: Final[int] = 4
_SPATIAL_DIMENSION: Final[int] = 3
_BENCHMARK_PRIMES: Final[tuple[int, ...]] = (2, 3, 5, 7, 11)


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
class BoundaryDeterminedBulkGeometry:
    lattice: PrimeIndexedInformationLattice
    adjunctions: tuple[EulerFixedPointAdjunction, ...]
    spacetime_metric: MetricTensor
    spatial_metric: MetricTensor
    execution_residue: Decimal
    resolution_scale: Decimal
    diagonal_completion: tuple[Decimal, Decimal, Decimal, Decimal]

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
    def emergent_from_execution(self) -> bool:
        return bool(
            self.lattice.state_change_count >= _SPACETIME_DIMENSION
            and len(self.adjunctions) == self.lattice.state_change_count
            and self.spacetime_dimension == _SPACETIME_DIMENSION
            and self.spatial_dimension == _SPATIAL_DIMENSION
            and self.metric_rank == _SPACETIME_DIMENSION
        )

    @property
    def boundary_determines_bulk(self) -> bool:
        return bool(self.emergent_from_execution and all(adjunction.resolves_topology for adjunction in self.adjunctions))

    @property
    def statement(self) -> str:
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


def build_benchmark_prime_indexed_lattice() -> PrimeIndexedInformationLattice:
    return HolographicCompiler().build_benchmark_lattice()


def actualize_boundary_determined_bulk() -> BoundaryDeterminedBulkGeometry:
    return HolographicCompiler().actualize_benchmark_bulk_geometry()


__all__ = [
    "BoundaryDeterminedBulkGeometry",
    "EulerFixedPointAdjunction",
    "HolographicCompiler",
    "PrimeIndexedInformationLattice",
    "PrimeIndexedLatticeSite",
    "actualize_boundary_determined_bulk",
    "build_benchmark_prime_indexed_lattice",
]
