from __future__ import annotations

"""Bridge static 26D boundary residues into 4D scale-dependent couplings.

This module packages the existing UV-to-IR cooling logic into a single
observer-facing transport bridge. The guiding picture is simple:

- the anomaly-free ``(26, 8, 312)`` boundary fixes a static ultraviolet
  electromagnetic residue ``alpha_UV^{-1}``,
- the projection into the 4D bulk discards 22 boundary dimensions and therefore
  loses a finite amount of inverse-coupling information, and
- the loss is distributed across scale by a monotonic cooling function so the
  infrared coupling becomes a mandatory consequence of the transport geometry
  rather than a fitted target.

The bridge exposes a callable ``ScaleDependentTransport`` operator for
single-scale queries, a dedicated ``CoolingFunction`` for the 26D -> 4D
projection, and an explicit experiment-facing verification against the
fine-structure constant.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np

from shbt.constants import CODATA_FINE_STRUCTURE_ALPHA_INVERSE, LEPTON_LEVEL, MZ_SCALE_GEV, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.derivation_api import DEFAULT_PRECISION, TopologicalVacuum, UniverseFactory
from shbt.core.renormalization_map import (
    BENCHMARK_BOUNDARY_BRANCH,
    BENCHMARK_UV_SCALE_GEV,
    BULK_DIMENSION,
    CODATA_ALIGNMENT_ABS_TOLERANCE,
    UVIRRendering,
    calculate_information_loss,
    holographic_smearing_fraction,
    map_alpha_uv_to_ir,
    uv_to_ir_rendering_function,
)


def _require_benchmark_vacuum(vacuum: TopologicalVacuum | None) -> TopologicalVacuum:
    resolved_vacuum = UniverseFactory.benchmark_vacuum() if vacuum is None else vacuum
    if resolved_vacuum.branch != BENCHMARK_BOUNDARY_BRANCH:
        raise ValueError(
            "Renormalization bridge is defined only on the anomaly-free (26, 8, 312) boundary manifold."
        )
    return resolved_vacuum


@dataclass(frozen=True)
class CoolingProjection:
    energy_gev: float
    alpha_uv_inverse: float
    boundary_dimension: int
    bulk_dimension: int
    discarded_dimensions: int
    smearing_fraction: float
    information_loss_budget_inverse: float
    information_loss_inverse: float
    retained_information_inverse: float
    retained_information_fraction: float
    boundary_bit_density_inverse: float
    bulk_bit_density_inverse: float
    smeared_bit_density_inverse: float

    @property
    def lost_information_fraction(self) -> float:
        return float(1.0 - self.retained_information_fraction)

    @property
    def fully_cooled(self) -> bool:
        return self.smearing_fraction >= 1.0


@dataclass(frozen=True)
class CoolingFunction:
    alpha_uv_inverse: float
    boundary_dimension: int = int(LEPTON_LEVEL)
    bulk_dimension: int = int(BULK_DIMENSION)
    uv_scale_gev: float = float(BENCHMARK_UV_SCALE_GEV)

    @property
    def discarded_dimensions(self) -> int:
        return int(self.boundary_dimension - self.bulk_dimension)

    @property
    def projection_ratio(self) -> float:
        return float(self.bulk_dimension / self.boundary_dimension)

    @property
    def information_loss_budget_inverse(self) -> float:
        return float(
            calculate_information_loss(
                0.0,
                alpha_uv=self.alpha_uv_inverse,
                uv_scale_gev=self.uv_scale_gev,
            )
        )

    def smearing_fraction(self, energy_gev: float) -> float:
        resolved_energy = float(energy_gev)
        if resolved_energy <= 0.0:
            return 1.0
        return float(holographic_smearing_fraction(resolved_energy, uv_scale_gev=self.uv_scale_gev))

    def project(self, energy_gev: float) -> CoolingProjection:
        resolved_energy = float(energy_gev)
        smearing_fraction = self.smearing_fraction(resolved_energy)
        information_loss_inverse = float(
            calculate_information_loss(
                resolved_energy,
                alpha_uv=self.alpha_uv_inverse,
                uv_scale_gev=self.uv_scale_gev,
            )
        )
        retained_information_inverse = float(self.alpha_uv_inverse - information_loss_inverse)
        if retained_information_inverse <= 0.0:
            raise ValueError("Cooling projection drove the inverse coupling below zero.")

        boundary_bit_density_inverse = float(self.alpha_uv_inverse / self.boundary_dimension)
        bulk_bit_density_inverse = float(retained_information_inverse / self.bulk_dimension)
        smeared_bit_density_inverse = float(
            information_loss_inverse / self.discarded_dimensions if self.discarded_dimensions > 0 else 0.0
        )
        retained_information_fraction = float(retained_information_inverse / self.alpha_uv_inverse)

        return CoolingProjection(
            energy_gev=resolved_energy,
            alpha_uv_inverse=float(self.alpha_uv_inverse),
            boundary_dimension=int(self.boundary_dimension),
            bulk_dimension=int(self.bulk_dimension),
            discarded_dimensions=self.discarded_dimensions,
            smearing_fraction=smearing_fraction,
            information_loss_budget_inverse=self.information_loss_budget_inverse,
            information_loss_inverse=information_loss_inverse,
            retained_information_inverse=retained_information_inverse,
            retained_information_fraction=retained_information_fraction,
            boundary_bit_density_inverse=boundary_bit_density_inverse,
            bulk_bit_density_inverse=bulk_bit_density_inverse,
            smeared_bit_density_inverse=smeared_bit_density_inverse,
        )

    def __call__(self, energy_gev: float) -> CoolingProjection:
        return self.project(energy_gev)


@dataclass(frozen=True)
class TransportScalePoint:
    step: int
    energy_gev: float
    cooling_fraction: float
    information_loss_inverse: float
    alpha_uv_inverse: float
    transported_alpha_inverse: float
    transported_alpha: float
    cooling: CoolingProjection
    rendering: UVIRRendering | None

    @property
    def has_running_rendering(self) -> bool:
        return self.rendering is not None

    @property
    def retained_information_inverse(self) -> float:
        return self.cooling.retained_information_inverse

    @property
    def retained_information_fraction(self) -> float:
        return self.cooling.retained_information_fraction

    @property
    def boundary_bit_density_inverse(self) -> float:
        return self.cooling.boundary_bit_density_inverse

    @property
    def bulk_bit_density_inverse(self) -> float:
        return self.cooling.bulk_bit_density_inverse

    @property
    def smeared_bit_density_inverse(self) -> float:
        return self.cooling.smeared_bit_density_inverse


@dataclass(frozen=True)
class TransportTrajectory:
    branch: tuple[int, int, int]
    boundary_dimension: int
    bulk_dimension: int
    uv_scale_gev: float
    ir_scale_gev: float
    alpha_uv_inverse: float
    points: tuple[TransportScalePoint, ...]

    @property
    def ir_point(self) -> TransportScalePoint:
        return self.points[-1]

    @property
    def ir_alpha_inverse(self) -> float:
        return self.ir_point.transported_alpha_inverse

    @property
    def ir_alpha(self) -> float:
        return self.ir_point.transported_alpha

    @property
    def information_loss_budget_inverse(self) -> float:
        return self.ir_point.cooling.information_loss_budget_inverse

    @property
    def monotonic_information_loss(self) -> bool:
        losses = [point.information_loss_inverse for point in self.points]
        return all(next_loss >= current_loss for current_loss, next_loss in zip(losses, losses[1:], strict=False))

    @property
    def monotonic_alpha_inverse_cooling(self) -> bool:
        alpha_inverses = [point.transported_alpha_inverse for point in self.points]
        return all(next_value <= current_value for current_value, next_value in zip(alpha_inverses, alpha_inverses[1:], strict=False))


@dataclass(frozen=True)
class TransportGeometryVerification:
    point: TransportScalePoint
    target_alpha: float
    target_alpha_inverse: float
    absolute_error_inverse: float
    relative_error_inverse: float
    tolerance_inverse: float
    target_used_in_transport: bool
    derived_from_geometry_only: bool
    aligns_with_experiment: bool


class ScaleDependentTransport:
    """Transport a static UV boundary residue through a 26D -> 4D cooling map."""

    boundary_dimension: Final[int] = int(LEPTON_LEVEL)
    bulk_dimension: Final[int] = int(BULK_DIMENSION)
    branch: Final[tuple[int, int, int]] = BENCHMARK_BOUNDARY_BRANCH

    def __init__(
        self,
        *,
        precision: int = DEFAULT_PRECISION,
        vacuum: TopologicalVacuum | None = None,
        uv_scale_gev: float = BENCHMARK_UV_SCALE_GEV,
        ir_scale_gev: float = 0.0,
    ) -> None:
        self.precision = max(int(precision), DEFAULT_PRECISION)
        self.vacuum = _require_benchmark_vacuum(vacuum)
        self.uv_scale_gev = float(uv_scale_gev)
        self.ir_scale_gev = float(ir_scale_gev)
        self.alpha_uv_inverse = float(
            UniverseFactory.derive_alpha_surface(
                precision=self.precision,
                vacuum=self.vacuum,
            ).alpha_inverse_decimal
        )
        self.alpha_uv = float(1.0 / self.alpha_uv_inverse)
        self.cooling_operator = CoolingFunction(
            alpha_uv_inverse=self.alpha_uv_inverse,
            boundary_dimension=self.boundary_dimension,
            bulk_dimension=self.bulk_dimension,
            uv_scale_gev=self.uv_scale_gev,
        )

    def __call__(self, energy_gev: float, *, step: int = 0) -> TransportScalePoint:
        return self.transport_to_scale(energy_gev, step=step)

    def cooling_function(self, energy_gev: float) -> float:
        return self.cooling_operator.smearing_fraction(energy_gev)

    def apply(self, energy_gev: float, *, step: int = 0) -> TransportScalePoint:
        return self.transport_to_scale(energy_gev, step=step)

    def transport_to_scale(self, energy_gev: float, *, step: int = 0) -> TransportScalePoint:
        resolved_energy = float(energy_gev)
        cooling = self.cooling_operator.project(resolved_energy)
        transported_alpha_inverse = float(
            map_alpha_uv_to_ir(
                self.alpha_uv_inverse,
                resolved_energy,
                uv_scale_gev=self.uv_scale_gev,
            )
        )
        if not np.isclose(
            transported_alpha_inverse,
            cooling.retained_information_inverse,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError("Cooling projection drifted from the renormalization-map transport.")
        transported_alpha = float(1.0 / transported_alpha_inverse)
        rendering = (
            uv_to_ir_rendering_function(
                resolved_energy,
                precision=self.precision,
                vacuum=self.vacuum,
            )
            if resolved_energy > 0.0
            else None
        )
        return TransportScalePoint(
            step=int(step),
            energy_gev=resolved_energy,
            cooling_fraction=cooling.smearing_fraction,
            information_loss_inverse=cooling.information_loss_inverse,
            alpha_uv_inverse=self.alpha_uv_inverse,
            transported_alpha_inverse=transported_alpha_inverse,
            transported_alpha=transported_alpha,
            cooling=cooling,
            rendering=rendering,
        )

    def derive_ir_alpha_inverse(self, energy_gev: float | None = None) -> float:
        resolved_energy = self.ir_scale_gev if energy_gev is None else float(energy_gev)
        return self.transport_to_scale(resolved_energy).transported_alpha_inverse

    def derive_ir_alpha(self, energy_gev: float | None = None) -> float:
        resolved_energy = self.ir_scale_gev if energy_gev is None else float(energy_gev)
        return self.transport_to_scale(resolved_energy).transported_alpha

    def verify_experimental_alignment(
        self,
        energy_gev: float | None = None,
        *,
        target_alpha_inverse: float = CODATA_FINE_STRUCTURE_ALPHA_INVERSE,
        tolerance_inverse: float = CODATA_ALIGNMENT_ABS_TOLERANCE,
    ) -> TransportGeometryVerification:
        resolved_energy = self.ir_scale_gev if energy_gev is None else float(energy_gev)
        point = self.transport_to_scale(resolved_energy)
        resolved_target_inverse = float(target_alpha_inverse)
        absolute_error_inverse = float(abs(point.transported_alpha_inverse - resolved_target_inverse))
        relative_error_inverse = float(absolute_error_inverse / resolved_target_inverse)
        return TransportGeometryVerification(
            point=point,
            target_alpha=float(1.0 / resolved_target_inverse),
            target_alpha_inverse=resolved_target_inverse,
            absolute_error_inverse=absolute_error_inverse,
            relative_error_inverse=relative_error_inverse,
            tolerance_inverse=float(tolerance_inverse),
            target_used_in_transport=False,
            derived_from_geometry_only=True,
            aligns_with_experiment=bool(absolute_error_inverse <= float(tolerance_inverse)),
        )

    def verify_codata_alignment(
        self,
        energy_gev: float | None = None,
        *,
        tolerance_inverse: float = CODATA_ALIGNMENT_ABS_TOLERANCE,
    ) -> TransportGeometryVerification:
        return self.verify_experimental_alignment(
            energy_gev,
            target_alpha_inverse=CODATA_FINE_STRUCTURE_ALPHA_INVERSE,
            tolerance_inverse=tolerance_inverse,
        )

    def simulate_transport(self, *, sample_count: int = 9) -> TransportTrajectory:
        resolved_sample_count = int(sample_count)
        if resolved_sample_count < 2:
            raise ValueError(f"sample_count must be at least 2, received {sample_count}")

        if self.ir_scale_gev > 0.0:
            energies = tuple(float(value) for value in np.geomspace(self.uv_scale_gev, self.ir_scale_gev, resolved_sample_count, dtype=float))
        else:
            positive_count = max(1, resolved_sample_count - 1)
            positive_energies = tuple(
                float(value)
                for value in np.geomspace(self.uv_scale_gev, MZ_SCALE_GEV, positive_count, dtype=float)
            )
            energies = positive_energies + (0.0,)
            if resolved_sample_count == 2:
                energies = (self.uv_scale_gev, 0.0)

        points = tuple(self.transport_to_scale(energy, step=index) for index, energy in enumerate(energies))
        return TransportTrajectory(
            branch=self.branch,
            boundary_dimension=self.boundary_dimension,
            bulk_dimension=self.bulk_dimension,
            uv_scale_gev=self.uv_scale_gev,
            ir_scale_gev=self.ir_scale_gev,
            alpha_uv_inverse=self.alpha_uv_inverse,
            points=points,
        )


def verify_scale_dependent_transport_experiment(
    *,
    precision: int = DEFAULT_PRECISION,
    vacuum: TopologicalVacuum | None = None,
    uv_scale_gev: float = BENCHMARK_UV_SCALE_GEV,
    ir_scale_gev: float = 0.0,
    target_alpha_inverse: float = CODATA_FINE_STRUCTURE_ALPHA_INVERSE,
    tolerance_inverse: float = CODATA_ALIGNMENT_ABS_TOLERANCE,
) -> TransportGeometryVerification:
    bridge = ScaleDependentTransport(
        precision=precision,
        vacuum=vacuum,
        uv_scale_gev=uv_scale_gev,
        ir_scale_gev=ir_scale_gev,
    )
    return bridge.verify_experimental_alignment(
        target_alpha_inverse=target_alpha_inverse,
        tolerance_inverse=tolerance_inverse,
    )


__all__ = [
    "CoolingFunction",
    "CoolingProjection",
    "ScaleDependentTransport",
    "TransportGeometryVerification",
    "TransportScalePoint",
    "TransportTrajectory",
    "verify_scale_dependent_transport_experiment",
]
