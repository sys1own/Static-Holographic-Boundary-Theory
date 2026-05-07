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

The bridge exposes a ``ScaleDependentTransport`` class for single-scale queries
and for full UV-to-IR cooling trajectories.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np

from shbt.constants import LEPTON_LEVEL, MZ_SCALE_GEV, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.derivation_api import DEFAULT_PRECISION, TopologicalVacuum, UniverseFactory
from shbt.core.renormalization_map import (
    BENCHMARK_BOUNDARY_BRANCH,
    BENCHMARK_UV_SCALE_GEV,
    BULK_DIMENSION,
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
class TransportScalePoint:
    step: int
    energy_gev: float
    cooling_fraction: float
    information_loss_inverse: float
    alpha_uv_inverse: float
    transported_alpha_inverse: float
    transported_alpha: float
    rendering: UVIRRendering | None

    @property
    def has_running_rendering(self) -> bool:
        return self.rendering is not None


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
    def monotonic_information_loss(self) -> bool:
        losses = [point.information_loss_inverse for point in self.points]
        return all(next_loss >= current_loss for current_loss, next_loss in zip(losses, losses[1:], strict=False))

    @property
    def monotonic_alpha_inverse_cooling(self) -> bool:
        alpha_inverses = [point.transported_alpha_inverse for point in self.points]
        return all(next_value <= current_value for current_value, next_value in zip(alpha_inverses, alpha_inverses[1:], strict=False))


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

    def cooling_function(self, energy_gev: float) -> float:
        resolved_energy = float(energy_gev)
        if resolved_energy <= 0.0:
            return 1.0
        return float(holographic_smearing_fraction(resolved_energy, uv_scale_gev=self.uv_scale_gev))

    def transport_to_scale(self, energy_gev: float, *, step: int = 0) -> TransportScalePoint:
        resolved_energy = float(energy_gev)
        cooling_fraction = self.cooling_function(resolved_energy)
        information_loss_inverse = float(
            calculate_information_loss(
                resolved_energy,
                alpha_uv=self.alpha_uv_inverse,
                precision=self.precision,
                vacuum=self.vacuum,
                uv_scale_gev=self.uv_scale_gev,
            )
        )
        transported_alpha_inverse = float(
            map_alpha_uv_to_ir(
                self.alpha_uv_inverse,
                resolved_energy,
                uv_scale_gev=self.uv_scale_gev,
            )
        )
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
            cooling_fraction=cooling_fraction,
            information_loss_inverse=information_loss_inverse,
            alpha_uv_inverse=self.alpha_uv_inverse,
            transported_alpha_inverse=transported_alpha_inverse,
            transported_alpha=transported_alpha,
            rendering=rendering,
        )

    def derive_ir_alpha_inverse(self, energy_gev: float | None = None) -> float:
        resolved_energy = self.ir_scale_gev if energy_gev is None else float(energy_gev)
        return self.transport_to_scale(resolved_energy).transported_alpha_inverse

    def derive_ir_alpha(self, energy_gev: float | None = None) -> float:
        resolved_energy = self.ir_scale_gev if energy_gev is None else float(energy_gev)
        return self.transport_to_scale(resolved_energy).transported_alpha

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

        points = tuple(
            self.transport_to_scale(energy, step=index)
            for index, energy in enumerate(energies)
        )
        return TransportTrajectory(
            branch=self.branch,
            boundary_dimension=self.boundary_dimension,
            bulk_dimension=self.bulk_dimension,
            uv_scale_gev=self.uv_scale_gev,
            ir_scale_gev=self.ir_scale_gev,
            alpha_uv_inverse=self.alpha_uv_inverse,
            points=points,
        )


__all__ = [
    "ScaleDependentTransport",
    "TransportScalePoint",
    "TransportTrajectory",
]
