from __future__ import annotations

"""High-level façade for SHBT scale-dependent renormalization.

This module packages the existing renormalization map and bridge into a single
transport-facing API. The physical picture is:

- the anomaly-free ``(26, 8, 312)`` boundary fixes a static ultraviolet residue
  ``alpha_UV^{-1}``,
- the 26D -> 4D projection is non-invertible, because 22 boundary dimensions
  are discarded into an aggregate cooling budget rather than a recoverable
  microscopic preimage, and
- the observed infrared coupling is the cooled image of that branch-fixed
  residue after the bulk smears the lost bit-density over scale.

The module re-exports the lower-level transport operator and provides a concise
``derive_running_couplings`` entrypoint for programmatic use.
"""

from dataclasses import dataclass

from shbt.constants import CODATA_FINE_STRUCTURE_ALPHA_INVERSE
from shbt.core.derivation_api import DEFAULT_PRECISION, TopologicalVacuum
from shbt.core.renormalization_bridge import (
    CoolingFunction,
    CoolingProjection,
    ScaleDependentTransport,
    TransportGeometryVerification,
    TransportScalePoint,
    TransportTrajectory,
)
from shbt.core.renormalization_map import (
    BENCHMARK_UV_SCALE_GEV,
    CODATA_ALIGNMENT_ABS_TOLERANCE,
    NonInvertibleTransport,
    NonInvertibleTransportState,
    noninvertible_transport,
)


@dataclass(frozen=True)
class RunningCouplingAudit:
    transport: ScaleDependentTransport
    uv_point: TransportScalePoint
    ir_point: TransportScalePoint
    trajectory: TransportTrajectory
    verification: TransportGeometryVerification
    noninvertible_ir_transport: NonInvertibleTransportState

    @property
    def alpha_uv_inverse(self) -> float:
        return float(self.uv_point.transported_alpha_inverse)

    @property
    def alpha_ir_inverse(self) -> float:
        return float(self.ir_point.transported_alpha_inverse)

    @property
    def alpha_shift_inverse(self) -> float:
        return float(self.alpha_uv_inverse - self.alpha_ir_inverse)

    @property
    def alpha_ir(self) -> float:
        return float(self.ir_point.transported_alpha)

    @property
    def bulk_dimension(self) -> int:
        return int(self.transport.bulk_dimension)

    @property
    def boundary_dimension(self) -> int:
        return int(self.transport.boundary_dimension)

    @property
    def discarded_dimensions(self) -> int:
        return int(self.boundary_dimension - self.bulk_dimension)

    @property
    def transport_is_noninvertible(self) -> bool:
        return True

    @property
    def cooling_is_monotonic(self) -> bool:
        return bool(
            self.trajectory.monotonic_information_loss
            and self.trajectory.monotonic_alpha_inverse_cooling
        )

    @property
    def smeared_bit_density_inverse(self) -> float:
        return float(self.ir_point.smeared_bit_density_inverse)

    @property
    def aligns_with_experiment(self) -> bool:
        return bool(self.verification.aligns_with_experiment)

    @property
    def statement(self) -> str:
        return (
            "The infrared fine-structure residue is the non-invertible, scale-cooled image of "
            "the branch-fixed 26D boundary coupling."
        )


def derive_running_couplings(
    *,
    precision: int = DEFAULT_PRECISION,
    vacuum: TopologicalVacuum | None = None,
    uv_scale_gev: float = BENCHMARK_UV_SCALE_GEV,
    ir_scale_gev: float = 0.0,
    sample_count: int = 9,
    target_alpha_inverse: float = CODATA_FINE_STRUCTURE_ALPHA_INVERSE,
    tolerance_inverse: float = CODATA_ALIGNMENT_ABS_TOLERANCE,
) -> RunningCouplingAudit:
    """Derive the UV -> IR running of the branch-fixed electromagnetic residue."""

    transport = ScaleDependentTransport(
        precision=precision,
        vacuum=vacuum,
        uv_scale_gev=uv_scale_gev,
        ir_scale_gev=ir_scale_gev,
    )
    uv_point = transport.transport_to_scale(transport.uv_scale_gev)
    ir_point = transport.transport_to_scale(transport.ir_scale_gev)
    trajectory = transport.simulate_transport(sample_count=sample_count)
    verification = transport.verify_experimental_alignment(
        target_alpha_inverse=target_alpha_inverse,
        tolerance_inverse=tolerance_inverse,
    )
    noninvertible_ir_transport = noninvertible_transport(
        transport.ir_scale_gev,
        precision=precision,
        vacuum=vacuum,
    )
    return RunningCouplingAudit(
        transport=transport,
        uv_point=uv_point,
        ir_point=ir_point,
        trajectory=trajectory,
        verification=verification,
        noninvertible_ir_transport=noninvertible_ir_transport,
    )


def simulate_boundary_cooling(
    *,
    precision: int = DEFAULT_PRECISION,
    vacuum: TopologicalVacuum | None = None,
    uv_scale_gev: float = BENCHMARK_UV_SCALE_GEV,
    ir_scale_gev: float = 0.0,
    sample_count: int = 9,
) -> TransportTrajectory:
    """Convenience wrapper returning the full UV -> IR cooling trajectory."""

    transport = ScaleDependentTransport(
        precision=precision,
        vacuum=vacuum,
        uv_scale_gev=uv_scale_gev,
        ir_scale_gev=ir_scale_gev,
    )
    return transport.simulate_transport(sample_count=sample_count)


__all__ = [
    "BENCHMARK_UV_SCALE_GEV",
    "CODATA_ALIGNMENT_ABS_TOLERANCE",
    "CoolingFunction",
    "CoolingProjection",
    "NonInvertibleTransport",
    "NonInvertibleTransportState",
    "RunningCouplingAudit",
    "ScaleDependentTransport",
    "TransportGeometryVerification",
    "TransportScalePoint",
    "TransportTrajectory",
    "derive_running_couplings",
    "noninvertible_transport",
    "simulate_boundary_cooling",
]
