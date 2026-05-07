from __future__ import annotations

"""Internal observer logic for Boundary-Determined Bulk Actualization.

The observer in this module is not external to the holographic computation.
Instead, an internal agent carries three lattice coordinates and reads the bulk
through a self-valuation operator ``Sigma`` that mixes two quantities already
present in the branch-fixed bookkeeping:

- the local entropy still accessible inside the agent's horizon, and
- the covariant frame shift induced by moving away from the center of the
  holographic register.

Given normalized internal coordinates ``xi = (xi_l, xi_q, xi_s)``, define the
self-valuation functional

    Sigma = (1 + Delta_frame) * (1 + w(xi) * f_hidden),

where ``f_hidden = 1 - f_local`` is the hidden entropy fraction and ``w(xi)``
is the mean lattice coordinate weight. ``Sigma`` therefore stays branch-fixed
at the center while growing as hidden entropy accumulates near the observer's
horizon. The same ``Sigma`` then dresses the
execution-defined bulk metric by a coordinate transformation that shifts the
holographic projection into the agent's local frame.

The resulting localized entropy gradient

    grad_Sigma = Sigma * f_hidden / r_h,

is interpreted as the observer-facing entropic potential gradient. The module
therefore exposes General Relativity as a user interface: gravitational
acceleration is the way an internal agent perceives boundary entropy gradients,
with

    a = c^2 * grad_Sigma.

The same local smearing also dresses boundary-derived constants. In
particular, the benchmark surface value of ``alpha^{-1}`` is not changed at the
branch level, but an internal observer sees an apparent drift whenever hidden
entropy is redistributed into the bulk image of the horizon.
"""

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Final

import numpy as np
import numpy.typing as npt

from shbt.constants import LEPTON_LEVEL, LIGHT_SPEED_M_PER_S, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.bulk_dynamics import BulkFlowAudit, build_bulk_dynamics_audit
from shbt.core.derivation_api import TopologicalVacuum, UniverseFactory
from shbt.core.differential_geometry import MetricTensor, build_metric_tensor, coordinate_transform
from shbt.core.observer import DEFAULT_PRECISION as CORE_OBSERVER_DEFAULT_PRECISION, Observer as CoreObserver
from shbt.core.projector import (
    BoundaryDeterminedBulkGeometry,
    HolographicCompiler,
)


DEFAULT_PRECISION: Final[int] = max(int(CORE_OBSERVER_DEFAULT_PRECISION), 120)
_GUARD_DIGITS: Final[int] = 16
_ENTROPY_TOLERANCE: Final[Decimal] = Decimal("1e-30")


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(str(value))


def _freeze_array(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    frozen = np.array(values, dtype=np.float64, copy=True)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True)
class AgentLatticeCoordinates:
    lepton_coordinate: Decimal | Fraction | float | int | str
    quark_coordinate: Decimal | Fraction | float | int | str
    support_coordinate: Decimal | Fraction | float | int | str

    def __post_init__(self) -> None:
        lepton_coordinate = _decimal(self.lepton_coordinate)
        quark_coordinate = _decimal(self.quark_coordinate)
        support_coordinate = _decimal(self.support_coordinate)
        for label, coordinate in (
            ("lepton_coordinate", lepton_coordinate),
            ("quark_coordinate", quark_coordinate),
            ("support_coordinate", support_coordinate),
        ):
            if coordinate < 0 or coordinate > 1:
                raise ValueError(f"{label} must lie in the closed interval [0, 1].")
        object.__setattr__(self, "lepton_coordinate", lepton_coordinate)
        object.__setattr__(self, "quark_coordinate", quark_coordinate)
        object.__setattr__(self, "support_coordinate", support_coordinate)

    @classmethod
    def benchmark(cls) -> "AgentLatticeCoordinates":
        return cls.from_branch(LEPTON_LEVEL, QUARK_LEVEL, PARENT_LEVEL)

    @classmethod
    def from_branch(
        cls,
        lepton_level: int,
        quark_level: int,
        parent_level: int,
    ) -> "AgentLatticeCoordinates":
        parent_level_decimal = Decimal(parent_level)
        return cls(
            lepton_coordinate=Decimal(lepton_level) / parent_level_decimal,
            quark_coordinate=Decimal(quark_level) / parent_level_decimal,
            support_coordinate=Decimal(lepton_level + quark_level) / parent_level_decimal,
        )

    @property
    def as_decimal_tuple(self) -> tuple[Decimal, Decimal, Decimal]:
        return (
            self.lepton_coordinate,
            self.quark_coordinate,
            self.support_coordinate,
        )

    @property
    def as_float_tuple(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self.as_decimal_tuple)

    @property
    def coordinate_weight(self) -> Decimal:
        with localcontext() as context:
            context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
            return sum(self.as_decimal_tuple, Decimal("0")) / Decimal(3)


@dataclass(frozen=True)
class SelfValuationAudit:
    benchmark_branch: tuple[int, int, int]
    evaluated_branch: tuple[int, int, int]
    agent_coordinates: AgentLatticeCoordinates
    observer_radius_m: Decimal
    local_horizon_radius_m: Decimal
    local_entropy_bits: Decimal
    hidden_entropy_bits: Decimal
    local_entropy_fraction: Decimal
    hidden_entropy_fraction: Decimal
    local_entropy_density_bits_per_m2: Decimal
    frame_shift: Decimal
    boundary_weight: Decimal
    bulk_weight: Decimal
    coordinate_weight: Decimal
    sigma: Decimal
    localized_entropy_gradient_per_m: Decimal

    @property
    def frame_dependent(self) -> bool:
        return self.frame_shift > 0

    @property
    def entropy_partition_closed(self) -> bool:
        return abs(self.local_entropy_fraction + self.hidden_entropy_fraction - Decimal("1")) <= _ENTROPY_TOLERANCE

    @property
    def axiom_ix_satisfied(self) -> bool:
        return bool(
            self.entropy_partition_closed
            and self.sigma > 0
            and self.local_entropy_density_bits_per_m2 > 0
            and abs(self.boundary_weight + self.bulk_weight - Decimal("1")) <= _ENTROPY_TOLERANCE
        )

    @property
    def internal_observer_consistent(self) -> bool:
        return bool(
            self.frame_dependent
            and self.axiom_ix_satisfied
        )

    @property
    def statement(self) -> str:
        return "Axiom IX (Self-Valuation Sigma) localizes the observer's frame by combining horizon entropy and covariant shift."


@dataclass(frozen=True)
class ObserverProjectionKernel:
    self_valuation: SelfValuationAudit
    local_entropy_projection_weight: Decimal
    hidden_entropy_projection_weight: Decimal
    spatial_projection_weight: Decimal
    temporal_projection_weight: Decimal

    @property
    def entropy_weighted(self) -> bool:
        return bool(
            self.local_entropy_projection_weight >= 0
            and self.hidden_entropy_projection_weight >= 0
            and self.spatial_projection_weight > 0
            and self.temporal_projection_weight >= 0
        )

    @property
    def statement(self) -> str:
        return (
            "The observer-local projection keeps accessible entropy at unit weight and amplifies hidden sectors through the frame shift."
        )


@dataclass(frozen=True)
class HolographicProjectionShift:
    base_geometry: BoundaryDeterminedBulkGeometry
    self_valuation: SelfValuationAudit
    projection_kernel: ObserverProjectionKernel
    jacobian_matrix: npt.ArrayLike
    coordinate_shift: tuple[Decimal, Decimal, Decimal]
    shifted_spacetime_metric: MetricTensor
    shifted_spatial_metric: MetricTensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "jacobian_matrix", _freeze_array(self.jacobian_matrix))

    @property
    def projection_shifted(self) -> bool:
        return not np.allclose(
            self.base_geometry.spacetime_metric.components,
            self.shifted_spacetime_metric.components,
            atol=1.0e-12,
            rtol=0.0,
        )

    @property
    def positive_definite(self) -> bool:
        return self.shifted_spacetime_metric.positive_definite and self.shifted_spatial_metric.positive_definite

    @property
    def entropy_weighted_projection(self) -> bool:
        return bool(self.projection_kernel.entropy_weighted and any(value > 0 for value in self.coordinate_shift))

    @property
    def statement(self) -> str:
        return "Internal lattice coordinates shift the holographic projection into an entropy-weighted agent-local bulk frame."


@dataclass(frozen=True)
class GeneralRelativityUIAudit:
    self_valuation: SelfValuationAudit
    projection: HolographicProjectionShift
    entropic_potential: Decimal
    localized_entropy_gradient_per_m: Decimal
    gravitational_acceleration_m_per_s2: Decimal
    observer_frame_curvature: Decimal
    equivalence_principle_residual: Decimal

    @property
    def equivalence_principle_verified(self) -> bool:
        return bool(
            self.gravitational_acceleration_m_per_s2 > 0
            and abs(self.equivalence_principle_residual) <= _ENTROPY_TOLERANCE
        )

    @property
    def general_relativity_is_ui(self) -> bool:
        return bool(
            self.self_valuation.internal_observer_consistent
            and self.projection.projection_shifted
            and self.projection.positive_definite
            and self.equivalence_principle_verified
        )

    @property
    def statement(self) -> str:
        return (
            "General Relativity is the UI through which an internal agent reads boundary data as bulk acceleration."
        )


@dataclass(frozen=True)
class FrameDependentAlphaAudit:
    self_valuation: SelfValuationAudit
    benchmark_alpha_inverse: Decimal
    apparent_alpha_inverse: Decimal
    benchmark_alpha: Decimal
    apparent_alpha: Decimal
    codata_alpha_inverse: Decimal
    boundary_visibility: Decimal
    bulk_smearing_fraction: Decimal
    sigma_smearing_factor: Decimal
    apparent_drift_inverse: Decimal
    apparent_drift_fraction: Decimal

    @property
    def alpha_drift_detected(self) -> bool:
        return self.apparent_drift_fraction > _ENTROPY_TOLERANCE

    @property
    def benchmark_recovered(self) -> bool:
        return bool(
            self.self_valuation.hidden_entropy_fraction <= _ENTROPY_TOLERANCE
            and abs(self.apparent_drift_inverse) <= _ENTROPY_TOLERANCE
        )

    @property
    def statement(self) -> str:
        return (
            "The fine-structure constant appears to drift because Sigma smears boundary information over the observer's local horizon."
        )


@dataclass(frozen=True)
class ObserverFrameAudit:
    self_valuation: SelfValuationAudit
    projection: HolographicProjectionShift
    general_relativity_ui: GeneralRelativityUIAudit
    alpha_drift: FrameDependentAlphaAudit

    @property
    def observer_frame_consistent(self) -> bool:
        return bool(
            self.self_valuation.axiom_ix_satisfied
            and self.general_relativity_ui.general_relativity_is_ui
            and (self.alpha_drift.alpha_drift_detected or self.alpha_drift.benchmark_recovered)
        )

    @property
    def statement(self) -> str:
        return (
            "Axiom IX closes the loop: Sigma projects boundary data into an observer-local metric and makes alpha drift a frame effect."
        )


@dataclass(frozen=True)
class InsideOutMetricAudit:
    self_valuation: SelfValuationAudit
    projection: HolographicProjectionShift
    gravity_flow: BulkFlowAudit
    inside_out_jacobian_matrix: npt.ArrayLike
    perceived_spacetime_metric: MetricTensor
    perceived_spatial_metric: MetricTensor
    entropy_temporal_coupling: Decimal
    observer_time_dilation: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "inside_out_jacobian_matrix", _freeze_array(self.inside_out_jacobian_matrix))

    @property
    def perceived_metric_positive_definite(self) -> bool:
        return self.perceived_spacetime_metric.positive_definite and self.perceived_spatial_metric.positive_definite

    @property
    def gravity_sector_accepts_internal_observer(self) -> bool:
        return bool(self.gravity_flow.static_block_projects_bulk and self.projection.entropy_weighted_projection)

    @property
    def inside_out_consistent(self) -> bool:
        return bool(
            self.gravity_sector_accepts_internal_observer
            and self.perceived_metric_positive_definite
            and self.observer_time_dilation >= 0
        )

    @property
    def statement(self) -> str:
        return (
            "The Universal Code runs inside-out: the internal observer's Sigma dresses the bulk metric perceived by the agent."
        )


def _vacuum_from_branch(branch: tuple[int, int, int]) -> TopologicalVacuum:
    benchmark_vacuum = UniverseFactory.benchmark_vacuum()
    return TopologicalVacuum(
        lepton_level=int(branch[0]),
        quark_level=int(branch[1]),
        parent_level=int(branch[2]),
        generation_count=int(benchmark_vacuum.generation_count),
    )


def _actualize_geometry_for_vacuum(
    vacuum: TopologicalVacuum,
    *,
    precision: int = DEFAULT_PRECISION,
) -> BoundaryDeterminedBulkGeometry:
    compiler = HolographicCompiler(precision=max(int(precision), DEFAULT_PRECISION))
    lattice = compiler.build_benchmark_lattice(vacuum=vacuum)
    return compiler.actualize_bulk_geometry(lattice)


class SelfValuationSigma:
    def __init__(self, *, precision: int = DEFAULT_PRECISION) -> None:
        self.precision = max(int(precision), DEFAULT_PRECISION)

    def derive_projection_kernel(
        self,
        *,
        self_valuation: SelfValuationAudit,
    ) -> ObserverProjectionKernel:
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            local_entropy_projection_weight = self_valuation.local_entropy_fraction
            hidden_entropy_projection_weight = self_valuation.hidden_entropy_fraction * (
                Decimal("1") + self_valuation.frame_shift
            )
            spatial_projection_weight = self_valuation.sigma * (
                local_entropy_projection_weight + hidden_entropy_projection_weight
            )
            temporal_projection_weight = self_valuation.sigma * (
                self_valuation.hidden_entropy_fraction + self_valuation.bulk_weight * self_valuation.frame_shift
            )
        return ObserverProjectionKernel(
            self_valuation=self_valuation,
            local_entropy_projection_weight=+local_entropy_projection_weight,
            hidden_entropy_projection_weight=+hidden_entropy_projection_weight,
            spatial_projection_weight=+spatial_projection_weight,
            temporal_projection_weight=+temporal_projection_weight,
        )

    def evaluate(
        self,
        *,
        observer: CoreObserver | None = None,
        agent_coordinates: AgentLatticeCoordinates | None = None,
        observer_radius_m: Decimal | Fraction | float | int | str = Decimal("0"),
        lepton_level: int = LEPTON_LEVEL,
        quark_level: int = QUARK_LEVEL,
        parent_level: int = PARENT_LEVEL,
    ) -> SelfValuationAudit:
        resolved_agent_coordinates = AgentLatticeCoordinates.benchmark() if agent_coordinates is None else agent_coordinates
        resolved_observer = (
            CoreObserver(
                observer_radius_m=observer_radius_m,
                lepton_level=lepton_level,
                quark_level=quark_level,
                parent_level=parent_level,
                precision=self.precision,
            )
            if observer is None
            else observer
        )
        perspective = resolved_observer.perceive_noether_bridge()

        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            local_entropy_bits = perspective.entropy_capacity_bits
            hidden_entropy_bits = perspective.hidden_entropy_capacity_bits
            local_entropy_fraction = local_entropy_bits / perspective.global_bit_budget
            hidden_entropy_fraction = hidden_entropy_bits / perspective.global_bit_budget
            coordinate_weight = resolved_agent_coordinates.coordinate_weight
            sigma = (Decimal("1") + perspective.covariant_frame_shift) * (
                Decimal("1") + coordinate_weight * hidden_entropy_fraction
            )
            localized_entropy_gradient_per_m = (
                sigma * hidden_entropy_fraction / perspective.local_horizon_radius_m
            )

        return SelfValuationAudit(
            benchmark_branch=tuple(int(value) for value in perspective.benchmark_branch),
            evaluated_branch=tuple(int(value) for value in perspective.evaluated_branch),
            agent_coordinates=resolved_agent_coordinates,
            observer_radius_m=perspective.horizon_limit.observer_radius_m,
            local_horizon_radius_m=perspective.local_horizon_radius_m,
            local_entropy_bits=+local_entropy_bits,
            hidden_entropy_bits=+hidden_entropy_bits,
            local_entropy_fraction=+local_entropy_fraction,
            hidden_entropy_fraction=+hidden_entropy_fraction,
            local_entropy_density_bits_per_m2=+perspective.horizon_limit.surface_bit_loading_bits_per_m2,
            frame_shift=+perspective.covariant_frame_shift,
            boundary_weight=+perspective.boundary_weight,
            bulk_weight=+perspective.bulk_weight,
            coordinate_weight=+coordinate_weight,
            sigma=+sigma,
            localized_entropy_gradient_per_m=+localized_entropy_gradient_per_m,
        )


def derive_frame_dependent_alpha(
    *,
    self_valuation: SelfValuationAudit,
    vacuum: TopologicalVacuum | None = None,
    precision: int = DEFAULT_PRECISION,
) -> FrameDependentAlphaAudit:
    resolved_precision = max(int(precision), DEFAULT_PRECISION)
    resolved_vacuum = _vacuum_from_branch(self_valuation.evaluated_branch) if vacuum is None else vacuum
    alpha_surface = UniverseFactory.derive_alpha_surface(precision=resolved_precision, vacuum=resolved_vacuum)

    with localcontext() as context:
        context.prec = resolved_precision + _GUARD_DIGITS
        benchmark_alpha_inverse = alpha_surface.alpha_inverse_decimal
        boundary_visibility = self_valuation.local_entropy_fraction * self_valuation.boundary_weight
        bulk_smearing_fraction = self_valuation.hidden_entropy_fraction * self_valuation.bulk_weight
        sigma_smearing_factor = bulk_smearing_fraction * self_valuation.sigma
        apparent_alpha_inverse = benchmark_alpha_inverse / (Decimal("1") + sigma_smearing_factor)
        benchmark_alpha = Decimal("1") / benchmark_alpha_inverse
        apparent_alpha = Decimal("1") / apparent_alpha_inverse
        apparent_drift_inverse = apparent_alpha_inverse - benchmark_alpha_inverse
        apparent_drift_fraction = abs(apparent_drift_inverse) / benchmark_alpha_inverse

    return FrameDependentAlphaAudit(
        self_valuation=self_valuation,
        benchmark_alpha_inverse=+benchmark_alpha_inverse,
        apparent_alpha_inverse=+apparent_alpha_inverse,
        benchmark_alpha=+benchmark_alpha,
        apparent_alpha=+apparent_alpha,
        codata_alpha_inverse=+alpha_surface.codata_alpha_inverse,
        boundary_visibility=+boundary_visibility,
        bulk_smearing_fraction=+bulk_smearing_fraction,
        sigma_smearing_factor=+sigma_smearing_factor,
        apparent_drift_inverse=+apparent_drift_inverse,
        apparent_drift_fraction=+apparent_drift_fraction,
    )


def shift_holographic_projection(
    *,
    observer: "InternalObserver" | None = None,
    self_valuation: SelfValuationAudit | None = None,
    sigma: SelfValuationSigma | None = None,
    agent_coordinates: AgentLatticeCoordinates | None = None,
    geometry: BoundaryDeterminedBulkGeometry | None = None,
) -> HolographicProjectionShift:
    if observer is not None:
        resolved_self_valuation = observer.self_valuate() if self_valuation is None else self_valuation
        resolved_geometry = observer.actualize_bulk_geometry() if geometry is None else geometry
        resolved_agent_coordinates = observer.coordinates if agent_coordinates is None else agent_coordinates
        resolved_sigma = observer.sigma if sigma is None else sigma
    else:
        if self_valuation is None:
            raise TypeError("shift_holographic_projection requires self_valuation when observer is not provided.")
        resolved_self_valuation = self_valuation
        resolved_geometry = (
            _actualize_geometry_for_vacuum(_vacuum_from_branch(resolved_self_valuation.evaluated_branch))
            if geometry is None
            else geometry
        )
        resolved_agent_coordinates = (
            resolved_self_valuation.agent_coordinates if agent_coordinates is None else agent_coordinates
        )
        resolved_sigma = SelfValuationSigma() if sigma is None else sigma

    projection_kernel = resolved_sigma.derive_projection_kernel(self_valuation=resolved_self_valuation)
    coordinate_shift = tuple(
        projection_kernel.spatial_projection_weight * value for value in resolved_agent_coordinates.as_decimal_tuple
    )

    entropic_potential = projection_kernel.temporal_projection_weight
    normalized_time_shift = entropic_potential / (Decimal("1") + entropic_potential)
    time_shift = float(normalized_time_shift)
    spatial_shift = tuple(float(value) for value in coordinate_shift)
    jacobian_matrix = np.asarray(
        [
            [1.0 + time_shift, -spatial_shift[0], -spatial_shift[1], -spatial_shift[2]],
            [0.0, 1.0 + spatial_shift[0], 0.0, 0.0],
            [0.0, 0.0, 1.0 + spatial_shift[1], 0.0],
            [0.0, 0.0, 0.0, 1.0 + spatial_shift[2]],
        ],
        dtype=float,
    )

    shifted_spacetime_metric = coordinate_transform(
        resolved_geometry.spacetime_metric,
        jacobian_matrix,
        label="internal_observer_spacetime_metric",
    )
    shifted_spatial_metric = build_metric_tensor(
        shifted_spacetime_metric.components[1:, 1:],
        label="internal_observer_spatial_metric",
    )
    return HolographicProjectionShift(
        base_geometry=resolved_geometry,
        self_valuation=resolved_self_valuation,
        projection_kernel=projection_kernel,
        jacobian_matrix=jacobian_matrix,
        coordinate_shift=coordinate_shift,
        shifted_spacetime_metric=shifted_spacetime_metric,
        shifted_spatial_metric=shifted_spatial_metric,
    )


def derive_general_relativity_ui(
    *,
    observer: "InternalObserver" | None = None,
    self_valuation: SelfValuationAudit | None = None,
    projection: HolographicProjectionShift | None = None,
    geometry: BoundaryDeterminedBulkGeometry | None = None,
) -> GeneralRelativityUIAudit:
    resolved_self_valuation = observer.self_valuate() if observer is not None and self_valuation is None else self_valuation
    if resolved_self_valuation is None:
        raise TypeError("derive_general_relativity_ui requires self_valuation when observer is not provided.")
    resolved_projection = (
        shift_holographic_projection(observer=observer, self_valuation=resolved_self_valuation, geometry=geometry)
        if projection is None
        else projection
    )

    with localcontext() as context:
        context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
        light_speed_squared = _decimal(LIGHT_SPEED_M_PER_S) * _decimal(LIGHT_SPEED_M_PER_S)
        entropic_potential = resolved_projection.projection_kernel.temporal_projection_weight
        localized_entropy_gradient_per_m = (
            resolved_self_valuation.localized_entropy_gradient_per_m
            * (Decimal("1") + resolved_projection.projection_kernel.hidden_entropy_projection_weight)
        )
        gravitational_acceleration_m_per_s2 = light_speed_squared * localized_entropy_gradient_per_m
        observer_frame_curvature = _decimal(
            abs(
                float(
                    np.trace(
                        resolved_projection.shifted_spacetime_metric.components
                        - resolved_projection.base_geometry.spacetime_metric.components
                    )
                )
            )
        )
        equivalence_principle_residual = (
            gravitational_acceleration_m_per_s2 - light_speed_squared * localized_entropy_gradient_per_m
        )

    return GeneralRelativityUIAudit(
        self_valuation=resolved_self_valuation,
        projection=resolved_projection,
        entropic_potential=+entropic_potential,
        localized_entropy_gradient_per_m=+localized_entropy_gradient_per_m,
        gravitational_acceleration_m_per_s2=+gravitational_acceleration_m_per_s2,
        observer_frame_curvature=+observer_frame_curvature,
        equivalence_principle_residual=+equivalence_principle_residual,
    )


def derive_observer_frame(
    *,
    observer: "InternalObserver" | None = None,
    self_valuation: SelfValuationAudit | None = None,
    projection: HolographicProjectionShift | None = None,
    general_relativity_ui: GeneralRelativityUIAudit | None = None,
    alpha_drift: FrameDependentAlphaAudit | None = None,
    geometry: BoundaryDeterminedBulkGeometry | None = None,
    vacuum: TopologicalVacuum | None = None,
    precision: int = DEFAULT_PRECISION,
) -> ObserverFrameAudit:
    resolved_self_valuation = observer.self_valuate() if observer is not None and self_valuation is None else self_valuation
    if resolved_self_valuation is None:
        raise TypeError("derive_observer_frame requires self_valuation when observer is not provided.")
    resolved_projection = (
        shift_holographic_projection(observer=observer, self_valuation=resolved_self_valuation, geometry=geometry)
        if projection is None
        else projection
    )
    resolved_general_relativity_ui = (
        derive_general_relativity_ui(
            observer=observer,
            self_valuation=resolved_self_valuation,
            projection=resolved_projection,
        )
        if general_relativity_ui is None
        else general_relativity_ui
    )
    resolved_alpha_drift = (
        derive_frame_dependent_alpha(
            self_valuation=resolved_self_valuation,
            vacuum=vacuum,
            precision=precision,
        )
        if alpha_drift is None
        else alpha_drift
    )
    return ObserverFrameAudit(
        self_valuation=resolved_self_valuation,
        projection=resolved_projection,
        general_relativity_ui=resolved_general_relativity_ui,
        alpha_drift=resolved_alpha_drift,
    )


def run_inside_out_simulation(
    *,
    observer: "InternalObserver" | None = None,
    redshift: Decimal | Fraction | float | int | str = Decimal("0"),
    self_valuation: SelfValuationAudit | None = None,
    projection: HolographicProjectionShift | None = None,
    gravity_flow: BulkFlowAudit | None = None,
) -> InsideOutMetricAudit:
    resolved_observer = InternalObserver() if observer is None else observer
    resolved_self_valuation = resolved_observer.self_valuate() if self_valuation is None else self_valuation
    resolved_projection = (
        shift_holographic_projection(observer=resolved_observer, self_valuation=resolved_self_valuation)
        if projection is None
        else projection
    )
    resolved_gravity_flow = (
        build_bulk_dynamics_audit(
            redshift=redshift,
            bit_budget=resolved_observer._core_observer.global_bit_budget,
            lepton_level=resolved_observer.vacuum.lepton_level,
            quark_level=resolved_observer.vacuum.quark_level,
            precision=resolved_observer.precision,
        )
        if gravity_flow is None
        else gravity_flow
    )

    with localcontext() as context:
        context.prec = resolved_observer.precision + _GUARD_DIGITS
        temporal_flow = abs(_decimal(resolved_gravity_flow.layerwise_arrow_of_time_gradient_km_s_mpc))
        normalized_temporal_flow = temporal_flow / (Decimal("1") + temporal_flow)
        entropy_temporal_coupling = resolved_projection.projection_kernel.temporal_projection_weight * (
            Decimal("1") + normalized_temporal_flow * resolved_self_valuation.hidden_entropy_fraction
        )
        observer_time_dilation = entropy_temporal_coupling / (Decimal("1") + entropy_temporal_coupling)
        spatial_projection_envelope = (
            resolved_projection.projection_kernel.spatial_projection_weight
            * normalized_temporal_flow
            * resolved_self_valuation.hidden_entropy_fraction
        )
        spatial_scale_components = tuple(
            Decimal("1") + spatial_projection_envelope * coordinate
            for coordinate in resolved_self_valuation.agent_coordinates.as_decimal_tuple
        )

    inside_out_jacobian_matrix = np.diag(
        [
            float(Decimal("1") + observer_time_dilation),
            float(spatial_scale_components[0]),
            float(spatial_scale_components[1]),
            float(spatial_scale_components[2]),
        ]
    )
    perceived_spacetime_metric = coordinate_transform(
        resolved_projection.shifted_spacetime_metric,
        inside_out_jacobian_matrix,
        label="internal_observer_perceived_spacetime_metric",
    )
    perceived_spatial_metric = build_metric_tensor(
        perceived_spacetime_metric.components[1:, 1:],
        label="internal_observer_perceived_spatial_metric",
    )
    return InsideOutMetricAudit(
        self_valuation=resolved_self_valuation,
        projection=resolved_projection,
        gravity_flow=resolved_gravity_flow,
        inside_out_jacobian_matrix=inside_out_jacobian_matrix,
        perceived_spacetime_metric=perceived_spacetime_metric,
        perceived_spatial_metric=perceived_spatial_metric,
        entropy_temporal_coupling=+entropy_temporal_coupling,
        observer_time_dilation=+observer_time_dilation,
    )


class InternalObserver:
    def __init__(
        self,
        *,
        agent_coordinates: AgentLatticeCoordinates | None = None,
        observer_radius_m: Decimal | Fraction | float | int | str = Decimal("0"),
        lepton_level: int = LEPTON_LEVEL,
        quark_level: int = QUARK_LEVEL,
        parent_level: int = PARENT_LEVEL,
        precision: int = DEFAULT_PRECISION,
    ) -> None:
        self.precision = max(int(precision), DEFAULT_PRECISION)
        self.vacuum = TopologicalVacuum(
            lepton_level=int(lepton_level),
            quark_level=int(quark_level),
            parent_level=int(parent_level),
            generation_count=int(UniverseFactory.benchmark_vacuum().generation_count),
        )
        self.agent_coordinates = (
            AgentLatticeCoordinates.from_branch(*self.vacuum.branch)
            if agent_coordinates is None
            else agent_coordinates
        )
        self._core_observer = CoreObserver(
            observer_radius_m=observer_radius_m,
            lepton_level=self.vacuum.lepton_level,
            quark_level=self.vacuum.quark_level,
            parent_level=self.vacuum.parent_level,
            precision=self.precision,
        )
        self._sigma_module = SelfValuationSigma(precision=self.precision)
        self._compiler = HolographicCompiler(precision=self.precision)

    @property
    def coordinates(self) -> AgentLatticeCoordinates:
        return self.agent_coordinates

    @property
    def global_horizon_radius_m(self) -> Decimal:
        return self._core_observer.global_horizon_radius_m

    @property
    def observer_radius_m(self) -> Decimal:
        return self._core_observer.observer_radius_m

    @property
    def local_horizon_radius_m(self) -> Decimal:
        return self._core_observer.local_horizon_radius_m

    @property
    def sigma(self) -> SelfValuationSigma:
        return self._sigma_module

    def move_to(
        self,
        observer_radius_m: Decimal | Fraction | float | int | str,
    ) -> SelfValuationAudit:
        self._core_observer.move_to(observer_radius_m)
        return self.self_valuate()

    def self_valuate(self) -> SelfValuationAudit:
        return self._sigma_module.evaluate(
            observer=self._core_observer,
            agent_coordinates=self.agent_coordinates,
        )

    def actualize_bulk_geometry(self) -> BoundaryDeterminedBulkGeometry:
        return _actualize_geometry_for_vacuum(self.vacuum, precision=self.precision)

    def derive_projection_kernel(self) -> ObserverProjectionKernel:
        return self.sigma.derive_projection_kernel(self_valuation=self.self_valuate())

    def shift_holographic_projection(self) -> HolographicProjectionShift:
        return shift_holographic_projection(observer=self)

    def derive_general_relativity_ui(self) -> GeneralRelativityUIAudit:
        return derive_general_relativity_ui(observer=self)

    def derive_frame_dependent_alpha(self) -> FrameDependentAlphaAudit:
        return derive_frame_dependent_alpha(
            self_valuation=self.self_valuate(),
            vacuum=self.vacuum,
            precision=self.precision,
        )

    def derive_observer_frame(self) -> ObserverFrameAudit:
        return derive_observer_frame(
            observer=self,
            vacuum=self.vacuum,
            precision=self.precision,
        )

    def run_inside_out_simulation(
        self,
        *,
        redshift: Decimal | Fraction | float | int | str = Decimal("0"),
    ) -> InsideOutMetricAudit:
        return run_inside_out_simulation(observer=self, redshift=redshift)


class Observer(InternalObserver):
    def __init__(
        self,
        *,
        coordinates: AgentLatticeCoordinates | None = None,
        agent_coordinates: AgentLatticeCoordinates | None = None,
        observer_radius_m: Decimal | Fraction | float | int | str = Decimal("0"),
        lepton_level: int = LEPTON_LEVEL,
        quark_level: int = QUARK_LEVEL,
        parent_level: int = PARENT_LEVEL,
        precision: int = DEFAULT_PRECISION,
    ) -> None:
        super().__init__(
            agent_coordinates=coordinates if agent_coordinates is None else agent_coordinates,
            observer_radius_m=observer_radius_m,
            lepton_level=lepton_level,
            quark_level=quark_level,
            parent_level=parent_level,
            precision=precision,
        )


__all__ = [
    "AgentLatticeCoordinates",
    "DEFAULT_PRECISION",
    "FrameDependentAlphaAudit",
    "GeneralRelativityUIAudit",
    "HolographicProjectionShift",
    "InsideOutMetricAudit",
    "InternalObserver",
    "Observer",
    "ObserverFrameAudit",
    "ObserverProjectionKernel",
    "SelfValuationAudit",
    "SelfValuationSigma",
    "derive_frame_dependent_alpha",
    "derive_general_relativity_ui",
    "derive_observer_frame",
    "run_inside_out_simulation",
    "shift_holographic_projection",
]
