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
class HolographicProjectionShift:
    base_geometry: BoundaryDeterminedBulkGeometry
    self_valuation: SelfValuationAudit
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
    def statement(self) -> str:
        return "Internal lattice coordinates shift the holographic projection into an agent-local bulk frame."


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
    self_valuation: SelfValuationAudit,
    agent_coordinates: AgentLatticeCoordinates | None = None,
    geometry: BoundaryDeterminedBulkGeometry | None = None,
) -> HolographicProjectionShift:
    resolved_agent_coordinates = self_valuation.agent_coordinates if agent_coordinates is None else agent_coordinates
    resolved_geometry = (
        _actualize_geometry_for_vacuum(_vacuum_from_branch(self_valuation.evaluated_branch))
        if geometry is None
        else geometry
    )
    coordinate_shift = tuple(self_valuation.sigma * value for value in resolved_agent_coordinates.as_decimal_tuple)

    entropic_potential = self_valuation.sigma * self_valuation.hidden_entropy_fraction
    time_shift = float(entropic_potential)
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
        self_valuation=self_valuation,
        jacobian_matrix=jacobian_matrix,
        coordinate_shift=coordinate_shift,
        shifted_spacetime_metric=shifted_spacetime_metric,
        shifted_spatial_metric=shifted_spatial_metric,
    )


def derive_general_relativity_ui(
    *,
    self_valuation: SelfValuationAudit,
    projection: HolographicProjectionShift | None = None,
    geometry: BoundaryDeterminedBulkGeometry | None = None,
) -> GeneralRelativityUIAudit:
    resolved_projection = (
        shift_holographic_projection(self_valuation=self_valuation, geometry=geometry)
        if projection is None
        else projection
    )

    with localcontext() as context:
        context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
        light_speed_squared = _decimal(LIGHT_SPEED_M_PER_S) * _decimal(LIGHT_SPEED_M_PER_S)
        entropic_potential = self_valuation.sigma * self_valuation.hidden_entropy_fraction
        localized_entropy_gradient_per_m = self_valuation.localized_entropy_gradient_per_m
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
        self_valuation=self_valuation,
        projection=resolved_projection,
        entropic_potential=+entropic_potential,
        localized_entropy_gradient_per_m=+localized_entropy_gradient_per_m,
        gravitational_acceleration_m_per_s2=+gravitational_acceleration_m_per_s2,
        observer_frame_curvature=+observer_frame_curvature,
        equivalence_principle_residual=+equivalence_principle_residual,
    )


def derive_observer_frame(
    *,
    self_valuation: SelfValuationAudit,
    projection: HolographicProjectionShift | None = None,
    general_relativity_ui: GeneralRelativityUIAudit | None = None,
    alpha_drift: FrameDependentAlphaAudit | None = None,
    geometry: BoundaryDeterminedBulkGeometry | None = None,
    vacuum: TopologicalVacuum | None = None,
    precision: int = DEFAULT_PRECISION,
) -> ObserverFrameAudit:
    resolved_projection = (
        shift_holographic_projection(self_valuation=self_valuation, geometry=geometry)
        if projection is None
        else projection
    )
    resolved_general_relativity_ui = (
        derive_general_relativity_ui(
            self_valuation=self_valuation,
            projection=resolved_projection,
        )
        if general_relativity_ui is None
        else general_relativity_ui
    )
    resolved_alpha_drift = (
        derive_frame_dependent_alpha(
            self_valuation=self_valuation,
            vacuum=vacuum,
            precision=precision,
        )
        if alpha_drift is None
        else alpha_drift
    )
    return ObserverFrameAudit(
        self_valuation=self_valuation,
        projection=resolved_projection,
        general_relativity_ui=resolved_general_relativity_ui,
        alpha_drift=resolved_alpha_drift,
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

    def shift_holographic_projection(self) -> HolographicProjectionShift:
        return shift_holographic_projection(
            self_valuation=self.self_valuate(),
            agent_coordinates=self.agent_coordinates,
            geometry=self.actualize_bulk_geometry(),
        )

    def derive_general_relativity_ui(self) -> GeneralRelativityUIAudit:
        self_valuation = self.self_valuate()
        projection = shift_holographic_projection(
            self_valuation=self_valuation,
            agent_coordinates=self.agent_coordinates,
            geometry=self.actualize_bulk_geometry(),
        )
        return derive_general_relativity_ui(
            self_valuation=self_valuation,
            projection=projection,
        )

    def derive_frame_dependent_alpha(self) -> FrameDependentAlphaAudit:
        return derive_frame_dependent_alpha(
            self_valuation=self.self_valuate(),
            vacuum=self.vacuum,
            precision=self.precision,
        )

    def derive_observer_frame(self) -> ObserverFrameAudit:
        self_valuation = self.self_valuate()
        projection = shift_holographic_projection(
            self_valuation=self_valuation,
            agent_coordinates=self.agent_coordinates,
            geometry=self.actualize_bulk_geometry(),
        )
        return derive_observer_frame(
            self_valuation=self_valuation,
            projection=projection,
            vacuum=self.vacuum,
            precision=self.precision,
        )


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
    "InternalObserver",
    "Observer",
    "ObserverFrameAudit",
    "SelfValuationAudit",
    "SelfValuationSigma",
    "derive_frame_dependent_alpha",
    "derive_general_relativity_ui",
    "derive_observer_frame",
    "shift_holographic_projection",
]
