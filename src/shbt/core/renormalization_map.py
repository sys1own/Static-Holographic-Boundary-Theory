from __future__ import annotations

"""Render static benchmark boundary residues into running electromagnetic couplings.

The anomaly-free ``(26, 8, 312)`` boundary fixes a static ultraviolet residue
``alpha_UV`` through the surface-tension gauge identity. This module turns that
static datum into a running observable by transporting the corresponding gauge
couplings across scales and smearing the branch-fixed closure residue over the
UV-to-IR logarithmic interval.

Unlike the legacy gauge audit in ``main.py``, the original verification target
here is the electroweak Standard-Model expectation at ``M_Z`` rather than the
zero-energy CODATA fine-structure comparator.

The module now also exposes a complementary “smoking gun” transport layer that
interprets the projection from the 26D boundary to the 4D bulk as a
noninvertible cooling map. In that picture, the branch-fixed ultraviolet
residue ``alpha_UV^{-1}≈137.647`` loses information as the discarded 22
dimensions are smeared over the logarithmic bulk interval, yielding an
infrared observable close to the CODATA comparator without ever using CODATA as
an input to the transport itself.
"""

from dataclasses import dataclass
import math
from typing import Final

import numpy as np

from shbt.constants import CODATA_FINE_STRUCTURE_ALPHA_INVERSE, LEPTON_LEVEL, MZ_SCALE_GEV, PARENT_LEVEL, PLANCK2018_ALPHA_EM_INV_MZ, QUARK_LEVEL
from shbt.core.derivation_api import DEFAULT_PRECISION, TopologicalVacuum, UniverseFactory
from shbt.main import (
    BENCHMARK_C_DARK_RESIDUE,
    BENCHMARK_VEV_RATIO,
    DEFAULT_TOPOLOGICAL_VACUUM,
    RunningCouplings,
    THEORETICAL_MATCHING_UNCERTAINTY_FRACTION,
    alpha_em_inverse_from_running_couplings,
    derive_running_couplings,
    integrate_sm_running_couplings,
)


BENCHMARK_BOUNDARY_BRANCH: Final[tuple[int, int, int]] = (
    int(LEPTON_LEVEL),
    int(QUARK_LEVEL),
    int(PARENT_LEVEL),
)
BENCHMARK_UV_SCALE_GEV: Final[float] = float(MZ_SCALE_GEV * DEFAULT_TOPOLOGICAL_VACUUM.scale_ratio)
BULK_DIMENSION: Final[int] = 4
CODATA_ALIGNMENT_ABS_TOLERANCE: Final[float] = 5.0e-4


def _require_benchmark_vacuum(vacuum: TopologicalVacuum | None) -> TopologicalVacuum:
    resolved_vacuum = UniverseFactory.benchmark_vacuum() if vacuum is None else vacuum
    if resolved_vacuum.branch != BENCHMARK_BOUNDARY_BRANCH:
        raise ValueError(
            "Renormalization rendering is defined only on the anomaly-free (26, 8, 312) boundary manifold."
        )
    return resolved_vacuum


@dataclass(frozen=True)
class UVIRRendering:
    vacuum: TopologicalVacuum
    energy_gev: float
    uv_scale_gev: float
    smearing_fraction: float
    structural_exponent: float
    delta_b_em: float
    alpha_uv: float
    alpha_uv_inverse: float
    alpha_raw: float
    alpha_raw_inverse: float
    vacuum_polarization_residue: float
    vacuum_polarization_residue_inverse: float
    holographic_smearing_residue_inverse: float
    delta_alpha_residue: float
    rendered_alpha: float
    rendered_alpha_inverse: float


@dataclass(frozen=True)
class RunningAlphaVerification:
    rendering: UVIRRendering
    target_alpha: float
    target_alpha_inverse: float
    absolute_error_inverse: float
    relative_error_inverse: float
    matching_sigma_inverse: float
    pull: float
    improves_over_raw: bool
    aligns_with_standard_model: bool


@dataclass(frozen=True)
class NonInvertibleTransportState:
    vacuum: TopologicalVacuum
    energy_gev: float
    uv_scale_gev: float
    boundary_dimension: int
    bulk_dimension: int
    discarded_dimensions: int
    visible_support_half: float
    vev_residue: float
    c_dark_completion: float
    alpha_uv_inverse: float
    smearing_fraction: float
    information_loss_budget_inverse: float
    information_loss_inverse: float
    retained_information_inverse: float
    projected_alpha_inverse: float
    projected_alpha: float


@dataclass(frozen=True)
class CoolingPoint:
    step: int
    transport: NonInvertibleTransportState


@dataclass(frozen=True)
class CoolingTrajectory:
    branch: tuple[int, int, int]
    uv_scale_gev: float
    ir_scale_gev: float
    points: tuple[CoolingPoint, ...]

    @property
    def monotonic_information_loss(self) -> bool:
        losses = [point.transport.information_loss_inverse for point in self.points]
        return all(next_loss >= current_loss for current_loss, next_loss in zip(losses, losses[1:], strict=False))

    @property
    def monotonic_alpha_inverse_cooling(self) -> bool:
        alpha_inverses = [point.transport.projected_alpha_inverse for point in self.points]
        return all(next_value <= current_value for current_value, next_value in zip(alpha_inverses, alpha_inverses[1:], strict=False))


@dataclass(frozen=True)
class NonInvertibleTransportVerification:
    transport: NonInvertibleTransportState
    codata_alpha_inverse: float
    absolute_error_inverse: float
    relative_error_inverse: float
    tolerance_inverse: float
    target_used_in_transport: bool
    aligns_with_codata: bool


def holographic_smearing_fraction(
    energy_gev: float,
    *,
    uv_scale_gev: float = BENCHMARK_UV_SCALE_GEV,
) -> float:
    """Return the log-interval fraction used to smear the closure residue to ``energy_gev``."""

    resolved_energy = float(energy_gev)
    resolved_uv_scale = float(uv_scale_gev)
    if resolved_energy <= 0.0:
        raise ValueError(f"Energy must be positive, received {resolved_energy}")
    if resolved_uv_scale <= MZ_SCALE_GEV:
        raise ValueError(
            f"UV scale must sit above M_Z to define holographic smearing, received {resolved_uv_scale}"
        )
    if resolved_energy >= resolved_uv_scale:
        return 0.0
    if resolved_energy <= MZ_SCALE_GEV:
        return 1.0
    numerator = math.log(resolved_uv_scale / resolved_energy)
    denominator = math.log(resolved_uv_scale / MZ_SCALE_GEV)
    return float(min(1.0, max(0.0, numerator / denominator)))


def _boundary_information_loss_budget_inverse(alpha_uv_inverse: float) -> float:
    """Return the branch-fixed inverse-coupling loss budget for the 26D -> 4D projection.

    The loss budget is derived only from benchmark residues:

    - the positive completion residue ``c_dark``;
    - the representational-admissibility residue ``64/312``; and
    - the discarded-dimensionality factor ``26-4=22`` normalized by the visible
      half-support ``(k_ell+k_q)/2 = 17``.

    No observational electromagnetic input is used in this construction.
    """

    del alpha_uv_inverse
    discarded_dimensions = int(LEPTON_LEVEL - BULK_DIMENSION)
    visible_support_half = 0.5 * float(LEPTON_LEVEL + QUARK_LEVEL)
    return float(
        float(BENCHMARK_C_DARK_RESIDUE)
        * float(BENCHMARK_VEV_RATIO)
        * discarded_dimensions
        / visible_support_half
    )


def noninvertible_transport(
    energy_gev: float,
    *,
    precision: int = DEFAULT_PRECISION,
    vacuum: TopologicalVacuum | None = None,
) -> NonInvertibleTransportState:
    """Project the static UV residue into the 4D bulk with irreversible information loss.

    The map is intentionally noninvertible: once the 26D boundary data are
    compressed into a 4D bulk observable, the discarded 22-dimensional support
    is remembered only through an aggregate loss budget rather than by a unique
    inverse preimage.
    """

    resolved_precision = max(int(precision), DEFAULT_PRECISION)
    resolved_vacuum = _require_benchmark_vacuum(vacuum)
    resolved_energy = float(energy_gev)
    alpha_uv_inverse = float(
        UniverseFactory.derive_alpha_surface(
            precision=resolved_precision,
            vacuum=resolved_vacuum,
        ).alpha_inverse_decimal
    )
    smearing_fraction = holographic_smearing_fraction(resolved_energy)
    information_loss_budget_inverse = _boundary_information_loss_budget_inverse(alpha_uv_inverse)
    information_loss_inverse = float(smearing_fraction * information_loss_budget_inverse)
    projected_alpha_inverse = float(alpha_uv_inverse - information_loss_inverse)
    if projected_alpha_inverse <= 0.0:
        raise ValueError("Noninvertible transport over-cooled the inverse coupling below zero.")
    projected_alpha = float(1.0 / projected_alpha_inverse)

    return NonInvertibleTransportState(
        vacuum=resolved_vacuum,
        energy_gev=resolved_energy,
        uv_scale_gev=BENCHMARK_UV_SCALE_GEV,
        boundary_dimension=int(LEPTON_LEVEL),
        bulk_dimension=BULK_DIMENSION,
        discarded_dimensions=int(LEPTON_LEVEL - BULK_DIMENSION),
        visible_support_half=0.5 * float(LEPTON_LEVEL + QUARK_LEVEL),
        vev_residue=float(BENCHMARK_VEV_RATIO),
        c_dark_completion=float(BENCHMARK_C_DARK_RESIDUE),
        alpha_uv_inverse=alpha_uv_inverse,
        smearing_fraction=smearing_fraction,
        information_loss_budget_inverse=information_loss_budget_inverse,
        information_loss_inverse=information_loss_inverse,
        retained_information_inverse=float(alpha_uv_inverse - information_loss_inverse),
        projected_alpha_inverse=projected_alpha_inverse,
        projected_alpha=projected_alpha,
    )


def NonInvertibleTransport(
    energy_gev: float,
    *,
    precision: int = DEFAULT_PRECISION,
    vacuum: TopologicalVacuum | None = None,
) -> NonInvertibleTransportState:
    """Backward-compatible publication alias for :func:`noninvertible_transport`."""

    return noninvertible_transport(
        energy_gev,
        precision=precision,
        vacuum=vacuum,
    )


def cooling_script(
    *,
    sample_count: int = 9,
    precision: int = DEFAULT_PRECISION,
    vacuum: TopologicalVacuum | None = None,
) -> CoolingTrajectory:
    """Simulate how boundary information smears across the 4D bulk energy interval."""

    if int(sample_count) < 2:
        raise ValueError(f"sample_count must be at least 2, received {sample_count}")

    energies = np.geomspace(BENCHMARK_UV_SCALE_GEV, MZ_SCALE_GEV, int(sample_count), dtype=float)
    points = tuple(
        CoolingPoint(
            step=index,
            transport=noninvertible_transport(
                float(energy),
                precision=precision,
                vacuum=vacuum,
            ),
        )
        for index, energy in enumerate(energies)
    )
    return CoolingTrajectory(
        branch=BENCHMARK_BOUNDARY_BRANCH,
        uv_scale_gev=float(BENCHMARK_UV_SCALE_GEV),
        ir_scale_gev=float(MZ_SCALE_GEV),
        points=points,
    )


def verify_noninvertible_transport_at_z_boson(
    *,
    precision: int = DEFAULT_PRECISION,
    vacuum: TopologicalVacuum | None = None,
    tolerance_inverse: float = CODATA_ALIGNMENT_ABS_TOLERANCE,
) -> NonInvertibleTransportVerification:
    """Check that the cooled projection lands near the CODATA comparator at ``M_Z``.

    CODATA appears only here as an external verification comparator; the
    transport itself depends solely on branch-fixed residues.
    """

    transport = noninvertible_transport(
        MZ_SCALE_GEV,
        precision=precision,
        vacuum=vacuum,
    )
    codata_alpha_inverse = float(CODATA_FINE_STRUCTURE_ALPHA_INVERSE)
    absolute_error_inverse = float(abs(transport.projected_alpha_inverse - codata_alpha_inverse))
    relative_error_inverse = float(absolute_error_inverse / codata_alpha_inverse)
    return NonInvertibleTransportVerification(
        transport=transport,
        codata_alpha_inverse=codata_alpha_inverse,
        absolute_error_inverse=absolute_error_inverse,
        relative_error_inverse=relative_error_inverse,
        tolerance_inverse=float(tolerance_inverse),
        target_used_in_transport=False,
        aligns_with_codata=bool(absolute_error_inverse <= float(tolerance_inverse)),
    )


def _surface_running_uv_inputs(alpha_uv_inverse: float) -> RunningCouplings:
    benchmark_uv_inputs = derive_running_couplings(
        BENCHMARK_UV_SCALE_GEV,
        solver_config=DEFAULT_TOPOLOGICAL_VACUUM.solver_config,
    )
    branch_weak_mixing = DEFAULT_TOPOLOGICAL_VACUUM.derive_gauge_strong_audit().sin2_theta_w
    alpha1_inverse = (3.0 / 5.0) * float(alpha_uv_inverse) * (1.0 - branch_weak_mixing)
    alpha2_inverse = float(alpha_uv_inverse) * branch_weak_mixing
    return RunningCouplings(
        top=benchmark_uv_inputs.top,
        bottom=benchmark_uv_inputs.bottom,
        tau=benchmark_uv_inputs.tau,
        g1=float(math.sqrt(4.0 * math.pi / alpha1_inverse)),
        g2=float(math.sqrt(4.0 * math.pi / alpha2_inverse)),
        g3=benchmark_uv_inputs.g3,
    )


def uv_to_ir_rendering_function(
    energy_gev: float,
    *,
    precision: int = DEFAULT_PRECISION,
    vacuum: TopologicalVacuum | None = None,
) -> UVIRRendering:
    """Render the benchmark UV boundary residue to a running electromagnetic coupling.

    The raw transport follows the coupled one-loop SM gauge flow, while the
    branch-fixed heavy-Higgs closure residue is distributed with the logarithmic
    holographic smearing fraction between the UV boundary and the electroweak
    anchor scale.
    """

    resolved_precision = max(int(precision), DEFAULT_PRECISION)
    resolved_vacuum = _require_benchmark_vacuum(vacuum)
    resolved_energy = float(energy_gev)
    alpha_uv_inverse = float(
        UniverseFactory.derive_alpha_surface(
            precision=resolved_precision,
            vacuum=resolved_vacuum,
        ).alpha_inverse_decimal
    )
    alpha_uv = float(1.0 / alpha_uv_inverse)
    uv_inputs = _surface_running_uv_inputs(alpha_uv_inverse)
    running_couplings = integrate_sm_running_couplings(
        BENCHMARK_UV_SCALE_GEV,
        resolved_energy,
        uv_inputs,
        solver_config=DEFAULT_TOPOLOGICAL_VACUUM.solver_config,
    )
    alpha_raw_inverse = float(alpha_em_inverse_from_running_couplings(running_couplings))
    alpha_raw = float(1.0 / alpha_raw_inverse)

    threshold_data = DEFAULT_TOPOLOGICAL_VACUUM.derive_rhn_threshold_data("lepton")
    _, full_hhd_delta_inverse, delta_b_em = DEFAULT_TOPOLOGICAL_VACUUM.apply_heavy_higgs_dilution_correction(
        alpha_raw_inverse,
        threshold_data=threshold_data,
    )
    smearing_fraction = holographic_smearing_fraction(resolved_energy)
    holographic_residue_inverse = float(smearing_fraction * full_hhd_delta_inverse)
    rendered_alpha_inverse = float(alpha_raw_inverse + holographic_residue_inverse)
    rendered_alpha = float(1.0 / rendered_alpha_inverse)
    delta_alpha_residue = float(rendered_alpha - alpha_uv)

    return UVIRRendering(
        vacuum=resolved_vacuum,
        energy_gev=resolved_energy,
        uv_scale_gev=BENCHMARK_UV_SCALE_GEV,
        smearing_fraction=smearing_fraction,
        structural_exponent=float(threshold_data.structural_exponent),
        delta_b_em=float(delta_b_em),
        alpha_uv=alpha_uv,
        alpha_uv_inverse=alpha_uv_inverse,
        alpha_raw=alpha_raw,
        alpha_raw_inverse=alpha_raw_inverse,
        vacuum_polarization_residue=float(alpha_raw - alpha_uv),
        vacuum_polarization_residue_inverse=float(alpha_raw_inverse - alpha_uv_inverse),
        holographic_smearing_residue_inverse=holographic_residue_inverse,
        delta_alpha_residue=delta_alpha_residue,
        rendered_alpha=rendered_alpha,
        rendered_alpha_inverse=rendered_alpha_inverse,
    )


def running_alpha_function(
    energy_gev: float,
    *,
    precision: int = DEFAULT_PRECISION,
    vacuum: TopologicalVacuum | None = None,
) -> float:
    """Return the rendered electromagnetic coupling ``alpha(E)``.

    By construction,

        alpha(E) = alpha_UV + Delta alpha_residue,

    where ``Delta alpha_residue`` is extracted from the bulk vacuum
    polarization plus the log-smeared branch-fixed closure residue.
    """

    rendering = uv_to_ir_rendering_function(
        energy_gev,
        precision=precision,
        vacuum=vacuum,
    )
    return float(rendering.alpha_uv + rendering.delta_alpha_residue)


def verify_rendered_alpha_at_z_boson(
    *,
    precision: int = DEFAULT_PRECISION,
    vacuum: TopologicalVacuum | None = None,
    tolerance_fraction: float = THEORETICAL_MATCHING_UNCERTAINTY_FRACTION,
) -> RunningAlphaVerification:
    """Verify that the rendered ``alpha(M_Z)`` aligns with the SM electroweak expectation."""

    rendering = uv_to_ir_rendering_function(
        MZ_SCALE_GEV,
        precision=precision,
        vacuum=vacuum,
    )
    target_alpha_inverse = float(PLANCK2018_ALPHA_EM_INV_MZ)
    target_alpha = float(1.0 / target_alpha_inverse)
    absolute_error_inverse = float(abs(rendering.rendered_alpha_inverse - target_alpha_inverse))
    relative_error_inverse = float(absolute_error_inverse / target_alpha_inverse)
    matching_sigma_inverse = float(
        max(np.finfo(float).eps, abs(target_alpha_inverse) * float(tolerance_fraction))
    )
    pull = float(absolute_error_inverse / matching_sigma_inverse)
    improves_over_raw = bool(
        absolute_error_inverse < abs(rendering.alpha_raw_inverse - target_alpha_inverse)
    )
    aligns_with_standard_model = bool(
        improves_over_raw
        and relative_error_inverse <= float(tolerance_fraction)
        and pull <= 1.0
    )
    return RunningAlphaVerification(
        rendering=rendering,
        target_alpha=target_alpha,
        target_alpha_inverse=target_alpha_inverse,
        absolute_error_inverse=absolute_error_inverse,
        relative_error_inverse=relative_error_inverse,
        matching_sigma_inverse=matching_sigma_inverse,
        pull=pull,
        improves_over_raw=improves_over_raw,
        aligns_with_standard_model=aligns_with_standard_model,
    )


__all__ = [
    "BENCHMARK_BOUNDARY_BRANCH",
    "BENCHMARK_UV_SCALE_GEV",
    "BULK_DIMENSION",
    "CODATA_ALIGNMENT_ABS_TOLERANCE",
    "CoolingPoint",
    "CoolingTrajectory",
    "NonInvertibleTransport",
    "NonInvertibleTransportState",
    "NonInvertibleTransportVerification",
    "RunningAlphaVerification",
    "UVIRRendering",
    "cooling_script",
    "holographic_smearing_fraction",
    "noninvertible_transport",
    "running_alpha_function",
    "uv_to_ir_rendering_function",
    "verify_noninvertible_transport_at_z_boson",
    "verify_rendered_alpha_at_z_boson",
]
