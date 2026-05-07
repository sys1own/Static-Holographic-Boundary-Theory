from __future__ import annotations

"""Render static benchmark boundary residues into running electromagnetic couplings.

The anomaly-free ``(26, 8, 312)`` boundary fixes a static ultraviolet residue
``alpha_UV`` through the surface-tension gauge identity. This module turns that
static datum into a running observable by transporting the corresponding gauge
couplings across scales and smearing the branch-fixed closure residue over the
UV-to-IR logarithmic interval.

Unlike the legacy gauge audit in ``main.py``, the verification target here is
the electroweak Standard-Model expectation at ``M_Z`` rather than the
zero-energy CODATA fine-structure comparator.
"""

from dataclasses import dataclass
import math
from typing import Final

import numpy as np

from shbt.constants import LEPTON_LEVEL, MZ_SCALE_GEV, PARENT_LEVEL, PLANCK2018_ALPHA_EM_INV_MZ, QUARK_LEVEL
from shbt.core.derivation_api import DEFAULT_PRECISION, TopologicalVacuum, UniverseFactory
from shbt.main import (
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
    "RunningAlphaVerification",
    "UVIRRendering",
    "holographic_smearing_fraction",
    "running_alpha_function",
    "uv_to_ir_rendering_function",
    "verify_rendered_alpha_at_z_boson",
]
