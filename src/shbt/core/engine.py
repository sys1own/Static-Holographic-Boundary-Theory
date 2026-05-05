from __future__ import annotations

from fractions import Fraction

import numpy as np

from shbt.constants import G_SM, HIGGS_POLE_MASS_GEV, HOLOGRAPHIC_BITS, KAPPA_D5, LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL, TOP_POLE_MASS_GEV
from shbt.core.algebra import jarlskog_invariant, pdg_parameters, pdg_unitary
from shbt.core.noether_bridge import branch_planck_mass_ev
from shbt.physics_engine import (
    charged_lepton_yukawa_diagonal,
    integrate_pmns_majorana_rge_numerically,
    majorana_mass_matrix_beta,
    majorana_mass_matrix_from_pmns,
    matrix_derived_lepton_betas,
    solve_ivp_with_fallback,
    sm_one_loop_running_betas,
    takagi_diagonalize_symmetric,
    wrapped_angle_difference_deg,
)
from shbt.runtime_config import DEFAULT_SOLVER_CONFIG, PhysicsDomainWarning, SolverConfig, solver_isclose


def quark_matching_thresholds(uv_scale_gev: float, ir_scale_gev: float) -> tuple[float, ...]:
    return tuple(
        scale
        for scale in (TOP_POLE_MASS_GEV, HIGGS_POLE_MASS_GEV)
        if ir_scale_gev < scale < uv_scale_gev
    )


def apply_quark_threshold_matching(state: np.ndarray, threshold_scale_gev: float) -> np.ndarray:
    matched_state = np.array(state, dtype=float, copy=True)
    if solver_isclose(threshold_scale_gev, TOP_POLE_MASS_GEV):
        matched_state[4] = 0.0
    if solver_isclose(threshold_scale_gev, HIGGS_POLE_MASS_GEV):
        matched_state[5] = 0.0
        matched_state[6] = 0.0
    return matched_state


def visible_level_density_ratio(
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
) -> float:
    resolved_parent_level = PARENT_LEVEL if parent_level is None else int(parent_level)
    resolved_lepton_level = LEPTON_LEVEL if lepton_level is None else int(lepton_level)
    resolved_quark_level = QUARK_LEVEL if quark_level is None else int(quark_level)
    visible_support = resolved_lepton_level + resolved_quark_level
    if visible_support <= 0:
        raise ValueError("Visible support count must be positive.")
    return float(resolved_parent_level / visible_support)


def surface_tension_gauge_alpha_inverse(
    parent_level: int | None = None,
    lepton_level: int | None = None,
    quark_level: int | None = None,
    generation_count: int = G_SM,
) -> float:
    return float(
        generation_count
        * visible_level_density_ratio(
            parent_level=parent_level,
            lepton_level=lepton_level,
            quark_level=quark_level,
        )
    )


def topological_planck_mass_ev() -> float:
    return float(branch_planck_mass_ev())


def topological_newton_coordinate_ev_minus2(*, branch_planck_mass_ev: float | None = None) -> float:
    resolved_branch_planck_mass_ev = topological_planck_mass_ev() if branch_planck_mass_ev is None else float(branch_planck_mass_ev)
    return float(1.0 / (resolved_branch_planck_mass_ev * resolved_branch_planck_mass_ev))


def topological_mass_coordinate_ev(
    bit_count: float = HOLOGRAPHIC_BITS,
    kappa_geometric: float = KAPPA_D5,
) -> float:
    resolved_bit_count = float(bit_count)
    if resolved_bit_count <= 0.0:
        raise ValueError("Holographic bit count must be positive.")
    return float(kappa_geometric * topological_planck_mass_ev() * resolved_bit_count ** (-0.25))


def wzw_central_charge_fraction(level: int, dimension: int, dual_coxeter: int) -> Fraction:
    resolved_level = int(level)
    resolved_dual_coxeter = int(dual_coxeter)
    denominator = resolved_level + resolved_dual_coxeter
    if denominator <= 0:
        raise ValueError("WZW central charge requires k + h^∨ > 0.")
    return Fraction(resolved_level * int(dimension), denominator)


def quark_branching_index(parent_level: int = PARENT_LEVEL, quark_level: int = QUARK_LEVEL) -> int:
    denominator = 3 * int(quark_level)
    if denominator <= 0:
        raise ValueError("Quark branching index requires positive k_q.")
    return int(int(parent_level) // denominator)


__all__ = [
    "DEFAULT_SOLVER_CONFIG",
    "PhysicsDomainWarning",
    "SolverConfig",
    "apply_quark_threshold_matching",
    "charged_lepton_yukawa_diagonal",
    "integrate_pmns_majorana_rge_numerically",
    "jarlskog_invariant",
    "majorana_mass_matrix_beta",
    "majorana_mass_matrix_from_pmns",
    "matrix_derived_lepton_betas",
    "pdg_parameters",
    "pdg_unitary",
    "quark_branching_index",
    "quark_matching_thresholds",
    "solve_ivp_with_fallback",
    "sm_one_loop_running_betas",
    "surface_tension_gauge_alpha_inverse",
    "takagi_diagonalize_symmetric",
    "topological_mass_coordinate_ev",
    "topological_newton_coordinate_ev_minus2",
    "topological_planck_mass_ev",
    "visible_level_density_ratio",
    "wzw_central_charge_fraction",
    "wrapped_angle_difference_deg",
]
