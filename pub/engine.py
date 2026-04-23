from __future__ import annotations

import numpy as np

from .algebra import jarlskog_invariant, pdg_parameters, pdg_unitary
from .constants import HIGGS_POLE_MASS_GEV, TOP_POLE_MASS_GEV
from .physics_engine import (
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
from .runtime_config import DEFAULT_SOLVER_CONFIG, PhysicsDomainWarning, SolverConfig, solver_isclose


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
    "quark_matching_thresholds",
    "solve_ivp_with_fallback",
    "sm_one_loop_running_betas",
    "takagi_diagonalize_symmetric",
    "wrapped_angle_difference_deg",
]
