from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

from .runtime_config import DEFAULT_SOLVER_CONFIG, SolverConfig, SolverException

ONE_LOOP_FACTOR = 16.0 * math.pi * math.pi
SAFE_LOG_MAGNITUDE_FLOOR = 1.0e-30


def solve_ivp_with_fallback(
    equations: Callable[[float, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    initial_state: np.ndarray,
    *,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    **kwargs,
):
    """Run ``solve_ivp`` with strictly pinned Radau IIA and raise on failure."""

    for reserved_keyword in ("method", "rtol", "atol"):
        if reserved_keyword in kwargs:
            raise TypeError(f"'{reserved_keyword}' is managed by solver_config and cannot be passed explicitly")

    method = str(solver_config.method).strip()
    if method != "Radau":
        raise SolverException(f"Radau IIA pinning violation: expected method='Radau', got {method!r}.")
    if solver_config.atol > 1.0e-12:
        raise SolverException(
            f"Radau IIA pinning violation: verifier requires atol <= 1e-12, got {solver_config.atol!r}."
        )

    try:
        solution = solve_ivp(
            equations,
            t_span,
            initial_state,
            method=method,
            rtol=solver_config.rtol,
            atol=solver_config.atol,
            **kwargs,
        )
    except ValueError as exc:
        raise SolverException(
            f"Radau IIA solve failed before convergence at atol={solver_config.atol:.1e}: {exc}"
        ) from exc

    if not solution.success:
        raise SolverException(
            f"Radau IIA solve failed to satisfy atol={solver_config.atol:.1e}: {solution.message}"
        )
    if solution.t.size == 0:
        raise SolverException("Radau IIA solve returned an empty time grid.")

    target_time = float(t_span[1])
    reached_time = float(solution.t[-1])
    if not math.isclose(reached_time, target_time, rel_tol=0.0, abs_tol=solver_config.atol):
        raise SolverException(
            f"Radau IIA solve terminated at t={reached_time:.12e} before reaching {target_time:.12e} within atol={solver_config.atol:.1e}."
        )
    if not np.all(np.isfinite(solution.y)):
        raise SolverException("Radau IIA solve produced non-finite state entries.")
    return solution


def sm_one_loop_running_betas(couplings):
    r"""Return the standard one-loop SM RGEs for $(y_t,y_b,y_\tau,g_1,g_2,g_3)$."""

    y_t, y_b, y_tau = couplings.top, couplings.bottom, couplings.tau
    g1, g2, g3 = couplings.g1, couplings.g2, couplings.g3
    coupling_type = type(couplings)
    return coupling_type(
        top=y_t * ((9.0 / 2.0) * y_t * y_t + (3.0 / 2.0) * y_b * y_b + y_tau * y_tau - (17.0 / 20.0) * g1 * g1 - (9.0 / 4.0) * g2 * g2 - 8.0 * g3 * g3),
        bottom=y_b * ((3.0 / 2.0) * y_t * y_t + (9.0 / 2.0) * y_b * y_b + y_tau * y_tau - 0.25 * g1 * g1 - (9.0 / 4.0) * g2 * g2 - 8.0 * g3 * g3),
        tau=y_tau * (3.0 * y_t * y_t + 3.0 * y_b * y_b + (5.0 / 2.0) * y_tau * y_tau - (9.0 / 4.0) * g1 * g1 - (9.0 / 4.0) * g2 * g2),
        g1=(41.0 / 10.0) * g1 * g1 * g1,
        g2=(-19.0 / 6.0) * g2 * g2 * g2,
        g3=-7.0 * g3 * g3 * g3,
    )


def charged_lepton_yukawa_diagonal(tau_yukawa: float, charged_lepton_yukawa_ratios: dict[str, float]) -> np.ndarray:
    return np.diag(
        [
            charged_lepton_yukawa_ratios["electron"] * tau_yukawa,
            charged_lepton_yukawa_ratios["muon"] * tau_yukawa,
            tau_yukawa,
        ]
    ).astype(float)


def majorana_mass_matrix_from_pmns(
    unitary: npt.NDArray[np.complexfloating | np.floating],
    masses_ev: npt.NDArray[np.float64],
    phase_proxies_rad: tuple[float, float],
) -> npt.NDArray[np.complex128]:
    phi_1, phi_2 = phase_proxies_rad
    majorana_diagonal = np.diag(
        [
            masses_ev[0] * np.exp(1j * phi_1),
            masses_ev[1] * np.exp(1j * phi_2),
            masses_ev[2],
        ]
    )
    raw_matrix = unitary @ majorana_diagonal @ unitary.T
    return 0.5 * (raw_matrix + raw_matrix.T)


def majorana_mass_matrix_beta(
    mass_matrix: npt.NDArray[np.complexfloating | np.floating],
    tau_yukawa: float,
    charged_lepton_yukawa_ratios: dict[str, float],
    sm_majorana_c_e: float,
    universal_gamma: float = 0.0,
) -> npt.NDArray[np.complex128]:
    ye = charged_lepton_yukawa_diagonal(tau_yukawa, charged_lepton_yukawa_ratios)
    projector = ye.T @ ye
    beta_matrix = universal_gamma * mass_matrix + sm_majorana_c_e * (projector @ mass_matrix + mass_matrix @ projector)
    return 0.5 * (beta_matrix + beta_matrix.T)


def takagi_diagonalize_symmetric(
    mass_matrix: npt.NDArray[np.complexfloating | np.floating],
    *,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.float64]]:
    symmetric_matrix = 0.5 * (mass_matrix + mass_matrix.T)
    left_unitary, singular_values, vh_matrix = np.linalg.svd(symmetric_matrix)
    right_unitary = vh_matrix.conjugate().T
    phase_alignment = left_unitary.conjugate().T @ right_unitary
    diagonal_phases = np.diag(phase_alignment)
    phase_roots = np.exp(
        -0.5j
        * np.angle(
            np.array(
                [
                    solver_config.stability_guard.clamp_nonzero_magnitude(
                        value,
                        coordinate=f"Takagi phase alignment[{index}]",
                        fallback=1.0 + 0.0j,
                    )
                    for index, value in enumerate(diagonal_phases)
                ],
                dtype=complex,
            )
        )
    )
    takagi_unitary = left_unitary @ np.diag(phase_roots)

    order = np.argsort(singular_values)
    takagi_unitary = takagi_unitary[:, order]

    diagonalized = takagi_unitary.T @ symmetric_matrix @ takagi_unitary
    residual_diagonal = np.diag(diagonalized)
    residual_phases = np.exp(
        -0.5j
        * np.angle(
            np.array(
                [
                    solver_config.stability_guard.clamp_nonzero_magnitude(
                        value,
                        coordinate=f"Takagi residual phase[{index}]",
                        fallback=1.0 + 0.0j,
                    )
                    for index, value in enumerate(residual_diagonal)
                ],
                dtype=complex,
            )
        )
    )
    takagi_unitary = takagi_unitary @ np.diag(residual_phases)
    masses = np.abs(
        np.real_if_close(np.diag(takagi_unitary.T @ symmetric_matrix @ takagi_unitary))
    ).astype(float)
    return takagi_unitary, masses


def wrapped_angle_difference_deg(updated_angle_deg: float, reference_angle_deg: float) -> float:
    return ((updated_angle_deg - reference_angle_deg + 180.0) % 360.0) - 180.0


def matrix_derived_lepton_betas(
    unitary: npt.NDArray[np.complexfloating | np.floating],
    masses_ev: npt.NDArray[np.float64],
    *,
    phase_proxies_rad: tuple[float, float],
    tau_yukawa: float,
    charged_lepton_yukawa_ratios: dict[str, float],
    sm_majorana_c_e: float,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    pdg_parameter_extractor: Callable[[np.ndarray], tuple[float, float, float, float, float]],
) -> tuple[npt.NDArray[np.float64], float]:
    theta12_deg, theta13_deg, theta23_deg, delta_deg, _ = pdg_parameter_extractor(unitary)
    mass_matrix = majorana_mass_matrix_from_pmns(unitary, masses_ev, phase_proxies_rad)
    beta_matrix = majorana_mass_matrix_beta(
        mass_matrix,
        tau_yukawa,
        charged_lepton_yukawa_ratios,
        sm_majorana_c_e,
    )
    finite_diff_step = float(solver_config.finite_diff_step)
    if finite_diff_step <= 0.0:
        raise ValueError(f"finite_diff_step must be positive, received {finite_diff_step}")
    evolved_mass_matrix_plus = mass_matrix + finite_diff_step * beta_matrix
    evolved_mass_matrix_minus = mass_matrix - finite_diff_step * beta_matrix
    evolved_unitary_plus, _ = takagi_diagonalize_symmetric(
        evolved_mass_matrix_plus,
        solver_config=solver_config,
    )
    evolved_unitary_minus, _ = takagi_diagonalize_symmetric(
        evolved_mass_matrix_minus,
        solver_config=solver_config,
    )
    theta12_plus, theta13_plus, theta23_plus, delta_plus, _ = pdg_parameter_extractor(evolved_unitary_plus)
    theta12_minus, theta13_minus, theta23_minus, delta_minus, _ = pdg_parameter_extractor(evolved_unitary_minus)
    theta_betas = np.array(
        [
            (theta12_plus - theta12_minus) / (2.0 * finite_diff_step),
            (theta13_plus - theta13_minus) / (2.0 * finite_diff_step),
            (theta23_plus - theta23_minus) / (2.0 * finite_diff_step),
        ],
        dtype=float,
    )
    delta_beta = wrapped_angle_difference_deg(delta_plus, delta_minus) / (2.0 * finite_diff_step)
    return theta_betas, float(delta_beta)


def _pack_complex_matrix(matrix: npt.NDArray[np.complexfloating | np.floating]) -> npt.NDArray[np.float64]:
    return np.concatenate([np.real(matrix).ravel(), np.imag(matrix).ravel()]).astype(float)


def _unpack_complex_matrix(state: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
    half = state.size // 2
    real_part = state[:half].reshape(3, 3)
    imag_part = state[half:].reshape(3, 3)
    return real_part + 1j * imag_part


def safe_log_magnitude(value: complex | float, *, floor: float = SAFE_LOG_MAGNITUDE_FLOOR) -> float:
    return math.log(max(abs(value), floor))


def quark_branching_pressure(
    visible_block: npt.NDArray[np.complexfloating | np.floating],
    rank_difference: int,
    *,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    floor: float = SAFE_LOG_MAGNITUDE_FLOOR,
) -> float:
    safe_reference_entry = solver_config.stability_guard.clamp_nonzero_magnitude(
        visible_block[0, 0],
        coordinate="quark branching pressure reference entry",
        fallback=complex(floor, 0.0),
        floor=floor,
    )
    return -(rank_difference / 8.0) * safe_log_magnitude(safe_reference_entry, floor=floor)


def integrate_pmns_majorana_rge_numerically(
    uv_matrix: np.ndarray,
    m_0_uv_ev: float,
    *,
    phase_proxies_rad: tuple[float, float],
    scale_ratio: float,
    mz_scale_gev: float,
    total_loop_time: float,
    mass_spectrum_builder: Callable[[float], np.ndarray],
    coupling_state_builder: Callable[[float], np.ndarray],
    running_coupling_betas: Callable[[np.ndarray], np.ndarray],
    gamma_0_one_loop: float,
    gamma_0_two_loop: float,
    charged_lepton_yukawa_ratios: dict[str, float],
    sm_majorana_c_e: float,
    max_step: float | None = None,
    solver_config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    pdg_parameter_extractor: Callable[[np.ndarray], tuple[float, float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    uv_scale_gev = mz_scale_gev * scale_ratio
    uv_mass_matrix = majorana_mass_matrix_from_pmns(uv_matrix, mass_spectrum_builder(m_0_uv_ev), phase_proxies_rad)
    coupling_state_uv = coupling_state_builder(uv_scale_gev)

    def transport_equations(loop_time: float, state: np.ndarray) -> np.ndarray:
        unpacked = _unpack_complex_matrix(state[:18])
        mass_matrix = 0.5 * (unpacked + unpacked.T)
        couplings = state[18:]
        universal_gamma = gamma_0_one_loop + 2.0 * loop_time * gamma_0_two_loop
        mass_beta = majorana_mass_matrix_beta(
            mass_matrix,
            tau_yukawa=float(couplings[2]),
            charged_lepton_yukawa_ratios=charged_lepton_yukawa_ratios,
            sm_majorana_c_e=sm_majorana_c_e,
            universal_gamma=universal_gamma,
        )
        coupling_betas = running_coupling_betas(couplings)
        return np.concatenate([_pack_complex_matrix(mass_beta), -coupling_betas]).astype(float)

    initial_state = np.concatenate([_pack_complex_matrix(uv_mass_matrix), coupling_state_uv]).astype(float)
    solve_kwargs = {} if max_step is None else {"max_step": max_step}
    solution = solve_ivp_with_fallback(
        transport_equations,
        (0.0, total_loop_time),
        initial_state,
        solver_config=solver_config,
        **solve_kwargs,
    )
    final_mass_matrix = 0.5 * (
        _unpack_complex_matrix(solution.y[:18, -1])
        + _unpack_complex_matrix(solution.y[:18, -1]).T
    )
    pmns_rg, masses_rg = takagi_diagonalize_symmetric(
        final_mass_matrix,
        solver_config=solver_config,
    )
    theta12_rg, theta13_rg, theta23_rg, delta_rg, _ = pdg_parameter_extractor(pmns_rg)
    return pmns_rg, np.array([theta12_rg, theta13_rg, theta23_rg], dtype=float), float(delta_rg % 360.0), float(masses_rg[0])
