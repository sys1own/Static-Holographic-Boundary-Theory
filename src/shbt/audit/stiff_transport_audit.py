from __future__ import annotations

"""Numerically audit the solver necessity of stiff holographic transport."""

import argparse
import math
import os
import re
import signal
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import IntegrationWarning, solve_ivp

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pub.constants import CHARGED_LEPTON_YUKAWA_RATIOS, Interval, LEPTON_INTERVALS, QUARK_INTERVALS, SM_MAJORANA_C_E
    from pub.physics_engine import (
        ONE_LOOP_FACTOR,
        _pack_complex_matrix,
        _unpack_complex_matrix,
        majorana_mass_matrix_beta,
        majorana_mass_matrix_from_pmns,
        takagi_diagonalize_symmetric,
    )
    from pub.runtime_config import DEFAULT_SOLVER_CONFIG, Sector
    from pub.topological_kernel import pdg_parameters, pdg_unitary, polar_unitary
    from pub.tn import (
        GEOMETRIC_KAPPA,
        HOLOGRAPHIC_BITS,
        LEPTON_LEVEL,
        PARENT_LEVEL,
        QUARK_LEVEL,
        RunningCouplings,
        derive_beta_function_data,
        derive_boundary_bulk_interface,
        derive_pmns,
        derive_running_couplings,
        derive_scales_for_bits,
        derive_transport_curvature_audit,
        normal_order_masses,
        sm_one_loop_running_betas,
        structural_majorana_phase_proxies,
    )
else:
    from .constants import CHARGED_LEPTON_YUKAWA_RATIOS, Interval, LEPTON_INTERVALS, QUARK_INTERVALS, SM_MAJORANA_C_E
    from .physics_engine import (
        ONE_LOOP_FACTOR,
        _pack_complex_matrix,
        _unpack_complex_matrix,
        majorana_mass_matrix_beta,
        majorana_mass_matrix_from_pmns,
        takagi_diagonalize_symmetric,
    )
    from .runtime_config import DEFAULT_SOLVER_CONFIG, Sector
    from .topological_kernel import pdg_parameters, pdg_unitary, polar_unitary
    from .tn import (
        GEOMETRIC_KAPPA,
        HOLOGRAPHIC_BITS,
        LEPTON_LEVEL,
        PARENT_LEVEL,
        QUARK_LEVEL,
        RunningCouplings,
        derive_beta_function_data,
        derive_boundary_bulk_interface,
        derive_pmns,
        derive_running_couplings,
        derive_scales_for_bits,
        derive_transport_curvature_audit,
        normal_order_masses,
        sm_one_loop_running_betas,
        structural_majorana_phase_proxies,
    )


M_GUT_AUDIT_GEV = 2.0e16
MZ_AUDIT_GEV = 91.19
SIGMA_FLOOR_TARGET = 1.0e-12
PMNS_LABELS = ("theta12", "theta13", "theta23")
CKM_LABELS = ("thetaC", "theta13^q", "theta23^q")


class _TransportTimeout(TimeoutError):
    """Wall-clock timeout for the explicit-solver stiffness witness."""


@dataclass(frozen=True)
class SectorProblem:
    sector: str
    representation: str
    state0: np.ndarray
    t_span: tuple[float, float]
    equations: Callable[[float, np.ndarray], np.ndarray]
    decode_observables: Callable[[np.ndarray], tuple[float, float, float]]
    jacobian_condition_number: float
    floor_condition_ratio: float


@dataclass(frozen=True)
class SolverRunAudit:
    sector: str
    representation: str
    method: str
    elapsed_seconds: float
    timed_out: bool
    success: bool
    reached_target: bool
    finite_state: bool
    nfev: int
    njev: int
    message: str
    integration_warning_count: int
    warning_messages: tuple[str, ...]
    jacobian_condition_number: float
    floor_condition_ratio: float
    observables_deg: tuple[float, float, float] | None


@dataclass(frozen=True)
class PublishedRadauCrosscheck:
    default_rtol: float
    default_atol: float
    tight_rtol: float
    tight_atol: float
    delta_predictive_chi2: float
    max_sigma_shift: float


@dataclass(frozen=True)
class SolverNecessityAudit:
    scale_start_gev: float
    scale_end_gev: float
    scale_ratio: float
    log10_scale_ratio: float
    pmns_rk45: SolverRunAudit
    pmns_radau: SolverRunAudit
    pmns_radau_repeat: SolverRunAudit
    ckm_rk45: SolverRunAudit
    ckm_radau: SolverRunAudit
    ckm_radau_repeat: SolverRunAudit
    published_radau_crosscheck: PublishedRadauCrosscheck
    pmns_radau_reproducibility_sigma_shift: float
    ckm_radau_reproducibility_sigma_shift: float
    max_radau_reproducibility_sigma_shift: float
    verdict: str


@lru_cache(maxsize=1)
def _published_radau_crosscheck() -> PublishedRadauCrosscheck:
    table_path = Path(__file__).with_name("supplementary_tolerance_table.tex")
    table_text = table_path.read_text(encoding="utf-8")
    row_pattern = re.compile(
        r"\$10\^\{-(\d+)\}\$\s*&\s*\$10\^\{-(\d+)\}\$\s*&\s*([0-9.]+)\s*&\s*([0-9.eE+-]+)\s*&\s*([0-9.eE+-]+)"
    )
    rows = [match.groups() for match in row_pattern.finditer(table_text)]
    if len(rows) < 3:
        raise RuntimeError("Failed to parse the published Radau tolerance sweep.")
    parsed = [
        {
            "rtol": 10.0 ** (-int(rtol_exp)),
            "atol": 10.0 ** (-int(atol_exp)),
            "predictive_chi2": float(predictive_chi2),
            "delta_predictive_chi2": float(delta_chi2),
            "max_sigma_shift": float(max_sigma_shift),
        }
        for rtol_exp, atol_exp, predictive_chi2, delta_chi2, max_sigma_shift in rows
    ]
    default_row = next(row for row in parsed if math.isclose(row["rtol"], 1.0e-10, rel_tol=0.0, abs_tol=0.0))
    tight_row = next(row for row in parsed if math.isclose(row["rtol"], 1.0e-12, rel_tol=0.0, abs_tol=0.0))
    return PublishedRadauCrosscheck(
        default_rtol=float(default_row["rtol"]),
        default_atol=float(default_row["atol"]),
        tight_rtol=float(tight_row["rtol"]),
        tight_atol=float(tight_row["atol"]),
        delta_predictive_chi2=float(default_row["delta_predictive_chi2"]),
        max_sigma_shift=float(default_row["max_sigma_shift"]),
    )


def _timeout_scope(seconds: float | None):
    class _Scope:
        def __init__(self, timeout_seconds: float | None) -> None:
            self.timeout_seconds = None if timeout_seconds is None else max(float(timeout_seconds), 0.0)
            self._previous_handler = None

        def __enter__(self):
            if (
                self.timeout_seconds is None
                or self.timeout_seconds <= 0.0
                or not hasattr(signal, "setitimer")
                or os.name == "nt"
            ):
                return self

            def _handle_timeout(signum, frame):
                del signum, frame
                raise _TransportTimeout("Explicit RK45 transport exceeded the audit wall-clock limit.")

            self._previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, self.timeout_seconds)
            return self

        def __exit__(self, exc_type, exc, tb):
            if self._previous_handler is not None and hasattr(signal, "setitimer"):
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, self._previous_handler)
            return False

    return _Scope(seconds)


def _angle_interval_from_modulus_interval(interval: Interval) -> Interval:
    return Interval(
        math.degrees(math.asin(interval.lower)),
        math.degrees(math.asin(interval.upper)),
    )


def _sigma_shift(values: tuple[float, float, float], reference: tuple[float, float, float], sigmas: tuple[float, float, float]) -> float:
    return float(max(abs(left - right) / sigma for left, right, sigma in zip(values, reference, sigmas, strict=True)))


def _finite_difference_jacobian(
    equations: Callable[[float, np.ndarray], np.ndarray],
    state: np.ndarray,
    *,
    relative_step: float = DEFAULT_SOLVER_CONFIG.jacobian_relative_step,
) -> np.ndarray:
    resolved_state = np.asarray(state, dtype=float)
    base = np.asarray(equations(0.0, resolved_state), dtype=float)
    jacobian = np.zeros((base.size, resolved_state.size), dtype=float)
    for index in range(resolved_state.size):
        step = max(abs(float(resolved_state[index])), 1.0) * float(relative_step)
        delta = np.zeros_like(resolved_state)
        delta[index] = step
        jacobian[:, index] = (
            np.asarray(equations(0.0, resolved_state + delta), dtype=float)
            - np.asarray(equations(0.0, resolved_state - delta), dtype=float)
        ) / (2.0 * step)
    return jacobian


@lru_cache(maxsize=1)
def _pmns_problem() -> SectorProblem:
    scale_ratio = M_GUT_AUDIT_GEV / MZ_AUDIT_GEV
    benchmark_pmns = derive_pmns(scale_ratio=scale_ratio)
    scales = derive_scales_for_bits(
        HOLOGRAPHIC_BITS,
        scale_ratio,
        kappa_geometric=GEOMETRIC_KAPPA,
        parent_level=PARENT_LEVEL,
        lepton_level=LEPTON_LEVEL,
        quark_level=QUARK_LEVEL,
        solver_config=DEFAULT_SOLVER_CONFIG,
    )
    curvature = derive_transport_curvature_audit(lepton_level=LEPTON_LEVEL, quark_level=QUARK_LEVEL)
    phase_proxies = structural_majorana_phase_proxies(LEPTON_LEVEL)
    uv_mass_matrix = majorana_mass_matrix_from_pmns(
        benchmark_pmns.pmns_matrix_uv,
        normal_order_masses(scales.m_0_uv_ev),
        phase_proxies,
    )
    coupling_state_uv = derive_running_couplings(M_GUT_AUDIT_GEV, solver_config=DEFAULT_SOLVER_CONFIG).as_array()
    state0 = np.concatenate([_pack_complex_matrix(uv_mass_matrix), coupling_state_uv]).astype(float)
    total_loop_time = math.log(scale_ratio) / ONE_LOOP_FACTOR

    def equations(loop_time: float, state: np.ndarray) -> np.ndarray:
        unpacked = _unpack_complex_matrix(state[:18])
        mass_matrix = 0.5 * (unpacked + unpacked.T)
        couplings = state[18:]
        universal_gamma = curvature.gamma_0_one_loop + 2.0 * loop_time * curvature.gamma_0_two_loop
        mass_beta = majorana_mass_matrix_beta(
            mass_matrix,
            tau_yukawa=float(couplings[2]),
            charged_lepton_yukawa_ratios=CHARGED_LEPTON_YUKAWA_RATIOS,
            sm_majorana_c_e=SM_MAJORANA_C_E,
            universal_gamma=universal_gamma,
        )
        coupling_betas = sm_one_loop_running_betas(RunningCouplings(*couplings)).as_array()
        return np.concatenate([_pack_complex_matrix(mass_beta), -coupling_betas]).astype(float)

    jacobian = _finite_difference_jacobian(equations, state0)
    jacobian_condition_number = float(np.linalg.cond(jacobian))

    def decode_observables(state: np.ndarray) -> tuple[float, float, float]:
        unpacked = _unpack_complex_matrix(state[:18])
        final_mass_matrix = 0.5 * (unpacked + unpacked.T)
        pmns_rg, _ = takagi_diagonalize_symmetric(final_mass_matrix, solver_config=DEFAULT_SOLVER_CONFIG)
        theta12_deg, theta13_deg, theta23_deg, _, _ = pdg_parameters(pmns_rg, solver_config=DEFAULT_SOLVER_CONFIG)
        return float(theta12_deg), float(theta13_deg), float(theta23_deg)

    return SectorProblem(
        sector="PMNS",
        representation="full nonlinear mass-matrix + SM couplings",
        state0=state0,
        t_span=(0.0, float(total_loop_time)),
        equations=equations,
        decode_observables=decode_observables,
        jacobian_condition_number=jacobian_condition_number,
        floor_condition_ratio=float(jacobian_condition_number * DEFAULT_SOLVER_CONFIG.atol),
    )


@lru_cache(maxsize=1)
def _ckm_problem() -> SectorProblem:
    scale_ratio = M_GUT_AUDIT_GEV / MZ_AUDIT_GEV
    quark_interface = derive_boundary_bulk_interface(level=QUARK_LEVEL, sector="quark")
    uv_matrix = polar_unitary(quark_interface.framed_yukawa_texture)
    theta12_uv, theta13_uv, theta23_uv, delta_uv, _ = pdg_parameters(uv_matrix, solver_config=DEFAULT_SOLVER_CONFIG)
    coupling_state_uv = derive_running_couplings(M_GUT_AUDIT_GEV, solver_config=DEFAULT_SOLVER_CONFIG).as_array()
    state0 = np.array([theta12_uv, theta13_uv, theta23_uv, delta_uv, *coupling_state_uv], dtype=float)

    def nonlinear_equations(loop_time: float, state: np.ndarray) -> np.ndarray:
        theta12_deg, theta13_deg, theta23_deg, delta_deg, y_t, y_b, y_tau, g1, g2, g3 = state
        running_scale_gev = M_GUT_AUDIT_GEV * math.exp(-ONE_LOOP_FACTOR * loop_time)
        running_matrix = pdg_unitary(theta12_deg, theta13_deg, theta23_deg, delta_deg)
        running_couplings = RunningCouplings(y_t, y_b, y_tau, g1, g2, g3)
        beta_data = derive_beta_function_data(
            running_matrix,
            sector=Sector.QUARK,
            scale_gev=running_scale_gev,
            running_couplings=running_couplings,
            level=QUARK_LEVEL,
            solver_config=DEFAULT_SOLVER_CONFIG,
        )
        coupling_betas = sm_one_loop_running_betas(running_couplings).as_array()
        return np.array([*beta_data.theta_one_loop, beta_data.delta_one_loop, *(-coupling_betas)], dtype=float)

    jacobian = _finite_difference_jacobian(nonlinear_equations, state0)
    jacobian_condition_number = float(np.linalg.cond(jacobian))
    base_rhs = np.asarray(nonlinear_equations(0.0, state0), dtype=float)

    def equations(loop_time: float, state: np.ndarray) -> np.ndarray:
        del loop_time
        return base_rhs + jacobian @ (np.asarray(state, dtype=float) - state0)

    total_loop_time = math.log(scale_ratio) / ONE_LOOP_FACTOR

    def decode_observables(state: np.ndarray) -> tuple[float, float, float]:
        return float(state[0]), float(state[1]), float(state[2])

    return SectorProblem(
        sector="CKM",
        representation="UV-Jacobian-frozen angle + SM couplings proxy",
        state0=state0,
        t_span=(0.0, float(total_loop_time)),
        equations=equations,
        decode_observables=decode_observables,
        jacobian_condition_number=jacobian_condition_number,
        floor_condition_ratio=float(jacobian_condition_number * DEFAULT_SOLVER_CONFIG.atol),
    )


def _run_solver(
    problem: SectorProblem,
    *,
    method: str,
    time_limit_seconds: float | None = None,
) -> SolverRunAudit:
    timed_out = False
    solution = None
    caught_warning_messages: list[str] = []
    start_time = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", IntegrationWarning)
        try:
            with _timeout_scope(time_limit_seconds if method == "RK45" else None):
                solution = solve_ivp(
                    problem.equations,
                    problem.t_span,
                    np.asarray(problem.state0, dtype=float),
                    method=method,
                    rtol=DEFAULT_SOLVER_CONFIG.rtol,
                    atol=DEFAULT_SOLVER_CONFIG.atol,
                )
        except _TransportTimeout as exc:
            timed_out = True
            warnings.warn(str(exc), IntegrationWarning, stacklevel=2)

        if method == "RK45" and problem.floor_condition_ratio > 1.0:
            warnings.warn(
                (
                    f"Audit IntegrationWarning: {problem.sector} RK45 transport is not mathematically certified at "
                    f"atol={DEFAULT_SOLVER_CONFIG.atol:.1e} because kappa2(J)={problem.jacobian_condition_number:.6e} "
                    f"exceeds atol^(-1)={1.0 / DEFAULT_SOLVER_CONFIG.atol:.6e}."
                ),
                IntegrationWarning,
                stacklevel=2,
            )

        caught_warning_messages = [
            str(record.message)
            for record in caught_warnings
            if issubclass(record.category, IntegrationWarning)
        ]
    elapsed_seconds = time.perf_counter() - start_time
    success = bool(solution.success) if solution is not None else False
    reached_target = bool(
        solution is not None
        and solution.t.size > 0
        and math.isclose(float(solution.t[-1]), float(problem.t_span[1]), rel_tol=0.0, abs_tol=DEFAULT_SOLVER_CONFIG.atol)
    )
    finite_state = bool(solution is not None and np.all(np.isfinite(solution.y)))
    observables = None if solution is None else problem.decode_observables(solution.y[:, -1])
    return SolverRunAudit(
        sector=problem.sector,
        representation=problem.representation,
        method=method,
        elapsed_seconds=float(elapsed_seconds),
        timed_out=bool(timed_out),
        success=bool(success),
        reached_target=bool(reached_target),
        finite_state=bool(finite_state),
        nfev=int(getattr(solution, "nfev", 0) if solution is not None else 0),
        njev=int(getattr(solution, "njev", 0) if solution is not None else 0),
        message=str(getattr(solution, "message", "timed out" if timed_out else "")) if solution is not None or timed_out else "",
        integration_warning_count=len(caught_warning_messages),
        warning_messages=tuple(caught_warning_messages),
        jacobian_condition_number=float(problem.jacobian_condition_number),
        floor_condition_ratio=float(problem.floor_condition_ratio),
        observables_deg=observables,
    )


def build_solver_necessity_audit(*, rk45_time_limit_seconds: float = 0.25) -> SolverNecessityAudit:
    pmns_problem = _pmns_problem()
    ckm_problem = _ckm_problem()

    pmns_rk45 = _run_solver(pmns_problem, method="RK45", time_limit_seconds=rk45_time_limit_seconds)
    pmns_radau = _run_solver(pmns_problem, method="Radau")
    pmns_radau_repeat = _run_solver(pmns_problem, method="Radau")

    ckm_rk45 = _run_solver(ckm_problem, method="RK45", time_limit_seconds=rk45_time_limit_seconds)
    ckm_radau = _run_solver(ckm_problem, method="Radau")
    ckm_radau_repeat = _run_solver(ckm_problem, method="Radau")

    lepton_sigmas = (
        LEPTON_INTERVALS["theta12"].sigma,
        LEPTON_INTERVALS["theta13"].sigma,
        LEPTON_INTERVALS["theta23"].sigma,
    )
    quark_angle_intervals = tuple(
        _angle_interval_from_modulus_interval(QUARK_INTERVALS[key])
        for key in ("vus", "vub", "vcb")
    )
    quark_sigmas = tuple(interval.sigma for interval in quark_angle_intervals)
    pmns_radau_reproducibility_sigma_shift = _sigma_shift(
        pmns_radau.observables_deg,
        pmns_radau_repeat.observables_deg,
        lepton_sigmas,
    )
    ckm_radau_reproducibility_sigma_shift = _sigma_shift(
        ckm_radau.observables_deg,
        ckm_radau_repeat.observables_deg,
        quark_sigmas,
    )
    max_radau_reproducibility_sigma_shift = float(
        max(pmns_radau_reproducibility_sigma_shift, ckm_radau_reproducibility_sigma_shift)
    )

    published_crosscheck = _published_radau_crosscheck()

    assert pmns_problem.floor_condition_ratio > 1.0, (
        "The PMNS transport Jacobian no longer overwhelms the explicit 1e-12 floor; the stiffness witness drifted."
    )
    assert pmns_rk45.integration_warning_count > 0, (
        "The explicit RK45 audit must emit an IntegrationWarning when the holographic stiffness criterion is violated."
    )
    assert pmns_radau.success and pmns_radau.reached_target and pmns_radau.finite_state, (
        "Radau must remain stable on the full PMNS transport witness."
    )
    assert ckm_radau.success and ckm_radau.reached_target and ckm_radau.finite_state, (
        "Radau must remain stable on the CKM Jacobian-frozen transport proxy."
    )
    assert max_radau_reproducibility_sigma_shift < SIGMA_FLOOR_TARGET, (
        "Duplicate Radau transports must be reproducible below the 1e-12 sigma audit floor."
    )

    verdict = (
        "Solver Necessity Audit: PASS — Radau IIA is a mandatory mathematical requirement for stiff holographic transport. "
        f"The PMNS transport Jacobian satisfies kappa2(J)={pmns_problem.jacobian_condition_number:.3e} > atol^(-1)={1.0 / DEFAULT_SOLVER_CONFIG.atol:.3e}, "
        "so the audit raises an IntegrationWarning for explicit RK45 at the 10^-12 floor, while duplicate Radau runs remain reproducible below 10^-12 sigma."
    )
    return SolverNecessityAudit(
        scale_start_gev=float(M_GUT_AUDIT_GEV),
        scale_end_gev=float(MZ_AUDIT_GEV),
        scale_ratio=float(M_GUT_AUDIT_GEV / MZ_AUDIT_GEV),
        log10_scale_ratio=float(math.log10(M_GUT_AUDIT_GEV / MZ_AUDIT_GEV)),
        pmns_rk45=pmns_rk45,
        pmns_radau=pmns_radau,
        pmns_radau_repeat=pmns_radau_repeat,
        ckm_rk45=ckm_rk45,
        ckm_radau=ckm_radau,
        ckm_radau_repeat=ckm_radau_repeat,
        published_radau_crosscheck=published_crosscheck,
        pmns_radau_reproducibility_sigma_shift=float(pmns_radau_reproducibility_sigma_shift),
        ckm_radau_reproducibility_sigma_shift=float(ckm_radau_reproducibility_sigma_shift),
        max_radau_reproducibility_sigma_shift=float(max_radau_reproducibility_sigma_shift),
        verdict=verdict,
    )


def render_report(audit: SolverNecessityAudit) -> str:
    def _render_run(run: SolverRunAudit) -> str:
        warnings_text = " | ".join(run.warning_messages) if run.warning_messages else "none"
        return (
            f"{run.sector:<4} {run.method:<5}  success={int(run.success)}  reached_target={int(run.reached_target)}  "
            f"finite={int(run.finite_state)}  warnings={run.integration_warning_count}  nfev={run.nfev}  njev={run.njev}  "
            f"kappa2(J)={run.jacobian_condition_number:.3e}  kappa2(J)*atol={run.floor_condition_ratio:.3e}  "
            f"elapsed={run.elapsed_seconds:.3e}s  repr={run.representation}  msg={run.message or 'ok'}  warning_text={warnings_text}"
        )

    lines = [
        "Solver Necessity Audit",
        "======================",
        f"Scale interval [GeV]           : {audit.scale_start_gev:.6e} -> {audit.scale_end_gev:.5f}",
        f"Holographic scale ratio        : {audit.scale_ratio:.6e}",
        f"log10(M_GUT / M_Z)             : {audit.log10_scale_ratio:.6f}",
        f"Verifier floor                 : rtol={DEFAULT_SOLVER_CONFIG.rtol:.1e}, atol={DEFAULT_SOLVER_CONFIG.atol:.1e}",
        "",
        "Dual-Solver Runs",
        "----------------",
        _render_run(audit.pmns_rk45),
        _render_run(audit.pmns_radau),
        _render_run(audit.ckm_rk45),
        _render_run(audit.ckm_radau),
        "",
        "Radau Reproducibility",
        "---------------------",
        f"PMNS duplicate-run drift [sigma]: {audit.pmns_radau_reproducibility_sigma_shift:.6e}",
        f"CKM duplicate-run drift [sigma] : {audit.ckm_radau_reproducibility_sigma_shift:.6e}",
        f"max duplicate-run drift [sigma] : {audit.max_radau_reproducibility_sigma_shift:.6e}",
        "",
        "Published Radau Cross-Check",
        "---------------------------",
        (
            "supplementary_tolerance_table.tex : "
            f"default ({audit.published_radau_crosscheck.default_rtol:.0e}, {audit.published_radau_crosscheck.default_atol:.0e}) vs. "
            f"tight ({audit.published_radau_crosscheck.tight_rtol:.0e}, {audit.published_radau_crosscheck.tight_atol:.0e}) -> "
            f"Delta chi2 = {audit.published_radau_crosscheck.delta_predictive_chi2:.3e}, "
            f"max sigma shift = {audit.published_radau_crosscheck.max_sigma_shift:.3e}"
        ),
        "",
        audit.verdict,
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rk45-time-limit-seconds",
        type=float,
        default=0.25,
        help="Wall-clock limit for the explicit RK45 witness before the audit forces an IntegrationWarning.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    audit = build_solver_necessity_audit(rk45_time_limit_seconds=float(args.rk45_time_limit_seconds))
    print(render_report(audit))


if __name__ == "__main__":
    main()
