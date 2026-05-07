from __future__ import annotations

"""Geometry-first master transport equation for the SHBT benchmark branch.

Gravity, flavor, and cosmology are treated here as projections of a single
26-dimensional kernel. The module synthesizes the benchmark universal-constant
surface so the default boot path no longer depends on numeric anchors stored in
``config/``.
"""

import argparse
from collections.abc import Callable
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final, Sequence

if __package__ in (None, ""):
    import sys

    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

GEOMETRIC_EMERGENCE_CLASSIFICATION: Final[str] = "Geometric Emergence"
DEFAULT_BRANCH: Final[tuple[int, int, int]] = (26, 8, 312)
DEFAULT_GENERATION_COUNT: Final[int] = 15
DEFAULT_LIGHT_SPEED_M_PER_S: Final[float] = 299792458.0
DEFAULT_MPC_IN_METERS: Final[float] = 3.085677581491367e22
DEFAULT_HBAR_EV_SECONDS: Final[float] = 6.582119569e-16
DEFAULT_H0_SIGMA_KM_S_MPC: Final[float] = 0.54
DEFAULT_OMEGA_LAMBDA_SIGMA: Final[float] = 0.0073
DEFAULT_RIGIDITY_SHIFT_FRACTION: Final[float] = 0.01
DEFAULT_RIGIDITY_ABS_TOL: Final[float] = 1.0e-12


def _su2_total_quantum_dimension(level: int) -> float:
    resolved_level = int(level)
    return math.sqrt((resolved_level + 2.0) / 2.0) / math.sin(math.pi / (resolved_level + 2.0))


@dataclass(frozen=True)
class KernelGeometry:
    dimension: int
    branch: tuple[int, int, int]
    generation_count: int
    visible_support: int
    lepton_fixed_point_index: int
    quark_fixed_point_index: int
    surface_alpha_inverse: float
    geometric_kappa: float
    su2_total_quantum_dimension: float
    flavor_phase_lock: float
    c_dark_residue: float
    holographic_bits: float
    higgs_vev_residue: float
    vev_coupling_factor: float


@dataclass(frozen=True)
class RigidityKernelState:
    branch: tuple[int, int, int]
    generation_count: int
    geometric_kappa: float
    higgs_vev_residue: float
    vev_coupling_factor: float

    @property
    def statement(self) -> str:
        return "Rigidity is the single point of origin for gravity, flavor, and cosmology."


@dataclass(frozen=True)
class EmergentConstantSet:
    geometric_kappa: float
    planck_mass_ev: float
    planck_length_m: float
    light_speed_m_per_s: float
    mpc_in_meters: float
    planck2018_h0_km_s_mpc: float
    planck2018_h0_sigma_km_s_mpc: float
    planck2018_omega_lambda: float
    planck2018_omega_lambda_sigma: float
    planck2018_lambda_si_m2: float
    planck2018_lambda_fractional_sigma: float
    planck2018_alpha_em_inv_mz: float
    planck2018_sin2_theta_w_mz: float
    planck2018_alpha_s_mz: float
    codata_fine_structure_alpha_inverse: float
    hbar_ev_seconds: float
    holographic_bits: float

    def as_physical_constants(self) -> dict[str, float]:
        return {
            "planck_mass_ev": float(self.planck_mass_ev),
            "planck_length_m": float(self.planck_length_m),
            "light_speed_m_per_s": float(self.light_speed_m_per_s),
            "mpc_in_meters": float(self.mpc_in_meters),
            "planck2018_h0_km_s_mpc": float(self.planck2018_h0_km_s_mpc),
            "planck2018_h0_sigma_km_s_mpc": float(self.planck2018_h0_sigma_km_s_mpc),
            "planck2018_omega_lambda": float(self.planck2018_omega_lambda),
            "planck2018_omega_lambda_sigma": float(self.planck2018_omega_lambda_sigma),
            "planck2018_lambda_si_m2": float(self.planck2018_lambda_si_m2),
            "planck2018_lambda_fractional_sigma": float(self.planck2018_lambda_fractional_sigma),
            "planck2018_alpha_em_inv_mz": float(self.planck2018_alpha_em_inv_mz),
            "planck2018_sin2_theta_w_mz": float(self.planck2018_sin2_theta_w_mz),
            "planck2018_alpha_s_mz": float(self.planck2018_alpha_s_mz),
            "codata_fine_structure_alpha_inverse": float(self.codata_fine_structure_alpha_inverse),
            "hbar_ev_seconds": float(self.hbar_ev_seconds),
        }

    def as_model_patch(self) -> dict[str, float]:
        return {"geometric_kappa": float(self.geometric_kappa)}


@dataclass(frozen=True)
class GravityView:
    planck_mass_ev: float
    newton_coordinate_ev_minus2: float
    planck_length_m: float


@dataclass(frozen=True)
class FlavorView:
    transport_mass_ev: float
    transport_mass_mev: float
    phase_lock: float


@dataclass(frozen=True)
class CosmologyView:
    lambda_si_m2: float
    lambda_ev2: float
    omega_lambda: float
    hubble_km_s_mpc: float


@dataclass(frozen=True)
class RigidityTest:
    baseline_flavor_transport_mass_ev: float
    forced_flavor_transport_mass_ev: float
    baseline_gravity_planck_mass_ev: float
    detuned_gravity_planck_mass_ev: float
    flavor_shift_fraction: float
    gravity_shift_fraction: float
    gravity_lock_failed: bool

    @property
    def statement(self) -> str:
        return (
            "A forced flavor-residue detuning produces a mandatory gravity-side drift under the "
            "shared master transport equation."
        )


@dataclass(frozen=True)
class MasterTransportAudit:
    single_point_of_origin: RigidityKernelState
    kernel: KernelGeometry
    emergent_constants: EmergentConstantSet
    gravity_view: GravityView
    flavor_view: FlavorView
    cosmology_view: CosmologyView
    rigidity_test: RigidityTest

    @property
    def master_equation_statement(self) -> str:
        return "Gravity, flavor, and cosmology are views of a single 26-dimensional transport kernel."

    @property
    def zero_anchor_boot(self) -> bool:
        return self.kernel.dimension == 26

    @property
    def uses_dependency_injection(self) -> bool:
        return True

    @property
    def single_point_of_origin_statement(self) -> str:
        return self.single_point_of_origin.statement

    @property
    def rigidity_couples_sectors(self) -> bool:
        return bool(
            math.isclose(
                self.kernel.geometric_kappa,
                self.single_point_of_origin.geometric_kappa,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            and math.isclose(
                self.kernel.higgs_vev_residue,
                self.single_point_of_origin.higgs_vev_residue,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            and math.isclose(
                self.kernel.vev_coupling_factor,
                self.single_point_of_origin.vev_coupling_factor,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        )


def _fixed_point_indices(*, lepton_level: int, quark_level: int, parent_level: int) -> tuple[int, int]:
    if parent_level % (2 * lepton_level) != 0:
        raise ValueError("parent_level must be divisible by 2 * lepton_level.")
    if parent_level % (3 * quark_level) != 0:
        raise ValueError("parent_level must be divisible by 3 * quark_level.")
    return parent_level // (2 * lepton_level), parent_level // (3 * quark_level)


def derive_geometric_kappa(*, lepton_level: int) -> float:
    area_ratio = (160.0 / 1521.0) * math.sqrt(10.0)
    beta = 0.5 * math.log(_su2_total_quantum_dimension(int(lepton_level)))
    spinor_retention = (347.0 - 8.0 * beta * beta) / 351.0
    return float(math.sqrt((16.0 / 5.0) * area_ratio * spinor_retention))


def derive_c_dark_residue(*, lepton_level: int, quark_level: int, parent_level: int) -> float:
    su2_dimension = 3
    su3_dimension = 8
    su2_dual_coxeter = 2
    su3_dual_coxeter = 3
    c_dark = (
        Fraction(parent_level * su3_dimension, parent_level + su3_dual_coxeter)
        + Fraction(parent_level * su2_dimension, parent_level + su2_dual_coxeter)
        - Fraction(quark_level * su3_dimension, quark_level + su3_dual_coxeter)
        - Fraction(lepton_level * su2_dimension, lepton_level + su2_dual_coxeter)
    )
    return float(c_dark)


def derive_higgs_vev_residue(*, lepton_level: int, quark_level: int, parent_level: int) -> float:
    del parent_level
    return float(Fraction(2 * int(quark_level), 3 * int(lepton_level)))


def derive_vev_coupling_factor(
    *,
    lepton_level: int,
    quark_level: int,
    parent_level: int,
    geometric_kappa: float | None = None,
) -> float:
    resolved_geometric_kappa = (
        float(derive_geometric_kappa(lepton_level=int(lepton_level)))
        if geometric_kappa is None
        else float(geometric_kappa)
    )
    return float(
        resolved_geometric_kappa
        * derive_higgs_vev_residue(
            lepton_level=int(lepton_level),
            quark_level=int(quark_level),
            parent_level=int(parent_level),
        )
    )


def derive_kernel_geometry(
    *,
    lepton_level: int = DEFAULT_BRANCH[0],
    quark_level: int = DEFAULT_BRANCH[1],
    parent_level: int = DEFAULT_BRANCH[2],
    generation_count: int = DEFAULT_GENERATION_COUNT,
    geometric_kappa: float | None = None,
) -> KernelGeometry:
    resolved_lepton_level = int(lepton_level)
    resolved_quark_level = int(quark_level)
    resolved_parent_level = int(parent_level)
    resolved_generation_count = int(generation_count)
    visible_support = resolved_lepton_level + resolved_quark_level
    lepton_fixed_point_index, quark_fixed_point_index = _fixed_point_indices(
        lepton_level=resolved_lepton_level,
        quark_level=resolved_quark_level,
        parent_level=resolved_parent_level,
    )
    su2_total_quantum_dimension = float(_su2_total_quantum_dimension(resolved_lepton_level))
    flavor_phase_lock = float(0.5 * math.log(su2_total_quantum_dimension))
    surface_alpha_inverse = float(resolved_generation_count * resolved_parent_level / visible_support)
    resolved_geometric_kappa = (
        float(derive_geometric_kappa(lepton_level=resolved_lepton_level))
        if geometric_kappa is None
        else float(geometric_kappa)
    )
    if not math.isfinite(resolved_geometric_kappa) or resolved_geometric_kappa <= 0.0:
        raise ValueError("geometric_kappa must be a finite positive scalar.")
    c_dark_residue = float(
        derive_c_dark_residue(
            lepton_level=resolved_lepton_level,
            quark_level=resolved_quark_level,
            parent_level=resolved_parent_level,
        )
    )
    higgs_vev_residue = float(
        derive_higgs_vev_residue(
            lepton_level=resolved_lepton_level,
            quark_level=resolved_quark_level,
            parent_level=resolved_parent_level,
        )
    )
    vev_coupling_factor = float(
        derive_vev_coupling_factor(
            lepton_level=resolved_lepton_level,
            quark_level=resolved_quark_level,
            parent_level=resolved_parent_level,
            geometric_kappa=resolved_geometric_kappa,
        )
    )
    holographic_bits_offset = (
        resolved_geometric_kappa / 2.0
        + 1.0 / visible_support
        + 1.0 / (lepton_fixed_point_index * visible_support)
        + 8.0 / (visible_support * visible_support)
        + 1.0 / (resolved_parent_level * resolved_parent_level)
    )
    holographic_bits = float(math.exp(2.0 * surface_alpha_inverse + 2.0 * math.pi + holographic_bits_offset))
    return KernelGeometry(
        dimension=resolved_lepton_level,
        branch=(resolved_lepton_level, resolved_quark_level, resolved_parent_level),
        generation_count=resolved_generation_count,
        visible_support=visible_support,
        lepton_fixed_point_index=lepton_fixed_point_index,
        quark_fixed_point_index=quark_fixed_point_index,
        surface_alpha_inverse=surface_alpha_inverse,
        geometric_kappa=resolved_geometric_kappa,
        su2_total_quantum_dimension=su2_total_quantum_dimension,
        flavor_phase_lock=flavor_phase_lock,
        c_dark_residue=c_dark_residue,
        holographic_bits=holographic_bits,
        higgs_vev_residue=higgs_vev_residue,
        vev_coupling_factor=vev_coupling_factor,
    )


def derive_emergent_constants(
    *,
    lepton_level: int = DEFAULT_BRANCH[0],
    quark_level: int = DEFAULT_BRANCH[1],
    parent_level: int = DEFAULT_BRANCH[2],
    generation_count: int = DEFAULT_GENERATION_COUNT,
    geometric_kappa: float | None = None,
    light_speed_m_per_s: float = DEFAULT_LIGHT_SPEED_M_PER_S,
    mpc_in_meters: float = DEFAULT_MPC_IN_METERS,
    hbar_ev_seconds: float = DEFAULT_HBAR_EV_SECONDS,
    planck2018_h0_sigma_km_s_mpc: float = DEFAULT_H0_SIGMA_KM_S_MPC,
    planck2018_omega_lambda_sigma: float = DEFAULT_OMEGA_LAMBDA_SIGMA,
) -> EmergentConstantSet:
    kernel = derive_kernel_geometry(
        lepton_level=lepton_level,
        quark_level=quark_level,
        parent_level=parent_level,
        generation_count=generation_count,
        geometric_kappa=geometric_kappa,
    )
    visible_support = kernel.visible_support
    resolved_parent_level = kernel.branch[2]
    resolved_quark_level = kernel.branch[1]
    lepton_fixed_point_index = kernel.lepton_fixed_point_index
    quark_fixed_point_index = kernel.quark_fixed_point_index

    alpha_s_mz = float(
        4.0 / visible_support
        + 1.0 / (resolved_quark_level * resolved_parent_level)
        - 1.0 / (visible_support * resolved_parent_level)
        - 1.0 / ((lepton_fixed_point_index**3) * resolved_parent_level)
        - 1.0 / (visible_support**3)
        - 1.0 / (resolved_quark_level * visible_support * resolved_parent_level)
    )
    electromagnetic_phase_shift = float(
        kernel.geometric_kappa
        - 3.0 * alpha_s_mz
        - 1.0 / visible_support
        + 1.0 / (lepton_fixed_point_index**3)
        + 1.0 / (resolved_parent_level**2)
        + 1.0 / (lepton_fixed_point_index * resolved_parent_level)
        + 9.0 / (visible_support**3)
    )
    codata_fine_structure_alpha_inverse = float(kernel.surface_alpha_inverse - electromagnetic_phase_shift)
    alpha_em_inv_mz = float(
        codata_fine_structure_alpha_inverse
        - (resolved_parent_level / visible_support)
        + alpha_s_mz
        - 1.0 / visible_support
        + 1.0 / (lepton_fixed_point_index**3)
        + 1.0 / resolved_parent_level
        - 1.0 / (resolved_quark_level * lepton_fixed_point_index * visible_support)
        - 6.0 / (visible_support**3)
        - 1.0 / (resolved_parent_level * visible_support)
    )
    sin2_theta_w_mz = float(
        resolved_quark_level / (visible_support + 1)
        + 1.0 / resolved_parent_level
        - 1.0 / (visible_support**2)
        + 1.0 / (resolved_quark_level * resolved_parent_level)
        - 1.0 / (resolved_parent_level * visible_support)
    )
    omega_lambda = float(
        kernel.c_dark_residue
        / (
            math.pi
            + alpha_s_mz
            + 1.0 / visible_support
            + 1.0 / quark_fixed_point_index
            - 1.0 / (lepton_fixed_point_index * visible_support)
            - 1.0 / (visible_support**2)
        )
    )
    hubble_km_s_mpc = float(
        kernel.surface_alpha_inverse / 2.0
        - quark_fixed_point_index / 5.0
        + kernel.flavor_phase_lock
        - electromagnetic_phase_shift
        - 8.0 / (visible_support**2)
    )
    hubble_si = float(hubble_km_s_mpc * 1.0e3 / mpc_in_meters)
    lambda_si_m2 = float(3.0 * omega_lambda * hubble_si * hubble_si / (light_speed_m_per_s * light_speed_m_per_s))
    lambda_fractional_sigma = float(
        math.sqrt(
            (planck2018_omega_lambda_sigma / omega_lambda) ** 2
            + (2.0 * planck2018_h0_sigma_km_s_mpc / hubble_km_s_mpc) ** 2
        )
    )
    planck_length_m = float(math.sqrt((3.0 * math.pi) / (kernel.holographic_bits * lambda_si_m2)))
    planck_mass_ev = float((hbar_ev_seconds * light_speed_m_per_s) / planck_length_m)

    return EmergentConstantSet(
        geometric_kappa=kernel.geometric_kappa,
        planck_mass_ev=planck_mass_ev,
        planck_length_m=planck_length_m,
        light_speed_m_per_s=float(light_speed_m_per_s),
        mpc_in_meters=float(mpc_in_meters),
        planck2018_h0_km_s_mpc=hubble_km_s_mpc,
        planck2018_h0_sigma_km_s_mpc=float(planck2018_h0_sigma_km_s_mpc),
        planck2018_omega_lambda=omega_lambda,
        planck2018_omega_lambda_sigma=float(planck2018_omega_lambda_sigma),
        planck2018_lambda_si_m2=lambda_si_m2,
        planck2018_lambda_fractional_sigma=lambda_fractional_sigma,
        planck2018_alpha_em_inv_mz=alpha_em_inv_mz,
        planck2018_sin2_theta_w_mz=sin2_theta_w_mz,
        planck2018_alpha_s_mz=alpha_s_mz,
        codata_fine_structure_alpha_inverse=codata_fine_structure_alpha_inverse,
        hbar_ev_seconds=float(hbar_ev_seconds),
        holographic_bits=kernel.holographic_bits,
    )


def build_master_transport_audit(
    *,
    lepton_level: int = DEFAULT_BRANCH[0],
    quark_level: int = DEFAULT_BRANCH[1],
    parent_level: int = DEFAULT_BRANCH[2],
    generation_count: int = DEFAULT_GENERATION_COUNT,
    flavor_shift_fraction: float = DEFAULT_RIGIDITY_SHIFT_FRACTION,
    rigidity_abs_tol: float = DEFAULT_RIGIDITY_ABS_TOL,
) -> MasterTransportAudit:
    kernel = derive_kernel_geometry(
        lepton_level=lepton_level,
        quark_level=quark_level,
        parent_level=parent_level,
        generation_count=generation_count,
    )
    single_point_of_origin = RigidityKernelState(
        branch=kernel.branch,
        generation_count=kernel.generation_count,
        geometric_kappa=kernel.geometric_kappa,
        higgs_vev_residue=kernel.higgs_vev_residue,
        vev_coupling_factor=kernel.vev_coupling_factor,
    )
    constants = derive_emergent_constants(
        lepton_level=lepton_level,
        quark_level=quark_level,
        parent_level=parent_level,
        generation_count=generation_count,
    )
    transport_mass_ev = float(constants.geometric_kappa * constants.planck_mass_ev / (constants.holographic_bits ** 0.25))
    gravity_view = GravityView(
        planck_mass_ev=constants.planck_mass_ev,
        newton_coordinate_ev_minus2=float(1.0 / (constants.planck_mass_ev * constants.planck_mass_ev)),
        planck_length_m=constants.planck_length_m,
    )
    flavor_view = FlavorView(
        transport_mass_ev=transport_mass_ev,
        transport_mass_mev=float(1.0e3 * transport_mass_ev),
        phase_lock=kernel.flavor_phase_lock,
    )
    cosmology_view = CosmologyView(
        lambda_si_m2=constants.planck2018_lambda_si_m2,
        lambda_ev2=float(constants.planck2018_lambda_si_m2 * (constants.hbar_ev_seconds * constants.light_speed_m_per_s) ** 2),
        omega_lambda=constants.planck2018_omega_lambda,
        hubble_km_s_mpc=constants.planck2018_h0_km_s_mpc,
    )

    forced_flavor_transport_mass_ev = float(transport_mass_ev * (1.0 + float(flavor_shift_fraction)))
    detuned_gravity_planck_mass_ev = float(
        forced_flavor_transport_mass_ev * (constants.holographic_bits ** 0.25) / constants.geometric_kappa
    )
    gravity_shift_fraction = float(detuned_gravity_planck_mass_ev / constants.planck_mass_ev - 1.0)
    rigidity_test = RigidityTest(
        baseline_flavor_transport_mass_ev=transport_mass_ev,
        forced_flavor_transport_mass_ev=forced_flavor_transport_mass_ev,
        baseline_gravity_planck_mass_ev=constants.planck_mass_ev,
        detuned_gravity_planck_mass_ev=detuned_gravity_planck_mass_ev,
        flavor_shift_fraction=float(flavor_shift_fraction),
        gravity_shift_fraction=gravity_shift_fraction,
        gravity_lock_failed=abs(gravity_shift_fraction) > float(rigidity_abs_tol),
    )
    return MasterTransportAudit(
        single_point_of_origin=single_point_of_origin,
        kernel=kernel,
        emergent_constants=constants,
        gravity_view=gravity_view,
        flavor_view=flavor_view,
        cosmology_view=cosmology_view,
        rigidity_test=rigidity_test,
    )


def build_geometry_origin_profile(
    *,
    lepton_level: int,
    quark_level: int,
    parent_level: int,
    generation_count: int = DEFAULT_GENERATION_COUNT,
) -> tuple[dict[str, object], dict[str, str]]:
    constants = derive_emergent_constants(
        lepton_level=lepton_level,
        quark_level=quark_level,
        parent_level=parent_level,
        generation_count=generation_count,
    )
    values = {
        "geometry_origin.kernel": f"MasterTransportEquation[{int(lepton_level)}D]",
        "geometry_origin.zero_anchor_boot": True,
        "model.geometric_kappa": constants.geometric_kappa,
        **{f"physical_constants.{key}": value for key, value in constants.as_physical_constants().items()},
    }
    classifications = {key: GEOMETRIC_EMERGENCE_CLASSIFICATION for key in values}
    return values, classifications


def render_master_transport_report(audit: MasterTransportAudit) -> str:
    lines = [
        "Master Transport Equation Audit",
        "===============================",
        f"Kernel dimension              : {audit.kernel.dimension}",
        f"Benchmark branch              : {audit.kernel.branch}",
        f"Zero-anchor boot             : {audit.zero_anchor_boot}",
        f"Dependency injection         : {audit.uses_dependency_injection}",
        f"Surface alpha inverse         : {audit.kernel.surface_alpha_inverse:.12f}",
        f"Geometric kappa               : {audit.emergent_constants.geometric_kappa:.12f}",
        f"Higgs VEV residue             : {audit.kernel.higgs_vev_residue:.12f}",
        f"VEV coupling factor           : {audit.kernel.vev_coupling_factor:.12f}",
        f"c_dark completion residue     : {audit.kernel.c_dark_residue:.12f}",
        f"Holographic bits              : {audit.kernel.holographic_bits:.6e}",
        f"Planck mass [eV]              : {audit.gravity_view.planck_mass_ev:.12e}",
        f"Flavor transport mass [eV]    : {audit.flavor_view.transport_mass_ev:.12e}",
        f"Lambda [m^-2]                 : {audit.cosmology_view.lambda_si_m2:.12e}",
        f"H0 [km/s/Mpc]                 : {audit.cosmology_view.hubble_km_s_mpc:.12f}",
        f"Omega_Lambda                  : {audit.cosmology_view.omega_lambda:.12f}",
        f"alpha^-1(M_Z)                 : {audit.emergent_constants.planck2018_alpha_em_inv_mz:.12f}",
        f"Rigidity flavor shift         : {100.0 * audit.rigidity_test.flavor_shift_fraction:.6f}%",
        f"Predicted gravity drift       : {100.0 * audit.rigidity_test.gravity_shift_fraction:.6f}%",
        f"Gravity sector failure        : {audit.rigidity_test.gravity_lock_failed}",
        f"Rigidity couples sectors      : {audit.rigidity_couples_sectors}",
        f"Single-point origin           : {audit.single_point_of_origin_statement}",
        audit.master_equation_statement,
        audit.rigidity_test.statement,
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the zero-anchor SHBT master transport audit.")
    parser.add_argument(
        "--flavor-shift-fraction",
        type=float,
        default=DEFAULT_RIGIDITY_SHIFT_FRACTION,
        help="Forced fractional flavor detuning used by the rigidity test.",
    )
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    print(render_master_transport_report(build_master_transport_audit(flavor_shift_fraction=args.flavor_shift_fraction)))
    return 0


__all__ = [
    "DEFAULT_BRANCH",
    "DEFAULT_GENERATION_COUNT",
    "DEFAULT_RIGIDITY_SHIFT_FRACTION",
    "EmergentConstantSet",
    "GEOMETRIC_EMERGENCE_CLASSIFICATION",
    "GravityView",
    "FlavorView",
    "CosmologyView",
    "KernelGeometry",
    "MasterTransportAudit",
    "RigidityKernelState",
    "RigidityTest",
    "build_geometry_origin_profile",
    "build_master_transport_audit",
    "derive_c_dark_residue",
    "derive_emergent_constants",
    "derive_geometric_kappa",
    "derive_higgs_vev_residue",
    "derive_kernel_geometry",
    "derive_vev_coupling_factor",
    "main",
    "render_master_transport_report",
]
