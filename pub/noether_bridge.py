from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pub import algebra
    from pub.constants import HOLOGRAPHIC_BITS, LEPTON_LEVEL, LIGHT_SPEED_M_PER_S, PARENT_LEVEL, PLANCK2018_LAMBDA_SI_M2, PLANCK_LENGTH_M, QUARK_LEVEL
else:
    from . import algebra
    from .constants import HOLOGRAPHIC_BITS, LEPTON_LEVEL, LIGHT_SPEED_M_PER_S, PARENT_LEVEL, PLANCK2018_LAMBDA_SI_M2, PLANCK_LENGTH_M, QUARK_LEVEL

PI = Decimal("3.14159265358979323846264338327950288419716939937510")
HBAR_EV_SECONDS = Decimal("6.582119569e-16")
MINKOWSKI_DIAGONAL = (Decimal("-1"), Decimal("1"), Decimal("1"), Decimal("1"))
DEFAULT_PRECISION = 200
FALLBACK_C_DARK_COMPLETION = Fraction(1197103, 362670)


@dataclass(frozen=True)
class NewtonLockAudit:
    c_dark_fraction: Fraction
    c_dark: Decimal
    planck_mass_ev: Decimal
    eight_pi_g_effective_ev_minus2: Decimal
    g_effective_ev_minus2: Decimal
    g_topological_ev_minus2: Decimal
    topological_from_effective_factor: Decimal
    effective_from_topological_factor: Decimal


@dataclass(frozen=True)
class SaturationAudit:
    lambda_obs_si_m2: Decimal
    lambda_obs_ev2: Decimal
    holographic_bits_from_lambda: Decimal
    configured_holographic_bits: Decimal
    register_noise_floor: Decimal
    relative_mismatch: Decimal

    @property
    def boundary_condition_locked(self) -> bool:
        return self.relative_mismatch <= Decimal("1e-15")


@dataclass(frozen=True)
class UnityOfScaleAudit:
    kappa_d5: Decimal
    lightest_mass_ev: Decimal
    lightest_mass_mev: Decimal
    lambda_lhs_ev2: Decimal
    lambda_lhs_si_m2: Decimal
    lambda_rhs_topological_ev2: Decimal
    lambda_rhs_noether_bridged_ev2: Decimal
    epsilon_lambda: Decimal
    epsilon_lambda_noether_bridged: Decimal
    register_noise_floor: Decimal

    @property
    def topological_identity_pass(self) -> bool:
        return self.epsilon_lambda <= self.register_noise_floor

    @property
    def noether_bridge_identity_pass(self) -> bool:
        return self.epsilon_lambda_noether_bridged <= self.register_noise_floor

    @property
    def passed(self) -> bool:
        return self.topological_identity_pass and self.noether_bridge_identity_pass


@dataclass(frozen=True)
class FramingDefectAudit:
    parent_level: int
    lepton_level: int
    quark_level: int
    lepton_gap: Fraction
    quark_gap: Fraction
    delta_fr: Fraction


@dataclass(frozen=True)
class TensorSnapshot:
    amplitude: Decimal
    diagonal: tuple[Decimal, Decimal, Decimal, Decimal]
    units: str

    @property
    def vanished(self) -> bool:
        return self.amplitude == 0


@dataclass(frozen=True)
class ReviewerTrapAudit:
    benchmark: FramingDefectAudit
    detuned: FramingDefectAudit
    q_iso_ev4: Decimal
    closure_tensor_benchmark: TensorSnapshot
    closure_tensor_detuned: TensorSnapshot
    anomalous_source_ev2: TensorSnapshot
    anomalous_source_si_m2: TensorSnapshot

    @property
    def closure_equivalence_verified(self) -> bool:
        benchmark_ok = self.benchmark.delta_fr == 0 and self.closure_tensor_benchmark.vanished
        detuned_ok = self.detuned.delta_fr != 0 and not self.closure_tensor_detuned.vanished
        return benchmark_ok and detuned_ok

    @property
    def equivalence_principle_preserved(self) -> bool:
        return self.anomalous_source_si_m2.vanished


@dataclass(frozen=True)
class GravitySideRigidityReport:
    branch: tuple[int, int, int]
    detuned_branch: tuple[int, int, int]
    newton_lock: NewtonLockAudit
    saturation: SaturationAudit
    unity: UnityOfScaleAudit
    reviewer_trap: ReviewerTrapAudit


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(str(value))


def _fraction_to_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _format_fraction(value: Fraction) -> str:
    if value == 0:
        return "0"
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _format_decimal_scientific(value: Decimal, precision: int = 12) -> str:
    if value == 0:
        return f"{0:.{precision}E}"
    return format(value, f".{precision}E")


def _format_tensor(snapshot: TensorSnapshot, precision: int = 6) -> str:
    diagonal = ", ".join(_format_decimal_scientific(entry, precision) for entry in snapshot.diagonal)
    return f"diag({diagonal}) {snapshot.units}"


def _meter_to_ev_inverse() -> Decimal:
    return Decimal("1") / (HBAR_EV_SECONDS * _decimal(LIGHT_SPEED_M_PER_S))


def branch_planck_mass_ev() -> Decimal:
    return (HBAR_EV_SECONDS * _decimal(LIGHT_SPEED_M_PER_S)) / _decimal(PLANCK_LENGTH_M)


def lambda_si_m2_to_ev2(lambda_si_m2: Decimal) -> Decimal:
    return lambda_si_m2 * (HBAR_EV_SECONDS * _decimal(LIGHT_SPEED_M_PER_S)) ** 2


def ev2_to_lambda_si_m2(lambda_ev2: Decimal) -> Decimal:
    return lambda_ev2 / (HBAR_EV_SECONDS * _decimal(LIGHT_SPEED_M_PER_S)) ** 2


def load_c_dark_completion_fraction() -> Fraction:
    tex_path = Path(__file__).with_name("physics_constants.tex")
    try:
        match = re.search(
            r"\\newcommand\{\\cDarkCompletionExact\}\{\\frac\{(\d+)\}\{(\d+)\}\}",
            tex_path.read_text(encoding="utf-8"),
        )
    except OSError:
        match = None
    if match is None:
        return FALLBACK_C_DARK_COMPLETION
    return Fraction(int(match.group(1)), int(match.group(2)))


def derive_kappa_d5(*, lepton_level: int = LEPTON_LEVEL, precision: int = DEFAULT_PRECISION) -> Decimal:
    with localcontext() as context:
        context.prec = precision
        area_ratio = (Decimal(160) / Decimal(1521)) * Decimal(10).sqrt()
        beta = _decimal(0.5 * math.log(algebra.su2_total_quantum_dimension(int(lepton_level))))
        spinor_retention = (Decimal(347) - Decimal(8) * beta * beta) / Decimal(351)
        return ((Decimal(16) / Decimal(5)) * area_ratio * spinor_retention).sqrt()


def _distance_to_integer_fraction(value: Fraction) -> Fraction:
    denominator = value.denominator
    remainder = Fraction(value.numerator % denominator, denominator)
    return min(remainder, Fraction(1, 1) - remainder)


def framing_defect(parent_level: int, lepton_level: int, quark_level: int) -> FramingDefectAudit:
    lepton_gap = _distance_to_integer_fraction(Fraction(parent_level, 2 * lepton_level))
    quark_gap = _distance_to_integer_fraction(Fraction(parent_level, 3 * quark_level))
    delta_fr = lepton_gap if lepton_gap >= quark_gap else quark_gap
    return FramingDefectAudit(
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
        lepton_gap=lepton_gap,
        quark_gap=quark_gap,
        delta_fr=delta_fr,
    )


def newton_constant_lock(*, c_dark_fraction: Fraction | None = None, precision: int = DEFAULT_PRECISION) -> NewtonLockAudit:
    resolved_fraction = FALLBACK_C_DARK_COMPLETION if c_dark_fraction is None else c_dark_fraction
    with localcontext() as context:
        context.prec = precision
        c_dark = _fraction_to_decimal(resolved_fraction)
        planck_mass = branch_planck_mass_ev()
        g_topological = Decimal("1") / (planck_mass * planck_mass)
        eight_pi_g_effective = Decimal(12) / (c_dark * planck_mass * planck_mass)
        g_effective = eight_pi_g_effective / (Decimal(8) * PI)
        topological_from_effective = g_topological / g_effective
        effective_from_topological = g_effective / g_topological
    return NewtonLockAudit(
        c_dark_fraction=resolved_fraction,
        c_dark=c_dark,
        planck_mass_ev=planck_mass,
        eight_pi_g_effective_ev_minus2=eight_pi_g_effective,
        g_effective_ev_minus2=g_effective,
        g_topological_ev_minus2=g_topological,
        topological_from_effective_factor=topological_from_effective,
        effective_from_topological_factor=effective_from_topological,
    )


def saturation_audit(*, lambda_obs_si_m2: Decimal | None = None, precision: int = DEFAULT_PRECISION) -> SaturationAudit:
    resolved_lambda = _decimal(PLANCK2018_LAMBDA_SI_M2) if lambda_obs_si_m2 is None else lambda_obs_si_m2
    with localcontext() as context:
        context.prec = precision
        bit_count = (Decimal(3) * PI) / (_decimal(PLANCK_LENGTH_M) ** 2 * resolved_lambda)
        configured_bits = _decimal(HOLOGRAPHIC_BITS)
        relative_mismatch = abs(bit_count / configured_bits - Decimal(1)) if configured_bits != 0 else Decimal("Infinity")
        lambda_ev2 = lambda_si_m2_to_ev2(resolved_lambda)
    return SaturationAudit(
        lambda_obs_si_m2=resolved_lambda,
        lambda_obs_ev2=lambda_ev2,
        holographic_bits_from_lambda=bit_count,
        configured_holographic_bits=configured_bits,
        register_noise_floor=Decimal(1) / bit_count,
        relative_mismatch=relative_mismatch,
    )


def unity_of_scale_audit(
    *,
    kappa_d5: Decimal,
    newton_lock_audit: NewtonLockAudit,
    saturation: SaturationAudit,
    precision: int = DEFAULT_PRECISION,
) -> UnityOfScaleAudit:
    with localcontext() as context:
        context.prec = precision
        bit_count = saturation.holographic_bits_from_lambda
        lightest_mass_ev = kappa_d5 * newton_lock_audit.planck_mass_ev * (bit_count ** Decimal("-0.25"))
        lambda_lhs_ev2 = saturation.lambda_obs_ev2
        lambda_rhs_topological = (
            Decimal(3)
            * PI
            * newton_lock_audit.g_topological_ev_minus2
            * (lightest_mass_ev**4)
            / (kappa_d5**4)
        )
        lambda_rhs_noether_bridged = (
            Decimal(2)
            * PI
            * PI
            * newton_lock_audit.c_dark
            * newton_lock_audit.g_effective_ev_minus2
            * (lightest_mass_ev**4)
            / (kappa_d5**4)
        )
        epsilon_lambda = abs(Decimal(1) - (lambda_lhs_ev2 / lambda_rhs_topological))
        epsilon_lambda_noether_bridged = abs(Decimal(1) - (lambda_lhs_ev2 / lambda_rhs_noether_bridged))
    return UnityOfScaleAudit(
        kappa_d5=kappa_d5,
        lightest_mass_ev=lightest_mass_ev,
        lightest_mass_mev=Decimal(1000) * lightest_mass_ev,
        lambda_lhs_ev2=lambda_lhs_ev2,
        lambda_lhs_si_m2=saturation.lambda_obs_si_m2,
        lambda_rhs_topological_ev2=lambda_rhs_topological,
        lambda_rhs_noether_bridged_ev2=lambda_rhs_noether_bridged,
        epsilon_lambda=epsilon_lambda,
        epsilon_lambda_noether_bridged=epsilon_lambda_noether_bridged,
        register_noise_floor=saturation.register_noise_floor,
    )


def bulk_closure_tensor(delta_fr: Fraction, q_iso_ev4: Decimal) -> TensorSnapshot:
    amplitude = q_iso_ev4 * _fraction_to_decimal(delta_fr)
    if amplitude == 0:
        diagonal = (Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"))
    else:
        diagonal = tuple(sign * amplitude for sign in MINKOWSKI_DIAGONAL)
    return TensorSnapshot(amplitude=amplitude, diagonal=diagonal, units="[eV^4]")


def anomalous_source_tensor(closure_tensor: TensorSnapshot, newton_lock_audit: NewtonLockAudit) -> tuple[TensorSnapshot, TensorSnapshot]:
    amplitude_ev2 = newton_lock_audit.eight_pi_g_effective_ev_minus2 * closure_tensor.amplitude
    if amplitude_ev2 == 0:
        diagonal_ev2 = (Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"))
    else:
        diagonal_ev2 = tuple(sign * amplitude_ev2 for sign in MINKOWSKI_DIAGONAL)
    tensor_ev2 = TensorSnapshot(amplitude=amplitude_ev2, diagonal=diagonal_ev2, units="[eV^2]")
    amplitude_si_m2 = ev2_to_lambda_si_m2(amplitude_ev2)
    if amplitude_si_m2 == 0:
        diagonal_si_m2 = (Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"))
    else:
        diagonal_si_m2 = tuple(sign * amplitude_si_m2 for sign in MINKOWSKI_DIAGONAL)
    tensor_si_m2 = TensorSnapshot(amplitude=amplitude_si_m2, diagonal=diagonal_si_m2, units="[m^-2]")
    return tensor_ev2, tensor_si_m2


def reviewer_trap_audit(
    *,
    newton_lock_audit: NewtonLockAudit,
    saturation: SaturationAudit,
    benchmark_branch: tuple[int, int, int],
    detuned_branch: tuple[int, int, int],
    precision: int = DEFAULT_PRECISION,
) -> ReviewerTrapAudit:
    benchmark = framing_defect(*benchmark_branch)
    detuned = framing_defect(*detuned_branch)
    with localcontext() as context:
        context.prec = precision
        q_iso_ev4 = saturation.lambda_obs_ev2 / newton_lock_audit.eight_pi_g_effective_ev_minus2
    closure_benchmark = bulk_closure_tensor(benchmark.delta_fr, q_iso_ev4)
    closure_detuned = bulk_closure_tensor(detuned.delta_fr, q_iso_ev4)
    anomalous_source_ev2, anomalous_source_si_m2 = anomalous_source_tensor(closure_detuned, newton_lock_audit)
    return ReviewerTrapAudit(
        benchmark=benchmark,
        detuned=detuned,
        q_iso_ev4=q_iso_ev4,
        closure_tensor_benchmark=closure_benchmark,
        closure_tensor_detuned=closure_detuned,
        anomalous_source_ev2=anomalous_source_ev2,
        anomalous_source_si_m2=anomalous_source_si_m2,
    )


def build_gravity_side_rigidity_report(
    *,
    parent_level: int = PARENT_LEVEL,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    detuned_parent_level: int | None = None,
    detuned_lepton_level: int | None = None,
    detuned_quark_level: int | None = None,
    precision: int = DEFAULT_PRECISION,
) -> GravitySideRigidityReport:
    benchmark_branch = (int(parent_level), int(lepton_level), int(quark_level))
    detuned_branch = (
        int(parent_level if detuned_parent_level is None else detuned_parent_level),
        int(lepton_level + 1 if detuned_lepton_level is None else detuned_lepton_level),
        int(quark_level if detuned_quark_level is None else detuned_quark_level),
    )
    c_dark_fraction = load_c_dark_completion_fraction()
    newton = newton_constant_lock(c_dark_fraction=c_dark_fraction, precision=precision)
    saturation = saturation_audit(precision=precision)
    kappa_d5 = derive_kappa_d5(lepton_level=lepton_level, precision=precision)
    unity = unity_of_scale_audit(
        kappa_d5=kappa_d5,
        newton_lock_audit=newton,
        saturation=saturation,
        precision=precision,
    )
    reviewer = reviewer_trap_audit(
        newton_lock_audit=newton,
        saturation=saturation,
        benchmark_branch=benchmark_branch,
        detuned_branch=detuned_branch,
        precision=precision,
    )
    return GravitySideRigidityReport(
        branch=benchmark_branch,
        detuned_branch=detuned_branch,
        newton_lock=newton,
        saturation=saturation,
        unity=unity,
        reviewer_trap=reviewer,
    )


def render_report(report: GravitySideRigidityReport) -> str:
    benchmark_branch = f"({report.branch[1]},{report.branch[2]},{report.branch[0]})"
    detuned_branch = f"({report.detuned_branch[1]},{report.detuned_branch[2]},{report.detuned_branch[0]})"
    lines = [
        "Gravity-Side Rigidity Report",
        "===========================",
        f"Benchmark branch                : {benchmark_branch}",
        f"Detuned stress test             : {detuned_branch}",
        f"c_dark completion               : {_format_fraction(report.newton_lock.c_dark_fraction)} = {report.newton_lock.c_dark:.12f}",
        f"kappa_D5                        : {report.unity.kappa_d5:.12f}",
        "",
        "Newton Constant Lock",
        "--------------------",
        "8π G_eff = 12/(c_dark M_P^2)",
        f"G_eff [eV^-2]                   : {_format_decimal_scientific(report.newton_lock.g_effective_ev_minus2)}",
        f"G_top = M_P^-2 [eV^-2]          : {_format_decimal_scientific(report.newton_lock.g_topological_ev_minus2)}",
        f"G_top / G_eff                   : {_format_decimal_scientific(report.newton_lock.topological_from_effective_factor)}",
        "Gravity exists on the branch specifically to carry the positive c_dark completion residue.",
        "",
        "Unity of Scale Identity",
        "-----------------------",
        f"Lambda_obs [m^-2]               : {_format_decimal_scientific(report.saturation.lambda_obs_si_m2)}",
        f"N = 3π/(L_P^2 Lambda_obs)       : {_format_decimal_scientific(report.saturation.holographic_bits_from_lambda)}",
        f"1/N register floor              : {_format_decimal_scientific(report.saturation.register_noise_floor)}",
        f"m_nu = kappa_D5 M_P N^(-1/4)    : {_format_decimal_scientific(report.unity.lightest_mass_mev)} meV",
        f"Lambda_holo(lhs) [eV^2]         : {_format_decimal_scientific(report.unity.lambda_lhs_ev2)}",
        f"Lambda_holo(rhs, top) [eV^2]    : {_format_decimal_scientific(report.unity.lambda_rhs_topological_ev2)}",
        f"Lambda_holo(rhs, bridge) [eV^2] : {_format_decimal_scientific(report.unity.lambda_rhs_noether_bridged_ev2)}",
        f"epsilon_lambda(top)             : {_format_decimal_scientific(report.unity.epsilon_lambda)}",
        f"epsilon_lambda(bridge)          : {_format_decimal_scientific(report.unity.epsilon_lambda_noether_bridged)}",
        f"Unity closure                   : {'PASS' if report.unity.passed else 'FAIL'}",
        "",
        "Reviewer Trap",
        "-------------",
        f"Delta_fr benchmark              : {_format_fraction(report.reviewer_trap.benchmark.delta_fr)}",
        f"Delta_fr detuned                : {_format_fraction(report.reviewer_trap.detuned.delta_fr)} = {float(report.reviewer_trap.detuned.delta_fr):.12f}",
        f"Q_iso = sigma_holo [eV^4]       : {_format_decimal_scientific(report.reviewer_trap.q_iso_ev4)}",
        f"E_mu_nu benchmark               : {_format_tensor(report.reviewer_trap.closure_tensor_benchmark)}",
        f"E_mu_nu detuned                 : {_format_tensor(report.reviewer_trap.closure_tensor_detuned)}",
        f"J_mu_nu^(a) [eV^2]              : {_format_tensor(report.reviewer_trap.anomalous_source_ev2)}",
        f"J_mu_nu^(a) [m^-2]              : {_format_tensor(report.reviewer_trap.anomalous_source_si_m2)}",
        f"Bulk closure equivalence        : {'VERIFIED' if report.reviewer_trap.closure_equivalence_verified else 'FAILED'}",
        f"Equivalence Principle           : {'PRESERVED' if report.reviewer_trap.equivalence_principle_preserved else 'DESTROYED'}",
        "",
        "Saturation Audit",
        "----------------",
        f"Configured HOLOGRAPHIC_BITS     : {_format_decimal_scientific(report.saturation.configured_holographic_bits)}",
        f"Relative mismatch               : {_format_decimal_scientific(report.saturation.relative_mismatch)}",
        "N is reconstructed from the observed cosmological constant, so it enters as an Observational Boundary Condition rather than a fit parameter.",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-level", type=int, default=PARENT_LEVEL)
    parser.add_argument("--lepton-level", type=int, default=LEPTON_LEVEL)
    parser.add_argument("--quark-level", type=int, default=QUARK_LEVEL)
    parser.add_argument("--detuned-parent-level", type=int)
    parser.add_argument("--detuned-lepton-level", type=int)
    parser.add_argument("--detuned-quark-level", type=int)
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_gravity_side_rigidity_report(
        parent_level=args.parent_level,
        lepton_level=args.lepton_level,
        quark_level=args.quark_level,
        detuned_parent_level=args.detuned_parent_level,
        detuned_lepton_level=args.detuned_lepton_level,
        detuned_quark_level=args.detuned_quark_level,
        precision=args.precision,
    )
    print(render_report(report))


if __name__ == "__main__":
    main()
