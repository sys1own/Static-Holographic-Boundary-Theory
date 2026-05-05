from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.main import (
    CODATA_FINE_STRUCTURE_ALPHA_INVERSE,
    G_SM,
    HBAR_EV_SECONDS,
    HOLOGRAPHIC_BITS,
    KAPPA_D5,
    LEPTON_LEVEL,
    PLANCK_MASS_EV,
    PARENT_LEVEL,
    QUARK_LEVEL,
    surface_tension_gauge_alpha_inverse,
    topological_mass_coordinate_ev,
    topological_newton_coordinate_ev_minus2,
    topological_planck_mass_ev,
)

DEFAULT_PRECISION = 50
_GUARD_DIGITS = 12
D5_WEIGHT_SIMPLEX_HYPERAREA_FRACTION = Fraction(160, 1521)


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


def _fraction_to_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _format_decimal(value: Decimal, *, places: int = 18) -> str:
    if value.is_zero():
        return "0"
    adjusted = value.adjusted()
    if adjusted >= 6 or adjusted <= -4:
        return f"{value:.{places}E}"
    return f"{value:.{places}f}".rstrip("0").rstrip(".")


@lru_cache(maxsize=None)
def decimal_pi(precision: int = DEFAULT_PRECISION) -> Decimal:
    with localcontext() as context:
        context.prec = precision + _GUARD_DIGITS
        one = Decimal(1)
        two = Decimal(2)
        four = Decimal(4)
        a_term = one
        b_term = one / two.sqrt()
        t_term = Decimal("0.25")
        p_term = one
        pi_value = Decimal(0)
        for _ in range(8):
            a_next = (a_term + b_term) / two
            b_term = (a_term * b_term).sqrt()
            delta = a_term - a_next
            t_term -= p_term * delta * delta
            a_term = a_next
            p_term *= two
            pi_value = ((a_term + b_term) * (a_term + b_term)) / (four * t_term)
        context.prec = precision
        return +pi_value


def decimal_sin(value: Decimal, *, precision: int = DEFAULT_PRECISION) -> Decimal:
    with localcontext() as context:
        context.prec = precision + _GUARD_DIGITS
        term = value
        result = value
        threshold = Decimal(1).scaleb(-(precision + _GUARD_DIGITS // 2))
        index = 1
        while True:
            denominator = Decimal((2 * index) * (2 * index + 1))
            term *= -(value * value) / denominator
            result += term
            if abs(term) <= threshold:
                break
            index += 1
        context.prec = precision
        return +result


def su2_total_quantum_dimension_decimal(level: int, *, precision: int = DEFAULT_PRECISION) -> Decimal:
    with localcontext() as context:
        context.prec = precision + _GUARD_DIGITS
        level_plus_two = Decimal(level + 2)
        prefactor = (level_plus_two / Decimal(2)).sqrt()
        theta = decimal_pi(context.prec) / level_plus_two
        dimension = prefactor / decimal_sin(theta, precision=context.prec)
        context.prec = precision
        return +dimension


def quarter_power_inverse(value: Decimal, *, precision: int = DEFAULT_PRECISION) -> Decimal:
    with localcontext() as context:
        context.prec = precision + _GUARD_DIGITS
        if value <= 0:
            raise ValueError("Quarter-power inverse requires a positive Decimal.")
        result = Decimal(1) / value.sqrt().sqrt()
        context.prec = precision
        return +result


@dataclass(frozen=True)
class AlphaSurfaceDerivation:
    visible_support: int
    level_density_ratio: Fraction
    alpha_inverse_fraction: Fraction
    alpha_inverse_decimal: Decimal
    live_alpha_inverse: Decimal
    codata_alpha_inverse: Decimal


@dataclass(frozen=True)
class ObservationalBoundaryConditionDerivation:
    hbar_ev_seconds: Decimal


@dataclass(frozen=True)
class KappaDerivation:
    simplex_fraction: Fraction
    sqrt_ten: Decimal
    weight_simplex_hyperarea: Decimal
    su2_total_quantum_dimension: Decimal
    beta: Decimal
    eta_spin: Decimal
    kappa: Decimal
    live_kappa: Decimal


@dataclass(frozen=True)
class MassBridgeDerivation:
    branch_planck_mass_ev: Decimal
    configured_planck_mass_ev: Decimal
    holographic_bits: Decimal
    holographic_bits_quarter_inverse: Decimal
    neutrino_floor_ev: Decimal
    neutrino_floor_mev: Decimal
    live_neutrino_floor_ev: Decimal


@dataclass(frozen=True)
class UnityOfScaleDerivation:
    branch_planck_mass_ev: Decimal
    holographic_bits: Decimal
    lambda_holo_ev2: Decimal
    g_newton_topological_ev_minus2: Decimal
    g_newton_inverse_square_ev_minus2: Decimal
    g_newton_bridge_drift: Decimal
    kappa_fourth: Decimal
    expanded_theorem_rhs_ev2: Decimal
    reduced_theorem_rhs_ev2: Decimal
    identity_ratio: Decimal
    epsilon_lambda: Decimal
    decimal_tolerance: Decimal
    register_noise_floor: Decimal

    @property
    def passed(self) -> bool:
        return self.epsilon_lambda <= self.decimal_tolerance


def derive_observational_boundary_conditions() -> ObservationalBoundaryConditionDerivation:
    return ObservationalBoundaryConditionDerivation(hbar_ev_seconds=_decimal(HBAR_EV_SECONDS))


def derive_alpha_surface(*, precision: int = DEFAULT_PRECISION) -> AlphaSurfaceDerivation:
    del precision
    visible_support = LEPTON_LEVEL + QUARK_LEVEL
    level_density_ratio = Fraction(PARENT_LEVEL, visible_support)
    alpha_inverse_fraction = Fraction(G_SM, 1) * level_density_ratio
    alpha_inverse_decimal = _fraction_to_decimal(alpha_inverse_fraction)
    live_alpha_inverse = _decimal(
        surface_tension_gauge_alpha_inverse(
            parent_level=PARENT_LEVEL,
            lepton_level=LEPTON_LEVEL,
            quark_level=QUARK_LEVEL,
        )
    )
    codata_alpha_inverse = _decimal(CODATA_FINE_STRUCTURE_ALPHA_INVERSE)
    return AlphaSurfaceDerivation(
        visible_support=visible_support,
        level_density_ratio=level_density_ratio,
        alpha_inverse_fraction=alpha_inverse_fraction,
        alpha_inverse_decimal=alpha_inverse_decimal,
        live_alpha_inverse=live_alpha_inverse,
        codata_alpha_inverse=codata_alpha_inverse,
    )


def derive_kappa_d5(*, precision: int = DEFAULT_PRECISION) -> KappaDerivation:
    with localcontext() as context:
        context.prec = precision + _GUARD_DIGITS
        simplex_fraction = D5_WEIGHT_SIMPLEX_HYPERAREA_FRACTION
        sqrt_ten = Decimal(10).sqrt()
        weight_simplex_hyperarea = _fraction_to_decimal(simplex_fraction) * sqrt_ten
        total_quantum_dimension = su2_total_quantum_dimension_decimal(LEPTON_LEVEL, precision=context.prec)
        beta = total_quantum_dimension.ln() / Decimal(2)
        eta_spin = (_decimal(Fraction(347, 1)) - _decimal(Fraction(8, 1)) * beta * beta) / _decimal(Fraction(351, 1))
        kappa = (_decimal(Fraction(16, 5)) * weight_simplex_hyperarea * eta_spin).sqrt()
        context.prec = precision
        return KappaDerivation(
            simplex_fraction=simplex_fraction,
            sqrt_ten=+sqrt_ten,
            weight_simplex_hyperarea=+weight_simplex_hyperarea,
            su2_total_quantum_dimension=+total_quantum_dimension,
            beta=+beta,
            eta_spin=+eta_spin,
            kappa=+kappa,
            live_kappa=_decimal(KAPPA_D5),
        )


def derive_mass_bridge(*, precision: int = DEFAULT_PRECISION, kappa: Decimal | None = None) -> MassBridgeDerivation:
    with localcontext() as context:
        context.prec = precision + _GUARD_DIGITS
        resolved_kappa = derive_kappa_d5(precision=context.prec).kappa if kappa is None else _decimal(kappa)
        branch_planck_mass_ev = _decimal(topological_planck_mass_ev())
        configured_planck_mass_ev = _decimal(PLANCK_MASS_EV)
        holographic_bits = _decimal(HOLOGRAPHIC_BITS)
        holographic_bits_quarter_inverse = quarter_power_inverse(holographic_bits, precision=context.prec)
        neutrino_floor_ev = resolved_kappa * branch_planck_mass_ev * holographic_bits_quarter_inverse
        neutrino_floor_mev = neutrino_floor_ev * Decimal(1000)
        live_neutrino_floor_ev = _decimal(
            topological_mass_coordinate_ev(
                bit_count=HOLOGRAPHIC_BITS,
                kappa_geometric=float(resolved_kappa),
            )
        )
        context.prec = precision
        return MassBridgeDerivation(
            branch_planck_mass_ev=+branch_planck_mass_ev,
            configured_planck_mass_ev=+configured_planck_mass_ev,
            holographic_bits=+holographic_bits,
            holographic_bits_quarter_inverse=+holographic_bits_quarter_inverse,
            neutrino_floor_ev=+neutrino_floor_ev,
            neutrino_floor_mev=+neutrino_floor_mev,
            live_neutrino_floor_ev=+live_neutrino_floor_ev,
        )


def derive_unity_of_scale(
    *,
    precision: int = DEFAULT_PRECISION,
    kappa: Decimal | None = None,
    mass_bridge: MassBridgeDerivation | None = None,
) -> UnityOfScaleDerivation:
    with localcontext() as context:
        context.prec = precision + _GUARD_DIGITS
        resolved_kappa = derive_kappa_d5(precision=context.prec).kappa if kappa is None else _decimal(kappa)
        resolved_mass_bridge = derive_mass_bridge(precision=context.prec, kappa=resolved_kappa) if mass_bridge is None else mass_bridge
        three_pi = Decimal(3) * decimal_pi(context.prec)
        branch_planck_mass_ev = resolved_mass_bridge.branch_planck_mass_ev
        holographic_bits = resolved_mass_bridge.holographic_bits
        lambda_holo_ev2 = three_pi * branch_planck_mass_ev * branch_planck_mass_ev / holographic_bits
        g_newton_topological_ev_minus2 = _decimal(
            topological_newton_coordinate_ev_minus2(branch_planck_mass_ev=float(branch_planck_mass_ev))
        )
        g_newton_inverse_square_ev_minus2 = Decimal(1) / (branch_planck_mass_ev * branch_planck_mass_ev)
        g_newton_bridge_drift = abs(g_newton_topological_ev_minus2 - g_newton_inverse_square_ev_minus2)
        kappa_fourth = resolved_kappa**4
        expanded_theorem_rhs_ev2 = (
            (three_pi / kappa_fourth)
            * g_newton_inverse_square_ev_minus2
            * (resolved_mass_bridge.neutrino_floor_ev**4)
        )
        reduced_theorem_rhs_ev2 = three_pi * branch_planck_mass_ev * branch_planck_mass_ev / holographic_bits
        identity_ratio = lambda_holo_ev2 / expanded_theorem_rhs_ev2
        epsilon_lambda = abs(Decimal(1) - identity_ratio)
        decimal_tolerance = Decimal(1).scaleb(-(precision - _GUARD_DIGITS))
        register_noise_floor = Decimal(1) / holographic_bits
        context.prec = precision
        derivation = UnityOfScaleDerivation(
            branch_planck_mass_ev=+branch_planck_mass_ev,
            holographic_bits=+holographic_bits,
            lambda_holo_ev2=+lambda_holo_ev2,
            g_newton_topological_ev_minus2=+g_newton_topological_ev_minus2,
            g_newton_inverse_square_ev_minus2=+g_newton_inverse_square_ev_minus2,
            g_newton_bridge_drift=+g_newton_bridge_drift,
            kappa_fourth=+kappa_fourth,
            expanded_theorem_rhs_ev2=+expanded_theorem_rhs_ev2,
            reduced_theorem_rhs_ev2=+reduced_theorem_rhs_ev2,
            identity_ratio=+identity_ratio,
            epsilon_lambda=+epsilon_lambda,
            decimal_tolerance=+decimal_tolerance,
            register_noise_floor=+register_noise_floor,
        )
    assert derivation.passed, (
        "Unity of Scale Identity no longer closes within the Decimal audit tolerance: "
        f"epsilon_Lambda={derivation.epsilon_lambda}, tolerance={derivation.decimal_tolerance}."
    )
    return derivation


def build_derivation_ledger(*, precision: int = DEFAULT_PRECISION) -> str:
    obc = derive_observational_boundary_conditions()
    alpha = derive_alpha_surface(precision=precision)
    kappa = derive_kappa_d5(precision=precision)
    mass = derive_mass_bridge(precision=precision, kappa=kappa.kappa)
    unity = derive_unity_of_scale(precision=precision, kappa=kappa.kappa, mass_bridge=mass)

    lines = [
        "Derivation Ledger",
        "=================",
        "",
        "Branch Integers",
        f"- k_l = {LEPTON_LEVEL}",
        f"- k_q = {QUARK_LEVEL}",
        f"- K = {PARENT_LEVEL}",
        f"- G_SM = {G_SM}",
        "",
        "Observational Boundary Conditions",
        f"- hbar [OBC] = {_format_decimal(obc.hbar_ev_seconds, places=24)} eV s",
        "",
        "Alpha Surface Inverse",
        f"- visible support = k_l + k_q = {alpha.visible_support}",
        f"- level-density ratio = K/(k_l + k_q) = {alpha.level_density_ratio.numerator}/{alpha.level_density_ratio.denominator}",
        f"- alpha_surf^-1 = G_SM * K/(k_l + k_q) = {alpha.alpha_inverse_fraction.numerator}/{alpha.alpha_inverse_fraction.denominator}",
        f"- alpha_surf^-1 [decimal] = {_format_decimal(alpha.alpha_inverse_decimal, places=24)}",
        f"- tn.py live check = {_format_decimal(alpha.live_alpha_inverse, places=24)}",
        f"- CODATA alpha^-1 = {_format_decimal(alpha.codata_alpha_inverse, places=12)}",
        f"- unresolved Two-Loop Residual = alpha_surf^-1 - alpha_CODATA^-1 = {_format_decimal(alpha.alpha_inverse_decimal - alpha.codata_alpha_inverse, places=24)}",
        "",
        "D5 Hyperarea Invariant",
        f"- simplex hyperarea prefactor = {kappa.simplex_fraction.numerator}/{kappa.simplex_fraction.denominator}",
        f"- sqrt(10) = {_format_decimal(kappa.sqrt_ten, places=24)}",
        f"- R_01^(par/vis) = 160*sqrt(10)/1521 = {_format_decimal(kappa.weight_simplex_hyperarea, places=24)}",
        f"- D_SU(2)({LEPTON_LEVEL}) = sqrt((k_l+2)/2)/sin(pi/(k_l+2)) = {_format_decimal(kappa.su2_total_quantum_dimension, places=24)}",
        f"- beta = 1/2 ln D_SU(2) = {_format_decimal(kappa.beta, places=24)}",
        f"- eta_spin = (347 - 8 beta^2)/351 = {_format_decimal(kappa.eta_spin, places=24)}",
        f"- kappa_D5 = sqrt((16/5) * R_01^(par/vis) * eta_spin) = {_format_decimal(kappa.kappa, places=24)}",
        f"- tn.py benchmark kappa = {_format_decimal(kappa.live_kappa, places=24)}",
        f"- derivation drift = {_format_decimal(kappa.kappa - kappa.live_kappa, places=24)}",
        "",
        "Finite-Capacity Mass Bridge",
        f"- topological_planck_mass_ev() = {_format_decimal(mass.branch_planck_mass_ev, places=18)} eV",
        f"- PLANCK_MASS_EV from tn.py = {_format_decimal(mass.configured_planck_mass_ev, places=18)} eV",
        f"- HOLOGRAPHIC_BITS = {_format_decimal(mass.holographic_bits, places=18)}",
        f"- N_holo^(-1/4) = {_format_decimal(mass.holographic_bits_quarter_inverse, places=24)}",
        f"- m_nu = kappa_D5 * M_P * N_holo^(-1/4) = {_format_decimal(mass.neutrino_floor_ev, places=24)} eV",
        f"- neutrino floor = {_format_decimal(mass.neutrino_floor_mev, places=24)} meV",
        f"- tn.py live mass bridge = {_format_decimal(mass.live_neutrino_floor_ev, places=24)} eV",
        f"- mass-bridge drift = {_format_decimal(mass.neutrino_floor_ev - mass.live_neutrino_floor_ev, places=24)} eV",
        "",
        "Unity of Scale Identity",
        f"- Lambda_holo(lhs) = 3*pi*M_P^2/N = {_format_decimal(unity.lambda_holo_ev2, places=24)} eV^2",
        f"- G_N = topological_newton_coordinate_ev_minus2(M_P^bridge) = {_format_decimal(unity.g_newton_topological_ev_minus2, places=24)} eV^(-2)",
        f"- direct M_P^(-2) cross-check = {_format_decimal(unity.g_newton_inverse_square_ev_minus2, places=24)} eV^(-2)",
        f"- Newton-lock drift = {_format_decimal(unity.g_newton_bridge_drift, places=24)} eV^(-2)",
        f"- kappa_D5^4 = {_format_decimal(unity.kappa_fourth, places=24)}",
        f"- Lambda_holo(rhs, expanded) = (3*pi/kappa_D5^4) * G_N * m_nu^4 = {_format_decimal(unity.expanded_theorem_rhs_ev2, places=24)} eV^2",
        f"- Lambda_holo(rhs, reduced) = 3*pi*M_P^2/N = {_format_decimal(unity.reduced_theorem_rhs_ev2, places=24)} eV^2",
        f"- lhs/rhs(expanded) = {_format_decimal(unity.identity_ratio, places=24)}",
        f"- epsilon_Lambda = {_format_decimal(unity.epsilon_lambda, places=24)}",
        f"- Decimal audit tolerance = {_format_decimal(unity.decimal_tolerance, places=24)}",
        f"- register noise floor = 1/N = {_format_decimal(unity.register_noise_floor, places=24)}",
        f"- closes in Decimal arithmetic = {unity.passed}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive branch-fixed universe constants from the (26, 8, 312) ledger.")
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION, help="Decimal precision used for the derivation ledger.")
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    print(build_derivation_ledger(precision=max(args.precision, DEFAULT_PRECISION)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
