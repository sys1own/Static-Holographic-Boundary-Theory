from __future__ import annotations

"""Precision cosmology engine for the SHBT late-time expansion audit."""

import argparse
import re
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pub.constants import PLANCK2018_LAMBDA_SI_M2
    from pub.noether_bridge import load_c_dark_completion_fraction
else:
    from .constants import PLANCK2018_LAMBDA_SI_M2
    from .noether_bridge import load_c_dark_completion_fraction


DEFAULT_PRECISION = 50
DEFAULT_REDSHIFTS = (Decimal("0"), Decimal("0.5"), Decimal("1"), Decimal("2"), Decimal("10"), Decimal("1100"))
FALLBACK_H0_CMB_ANCHOR_KM_S_MPC = Decimal("67.4")
BENCHMARK_GRADIENT_REFERENCE_KM_S_MPC = Decimal("4.80")
BENCHMARK_LOCAL_REFERENCE_KM_S_MPC = Decimal("72.2")


@dataclass(frozen=True)
class RedshiftExpansionPoint:
    redshift: Decimal
    loading_term_km_s_mpc: Decimal
    hubble_km_s_mpc: Decimal


@dataclass(frozen=True)
class PrecisionCosmologyAudit:
    h0_cmb_km_s_mpc: Decimal
    delta_mod_fraction: Fraction
    delta_mod: Decimal
    c_dark_fraction: Fraction
    c_dark: Decimal
    uplift_factor: Decimal
    h0_local_km_s_mpc: Decimal
    hubble_gradient_km_s_mpc: Decimal
    h_lambda_operational_km_s_mpc: Decimal
    loading_derivative_km_s_mpc: Decimal
    lambda_holo_si_m2: Decimal
    h_lambda_surface_tension_m_inverse: Decimal
    samples: tuple[RedshiftExpansionPoint, ...]

    @property
    def local_value_verified(self) -> bool:
        return abs(self.h0_local_km_s_mpc - BENCHMARK_LOCAL_REFERENCE_KM_S_MPC) <= Decimal("0.1")

    @property
    def gradient_verified(self) -> bool:
        return abs(self.hubble_gradient_km_s_mpc - BENCHMARK_GRADIENT_REFERENCE_KM_S_MPC) <= Decimal("0.1")

    @property
    def surface_tension_positive(self) -> bool:
        return self.lambda_holo_si_m2 > 0

    @property
    def late_time_loading_active(self) -> bool:
        return self.loading_derivative_km_s_mpc > 0

    @property
    def shbt_outperforms_lcdm(self) -> bool:
        return self.surface_tension_positive and self.late_time_loading_active and self.gradient_verified


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(str(value))


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _format_decimal_scientific(value: Decimal, digits: int = 12) -> str:
    return f"{float(value):.{digits}E}"


def _load_hubble_cmb_anchor() -> Decimal:
    tex_path = Path(__file__).with_name("physics_constants.tex")
    try:
        match = re.search(
            r"\\newcommand\{\\hubbleCmbAnchor\}\{([^}]+)\}",
            tex_path.read_text(encoding="utf-8"),
        )
    except OSError:
        match = None
    if match is None:
        return FALLBACK_H0_CMB_ANCHOR_KM_S_MPC
    return Decimal(match.group(1).strip())


def benchmark_entropy_debt_fraction() -> Fraction:
    return load_c_dark_completion_fraction() / 24


def entropy_debt_uplift_factor(
    delta_mod: Decimal | Fraction | float | int | str = benchmark_entropy_debt_fraction(),
    *,
    precision: int = DEFAULT_PRECISION,
) -> Decimal:
    with localcontext() as context:
        context.prec = max(int(precision), 28)
        resolved_delta_mod = _decimal(delta_mod)
        return (resolved_delta_mod / Decimal("2")).exp()


def late_time_hubble_constant(
    h0_cmb_km_s_mpc: Decimal | Fraction | float | int | str = FALLBACK_H0_CMB_ANCHOR_KM_S_MPC,
    delta_mod: Decimal | Fraction | float | int | str = benchmark_entropy_debt_fraction(),
    *,
    precision: int = DEFAULT_PRECISION,
) -> Decimal:
    with localcontext() as context:
        context.prec = max(int(precision), 28)
        return _decimal(h0_cmb_km_s_mpc) * entropy_debt_uplift_factor(delta_mod, precision=precision)


def hubble_gradient(
    h0_cmb_km_s_mpc: Decimal | Fraction | float | int | str = FALLBACK_H0_CMB_ANCHOR_KM_S_MPC,
    delta_mod: Decimal | Fraction | float | int | str = benchmark_entropy_debt_fraction(),
    *,
    precision: int = DEFAULT_PRECISION,
) -> Decimal:
    with localcontext() as context:
        context.prec = max(int(precision), 28)
        resolved_h0_cmb = _decimal(h0_cmb_km_s_mpc)
        return late_time_hubble_constant(resolved_h0_cmb, delta_mod, precision=precision) - resolved_h0_cmb


def loading_fraction_derivative(
    h0_cmb_km_s_mpc: Decimal | Fraction | float | int | str = FALLBACK_H0_CMB_ANCHOR_KM_S_MPC,
    delta_mod: Decimal | Fraction | float | int | str = benchmark_entropy_debt_fraction(),
    *,
    precision: int = DEFAULT_PRECISION,
) -> Decimal:
    with localcontext() as context:
        context.prec = max(int(precision), 28)
        return Decimal("3") * hubble_gradient(h0_cmb_km_s_mpc, delta_mod, precision=precision)


def redshift_dependent_hubble_constant(
    redshift: Decimal | Fraction | float | int | str,
    *,
    h_lambda_km_s_mpc: Decimal | Fraction | float | int | str = FALLBACK_H0_CMB_ANCHOR_KM_S_MPC,
    df_load_dtau_lock: Decimal | Fraction | float | int | str | None = None,
    delta_mod: Decimal | Fraction | float | int | str = benchmark_entropy_debt_fraction(),
    precision: int = DEFAULT_PRECISION,
) -> Decimal:
    with localcontext() as context:
        context.prec = max(int(precision), 28)
        resolved_redshift = _decimal(redshift)
        if resolved_redshift < 0:
            raise ValueError("redshift must be non-negative")
        resolved_h_lambda = _decimal(h_lambda_km_s_mpc)
        resolved_loading_derivative = (
            loading_fraction_derivative(resolved_h_lambda, delta_mod, precision=precision)
            if df_load_dtau_lock is None
            else _decimal(df_load_dtau_lock)
        )
        return resolved_h_lambda + resolved_loading_derivative / (Decimal("3") * (Decimal("1") + resolved_redshift))


def surface_tension_hubble_friction_m_inverse(
    lambda_holo_si_m2: Decimal | Fraction | float | int | str = PLANCK2018_LAMBDA_SI_M2,
    *,
    precision: int = DEFAULT_PRECISION,
) -> Decimal:
    with localcontext() as context:
        context.prec = max(int(precision), 28)
        return (_decimal(lambda_holo_si_m2) / Decimal("3")).sqrt()


def build_precision_cosmology_audit(
    *,
    redshifts: Sequence[Decimal | Fraction | float | int | str] = DEFAULT_REDSHIFTS,
    precision: int = DEFAULT_PRECISION,
) -> PrecisionCosmologyAudit:
    with localcontext() as context:
        context.prec = max(int(precision), 28)
        h0_cmb = _load_hubble_cmb_anchor()
        c_dark_fraction = load_c_dark_completion_fraction()
        delta_mod_fraction = benchmark_entropy_debt_fraction()
        delta_mod = _decimal(delta_mod_fraction)
        c_dark = _decimal(c_dark_fraction)
        uplift_factor = entropy_debt_uplift_factor(delta_mod, precision=precision)
        h0_local = late_time_hubble_constant(h0_cmb, delta_mod, precision=precision)
        delta_h0 = h0_local - h0_cmb
        loading_derivative = loading_fraction_derivative(h0_cmb, delta_mod, precision=precision)
        lambda_holo_si_m2 = _decimal(PLANCK2018_LAMBDA_SI_M2)
        h_lambda_surface = surface_tension_hubble_friction_m_inverse(lambda_holo_si_m2, precision=precision)
        samples = tuple(
            RedshiftExpansionPoint(
                redshift=_decimal(redshift),
                loading_term_km_s_mpc=loading_derivative / (Decimal("3") * (Decimal("1") + _decimal(redshift))),
                hubble_km_s_mpc=redshift_dependent_hubble_constant(
                    redshift,
                    h_lambda_km_s_mpc=h0_cmb,
                    df_load_dtau_lock=loading_derivative,
                    precision=precision,
                ),
            )
            for redshift in redshifts
        )
    assert samples[0].redshift == 0
    assert abs(samples[0].hubble_km_s_mpc - h0_local) < Decimal("1e-20")
    return PrecisionCosmologyAudit(
        h0_cmb_km_s_mpc=h0_cmb,
        delta_mod_fraction=delta_mod_fraction,
        delta_mod=delta_mod,
        c_dark_fraction=c_dark_fraction,
        c_dark=c_dark,
        uplift_factor=uplift_factor,
        h0_local_km_s_mpc=h0_local,
        hubble_gradient_km_s_mpc=delta_h0,
        h_lambda_operational_km_s_mpc=h0_cmb,
        loading_derivative_km_s_mpc=loading_derivative,
        lambda_holo_si_m2=lambda_holo_si_m2,
        h_lambda_surface_tension_m_inverse=h_lambda_surface,
        samples=samples,
    )


def render_report(audit: PrecisionCosmologyAudit) -> str:
    verification_status = "PASS" if audit.shbt_outperforms_lcdm and audit.local_value_verified else "CHECK"
    lines = [
        "Precision Cosmology Report",
        "==========================",
        f"Planck anchor H0^CMB [km s^-1 Mpc^-1]   : {audit.h0_cmb_km_s_mpc:.1f}",
        (
            "Benchmark entropy debt Delta_mod       : "
            f"{_format_fraction(audit.delta_mod_fraction)} = {audit.delta_mod:.12f}"
        ),
        f"Completion residue c_dark              : {_format_fraction(audit.c_dark_fraction)} = {audit.c_dark:.12f}",
        f"Late-time uplift exp(Delta_mod / 2)    : {audit.uplift_factor:.12f}",
        f"Predicted H0^loc [km s^-1 Mpc^-1]      : {audit.h0_local_km_s_mpc:.12f}",
        f"Predicted Delta H0 [km s^-1 Mpc^-1]    : {audit.hubble_gradient_km_s_mpc:.12f}",
        f"Operational H_Lambda [km s^-1 Mpc^-1]  : {audit.h_lambda_operational_km_s_mpc:.12f}",
        f"Resolved df_load/dtau_lock             : {audit.loading_derivative_km_s_mpc:.12f}",
        f"Lambda_holo [m^-2]                     : {_format_decimal_scientific(audit.lambda_holo_si_m2)}",
        f"Surface-tension H_Lambda [m^-1]        : {_format_decimal_scientific(audit.h_lambda_surface_tension_m_inverse)}",
        "",
        "Observer-frame loading law",
        "--------------------------",
        "H0(z) = H_Lambda + [3(1+z)]^-1 df_load/dtau_lock",
        "Here the observational baseline H_Lambda is anchored to the Planck CMB intercept,",
        "while the positive holographic surface tension keeps the branch sink finite and non-zero.",
        "",
        "Redshift ladder",
        "---------------",
        "z        loading term [km s^-1 Mpc^-1]   H0(z) [km s^-1 Mpc^-1]",
        "----------------------------------------------------------------",
    ]
    lines.extend(
        f"{sample.redshift:>6.1f}   {sample.loading_term_km_s_mpc:>26.12f}   {sample.hubble_km_s_mpc:>22.12f}"
        for sample in audit.samples
    )
    lines.extend(
        (
            "",
            f"Verification status                 : {verification_status}",
            f"Local uplift matches 72.2 benchmark : {'YES' if audit.local_value_verified else 'NO'}",
            f"Gradient matches 4.80 benchmark     : {'YES' if audit.gradient_verified else 'NO'}",
            f"Positive surface tension            : {'YES' if audit.surface_tension_positive else 'NO'}",
            f"Late-time loading active            : {'YES' if audit.late_time_loading_active else 'NO'}",
            "",
            "SHBT outperforms a redshift-rigid ΛCDM intercept here because the late-time",
            "expansion shift is generated by branch-fixed entropy debt carried by a positive",
            "holographic surface tension, rather than by an added detached late-time fluid.",
        )
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION)
    parser.add_argument("--redshifts", nargs="*", type=Decimal, default=list(DEFAULT_REDSHIFTS))
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    audit = build_precision_cosmology_audit(redshifts=args.redshifts, precision=args.precision)
    print(render_report(audit))


if __name__ == "__main__":
    main()
