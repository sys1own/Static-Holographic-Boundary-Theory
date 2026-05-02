from __future__ import annotations

"""Publication-facing verifier for the Unity of Scale Identity."""

import argparse
import math
import re
from dataclasses import dataclass
from decimal import Decimal, localcontext
from functools import lru_cache
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pub.constants import (
        ATMOSPHERIC_MASS_SPLITTING_NO_EV2,
        LEPTON_LEVEL,
        PARENT_LEVEL,
        PLANCK2018_LAMBDA_SI_M2,
        QUARK_LEVEL,
        SOLAR_MASS_SPLITTING_EV2,
    )
    from pub.noether_bridge import (
        DEFAULT_PRECISION,
        NewtonLockAudit,
        PI,
        SaturationAudit,
        UnityOfScaleAudit,
        derive_kappa_d5,
        newton_constant_lock,
        saturation_audit,
        unity_of_scale_audit,
    )
else:
    from .constants import (
        ATMOSPHERIC_MASS_SPLITTING_NO_EV2,
        LEPTON_LEVEL,
        PARENT_LEVEL,
        PLANCK2018_LAMBDA_SI_M2,
        QUARK_LEVEL,
        SOLAR_MASS_SPLITTING_EV2,
    )
    from .noether_bridge import (
        DEFAULT_PRECISION,
        NewtonLockAudit,
        PI,
        SaturationAudit,
        UnityOfScaleAudit,
        derive_kappa_d5,
        newton_constant_lock,
        saturation_audit,
        unity_of_scale_audit,
    )


EXPECTED_BRANCH = (26, 8, 312)
PLANCK2018_SUM_OF_MASSES_BOUND_EV = 0.12
EXACT_KAPPA_D5_REFERENCE = Decimal("0.988771051266")
_LATEX_SCIENTIFIC_PATTERN = re.compile(r"^(?P<coefficient>-?\d+(?:\.\d+)?)\\times10\^\{(?P<exponent>-?\d+)\}$")
_MACRO_PREFIX = "\\newcommand{\\"


@dataclass(frozen=True)
class PublicationScaleConstants:
    rounded_kappa_d5: float
    transported_lightest_mass_mev: float
    planck_sum_bound_ev: float


@dataclass(frozen=True)
class ScaleSynthesisAudit:
    lambda_from_mass_ev2: Decimal
    capacity_from_neutrino_floor: Decimal
    lambda_relative_mismatch: Decimal
    capacity_relative_mismatch: Decimal

    @property
    def saturated(self) -> bool:
        return self.lambda_relative_mismatch < Decimal("1e-50") and self.capacity_relative_mismatch < Decimal("1e-50")


@dataclass(frozen=True)
class ScaleTrapSample:
    lambda_scale_factor: Decimal
    lambda_si_m2: Decimal
    holographic_bits: Decimal
    register_noise_floor: Decimal
    lightest_mass_mev: Decimal


@dataclass(frozen=True)
class ScaleTrapDefenseAudit:
    samples: tuple[ScaleTrapSample, ...]
    capacity_grows_without_bound: bool
    lightest_mass_collapses_to_zero: bool
    one_copy_dictionary_collapses: bool


@dataclass(frozen=True)
class PlanckBoundAudit:
    transported_lightest_mass_mev: float
    masses_ev: tuple[float, float, float]
    sum_masses_ev: float
    sum_bound_ev: float
    headroom_ev: float

    @property
    def consistent_with_planck_2018(self) -> bool:
        return self.sum_masses_ev < self.sum_bound_ev


@dataclass(frozen=True)
class TensionSaturationAudit:
    benchmark_branch: tuple[int, int, int]
    publication: PublicationScaleConstants
    newton_lock: NewtonLockAudit
    saturation: SaturationAudit
    unity: UnityOfScaleAudit
    scale_synthesis: ScaleSynthesisAudit
    scale_trap: ScaleTrapDefenseAudit
    planck_bound: PlanckBoundAudit
    verdict: str

    @property
    def finite_capacity_requires_nonzero_mass(self) -> bool:
        return (
            self.saturation.boundary_condition_locked
            and self.unity.passed
            and self.scale_trap.one_copy_dictionary_collapses
            and self.planck_bound.consistent_with_planck_2018
            and self.unity.kappa_d5 > 0
            and self.saturation.holographic_bits_from_lambda > 0
        )


def _format_decimal_scientific(value: Decimal, precision: int = 12) -> str:
    if value == 0:
        return f"{0:.{precision}E}"
    return format(value, f".{precision}E")


def _parse_latex_number(text: str) -> float:
    cleaned = text.strip().strip("$")
    scientific_match = _LATEX_SCIENTIFIC_PATTERN.fullmatch(cleaned)
    if scientific_match is not None:
        coefficient = float(scientific_match.group("coefficient"))
        exponent = int(scientific_match.group("exponent"))
        return coefficient * (10.0 ** exponent)
    return float(cleaned)


@lru_cache(maxsize=1)
def _load_physics_constant_macros() -> dict[str, str]:
    macro_path = Path(__file__).with_name("physics_constants.tex")
    macros: dict[str, str] = {}
    for raw_line in macro_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith(_MACRO_PREFIX) or "}{" not in line or not line.endswith("}"):
            continue
        prefix, value = line.split("}{", 1)
        macro_name = prefix[len(_MACRO_PREFIX) :]
        macros[macro_name] = value[:-1]
    if not macros:
        raise RuntimeError("Failed to parse publication macros from physics_constants.tex.")
    return macros


@lru_cache(maxsize=1)
def load_publication_scale_constants() -> PublicationScaleConstants:
    macros = _load_physics_constant_macros()
    try:
        rounded_kappa_d5 = _parse_latex_number(macros["simplexPrefactor"])
        transported_lightest_mass_mev = _parse_latex_number(macros["mZeroBenchmarkMeV"])
    except KeyError as exc:
        raise RuntimeError(f"Missing publication macro for the tension audit: {exc.args[0]}") from exc
    return PublicationScaleConstants(
        rounded_kappa_d5=rounded_kappa_d5,
        transported_lightest_mass_mev=transported_lightest_mass_mev,
        planck_sum_bound_ev=PLANCK2018_SUM_OF_MASSES_BOUND_EV,
    )


def _normal_order_masses(lightest_mass_ev: float) -> tuple[float, float, float]:
    lightest_mass = max(0.0, float(lightest_mass_ev))
    m2_ev = math.sqrt(max(0.0, lightest_mass * lightest_mass + SOLAR_MASS_SPLITTING_EV2))
    m3_ev = math.sqrt(max(0.0, lightest_mass * lightest_mass + ATMOSPHERIC_MASS_SPLITTING_NO_EV2))
    return (lightest_mass, m2_ev, m3_ev)


def synthesize_lambda_holo_ev2(*, kappa_d5: Decimal, g_newton_ev_minus2: Decimal, lightest_mass_ev: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION
        return (Decimal(3) * PI * g_newton_ev_minus2 * (lightest_mass_ev**4)) / (kappa_d5**4)


def invert_capacity_from_neutrino_floor(*, kappa_d5: Decimal, planck_mass_ev: Decimal, lightest_mass_ev: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION
        return ((kappa_d5 * planck_mass_ev) / lightest_mass_ev) ** Decimal(4)


def build_scale_trap_defense_audit(
    *,
    kappa_d5: Decimal,
    newton_lock: NewtonLockAudit,
    precision: int = DEFAULT_PRECISION,
) -> ScaleTrapDefenseAudit:
    benchmark_lambda = Decimal(str(PLANCK2018_LAMBDA_SI_M2))
    samples: list[ScaleTrapSample] = []
    for factor in (Decimal("1"), Decimal("1e-4"), Decimal("1e-8"), Decimal("1e-16")):
        saturation = saturation_audit(lambda_obs_si_m2=benchmark_lambda * factor, precision=precision)
        unity = unity_of_scale_audit(
            kappa_d5=kappa_d5,
            newton_lock_audit=newton_lock,
            saturation=saturation,
            precision=precision,
        )
        samples.append(
            ScaleTrapSample(
                lambda_scale_factor=factor,
                lambda_si_m2=saturation.lambda_obs_si_m2,
                holographic_bits=saturation.holographic_bits_from_lambda,
                register_noise_floor=saturation.register_noise_floor,
                lightest_mass_mev=unity.lightest_mass_mev,
            )
        )
    audited_samples = tuple(samples)
    capacity_grows_without_bound = all(
        previous.holographic_bits < current.holographic_bits
        for previous, current in zip(audited_samples, audited_samples[1:])
    )
    lightest_mass_collapses_to_zero = all(
        previous.lightest_mass_mev > current.lightest_mass_mev
        for previous, current in zip(audited_samples, audited_samples[1:])
    )
    return ScaleTrapDefenseAudit(
        samples=audited_samples,
        capacity_grows_without_bound=capacity_grows_without_bound,
        lightest_mass_collapses_to_zero=lightest_mass_collapses_to_zero,
        one_copy_dictionary_collapses=capacity_grows_without_bound and lightest_mass_collapses_to_zero,
    )


def build_planck_bound_audit(*, transported_lightest_mass_mev: float) -> PlanckBoundAudit:
    masses_ev = _normal_order_masses(transported_lightest_mass_mev * 1.0e-3)
    sum_masses_ev = float(sum(masses_ev))
    return PlanckBoundAudit(
        transported_lightest_mass_mev=transported_lightest_mass_mev,
        masses_ev=masses_ev,
        sum_masses_ev=sum_masses_ev,
        sum_bound_ev=PLANCK2018_SUM_OF_MASSES_BOUND_EV,
        headroom_ev=float(PLANCK2018_SUM_OF_MASSES_BOUND_EV - sum_masses_ev),
    )


def build_tension_saturation_audit(*, precision: int = DEFAULT_PRECISION) -> TensionSaturationAudit:
    benchmark_branch = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))
    assert benchmark_branch == EXPECTED_BRANCH, (
        f"The holographic-tension verifier is locked to the benchmark branch {EXPECTED_BRANCH}, received {benchmark_branch}."
    )

    publication = load_publication_scale_constants()
    kappa_d5 = derive_kappa_d5(lepton_level=LEPTON_LEVEL, precision=precision)
    assert math.isclose(float(kappa_d5), float(EXACT_KAPPA_D5_REFERENCE), rel_tol=0.0, abs_tol=1.0e-12), (
        f"The D5 simplex residue drifted: expected {EXACT_KAPPA_D5_REFERENCE}, received {kappa_d5}."
    )
    assert round(float(kappa_d5), 8) == 0.98877105, "The benchmark residue must round to kappa_D5≈0.98877105."
    assert math.isclose(float(kappa_d5), publication.rounded_kappa_d5, rel_tol=0.0, abs_tol=1.0e-5), (
        f"Rounded publication residue {publication.rounded_kappa_d5} no longer matches the exact branch kappa_D5={kappa_d5}."
    )

    newton_lock = newton_constant_lock(precision=precision)
    saturation = saturation_audit(precision=precision)
    unity = unity_of_scale_audit(
        kappa_d5=kappa_d5,
        newton_lock_audit=newton_lock,
        saturation=saturation,
        precision=precision,
    )
    lambda_from_mass_ev2 = synthesize_lambda_holo_ev2(
        kappa_d5=kappa_d5,
        g_newton_ev_minus2=newton_lock.g_topological_ev_minus2,
        lightest_mass_ev=unity.lightest_mass_ev,
    )
    capacity_from_neutrino_floor = invert_capacity_from_neutrino_floor(
        kappa_d5=kappa_d5,
        planck_mass_ev=newton_lock.planck_mass_ev,
        lightest_mass_ev=unity.lightest_mass_ev,
    )
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION
        lambda_relative_mismatch = unity.epsilon_lambda
        capacity_relative_mismatch = abs((capacity_from_neutrino_floor / saturation.holographic_bits_from_lambda) - Decimal(1))

    scale_synthesis = ScaleSynthesisAudit(
        lambda_from_mass_ev2=lambda_from_mass_ev2,
        capacity_from_neutrino_floor=capacity_from_neutrino_floor,
        lambda_relative_mismatch=lambda_relative_mismatch,
        capacity_relative_mismatch=capacity_relative_mismatch,
    )
    assert abs(lambda_from_mass_ev2 - unity.lambda_rhs_topological_ev2) <= Decimal("1e-110"), (
        "The explicit Lambda_holo synthesis drifted away from the bridge's topological route."
    )
    assert lambda_relative_mismatch <= unity.register_noise_floor, (
        "The Unity of Scale Identity must close below the one-register floor on the benchmark branch."
    )
    assert capacity_relative_mismatch <= saturation.register_noise_floor, (
        "The neutrino-floor inversion must reproduce the benchmark capacity below one register bit."
    )

    scale_trap = build_scale_trap_defense_audit(
        kappa_d5=kappa_d5,
        newton_lock=newton_lock,
        precision=precision,
    )
    assert scale_trap.one_copy_dictionary_collapses, (
        "The sampled void-limit sequence must show N growing and m_nu collapsing as Lambda_holo is driven toward zero."
    )

    planck_bound = build_planck_bound_audit(
        transported_lightest_mass_mev=publication.transported_lightest_mass_mev,
    )
    assert planck_bound.consistent_with_planck_2018, (
        f"The transported spectrum must satisfy the Planck 2018 bound, received sum m_nu={planck_bound.sum_masses_ev:.6f} eV."
    )

    verdict = (
        "Tension Saturation Report: PASS — the Unity of Scale Identity closes on the anomaly-free branch "
        f"{benchmark_branch}, kappa_D5={float(kappa_d5):.12f} remains the branch-fixed geometric residue, "
        f"and the transported low-scale spectrum m1≈{publication.transported_lightest_mass_mev:.2f} meV "
        f"with sum m_nu≈{planck_bound.sum_masses_ev:.3f} eV stays below the Planck 2018 bound. "
        "Non-zero neutrino mass is therefore a mandatory consequence of finite capacity."
    )
    return TensionSaturationAudit(
        benchmark_branch=benchmark_branch,
        publication=publication,
        newton_lock=newton_lock,
        saturation=saturation,
        unity=unity,
        scale_synthesis=scale_synthesis,
        scale_trap=scale_trap,
        planck_bound=planck_bound,
        verdict=verdict,
    )


def render_report(audit: TensionSaturationAudit) -> str:
    lines = [
        "Tension Saturation Report",
        "=========================",
        f"Benchmark branch                  : {audit.benchmark_branch}",
        f"kappa_D5 (exact)                 : {float(audit.unity.kappa_d5):.12f}",
        f"kappa_D5 (publication rounded)   : {audit.publication.rounded_kappa_d5:.5f}",
        "",
        "Unity of Scale Identity",
        "-----------------------",
        "Lambda_holo = (3 pi / kappa_D5^4) G_N m_nu^4",
        f"Lambda_obs [m^-2]                : {_format_decimal_scientific(audit.saturation.lambda_obs_si_m2)}",
        f"Lambda_holo(lhs) [eV^2]          : {_format_decimal_scientific(audit.unity.lambda_lhs_ev2)}",
        f"Lambda_holo(rhs) [eV^2]          : {_format_decimal_scientific(audit.scale_synthesis.lambda_from_mass_ev2)}",
        f"N_Lambda                         : {_format_decimal_scientific(audit.saturation.holographic_bits_from_lambda)}",
        f"N_nu                             : {_format_decimal_scientific(audit.scale_synthesis.capacity_from_neutrino_floor)}",
        f"m_nu(branch bookkeeping)         : {_format_decimal_scientific(audit.unity.lightest_mass_mev)} meV",
        f"epsilon_lambda                   : {_format_decimal_scientific(audit.unity.epsilon_lambda)}",
        f"register floor                   : {_format_decimal_scientific(audit.unity.register_noise_floor)}",
        f"Unity closure                    : {'PASS' if audit.unity.passed else 'FAIL'}",
        "",
        "Scale Trap Defense",
        "------------------",
        "Driving Lambda_holo toward zero forces N upward and m_nu downward.",
        "scale factor     N_holo                    m_nu [meV]",
        "---------------------------------------------------------------",
    ]
    for sample in audit.scale_trap.samples:
        lines.append(
            f"{str(sample.lambda_scale_factor):<14} {_format_decimal_scientific(sample.holographic_bits, precision=6):<25} "
            f"{_format_decimal_scientific(sample.lightest_mass_mev, precision=6)}"
        )
    lines.extend(
        (
            f"Capacity-growth check             : {'PASS' if audit.scale_trap.capacity_grows_without_bound else 'FAIL'}",
            f"Mass-collapse check               : {'PASS' if audit.scale_trap.lightest_mass_collapses_to_zero else 'FAIL'}",
            f"One-copy dictionary               : {'COLLAPSES in void limit' if audit.scale_trap.one_copy_dictionary_collapses else 'NOT SHOWN'}",
            "",
            "Cosmological Bound Audit",
            "------------------------",
            f"Transported m1(M_Z)               : {audit.planck_bound.transported_lightest_mass_mev:.2f} meV",
            (
                "Transported masses [eV]           : "
                f"{audit.planck_bound.masses_ev[0]:.6f}, {audit.planck_bound.masses_ev[1]:.6f}, {audit.planck_bound.masses_ev[2]:.6f}"
            ),
            f"sum m_nu [eV]                     : {audit.planck_bound.sum_masses_ev:.6f}",
            f"Planck 2018 bound [eV]            : {audit.planck_bound.sum_bound_ev:.2f}",
            f"Headroom [eV]                     : {audit.planck_bound.headroom_ev:.6f}",
            f"Planck consistency                : {'PASS' if audit.planck_bound.consistent_with_planck_2018 else 'FAIL'}",
            "",
            audit.verdict,
        )
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    audit = build_tension_saturation_audit(precision=args.precision)
    print(render_report(audit))


if __name__ == "__main__":
    main()
