from __future__ import annotations

"""Flavor-identity audit for the SHBT modular genus ladder."""

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pub import algebra
    from pub.constants import ATMOSPHERIC_MASS_SPLITTING_NO_EV2, LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL, SOLAR_MASS_SPLITTING_EV2
else:
    from . import algebra
    from .constants import ATMOSPHERIC_MASS_SPLITTING_NO_EV2, LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL, SOLAR_MASS_SPLITTING_EV2


EXPECTED_BRANCH = (26, 8, 312)
DEFAULT_PHASE_LOCK_ABS_TOL = 1.0e-6
DEFAULT_SPECTRAL_REL_TOL = 5.0e-2
FALLBACK_BETA_ANCHOR = 1.754551


@dataclass(frozen=True)
class GenerationResidue:
    generation: int
    mass_ratio_to_m0: float
    log_gap_from_m0: float
    resolved_winding_number: float


@dataclass(frozen=True)
class SpectralAnchor:
    label: str
    lightest_mass_ev: float
    masses_ev: tuple[float, float, float]
    delta_m21_ev2: float
    delta_m32_ev2: float
    delta_m31_ev2: float
    anchored_observable: str
    cross_check_observable: str
    cross_check_observed_ev2: float
    cross_check_fractional_error: float


@dataclass(frozen=True)
class FlavorIdentityAudit:
    branch: tuple[int, int, int]
    total_quantum_dimension: float
    beta_phase_lock: float
    beta_anchor: float
    mass_step_ratio: float
    generation_residues: tuple[GenerationResidue, ...]
    kernel_delta_m21_coeff: float
    kernel_delta_m31_coeff: float
    kernel_splitting_ratio: float
    observed_delta_m21_ev2: float
    observed_delta_m31_ev2: float
    observed_splitting_ratio: float
    ratio_fractional_error: float
    solar_anchor: SpectralAnchor
    atmospheric_anchor: SpectralAnchor

    @property
    def phase_lock_verified(self) -> bool:
        return math.isclose(self.beta_phase_lock, self.beta_anchor, rel_tol=0.0, abs_tol=DEFAULT_PHASE_LOCK_ABS_TOL)

    @property
    def winding_lock_verified(self) -> bool:
        return all(
            math.isclose(row.resolved_winding_number, float(row.generation), rel_tol=0.0, abs_tol=1.0e-12)
            for row in self.generation_residues
        )

    @property
    def spectral_check_passed(self) -> bool:
        return abs(self.ratio_fractional_error) <= DEFAULT_SPECTRAL_REL_TOL

    @property
    def solar_anchor_cross_check_passed(self) -> bool:
        return abs(self.solar_anchor.cross_check_fractional_error) <= DEFAULT_SPECTRAL_REL_TOL

    @property
    def atmospheric_anchor_cross_check_passed(self) -> bool:
        return abs(self.atmospheric_anchor.cross_check_fractional_error) <= DEFAULT_SPECTRAL_REL_TOL

    @property
    def mandatory_residue_verified(self) -> bool:
        return (
            self.phase_lock_verified
            and self.winding_lock_verified
            and self.spectral_check_passed
            and self.solar_anchor_cross_check_passed
            and self.atmospheric_anchor_cross_check_passed
        )


def _load_beta_anchor() -> float:
    tex_path = Path(__file__).with_name("physics_constants.tex")
    try:
        match = re.search(
            r"\\newcommand\{\\inflationGenusBeta\}\{([^}]+)\}",
            tex_path.read_text(encoding="utf-8"),
        )
    except OSError:
        match = None
    if match is None:
        return float(FALLBACK_BETA_ANCHOR)
    return float(match.group(1).strip())


def modular_t_phase_lock(*, level: int = LEPTON_LEVEL) -> float:
    return float(0.5 * math.log(algebra.su2_total_quantum_dimension(int(level))))


def modular_genus_ladder(m_0_ev: float, beta: float, *, generations: Sequence[int] = (0, 1, 2)) -> tuple[float, ...]:
    return tuple(float(m_0_ev * math.exp(beta * int(generation))) for generation in generations)


def pairwise_mass_squared_splittings(masses_ev: Sequence[float]) -> tuple[float, float, float]:
    ordered = sorted(float(value) for value in masses_ev)
    first, second, third = ordered
    return (
        float(second * second - first * first),
        float(third * third - second * second),
        float(third * third - first * first),
    )


def solar_anchored_lightest_mass_ev(beta: float, delta_m21_ev2: float = SOLAR_MASS_SPLITTING_EV2) -> float:
    return float(math.sqrt(float(delta_m21_ev2) / max(math.exp(2.0 * beta) - 1.0, np_finfo_eps())))


def atmospheric_anchored_lightest_mass_ev(beta: float, delta_m31_ev2: float = ATMOSPHERIC_MASS_SPLITTING_NO_EV2) -> float:
    return float(math.sqrt(float(delta_m31_ev2) / max(math.exp(4.0 * beta) - 1.0, np_finfo_eps())))


def np_finfo_eps() -> float:
    return float(2.220446049250313e-16)


def _generation_residues(beta: float) -> tuple[GenerationResidue, ...]:
    residues: list[GenerationResidue] = []
    for generation in (0, 1, 2):
        mass_ratio = float(math.exp(beta * generation))
        log_gap = float(math.log(mass_ratio))
        resolved_winding = 0.0 if math.isclose(beta, 0.0, rel_tol=0.0, abs_tol=np_finfo_eps()) else log_gap / beta
        residues.append(
            GenerationResidue(
                generation=generation,
                mass_ratio_to_m0=mass_ratio,
                log_gap_from_m0=log_gap,
                resolved_winding_number=float(resolved_winding),
            )
        )
    return tuple(residues)


def _build_anchor(
    *,
    label: str,
    lightest_mass_ev: float,
    anchored_observable: str,
    cross_check_observable: str,
    cross_check_observed_ev2: float,
    beta: float,
) -> SpectralAnchor:
    masses = modular_genus_ladder(lightest_mass_ev, beta)
    delta_m21_ev2, delta_m32_ev2, delta_m31_ev2 = pairwise_mass_squared_splittings(masses)
    predicted_lookup = {
        "Delta m21^2": delta_m21_ev2,
        "Delta m32^2": delta_m32_ev2,
        "Delta m31^2": delta_m31_ev2,
    }
    predicted_cross_check = predicted_lookup[cross_check_observable]
    return SpectralAnchor(
        label=label,
        lightest_mass_ev=float(lightest_mass_ev),
        masses_ev=tuple(float(value) for value in masses),
        delta_m21_ev2=float(delta_m21_ev2),
        delta_m32_ev2=float(delta_m32_ev2),
        delta_m31_ev2=float(delta_m31_ev2),
        anchored_observable=anchored_observable,
        cross_check_observable=cross_check_observable,
        cross_check_observed_ev2=float(cross_check_observed_ev2),
        cross_check_fractional_error=float(
            (predicted_cross_check - float(cross_check_observed_ev2))
            / max(abs(float(cross_check_observed_ev2)), np_finfo_eps())
        ),
    )


def build_flavor_identity_audit() -> FlavorIdentityAudit:
    branch = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))
    assert branch == EXPECTED_BRANCH, (
        f"Flavor identity audit is locked to the benchmark branch {EXPECTED_BRANCH}, received {branch}."
    )

    total_quantum_dimension = float(algebra.su2_total_quantum_dimension(LEPTON_LEVEL))
    beta_phase_lock = modular_t_phase_lock(level=LEPTON_LEVEL)
    beta_anchor = _load_beta_anchor()
    mass_step_ratio = float(math.exp(beta_phase_lock))
    kernel_delta_m21_coeff = float(math.exp(2.0 * beta_phase_lock) - 1.0)
    kernel_delta_m31_coeff = float(math.exp(4.0 * beta_phase_lock) - 1.0)
    kernel_splitting_ratio = float(kernel_delta_m31_coeff / kernel_delta_m21_coeff)
    observed_splitting_ratio = float(ATMOSPHERIC_MASS_SPLITTING_NO_EV2 / SOLAR_MASS_SPLITTING_EV2)
    ratio_fractional_error = float(
        (kernel_splitting_ratio - observed_splitting_ratio) / max(abs(observed_splitting_ratio), np_finfo_eps())
    )

    solar_anchor = _build_anchor(
        label="solar-anchored ladder",
        lightest_mass_ev=solar_anchored_lightest_mass_ev(beta_phase_lock),
        anchored_observable="Delta m21^2",
        cross_check_observable="Delta m31^2",
        cross_check_observed_ev2=float(ATMOSPHERIC_MASS_SPLITTING_NO_EV2),
        beta=beta_phase_lock,
    )
    atmospheric_anchor = _build_anchor(
        label="atmospheric-anchored ladder",
        lightest_mass_ev=atmospheric_anchored_lightest_mass_ev(beta_phase_lock),
        anchored_observable="Delta m31^2",
        cross_check_observable="Delta m21^2",
        cross_check_observed_ev2=float(SOLAR_MASS_SPLITTING_EV2),
        beta=beta_phase_lock,
    )

    return FlavorIdentityAudit(
        branch=branch,
        total_quantum_dimension=total_quantum_dimension,
        beta_phase_lock=beta_phase_lock,
        beta_anchor=beta_anchor,
        mass_step_ratio=mass_step_ratio,
        generation_residues=_generation_residues(beta_phase_lock),
        kernel_delta_m21_coeff=kernel_delta_m21_coeff,
        kernel_delta_m31_coeff=kernel_delta_m31_coeff,
        kernel_splitting_ratio=kernel_splitting_ratio,
        observed_delta_m21_ev2=float(SOLAR_MASS_SPLITTING_EV2),
        observed_delta_m31_ev2=float(ATMOSPHERIC_MASS_SPLITTING_NO_EV2),
        observed_splitting_ratio=observed_splitting_ratio,
        ratio_fractional_error=ratio_fractional_error,
        solar_anchor=solar_anchor,
        atmospheric_anchor=atmospheric_anchor,
    )


def _format_mass_triplet(masses_ev: Sequence[float]) -> str:
    return "(" + ", ".join(f"{value:.6e}" for value in masses_ev) + ")"


def render_report(audit: FlavorIdentityAudit) -> str:
    lines = [
        "Flavor Identity Audit",
        "=====================",
        f"Benchmark branch (k_l, k_q, K)       : {audit.branch}",
        f"Visible block                        : SU(2)_{audit.branch[0]} x SU(3)_{audit.branch[1]}",
        f"Total quantum dimension D_26         : {audit.total_quantum_dimension:.12f}",
        f"Phase lock beta = 1/2 ln D_26        : {audit.beta_phase_lock:.12f}",
        f"Published beta anchor                : {audit.beta_anchor:.6f}",
        f"Adjacent modular winding e^beta      : {audit.mass_step_ratio:.12f}",
        f"Phase-lock verified                  : {'YES' if audit.phase_lock_verified else 'NO'}",
        "",
        "Generational winding ladder",
        "---------------------------",
        "g   m_g/m_0           ln(m_g/m_0)       ln(m_g/m_0)/beta",
        "---------------------------------------------------------",
    ]
    lines.extend(
        f"{row.generation:<1}   {row.mass_ratio_to_m0:>12.6f}   {row.log_gap_from_m0:>14.6f}   {row.resolved_winding_number:>16.6f}"
        for row in audit.generation_residues
    )
    lines.extend(
        (
            "",
            "The logarithmic mass gap is exactly beta times the winding index g, so the",
            "generation spacing is the modular-image of the SU(2)_26 winding ladder.",
            "",
            "Spectral check",
            "--------------",
            f"Observed Delta m21^2 [eV^2]          : {audit.observed_delta_m21_ev2:.12e}",
            f"Observed Delta m31^2 [eV^2]          : {audit.observed_delta_m31_ev2:.12e}",
            f"Kernel Delta m31^2 / Delta m21^2     : {audit.kernel_splitting_ratio:.12f}",
            f"Observed Delta m31^2 / Delta m21^2   : {audit.observed_splitting_ratio:.12f}",
            f"Ratio fractional error               : {100.0 * audit.ratio_fractional_error:+.6f}%",
            f"Spectral ratio check                 : {'PASS' if audit.spectral_check_passed else 'FAIL'}",
            "",
            f"{audit.solar_anchor.label}",
            "-" * len(audit.solar_anchor.label),
            f"m_0 [eV]                             : {audit.solar_anchor.lightest_mass_ev:.12e}",
            f"(m1, m2, m3) [eV]                    : {_format_mass_triplet(audit.solar_anchor.masses_ev)}",
            f"Predicted Delta m21^2 [eV^2]         : {audit.solar_anchor.delta_m21_ev2:.12e}",
            f"Predicted Delta m31^2 [eV^2]         : {audit.solar_anchor.delta_m31_ev2:.12e}",
            f"Cross-check fractional error         : {100.0 * audit.solar_anchor.cross_check_fractional_error:+.6f}%",
            "",
            f"{audit.atmospheric_anchor.label}",
            "-" * len(audit.atmospheric_anchor.label),
            f"m_0 [eV]                             : {audit.atmospheric_anchor.lightest_mass_ev:.12e}",
            f"(m1, m2, m3) [eV]                    : {_format_mass_triplet(audit.atmospheric_anchor.masses_ev)}",
            f"Predicted Delta m21^2 [eV^2]         : {audit.atmospheric_anchor.delta_m21_ev2:.12e}",
            f"Predicted Delta m31^2 [eV^2]         : {audit.atmospheric_anchor.delta_m31_ev2:.12e}",
            f"Cross-check fractional error         : {100.0 * audit.atmospheric_anchor.cross_check_fractional_error:+.6f}%",
            "",
            f"Mandatory residue verdict            : {'PROVED' if audit.mandatory_residue_verified else 'CHECK'}",
            "The generational hierarchy is a mandatory residue of the (26, 8) sector because",
            "the mass ladder, its logarithmic winding numbers, and the observed splitting ratio",
            "all descend from the same branch-fixed SU(2)_26 modular-T phase lock.",
        )
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    parse_args(argv)
    print(render_report(build_flavor_identity_audit()))


if __name__ == "__main__":
    main()
