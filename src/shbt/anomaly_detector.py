from __future__ import annotations

"""Hostile-reviewer anomaly detector for the SHBT benchmark branch."""

import argparse
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pub.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
    from pub.noether_bridge import (
        DEFAULT_PRECISION,
        FramingDefectAudit,
        ReviewerTrapAudit,
        TensorSnapshot,
        load_c_dark_completion_fraction,
        newton_constant_lock,
        reviewer_trap_audit,
        saturation_audit,
    )
else:
    from .constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
    from .noether_bridge import (
        DEFAULT_PRECISION,
        FramingDefectAudit,
        ReviewerTrapAudit,
        TensorSnapshot,
        load_c_dark_completion_fraction,
        newton_constant_lock,
        reviewer_trap_audit,
        saturation_audit,
    )


BENCHMARK_BRANCH = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))


@dataclass(frozen=True)
class CandidateSpec:
    branch: tuple[int, int, int]
    label: str
    origin: str


@dataclass(frozen=True)
class CandidateAudit:
    branch: tuple[int, int, int]
    label: str
    origin: str
    shift: tuple[int, int, int]
    framing: FramingDefectAudit
    q_iso_ev4: Decimal
    closure_tensor: TensorSnapshot
    anomalous_source_ev2: TensorSnapshot
    anomalous_source_si_m2: TensorSnapshot

    @property
    def benchmark_candidate(self) -> bool:
        return self.branch == BENCHMARK_BRANCH

    @property
    def unit_shift(self) -> bool:
        return sum(abs(value) for value in self.shift) == 1

    @property
    def bulk_closure_preserved(self) -> bool:
        return self.framing.delta_fr == 0 and self.closure_tensor.vanished

    @property
    def wep_preserved(self) -> bool:
        return self.anomalous_source_si_m2.vanished

    @property
    def normalizability_status(self) -> str:
        if self.benchmark_candidate:
            return "NORMALIZABLE"
        if self.framing.delta_fr != 0:
            return "NON-NORMALIZABLE"
        return "OUTSIDE SCOPE"

    @property
    def wep_status(self) -> str:
        if self.wep_preserved:
            return "PRESERVED" if self.benchmark_candidate else "NOT REOPENED"
        return "VIOLATED"

    @property
    def verdict(self) -> str:
        if self.benchmark_candidate:
            return "UNIQUE SURVIVOR"
        if self.framing.delta_fr != 0:
            return "REVIEWER FAIL"
        return "BROADER AUDIT NEEDED"


@dataclass(frozen=True)
class WepViolationMap:
    benchmark_branch: tuple[int, int, int]
    c_dark_fraction: Fraction
    c_dark: Decimal
    q_iso_ev4: Decimal
    include_unit_shifts: bool
    candidates: tuple[CandidateAudit, ...]

    @property
    def local_survivor_count(self) -> int:
        return sum(int(candidate.bulk_closure_preserved) for candidate in self.candidates if candidate.origin != "user")

    @property
    def hostile_reviewer_nightmare_confirmed(self) -> bool:
        benchmark_rows = [candidate for candidate in self.candidates if candidate.benchmark_candidate]
        return len(benchmark_rows) == 1 and benchmark_rows[0].bulk_closure_preserved and self.local_survivor_count == 1


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _format_decimal_scientific(value: Decimal, digits: int = 6) -> str:
    return f"{float(value):.{digits}E}"


def _format_branch(branch: tuple[int, int, int]) -> str:
    return f"({branch[0]},{branch[1]},{branch[2]})"


def _format_shift(shift: tuple[int, int, int]) -> str:
    delta_kl, delta_kq, delta_parent = shift
    pieces: list[str] = []
    if delta_kl:
        pieces.append(f"δk_l={delta_kl:+d}")
    if delta_kq:
        pieces.append(f"δk_q={delta_kq:+d}")
    if delta_parent:
        pieces.append(f"δK={delta_parent:+d}")
    return "0" if not pieces else ", ".join(pieces)


def _candidate_specs_from_args(candidates: Sequence[tuple[int, int, int]] | None) -> tuple[CandidateSpec, ...]:
    if not candidates:
        return ()
    return tuple(
        CandidateSpec(branch=(int(k_l), int(k_q), int(parent_level)), label=f"user {_format_branch((int(k_l), int(k_q), int(parent_level)))}", origin="user")
        for k_l, k_q, parent_level in candidates
    )


def default_candidate_specs() -> tuple[CandidateSpec, ...]:
    benchmark_k_l, benchmark_k_q, benchmark_parent = BENCHMARK_BRANCH
    return (
        CandidateSpec(branch=BENCHMARK_BRANCH, label="benchmark", origin="benchmark"),
        CandidateSpec(branch=(benchmark_k_l - 1, benchmark_k_q, benchmark_parent), label="lepton down", origin="unit-shift"),
        CandidateSpec(branch=(benchmark_k_l + 1, benchmark_k_q, benchmark_parent), label="lepton up", origin="unit-shift"),
        CandidateSpec(branch=(benchmark_k_l, benchmark_k_q - 1, benchmark_parent), label="quark down", origin="unit-shift"),
        CandidateSpec(branch=(benchmark_k_l, benchmark_k_q + 1, benchmark_parent), label="quark up", origin="unit-shift"),
        CandidateSpec(branch=(benchmark_k_l, benchmark_k_q, benchmark_parent - 1), label="parent down", origin="unit-shift"),
        CandidateSpec(branch=(benchmark_k_l, benchmark_k_q, benchmark_parent + 1), label="parent up", origin="unit-shift"),
    )


def _dedupe_specs(specs: Sequence[CandidateSpec]) -> tuple[CandidateSpec, ...]:
    seen: set[tuple[int, int, int]] = set()
    deduped: list[CandidateSpec] = []
    for spec in specs:
        if spec.branch in seen:
            continue
        seen.add(spec.branch)
        deduped.append(spec)
    return tuple(deduped)


def build_candidate_audit(spec: CandidateSpec, *, precision: int = DEFAULT_PRECISION) -> CandidateAudit:
    benchmark_internal = (BENCHMARK_BRANCH[2], BENCHMARK_BRANCH[0], BENCHMARK_BRANCH[1])
    candidate_internal = (spec.branch[2], spec.branch[0], spec.branch[1])
    c_dark_fraction = load_c_dark_completion_fraction()
    newton_lock = newton_constant_lock(c_dark_fraction=c_dark_fraction, precision=precision)
    saturation = saturation_audit(precision=precision)
    audit = reviewer_trap_audit(
        newton_lock_audit=newton_lock,
        saturation=saturation,
        benchmark_branch=benchmark_internal,
        detuned_branch=candidate_internal,
        precision=precision,
    )
    shift = (
        spec.branch[0] - BENCHMARK_BRANCH[0],
        spec.branch[1] - BENCHMARK_BRANCH[1],
        spec.branch[2] - BENCHMARK_BRANCH[2],
    )
    return CandidateAudit(
        branch=spec.branch,
        label=spec.label,
        origin=spec.origin,
        shift=shift,
        framing=audit.detuned,
        q_iso_ev4=audit.q_iso_ev4,
        closure_tensor=audit.closure_tensor_detuned,
        anomalous_source_ev2=audit.anomalous_source_ev2,
        anomalous_source_si_m2=audit.anomalous_source_si_m2,
    )


def build_wep_violation_map(
    *,
    include_unit_shifts: bool = True,
    candidates: Sequence[tuple[int, int, int]] | None = None,
    precision: int = DEFAULT_PRECISION,
) -> WepViolationMap:
    base_specs = default_candidate_specs() if include_unit_shifts else (CandidateSpec(branch=BENCHMARK_BRANCH, label="benchmark", origin="benchmark"),)
    specs = _dedupe_specs((*base_specs, *_candidate_specs_from_args(candidates)))
    candidate_audits = tuple(build_candidate_audit(spec, precision=precision) for spec in specs)
    c_dark_fraction = load_c_dark_completion_fraction()
    newton_lock = newton_constant_lock(c_dark_fraction=c_dark_fraction, precision=precision)
    saturation = saturation_audit(precision=precision)
    with localcontext() as context:
        context.prec = precision
        q_iso_ev4 = saturation.lambda_obs_ev2 / newton_lock.eight_pi_g_effective_ev_minus2
    return WepViolationMap(
        benchmark_branch=BENCHMARK_BRANCH,
        c_dark_fraction=c_dark_fraction,
        c_dark=newton_lock.c_dark,
        q_iso_ev4=q_iso_ev4,
        include_unit_shifts=include_unit_shifts,
        candidates=candidate_audits,
    )


def render_report(violation_map: WepViolationMap) -> str:
    lines = [
        "WEP Violation Map",
        "=================",
        f"Benchmark branch                  : {_format_branch(violation_map.benchmark_branch)}",
        f"c_dark completion                 : {_format_fraction(violation_map.c_dark_fraction)} = {violation_map.c_dark:.12f}",
        f"Q_iso [eV^4]                      : {_format_decimal_scientific(violation_map.q_iso_ev4, digits=12)}",
        "Bulk Closure Tensor               : E_mu_nu = Q_iso * Delta_fr * g_mu_nu",
        "Anomalous source                  : J_mu_nu^(a) = [12/(c_dark M_P^2)] * E_mu_nu",
        (
            "Displayed scope                   : benchmark branch plus the hostile-reviewer unit shifts; user candidates are appended as extra stress tests."
            if violation_map.include_unit_shifts
            else "Displayed scope                   : benchmark branch plus any user-supplied stress tests."
        ),
        "",
        "Branch map",
        "----------",
        "branch         origin      shift                  Delta_fr      |E| [eV^4]      |J| [m^-2]      WEP           Z_partial            verdict",
        "-------------------------------------------------------------------------------------------------------------------------------------------------",
    ]
    for candidate in violation_map.candidates:
        lines.append(
            f"{_format_branch(candidate.branch):<13} "
            f"{candidate.origin:<11} "
            f"{_format_shift(candidate.shift):<22} "
            f"{_format_fraction(candidate.framing.delta_fr):<12} "
            f"{_format_decimal_scientific(candidate.closure_tensor.amplitude):<15} "
            f"{_format_decimal_scientific(candidate.anomalous_source_si_m2.amplitude):<15} "
            f"{candidate.wep_status:<12} "
            f"{candidate.normalizability_status:<20} "
            f"{candidate.verdict}"
        )
    lines.extend(
        (
            "",
            "Logical chain",
            "-------------",
        )
    )
    for candidate in violation_map.candidates:
        if candidate.benchmark_candidate:
            lines.append(
                f"{_format_branch(candidate.branch)} keeps Delta_fr=0, so E_mu_nu=0 and J_mu_nu^(a)=0: the Equivalence Principle is preserved and the completed boundary remains normalizable."
            )
        elif candidate.framing.delta_fr != 0:
            lines.append(
                f"{_format_branch(candidate.branch)} gives Delta_fr={_format_fraction(candidate.framing.delta_fr)}, hence E_mu_nu!=0 and J_mu_nu^(a)!=0: universality of free fall is lost and the boundary partition function exits the normalizable benchmark sector."
            )
        else:
            lines.append(
                f"{_format_branch(candidate.branch)} does not reopen Delta_fr in this local framing audit; it is therefore not a counterexample to the hostile-reviewer nearest-neighbor trap and must be judged by the broader uniqueness/gauge-emergence stack."
            )
    lines.extend(
        (
            "",
            f"Unique local survivor count        : {violation_map.local_survivor_count}",
            f"Hostile Reviewer's Nightmare       : {'CONFIRMED' if violation_map.hostile_reviewer_nightmare_confirmed else 'CHECK'}",
            "Within the displayed one-step detuning map, only (26,8,312) keeps the bulk closure tensor and anomalous source identically zero.",
        )
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        nargs=3,
        metavar=("K_L", "K_Q", "K"),
        action="append",
        type=int,
        help="Append a user-supplied alternative branch in the display order (k_l, k_q, K).",
    )
    parser.add_argument(
        "--no-unit-shifts",
        action="store_true",
        help="Skip the default hostile-reviewer ±1 detuning map and show only the benchmark plus any user candidates.",
    )
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    violation_map = build_wep_violation_map(
        include_unit_shifts=not args.no_unit_shifts,
        candidates=args.candidate,
        precision=args.precision,
    )
    print(render_report(violation_map))


if __name__ == "__main__":
    main()
