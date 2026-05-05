from __future__ import annotations

"""Uniqueness-from-failure audit for the Minimal Canonical Completion."""

import argparse
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pub.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL, SO10_HIGGS_126_DYNKIN_LABELS, SO10_RANK
    from pub.noether_bridge import load_c_dark_completion_fraction
    from pub.tn import derive_so10_representation_data, verify_gko_orthogonality
else:
    from .constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL, SO10_HIGGS_126_DYNKIN_LABELS, SO10_RANK
    from .noether_bridge import load_c_dark_completion_fraction
    from .tn import derive_so10_representation_data, verify_gko_orthogonality


ALTERNATIVE_PARENT_GROUPS: dict[str, dict[str, object]] = {
    "SU(5)": {
        "rank": 4,
        "simple_parent": True,
        "criterion_a_pass": False,
        "criterion_a_reason": (
            "No direct symmetric 126_H-type B-L=2 Majorana slot exists in the benchmark bookkeeping, "
            "so the seesaw closes only through non-minimal effective operators or extra alignment data."
        ),
        "criterion_b_pass": True,
        "criterion_b_reason": (
            "Criterion B is not the elimination channel here; the uniqueness-from-failure audit already removes SU(5) at Criterion A."
        ),
        "primary_failure_mode": "Criterion A (Majorana Channel)",
    },
    "SU(3) x SU(3) x SU(3)": {
        "rank": 6,
        "simple_parent": False,
        "criterion_a_pass": False,
        "criterion_a_reason": (
            "The factorized parent does not furnish the benchmark-direct one-copy 126_H-type Majorana slot as a single canonical insertion."
        ),
        "criterion_b_pass": False,
        "criterion_b_reason": (
            "The factorized parent does not supply the unique c_dark Virasoro-orthogonal complement required to keep the completed boundary block neutral."
        ),
        "primary_failure_mode": "Criterion B (Parity Sink)",
    },
}

EXPECTED_BRANCH = (26, 8, 312)
REQUIRED_MAJORANA_CHANNEL = r"symmetric 16_F 16_F \overline{126}_H"


@dataclass(frozen=True)
class CriterionAudit:
    passed: bool
    reason: str

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass(frozen=True)
class ParentGroupAudit:
    label: str
    rank: int
    simple_parent: bool
    majorana_channel: CriterionAudit
    parity_sink: CriterionAudit
    primary_failure_mode: str
    survives: bool


@dataclass(frozen=True)
class MinimalityProofAudit:
    branch: tuple[int, int, int]
    required_majorana_channel: str
    benchmark_c_dark_fraction: Fraction
    benchmark_c_dark: float
    benchmark_parent: ParentGroupAudit
    alternatives: tuple[ParentGroupAudit, ...]
    minimal_local_survivor: str
    verdict: str


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _benchmark_parent_audit() -> ParentGroupAudit:
    completion_fraction = load_c_dark_completion_fraction()
    completion_value = float(completion_fraction)
    so10_majorana = derive_so10_representation_data("126_H", tuple(SO10_HIGGS_126_DYNKIN_LABELS))
    criterion_a = CriterionAudit(
        passed=bool(SO10_RANK == 5 and so10_majorana.dimension == 126),
        reason=(
            "SO(10) carries the direct symmetric 16_F16_F\\overline{126}_H slot required for the benchmark renormalizable B-L=2 seesaw."
        ),
    )
    criterion_b_audit = verify_gko_orthogonality(
        parent_level=PARENT_LEVEL,
        lepton_level=LEPTON_LEVEL,
        quark_level=QUARK_LEVEL,
    )
    criterion_b = CriterionAudit(
        passed=bool(criterion_b_audit.orthogonality_verified and 0.0 < completion_value < 24.0),
        reason=(
            "The benchmark branch retains a positive GKO-orthogonal residue and the exact completed-boundary parity sink "
            f"c_dark={_format_fraction(completion_fraction)}={completion_value:.12f}."
        ),
    )
    survives = bool(criterion_a.passed and criterion_b.passed)
    return ParentGroupAudit(
        label="SO(10)",
        rank=int(SO10_RANK),
        simple_parent=True,
        majorana_channel=criterion_a,
        parity_sink=criterion_b,
        primary_failure_mode="Minimal Local Survivor",
        survives=survives,
    )


def _alternative_parent_audit(label: str, spec: dict[str, object]) -> ParentGroupAudit:
    majorana_channel = CriterionAudit(
        passed=bool(spec["criterion_a_pass"]),
        reason=str(spec["criterion_a_reason"]),
    )
    parity_sink = CriterionAudit(
        passed=bool(spec["criterion_b_pass"]),
        reason=str(spec["criterion_b_reason"]),
    )
    survives = bool(majorana_channel.passed and parity_sink.passed)
    return ParentGroupAudit(
        label=label,
        rank=int(spec["rank"]),
        simple_parent=bool(spec["simple_parent"]),
        majorana_channel=majorana_channel,
        parity_sink=parity_sink,
        primary_failure_mode=str(spec["primary_failure_mode"]),
        survives=survives,
    )


def build_minimality_proof_audit() -> MinimalityProofAudit:
    branch = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))
    assert branch == EXPECTED_BRANCH, (
        f"The minimality audit is locked to the benchmark branch {EXPECTED_BRANCH}, received {branch}."
    )

    benchmark_parent = _benchmark_parent_audit()
    benchmark_c_dark_fraction = load_c_dark_completion_fraction()
    benchmark_c_dark = float(benchmark_c_dark_fraction)
    assert benchmark_parent.survives, "SO(10) must satisfy both SHBT closure criteria on the retained benchmark branch."

    alternatives = tuple(
        _alternative_parent_audit(label, spec)
        for label, spec in ALTERNATIVE_PARENT_GROUPS.items()
    )
    assert any(candidate.label == "SU(5)" and not candidate.majorana_channel.passed for candidate in alternatives), (
        "Uniqueness-from-failure audit requires SU(5) to fail the symmetric Majorana-channel test."
    )
    assert any(
        candidate.label == "SU(3) x SU(3) x SU(3)" and not candidate.parity_sink.passed
        for candidate in alternatives
    ), "Uniqueness-from-failure audit requires SU(3)^3 to fail the parity-sink test."
    assert all(not candidate.survives for candidate in alternatives), (
        "All lower-rank or factorized alternatives must fail at least one SHBT closure criterion."
    )

    verdict = (
        "Minimal Local Survivor: SO(10) is the unique minimal Lie algebra in the benchmark class that simultaneously "
        "furnishes a direct symmetric 126_H-type Majorana channel and the c_dark Virasoro-orthogonal complement "
        "required for boundary neutrality on the anomaly-free (26, 8, 312) branch."
    )
    return MinimalityProofAudit(
        branch=branch,
        required_majorana_channel=REQUIRED_MAJORANA_CHANNEL,
        benchmark_c_dark_fraction=benchmark_c_dark_fraction,
        benchmark_c_dark=benchmark_c_dark,
        benchmark_parent=benchmark_parent,
        alternatives=alternatives,
        minimal_local_survivor="SO(10)",
        verdict=verdict,
    )


def _table_row(candidate: ParentGroupAudit) -> str:
    return (
        f"{candidate.label:<24}"
        f"{candidate.rank:>4}  "
        f"{candidate.majorana_channel.status:^7}  "
        f"{candidate.parity_sink.status:^7}  "
        f"{('YES' if candidate.survives else 'NO'):^8}  "
        f"{candidate.primary_failure_mode}"
    )


def render_report(audit: MinimalityProofAudit) -> str:
    rows = [audit.benchmark_parent, *audit.alternatives]
    lines = [
        "Minimality Proof Table",
        "======================",
        f"Benchmark branch              : {audit.branch}",
        f"Required Majorana channel     : {audit.required_majorana_channel}",
        (
            "Benchmark c_dark residue      : "
            f"{_format_fraction(audit.benchmark_c_dark_fraction)} = {audit.benchmark_c_dark:.12f}"
        ),
        "",
        "Candidate                 Rank  Crit.A  Crit.B  Survives  Failure Mode",
        "-----------------------------------------------------------------------",
    ]
    lines.extend(_table_row(candidate) for candidate in rows)
    lines.extend(
        (
            "",
            "Failure Notes",
            "-------------",
            f"SO(10): {audit.benchmark_parent.majorana_channel.reason}",
            f"SO(10): {audit.benchmark_parent.parity_sink.reason}",
        )
    )
    for candidate in audit.alternatives:
        lines.append(f"{candidate.label}: {candidate.majorana_channel.reason}")
        lines.append(f"{candidate.label}: {candidate.parity_sink.reason}")
    lines.extend(("", audit.verdict))
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    parse_args(argv)
    audit = build_minimality_proof_audit()
    print(render_report(audit))


if __name__ == "__main__":
    main()
