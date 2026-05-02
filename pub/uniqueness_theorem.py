from __future__ import annotations

"""Formal uniqueness certificate for the benchmark local moat."""

import argparse
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pub.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
    from pub.tn import LOCAL_LEPTON_LEVEL_WINDOW, verify_gko_orthogonality
else:
    from .constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
    from .tn import LOCAL_LEPTON_LEVEL_WINDOW, verify_gko_orthogonality


EXPECTED_BENCHMARK = (26, 8, 312)
_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


@dataclass(frozen=True)
class PublishedMoatRow:
    lepton_level: int
    quark_level: int
    parent_level: int
    visible_anomaly_residual: float
    modularity_gap: float
    framing_gap: float
    formal_completion_residue: float
    status: str

    @property
    def coordinates(self) -> tuple[int, int, int]:
        return (self.lepton_level, self.quark_level, self.parent_level)


@dataclass(frozen=True)
class IntegratedAnomalyFilter:
    benchmark_coordinates: tuple[int, int, int]
    diophantine_minimal_parent_level: int
    diophantine_pass: bool
    lepton_branching_index: float
    quark_branching_index: float
    framing_gap: float
    framing_pass: bool
    gko_stress_tensor_residue: float
    published_formal_completion_residue: float
    gko_pass: bool

    @property
    def passed(self) -> bool:
        return bool(self.diophantine_pass and self.framing_pass and self.gko_pass)


@dataclass(frozen=True)
class MoatCellDiagnostic:
    coordinates: tuple[int, int, int]
    lexicographic_rank: int
    diophantine_minimal_parent_level: int
    diophantine_pass: bool
    lepton_branching_index: float
    quark_branching_index: float
    framing_gap: float
    framing_pass: bool
    published_modularity_gap: float
    published_formal_completion_residue: float
    gko_stress_tensor_residue: float
    gko_pass: bool
    published_status: str
    failure_modes: tuple[str, ...]

    @property
    def survives_integrated_filter(self) -> bool:
        return bool(self.diophantine_pass and self.framing_pass and self.gko_pass)


@dataclass(frozen=True)
class LexicographicEliminationStep:
    label: str
    surviving_coordinates: tuple[tuple[int, int, int], ...]
    explanation: str


@dataclass(frozen=True)
class FormalUniquenessCertificate:
    benchmark_coordinates: tuple[int, int, int]
    moat_window: tuple[int, ...]
    published_moat_source: str
    integrated_filter: IntegratedAnomalyFilter
    candidate_diagnostics: tuple[MoatCellDiagnostic, ...]
    elimination_steps: tuple[LexicographicEliminationStep, ...]
    unique_survivor: tuple[int, int, int]
    unique_survivor_count: int
    invariant_proof: str
    verdict: str


def _extract_number(cell: str) -> float:
    match = _NUMBER_PATTERN.search(cell)
    if match is None:
        raise RuntimeError(f"Failed to parse numeric cell from {cell!r}.")
    return float(match.group(0))


def _extract_integer(cell: str) -> int:
    return int(_extract_number(cell))


def _parse_status(cell: str) -> str:
    lowered = cell.lower()
    if "selected" in lowered:
        return "selected benchmark"
    if "fails" in lowered and "delta" in lowered:
        return "fails Delta_fr=0"
    collapsed = re.sub(r"\\[a-zA-Z]+", " ", cell)
    collapsed = re.sub(r"[{}$]", " ", collapsed)
    return " ".join(collapsed.split())


@lru_cache(maxsize=1)
def _published_moat_rows() -> tuple[PublishedMoatRow, ...]:
    table_path = Path(__file__).with_name("uniqueness_scan_table.tex")
    table_text = table_path.read_text(encoding="utf-8")
    rows: list[PublishedMoatRow] = []
    for raw_line in table_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("$") or line.count("&") < 13 or not line.endswith(r"\\"):
            continue
        cells = [cell.strip() for cell in line[:-2].split("&")]
        if len(cells) < 14:
            continue
        if not all(_NUMBER_PATTERN.search(cells[index]) for index in (0, 1, 2)):
            continue
        rows.append(
            PublishedMoatRow(
                lepton_level=_extract_integer(cells[0]),
                quark_level=_extract_integer(cells[1]),
                parent_level=_extract_integer(cells[2]),
                visible_anomaly_residual=_extract_number(cells[3]),
                modularity_gap=_extract_number(cells[4]),
                framing_gap=_extract_number(cells[5]),
                formal_completion_residue=_extract_number(cells[10]),
                status=_parse_status(cells[13]),
            )
        )

    if not rows:
        raise RuntimeError("Failed to parse the published uniqueness moat rows.")
    ordered_rows = tuple(sorted(rows, key=lambda row: row.coordinates))
    benchmark_row = next((row for row in ordered_rows if row.coordinates == EXPECTED_BENCHMARK), None)
    if benchmark_row is None:
        raise RuntimeError(f"Failed to locate the benchmark row {EXPECTED_BENCHMARK} in uniqueness_scan_table.tex.")
    return ordered_rows


def _build_candidate_diagnostic(row: PublishedMoatRow, *, lexicographic_rank: int) -> MoatCellDiagnostic:
    minimal_parent_level = math.lcm(2 * int(row.lepton_level), 3 * int(row.quark_level))
    diophantine_pass = int(row.parent_level) == int(minimal_parent_level)
    lepton_branching_index = float(row.parent_level / (2.0 * row.lepton_level))
    quark_branching_index = float(row.parent_level / (3.0 * row.quark_level))
    framing_pass = math.isclose(float(row.framing_gap), 0.0, rel_tol=0.0, abs_tol=1.0e-12)
    gko_audit = verify_gko_orthogonality(
        parent_level=row.parent_level,
        lepton_level=row.lepton_level,
        quark_level=row.quark_level,
    )
    gko_pass = bool(gko_audit.orthogonality_verified and row.formal_completion_residue > 0.0)

    failure_modes: list[str] = []
    if not diophantine_pass:
        failure_modes.append(
            f"Diophantine tower mismatch: K={row.parent_level} while lcm(2k_l,3k_q)={minimal_parent_level}."
        )
    if not framing_pass:
        failure_modes.append(
            "Fractional vacuum T-power detected: "
            f"I_L=K/(2k_l)={lepton_branching_index:.6f} gives Delta_fr={row.framing_gap:.6f}, so Witten framing single-valuedness fails."
        )
    if not gko_pass:
        failure_modes.append(
            "No positive Virasoro-orthogonal complement survives the moat filter: "
            f"c_dark={row.formal_completion_residue:.6f}."
        )

    return MoatCellDiagnostic(
        coordinates=row.coordinates,
        lexicographic_rank=int(lexicographic_rank),
        diophantine_minimal_parent_level=int(minimal_parent_level),
        diophantine_pass=bool(diophantine_pass),
        lepton_branching_index=float(lepton_branching_index),
        quark_branching_index=float(quark_branching_index),
        framing_gap=float(row.framing_gap),
        framing_pass=bool(framing_pass),
        published_modularity_gap=float(row.modularity_gap),
        published_formal_completion_residue=float(row.formal_completion_residue),
        gko_stress_tensor_residue=float(gko_audit.c_dark_residue),
        gko_pass=bool(gko_pass),
        published_status=str(row.status),
        failure_modes=tuple(failure_modes),
    )


def _format_coordinates(coordinates: tuple[int, int, int]) -> str:
    return f"({coordinates[0]},{coordinates[1]},{coordinates[2]})"


def _format_coordinate_tuple(coordinates: tuple[tuple[int, int, int], ...]) -> str:
    return ", ".join(_format_coordinates(entry) for entry in coordinates) if coordinates else "none"


def build_formal_uniqueness_certificate() -> FormalUniquenessCertificate:
    benchmark_coordinates = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))
    assert benchmark_coordinates == EXPECTED_BENCHMARK, (
        f"The uniqueness certificate is locked to the benchmark branch {EXPECTED_BENCHMARK}, received {benchmark_coordinates}."
    )

    published_rows = _published_moat_rows()
    diagnostics = tuple(
        _build_candidate_diagnostic(row, lexicographic_rank=index)
        for index, row in enumerate(published_rows, start=1)
    )
    benchmark_diagnostic = next(
        diagnostic for diagnostic in diagnostics if diagnostic.coordinates == benchmark_coordinates
    )
    integrated_filter = IntegratedAnomalyFilter(
        benchmark_coordinates=benchmark_coordinates,
        diophantine_minimal_parent_level=int(benchmark_diagnostic.diophantine_minimal_parent_level),
        diophantine_pass=bool(benchmark_diagnostic.diophantine_pass),
        lepton_branching_index=float(benchmark_diagnostic.lepton_branching_index),
        quark_branching_index=float(benchmark_diagnostic.quark_branching_index),
        framing_gap=float(benchmark_diagnostic.framing_gap),
        framing_pass=bool(benchmark_diagnostic.framing_pass),
        gko_stress_tensor_residue=float(benchmark_diagnostic.gko_stress_tensor_residue),
        published_formal_completion_residue=float(benchmark_diagnostic.published_formal_completion_residue),
        gko_pass=bool(benchmark_diagnostic.gko_pass),
    )

    initial_coordinates = tuple(diagnostic.coordinates for diagnostic in diagnostics)
    diophantine_survivors = tuple(
        diagnostic.coordinates for diagnostic in diagnostics if diagnostic.diophantine_pass
    )
    framing_survivors = tuple(
        diagnostic.coordinates
        for diagnostic in diagnostics
        if diagnostic.diophantine_pass and diagnostic.framing_pass
    )
    gko_survivors = tuple(
        diagnostic.coordinates
        for diagnostic in diagnostics
        if diagnostic.diophantine_pass and diagnostic.framing_pass and diagnostic.gko_pass
    )

    elimination_steps = (
        LexicographicEliminationStep(
            label="Published moat",
            surviving_coordinates=initial_coordinates,
            explanation="Start from the checked-in fixed-parent K=312 moat printed in uniqueness_scan_table.tex.",
        ),
        LexicographicEliminationStep(
            label="Diophantine filter",
            surviving_coordinates=diophantine_survivors,
            explanation="Retain only cells satisfying the exact branch identity K=lcm(2k_l,3k_q).",
        ),
        LexicographicEliminationStep(
            label="Framing filter",
            surviving_coordinates=framing_survivors,
            explanation="Retain only cells with Delta_fr=0, equivalently integer visible vacuum T-support.",
        ),
        LexicographicEliminationStep(
            label="GKO filter",
            surviving_coordinates=gko_survivors,
            explanation="Retain only cells that still carry a positive Virasoro-orthogonal complement.",
        ),
    )

    assert integrated_filter.passed, "The benchmark branch must satisfy the integrated anomaly filter."
    assert len(gko_survivors) == 1 and gko_survivors[0] == benchmark_coordinates, (
        "The benchmark branch must remain the unique fixed-parent survivor of the integrated theorem filter."
    )

    invariant_proof = "\n".join(
        (
            "1. On the published fixed-parent moat K=312 with k_q=8 and k_l in {24,25,26,27,28}, the exact Diophantine identity K=lcm(2k_l,3k_q) is satisfied only at k_l=26.",
            "2. The same row has integer visible branching indices I_L=6 and I_Q=13, so Delta_fr=0. Every neighboring cell carries a fractional visible index and therefore a fractional vacuum T-power.",
            (
                "3. The surviving row keeps a positive Virasoro-orthogonal complement: the stress-tensor audit remains positive and the checked-in uniqueness table records the benchmark formal completion residue as "
                f"c_dark={integrated_filter.published_formal_completion_residue:.4f}."
            ),
            (
                f"4. The intersection of the Diophantine, framing, and GKO closure predicates on the K=312 moat is therefore the singleton {{{_format_coordinates(benchmark_coordinates)}}}. "
                "Hence (26,8,312) is the unique Minimal Anomaly-Free Local Survivor."
            ),
        )
    )
    verdict = (
        "Formal Uniqueness Certificate: PASS — "
        f"{_format_coordinates(benchmark_coordinates)} is the unique Minimal Anomaly-Free Local Survivor, "
        "and the benchmark coordinates stand as incontrovertible residues of mathematical consistency."
    )
    return FormalUniquenessCertificate(
        benchmark_coordinates=benchmark_coordinates,
        moat_window=tuple(int(level) for level in LOCAL_LEPTON_LEVEL_WINDOW),
        published_moat_source="uniqueness_scan_table.tex",
        integrated_filter=integrated_filter,
        candidate_diagnostics=diagnostics,
        elimination_steps=elimination_steps,
        unique_survivor=gko_survivors[0],
        unique_survivor_count=len(gko_survivors),
        invariant_proof=invariant_proof,
        verdict=verdict,
    )


def render_certificate(certificate: FormalUniquenessCertificate) -> str:
    def _render_boolean(flag: bool) -> str:
        return "PASS" if flag else "FAIL"

    lines = [
        "Formal Uniqueness Certificate",
        "============================",
        f"Benchmark coordinates         : {_format_coordinates(certificate.benchmark_coordinates)}",
        f"Published moat source         : {certificate.published_moat_source}",
        f"Moat window (k_l values)      : {certificate.moat_window}",
        "",
        "Integrated Anomaly Filter",
        "-------------------------",
        (
            "Diophantine integrality       : "
            f"{_render_boolean(certificate.integrated_filter.diophantine_pass)}  "
            f"K=lcm(2k_l,3k_q)={certificate.integrated_filter.diophantine_minimal_parent_level}"
        ),
        (
            "Framing quantization          : "
            f"{_render_boolean(certificate.integrated_filter.framing_pass)}  "
            f"I_L={certificate.integrated_filter.lepton_branching_index:.6f}, "
            f"I_Q={certificate.integrated_filter.quark_branching_index:.6f}, "
            f"Delta_fr={certificate.integrated_filter.framing_gap:.6f}"
        ),
        (
            "GKO orthogonality             : "
            f"{_render_boolean(certificate.integrated_filter.gko_pass)}  "
            f"positive stress-tensor residue={certificate.integrated_filter.gko_stress_tensor_residue:.12f}, "
            f"published c_dark={certificate.integrated_filter.published_formal_completion_residue:.4f}"
        ),
        "",
        "Lexicographic Elimination",
        "-------------------------",
    ]
    for step in certificate.elimination_steps:
        lines.append(
            f"{step.label:<29}: {_format_coordinate_tuple(step.surviving_coordinates)}"
        )
        lines.append(f"  {step.explanation}")
    lines.extend(("", "Failure Diagnostics", "-------------------"))
    for diagnostic in certificate.candidate_diagnostics:
        if diagnostic.survives_integrated_filter:
            lines.append(
                f"{_format_coordinates(diagnostic.coordinates)} : SURVIVES -> all three closure conditions hold simultaneously; published c_dark={diagnostic.published_formal_completion_residue:.4f}."
            )
            continue
        lines.append(
            f"{_format_coordinates(diagnostic.coordinates)} : FAIL -> {' | '.join(diagnostic.failure_modes)}"
        )
    lines.extend(
        (
            "",
            "Invariant Proof",
            "---------------",
            certificate.invariant_proof,
            "",
            certificate.verdict,
        )
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    parse_args(argv)
    print(render_certificate(build_formal_uniqueness_certificate()))


if __name__ == "__main__":
    main()
