from __future__ import annotations

"""Publication-facing proton-stability audit for the SHBT benchmark branch."""

import argparse
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pub.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
    from pub.noether_bridge import framing_defect
else:
    from .constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
    from .noether_bridge import framing_defect


EXPECTED_BRANCH = (26, 8, 312)
PROTON_BOUNDARY_PIXEL_SCALE_GEV = 0.93827208816
HBAR_GEV_SECONDS = 6.582119569e-25
SECONDS_PER_JULIAN_YEAR = 365.25 * 24.0 * 60.0 * 60.0
_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_LATEX_SCIENTIFIC_PATTERN = re.compile(r"^(?P<coefficient>-?\d+(?:\.\d+)?)\\times10\^\{(?P<exponent>-?\d+)\}$")
_MACRO_PREFIX = "\\newcommand{\\"


@dataclass(frozen=True)
class PublicationLifetimeConstants:
    alpha_gut_inverse: float
    effective_gauge_mass_gev: float
    published_dimension_six_lifetime_years: float
    rapid_dimension_five_ceiling_years: float


@dataclass(frozen=True)
class PublishedProtonWindowRow:
    branch: tuple[int, int, int]
    published_framing_gap: float
    published_lifetime_label: str
    triple_lock_label: str


@dataclass(frozen=True)
class ProtonStabilityCellAudit:
    branch: tuple[int, int, int]
    delta_fr: Fraction
    published_framing_gap: float
    theorem_closed: bool
    dimension_five_suppressed: bool
    protected_dimension_six_lifetime_years: float | None
    rapid_dimension_five_ceiling_years: float | None
    published_lifetime_label: str
    triple_lock_label: str

    @property
    def verdict(self) -> str:
        return "PROTECTED" if self.dimension_five_suppressed else "RAPID d=5 LEAKAGE"

    @property
    def dominant_lifetime_text(self) -> str:
        if self.dimension_five_suppressed:
            protected_lifetime = self.protected_dimension_six_lifetime_years
            assert protected_lifetime is not None, "Protected branches must carry the d=6 reference lifetime."
            return f"{protected_lifetime:.2e} yr (protected d=6 holographic channel)"
        rapid_ceiling = self.rapid_dimension_five_ceiling_years
        assert rapid_ceiling is not None, "Off-shell branches must carry the rapid d=5 decay ceiling."
        return f"<{rapid_ceiling:.2e} yr (rapid d=5 leakage)"


@dataclass(frozen=True)
class ProtonStabilityAudit:
    benchmark_branch: tuple[int, int, int]
    fixed_parent_window: tuple[int, ...]
    theorem_prerequisite: str
    publication: PublicationLifetimeConstants
    reconstructed_dimension_six_lifetime_years: float
    cells: tuple[ProtonStabilityCellAudit, ...]
    unique_protector: tuple[int, int, int]
    unique_protector_count: int
    verdict: str

    @property
    def benchmark_cell(self) -> ProtonStabilityCellAudit:
        return next(cell for cell in self.cells if cell.branch == self.benchmark_branch)

    @property
    def off_shell_cells(self) -> tuple[ProtonStabilityCellAudit, ...]:
        return tuple(cell for cell in self.cells if cell.branch != self.benchmark_branch)

    @property
    def publication_reconstruction_matches(self) -> bool:
        return math.isclose(
            self.reconstructed_dimension_six_lifetime_years,
            self.publication.published_dimension_six_lifetime_years,
            rel_tol=2.0e-2,
            abs_tol=0.0,
        )


def _extract_number(cell: str) -> float:
    match = _NUMBER_PATTERN.search(cell)
    if match is None:
        raise RuntimeError(f"Failed to parse numeric cell from {cell!r}.")
    return float(match.group(0))


def _extract_integer(cell: str) -> int:
    return int(_extract_number(cell))


def _format_branch(branch: tuple[int, int, int]) -> str:
    return f"({branch[0]}, {branch[1]}, {branch[2]})"


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


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


def _published_rapid_decay_ceiling_years() -> float:
    supplementary_path = Path(__file__).with_name("supplementary.tex")
    for raw_line in supplementary_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if "rapid $d=5$ leakage" not in line:
            continue
        match = re.search(r"10\^\{(?P<exponent>-?\d+)\}", line)
        if match is None:
            break
        exponent = int(match.group("exponent"))
        return 10.0 ** exponent
    raise RuntimeError("Failed to extract the rapid d=5 leakage ceiling from supplementary.tex.")


@lru_cache(maxsize=1)
def load_publication_lifetime_constants() -> PublicationLifetimeConstants:
    macros = _load_physics_constant_macros()
    try:
        alpha_gut_inverse = _parse_latex_number(macros["alphaGutInverse"])
        effective_gauge_mass_gev = _parse_latex_number(macros["protonGaugeMassGeV"])
        published_dimension_six_lifetime_years = _parse_latex_number(macros["protonLifetimeYears"])
    except KeyError as exc:
        raise RuntimeError(f"Missing publication proton-stability macro: {exc.args[0]}") from exc
    return PublicationLifetimeConstants(
        alpha_gut_inverse=alpha_gut_inverse,
        effective_gauge_mass_gev=effective_gauge_mass_gev,
        published_dimension_six_lifetime_years=published_dimension_six_lifetime_years,
        rapid_dimension_five_ceiling_years=_published_rapid_decay_ceiling_years(),
    )


def compute_dimension_six_reference_lifetime_years(*, alpha_gut_inverse: float, effective_gauge_mass_gev: float) -> float:
    alpha_gut = 1.0 / alpha_gut_inverse
    dimension_six_width_gev = (
        alpha_gut
        * alpha_gut
        * PROTON_BOUNDARY_PIXEL_SCALE_GEV**5
        / effective_gauge_mass_gev**4
    )
    return HBAR_GEV_SECONDS / (dimension_six_width_gev * SECONDS_PER_JULIAN_YEAR)


@lru_cache(maxsize=1)
def _published_proton_window_rows() -> tuple[PublishedProtonWindowRow, ...]:
    table_path = Path(__file__).with_name("reviewer_audit_packet").joinpath("uniqueness_scan_table.tex")
    rows: list[PublishedProtonWindowRow] = []
    for raw_line in table_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.count("&") < 11 or not line.endswith(r"\\"):
            continue
        cells = [cell.strip() for cell in line[:-2].split("&")]
        if len(cells) < 12:
            continue
        if not all(_NUMBER_PATTERN.search(cells[index]) for index in (0, 1, 2, 5)):
            continue
        rows.append(
            PublishedProtonWindowRow(
                branch=(
                    _extract_integer(cells[0]),
                    _extract_integer(cells[1]),
                    _extract_integer(cells[2]),
                ),
                published_framing_gap=_extract_number(cells[5]),
                published_lifetime_label=cells[9],
                triple_lock_label=cells[11],
            )
        )
    if not rows:
        raise RuntimeError("Failed to parse the reviewer proton-stability moat rows.")
    return tuple(sorted(rows, key=lambda row: row.branch))


def build_proton_stability_audit() -> ProtonStabilityAudit:
    benchmark_branch = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))
    assert benchmark_branch == EXPECTED_BRANCH, (
        f"The proton-stability audit is locked to the benchmark branch {EXPECTED_BRANCH}, received {benchmark_branch}."
    )

    publication = load_publication_lifetime_constants()
    reconstructed_dimension_six_lifetime_years = compute_dimension_six_reference_lifetime_years(
        alpha_gut_inverse=publication.alpha_gut_inverse,
        effective_gauge_mass_gev=publication.effective_gauge_mass_gev,
    )
    assert math.isclose(
        reconstructed_dimension_six_lifetime_years,
        publication.published_dimension_six_lifetime_years,
        rel_tol=2.0e-2,
        abs_tol=0.0,
    ), "The publication-facing d=6 lifetime reconstruction drifted away from the checked-in manuscript value."

    published_rows = _published_proton_window_rows()
    fixed_parent_window = tuple(row.branch[0] for row in published_rows)
    expected_window = tuple(range(max(2, benchmark_branch[0] - 2), benchmark_branch[0] + 3))
    assert fixed_parent_window == expected_window, (
        f"Expected the proton moat window {expected_window}, received {fixed_parent_window}."
    )

    cells: list[ProtonStabilityCellAudit] = []
    for row in published_rows:
        lepton_level, quark_level, parent_level = row.branch
        defect = framing_defect(parent_level=parent_level, lepton_level=lepton_level, quark_level=quark_level)
        theorem_closed = defect.delta_fr == 0
        normalized_lifetime_label = " ".join(row.published_lifetime_label.split())
        normalized_triple_lock_label = " ".join(row.triple_lock_label.split()).lower()

        assert math.isclose(float(defect.delta_fr), row.published_framing_gap, rel_tol=0.0, abs_tol=1.0e-6), (
            f"Published Delta_fr drift for branch {row.branch}: computed {float(defect.delta_fr):.12f}, "
            f"published {row.published_framing_gap:.12f}."
        )

        if theorem_closed:
            assert row.branch == benchmark_branch, "Only the benchmark branch may close Delta_fr=0 in the fixed-parent moat."
            assert "3.86" in normalized_lifetime_label and "10^{36}" in normalized_lifetime_label, (
                f"Protected branch {row.branch} lost the publication lifetime label {row.published_lifetime_label!r}."
            )
            assert "locked" in normalized_triple_lock_label, (
                f"Protected branch {row.branch} must remain triple-locked, received {row.triple_lock_label!r}."
            )
            cells.append(
                ProtonStabilityCellAudit(
                    branch=row.branch,
                    delta_fr=defect.delta_fr,
                    published_framing_gap=row.published_framing_gap,
                    theorem_closed=True,
                    dimension_five_suppressed=True,
                    protected_dimension_six_lifetime_years=publication.published_dimension_six_lifetime_years,
                    rapid_dimension_five_ceiling_years=None,
                    published_lifetime_label=row.published_lifetime_label,
                    triple_lock_label=row.triple_lock_label,
                )
            )
            continue

        assert "10^{34}" in normalized_lifetime_label and "rapid $d=5$" in normalized_lifetime_label, (
            f"Off-shell branch {row.branch} lost the rapid d=5 lifetime label {row.published_lifetime_label!r}."
        )
        assert "broken" in normalized_triple_lock_label, (
            f"Off-shell branch {row.branch} must remain broken, received {row.triple_lock_label!r}."
        )
        cells.append(
            ProtonStabilityCellAudit(
                branch=row.branch,
                delta_fr=defect.delta_fr,
                published_framing_gap=row.published_framing_gap,
                theorem_closed=False,
                dimension_five_suppressed=False,
                protected_dimension_six_lifetime_years=None,
                rapid_dimension_five_ceiling_years=publication.rapid_dimension_five_ceiling_years,
                published_lifetime_label=row.published_lifetime_label,
                triple_lock_label=row.triple_lock_label,
            )
        )

    audited_cells = tuple(cells)
    unique_protectors = tuple(cell.branch for cell in audited_cells if cell.dimension_five_suppressed)
    assert unique_protectors == (benchmark_branch,), (
        "The fixed-parent proton-stability audit requires a single protector: the anomaly-free branch (26, 8, 312)."
    )
    assert all(cell.delta_fr != 0 for cell in audited_cells if cell.branch != benchmark_branch), (
        "Every off-shell cell must reopen the framing defect and therefore fail the d=5 suppression audit."
    )

    verdict = (
        "Baryon Stability Verdict: PASS — "
        f"{_format_branch(benchmark_branch)} is the unique protector of matter on the fixed-parent K=312 moat. "
        "It alone satisfies the theorem-level prerequisite Delta_fr=0, suppresses the d=5 channel, and retains "
        f"the protected d=6 holographic reference lifetime tau_p≈{publication.published_dimension_six_lifetime_years:.2e} yr; "
        f"every off-shell neighbor has Delta_fr!=0 and falls into the rapid-decay regime tau_p<"
        f"{publication.rapid_dimension_five_ceiling_years:.2e} yr."
    )
    return ProtonStabilityAudit(
        benchmark_branch=benchmark_branch,
        fixed_parent_window=fixed_parent_window,
        theorem_prerequisite="Delta_fr = 0",
        publication=publication,
        reconstructed_dimension_six_lifetime_years=reconstructed_dimension_six_lifetime_years,
        cells=audited_cells,
        unique_protector=benchmark_branch,
        unique_protector_count=len(unique_protectors),
        verdict=verdict,
    )


def render_report(audit: ProtonStabilityAudit) -> str:
    lines = [
        "Proton Stability Audit",
        "======================",
        f"Benchmark branch               : {_format_branch(audit.benchmark_branch)}",
        f"Fixed-parent moat (k_l)        : {audit.fixed_parent_window}",
        f"Theorem prerequisite           : {audit.theorem_prerequisite}",
        f"Published alpha_GUT^-1         : {audit.publication.alpha_gut_inverse:.2f}",
        f"Published M_X [GeV]            : {audit.publication.effective_gauge_mass_gev:.2e}",
        f"Reconstructed tau_p^(d=6) [yr] : {audit.reconstructed_dimension_six_lifetime_years:.2e}",
        f"Published tau_p^(d=6) [yr]     : {audit.publication.published_dimension_six_lifetime_years:.2e}",
        f"Rapid d=5 leakage bound [yr]   : <{audit.publication.rapid_dimension_five_ceiling_years:.2e}",
        "",
        "Branch           Delta_fr   d=5 status        Dominant lifetime",
        "--------------------------------------------------------------------------",
    ]
    for cell in audit.cells:
        status = "suppressed" if cell.dimension_five_suppressed else "open"
        lines.append(
            f"{_format_branch(cell.branch):<16}  {_format_fraction(cell.delta_fr):<8}  {status:<15}  {cell.dominant_lifetime_text}"
        )
    lines.extend(("", audit.verdict))
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    parse_args(argv)
    audit = build_proton_stability_audit()
    print(render_report(audit))


if __name__ == "__main__":
    main()
