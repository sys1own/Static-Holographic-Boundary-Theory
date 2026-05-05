from __future__ import annotations

"""Numerically enforce the reviewer trap on the focused 3 x 3 visible moat.

Each branch is seeded with ``tn.calculate_efe_violation_tensor()`` and then
lifted to the full visible bulk-closure tensor by enforcing the shared visible
framing audit on both ``K/(2k_ell)`` and ``K/(3k_q)``. The trap passes only if
the resulting ``E_mu_nu`` vanishes uniquely at ``(26, 8, 312)`` across the
benchmark-centered 3 x 3 visible-coordinate window with fixed parent level.
"""

import argparse
import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import pub.tn as tn
    from pub.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
    from pub.noether_bridge import (
        DEFAULT_PRECISION,
        TensorSnapshot,
        bulk_closure_tensor,
        framing_defect,
        load_c_dark_completion_fraction,
        newton_constant_lock,
        saturation_audit,
    )
else:
    from . import tn
    from .constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
    from .noether_bridge import (
        DEFAULT_PRECISION,
        TensorSnapshot,
        bulk_closure_tensor,
        framing_defect,
        load_c_dark_completion_fraction,
        newton_constant_lock,
        saturation_audit,
    )


EXPECTED_BENCHMARK = (26, 8, 312)
ZERO_TOLERANCE = 1.0e-15
VISIBLE_MOAT_HALF_WINDOW = 1
VISIBLE_MOAT_DIMENSION = 2 * VISIBLE_MOAT_HALF_WINDOW + 1
VISIBLE_MOAT_CELL_COUNT = VISIBLE_MOAT_DIMENSION * VISIBLE_MOAT_DIMENSION


@dataclass(frozen=True)
class StressWitness:
    branch: tuple[int, int, int]
    shift: tuple[int, int, int]
    lepton_efe_violation: float
    lepton_gap: Fraction
    quark_gap: Fraction
    delta_fr: Fraction
    closure_tensor: TensorSnapshot

    @property
    def benchmark_cell(self) -> bool:
        return self.branch == EXPECTED_BENCHMARK

    @property
    def physically_void(self) -> bool:
        return not self.benchmark_cell and self.delta_fr != 0 and not self.closure_tensor.vanished


@dataclass(frozen=True)
class MoatStressAudit:
    benchmark_branch: tuple[int, int, int]
    q_iso_ev4: Decimal
    witnesses: tuple[StressWitness, ...]

    @property
    def zero_tensor_cells(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(witness.branch for witness in self.witnesses if witness.closure_tensor.vanished)

    @property
    def detuned_witnesses(self) -> tuple[StressWitness, ...]:
        return tuple(witness for witness in self.witnesses if witness.branch != self.benchmark_branch)

    @property
    def strictly_nonzero_detuned_cells(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(witness.branch for witness in self.detuned_witnesses if witness.closure_tensor.amplitude != 0)

    @property
    def reviewer_trap_enforced(self) -> bool:
        return self.zero_tensor_cells == (self.benchmark_branch,) and all(
            witness.physically_void for witness in self.detuned_witnesses
        ) and self.strictly_nonzero_detuned_cells == tuple(
            witness.branch for witness in self.detuned_witnesses
        )


def _benchmark_branch() -> tuple[int, int, int]:
    benchmark = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))
    assert benchmark == EXPECTED_BENCHMARK, (
        f"The moat stress tester is locked to the published branch {EXPECTED_BENCHMARK}, received {benchmark}."
    )
    return benchmark


def _stress_branches(benchmark_branch: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
    lepton_level, quark_level, parent_level = benchmark_branch
    branches = tuple(
        (resolved_lepton_level, resolved_quark_level, parent_level)
        for resolved_lepton_level in range(
            lepton_level - VISIBLE_MOAT_HALF_WINDOW,
            lepton_level + VISIBLE_MOAT_HALF_WINDOW + 1,
        )
        for resolved_quark_level in range(
            quark_level - VISIBLE_MOAT_HALF_WINDOW,
            quark_level + VISIBLE_MOAT_HALF_WINDOW + 1,
        )
    )
    assert len(branches) == VISIBLE_MOAT_CELL_COUNT
    return branches


def _fraction_to_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


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


def _format_fraction(value: Fraction) -> str:
    if value == 0:
        return "0"
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _format_decimal_scientific(value: Decimal, digits: int = 6) -> str:
    return f"{float(value):.{digits}E}"


def calculate_visible_bulk_closure_tensor(
    branch: tuple[int, int, int],
    *,
    q_iso_ev4: Decimal,
) -> tuple[float, Fraction, Fraction, Fraction, TensorSnapshot]:
    """Lift the scalar EFE audit to the full visible reviewer-trap tensor."""

    lepton_level, quark_level, parent_level = branch
    model = tn.TopologicalVacuum(k_l=lepton_level, k_q=quark_level, parent_level=parent_level)
    lepton_efe_violation = float(tn.calculate_efe_violation_tensor(model=model))
    defect = framing_defect(parent_level, lepton_level, quark_level)
    lepton_gap_float = float(_fraction_to_decimal(defect.lepton_gap))
    assert math.isclose(lepton_efe_violation, lepton_gap_float, rel_tol=0.0, abs_tol=ZERO_TOLERANCE), (
        "calculate_efe_violation_tensor() must reproduce the lepton-side framing gap on every tested branch. "
        f"Branch {_format_branch(branch)} returned {lepton_efe_violation:.18e} vs {lepton_gap_float:.18e}."
    )
    return (
        lepton_efe_violation,
        defect.lepton_gap,
        defect.quark_gap,
        defect.delta_fr,
        bulk_closure_tensor(defect.delta_fr, q_iso_ev4),
    )


def build_moat_stress_audit(*, precision: int = DEFAULT_PRECISION) -> MoatStressAudit:
    benchmark_branch = _benchmark_branch()
    c_dark_fraction = load_c_dark_completion_fraction()
    newton_lock = newton_constant_lock(c_dark_fraction=c_dark_fraction, precision=precision)
    saturation = saturation_audit(precision=precision)
    with localcontext() as context:
        context.prec = precision
        q_iso_ev4 = saturation.lambda_obs_ev2 / newton_lock.eight_pi_g_effective_ev_minus2

    witnesses: list[StressWitness] = []
    for branch in _stress_branches(benchmark_branch):
        lepton_efe_violation, lepton_gap, quark_gap, delta_fr, closure_tensor = calculate_visible_bulk_closure_tensor(
            branch,
            q_iso_ev4=q_iso_ev4,
        )
        shift = (
            branch[0] - benchmark_branch[0],
            branch[1] - benchmark_branch[1],
            branch[2] - benchmark_branch[2],
        )
        witnesses.append(
            StressWitness(
                branch=branch,
                shift=shift,
                lepton_efe_violation=lepton_efe_violation,
                lepton_gap=lepton_gap,
                quark_gap=quark_gap,
                delta_fr=delta_fr,
                closure_tensor=closure_tensor,
            )
        )
    return MoatStressAudit(
        benchmark_branch=benchmark_branch,
        q_iso_ev4=q_iso_ev4,
        witnesses=tuple(witnesses),
    )


def enforce_reviewer_trap(audit: MoatStressAudit) -> MoatStressAudit:
    assert len(audit.witnesses) == VISIBLE_MOAT_CELL_COUNT, (
        f"Expected a {VISIBLE_MOAT_DIMENSION} x {VISIBLE_MOAT_DIMENSION} visible moat, received {len(audit.witnesses)} cells."
    )
    assert audit.zero_tensor_cells == (audit.benchmark_branch,), (
        "Bulk Closure Tensor must vanish exactly at the benchmark cell. "
        f"Received zero-tensor cells {audit.zero_tensor_cells}."
    )

    benchmark_witness = next(witness for witness in audit.witnesses if witness.branch == audit.benchmark_branch)
    assert math.isclose(benchmark_witness.lepton_efe_violation, 0.0, rel_tol=0.0, abs_tol=ZERO_TOLERANCE), (
        "Benchmark EFE audit must vanish exactly. "
        f"Received {benchmark_witness.lepton_efe_violation:.18e}."
    )
    assert benchmark_witness.delta_fr == 0, (
        f"Benchmark visible framing defect must close exactly, received {benchmark_witness.delta_fr}."
    )

    for witness in audit.detuned_witnesses:
        assert witness.delta_fr != 0, (
            f"Every detuned branch must reopen the visible framing anomaly; {_format_branch(witness.branch)} gave Delta_fr=0."
        )
        assert witness.closure_tensor.amplitude != 0, (
            f"Every detuned branch must yield a strictly non-zero bulk closure tensor; {_format_branch(witness.branch)} returned E_mu_nu=0."
        )
    return audit


def render_report(audit: MoatStressAudit) -> str:
    zero_tensor_cell = _format_branch(audit.zero_tensor_cells[0])
    lines = [
        "Reviewer Trap Stress Test",
        "==========================",
        f"Benchmark branch                  : {_format_branch(audit.benchmark_branch)}",
        f"Differential violation            : scan the {VISIBLE_MOAT_DIMENSION} x {VISIBLE_MOAT_DIMENSION} visible moat with fixed K={audit.benchmark_branch[2]}",
        "Lepton EFE seed                   : calculate_efe_violation_tensor()",
        "Bulk closure tensor               : E_mu_nu = Q_iso * Delta_fr * g_mu_nu",
        f"Q_iso [eV^4]                      : {_format_decimal_scientific(audit.q_iso_ev4, digits=12)}",
        "",
        "Branch audit",
        "------------",
        "branch         shift                  EFE_seed     lepton_gap   quark_gap    Delta_fr      |E| [eV^4]      verdict",
        "-------------------------------------------------------------------------------------------------------------------",
    ]
    for witness in audit.witnesses:
        verdict = "PASS" if witness.benchmark_cell else "PHYSICALLY VOID"
        lines.append(
            f"{_format_branch(witness.branch):<13} "
            f"{_format_shift(witness.shift):<22} "
            f"{witness.lepton_efe_violation:<12.6e} "
            f"{_format_fraction(witness.lepton_gap):<12} "
            f"{_format_fraction(witness.quark_gap):<12} "
            f"{_format_fraction(witness.delta_fr):<12} "
            f"{_format_decimal_scientific(witness.closure_tensor.amplitude):<15} "
            f"{verdict}"
        )
    lines.extend(
        (
            "",
            "Conclusion",
            "----------",
            f"Unique zero-tensor cell          : {zero_tensor_cell}",
            f"Strictly nonzero detunings       : {sum(int(not witness.closure_tensor.vanished) for witness in audit.detuned_witnesses)}/{len(audit.detuned_witnesses)}",
            "Detuning verdict                 : physically void",
            f"Proof                            : every off-shell cell in the {VISIBLE_MOAT_DIMENSION} x {VISIBLE_MOAT_DIMENSION} visible moat reintroduces Delta_fr != 0, so the framing anomaly is non-vanishing and E_mu_nu immediately reopens.",
            "Reviewer trap nuance             : k_q detunings keep the lepton EFE seed at zero but still fail because the visible quark framing gap makes Delta_fr > 0.",
        )
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = enforce_reviewer_trap(build_moat_stress_audit(precision=args.precision))
    print(render_report(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
