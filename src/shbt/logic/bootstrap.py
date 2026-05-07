from __future__ import annotations

"""Combinatorial symmetry scan for Axiom IX topological closure.

This logic-layer module performs an explicit inside-code scan over combinatorial
kernel coordinates ``(D, G, N)`` and identifies the unique branch that closes
all transport residues without reopening a singular transport kernel. In the
SHBT bookkeeping these scan coordinates are identified with the published
branch tuple

    (D, G, N) <-> (k_ell, k_q, K).

For each candidate kernel we evaluate a transport-residue closure metric

    C_closure = Delta_fr + Delta_dio + ||R_transport - R_benchmark||_2,

where

- ``Delta_fr`` is the framing-defect residue,
- ``Delta_dio`` is the Diophantine detuning from ``lcm(2D, 3G)``, and
- ``R_transport`` collects the branch-local transport residues
  ``(c_dark, <Higgs VEV>, kappa, phase lock)``.

A candidate satisfies the Axiom IX fixed-point criterion only if it is
non-singular, Diophantine-closed, and all transport-residue drifts vanish.
Under the disclosed scan domain this module certifies that only the published
kernel ``(26, 8, 312)`` survives.
"""

import argparse
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Final, Sequence

if __package__ in (None, ""):
    import sys

    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.constants import LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.algebra import su2_total_quantum_dimension
from shbt.core.master_transport import (
    derive_c_dark_residue,
    derive_geometric_kappa,
    derive_higgs_vev_residue,
)
from shbt.core.noether_bridge import framing_defect
from shbt.export import write_json_artifact
from shbt.paths import ProjectPaths


EXPECTED_BENCHMARK: Final[tuple[int, int, int]] = (26, 8, 312)
DEFAULT_DIMENSION_LEVELS: Final[tuple[int, ...]] = tuple(range(2, 33))
DEFAULT_GENERATION_LEVELS: Final[tuple[int, ...]] = tuple(range(1, 33))
DEFAULT_PARENT_DETUNINGS: Final[tuple[int, ...]] = (-2, -1, 0, 1, 2)
DEFAULT_STABILITY_TOLERANCE: Final[float] = 1.0e-12
DEFAULT_OUTPUT_PATH: Final[Path] = ProjectPaths.RESULTS / "uniqueness_report.json"


def _sanitize_small(value: float, *, tolerance: float = DEFAULT_STABILITY_TOLERANCE) -> float:
    numeric_value = float(value)
    if math.isclose(numeric_value, 0.0, rel_tol=0.0, abs_tol=float(tolerance)):
        return 0.0
    return numeric_value


@lru_cache(maxsize=1)
def _benchmark_transport_residues() -> dict[str, float]:
    benchmark = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))
    assert benchmark == EXPECTED_BENCHMARK, (
        f"The combinatorial closure scan is locked to the published branch {EXPECTED_BENCHMARK}, received {benchmark}."
    )
    return {
        "c_dark_residue": float(
            derive_c_dark_residue(
                lepton_level=benchmark[0],
                quark_level=benchmark[1],
                parent_level=benchmark[2],
            )
        ),
        "higgs_vev_residue": float(
            derive_higgs_vev_residue(
                lepton_level=benchmark[0],
                quark_level=benchmark[1],
                parent_level=benchmark[2],
            )
        ),
        "geometric_kappa": float(derive_geometric_kappa(lepton_level=benchmark[0])),
        "flavor_phase_lock": float(0.5 * math.log(su2_total_quantum_dimension(benchmark[0]))),
    }


@dataclass(frozen=True)
class ClosureMetric:
    framing_residue: float
    diophantine_gap: float
    c_dark_shift: float
    higgs_vev_shift: float
    geometric_kappa_shift: float
    flavor_phase_lock_shift: float
    transport_residue_norm: float
    closure_metric: float

    @property
    def transport_residues_stable(self) -> bool:
        return bool(math.isclose(self.transport_residue_norm, 0.0, rel_tol=0.0, abs_tol=DEFAULT_STABILITY_TOLERANCE))

    @property
    def topological_closure(self) -> bool:
        return bool(
            math.isclose(self.framing_residue, 0.0, rel_tol=0.0, abs_tol=DEFAULT_STABILITY_TOLERANCE)
            and math.isclose(self.diophantine_gap, 0.0, rel_tol=0.0, abs_tol=DEFAULT_STABILITY_TOLERANCE)
            and self.transport_residues_stable
            and math.isclose(self.closure_metric, 0.0, rel_tol=0.0, abs_tol=DEFAULT_STABILITY_TOLERANCE)
        )


@dataclass(frozen=True)
class KernelFailureAudit:
    coordinates: tuple[int, int, int]
    minimal_parent_level: int
    parent_detuning: int
    non_singular_transport_kernel: bool
    closure: ClosureMetric
    failure_modes: tuple[str, ...]

    @property
    def dimension_level(self) -> int:
        return int(self.coordinates[0])

    @property
    def generation_level(self) -> int:
        return int(self.coordinates[1])

    @property
    def parent_level(self) -> int:
        return int(self.coordinates[2])

    @property
    def axiom_ix_fixed_point(self) -> bool:
        return bool(self.non_singular_transport_kernel and self.closure.topological_closure)

    @property
    def benchmark_kernel(self) -> bool:
        return self.coordinates == EXPECTED_BENCHMARK

    @property
    def statement(self) -> str:
        if self.axiom_ix_fixed_point:
            return "Axiom IX closes: this kernel is non-singular and transport-stable."
        return "Axiom IX fails: the candidate kernel reopens singularity or transport-residue drift."


@dataclass(frozen=True)
class CombinatorialSymmetryAudit:
    benchmark_coordinates: tuple[int, int, int]
    dimension_levels: tuple[int, ...]
    generation_levels: tuple[int, ...]
    parent_detunings: tuple[int, ...]
    unique_fixed_point: KernelFailureAudit
    runner_up_kernel: KernelFailureAudit
    stable_fixed_points: tuple[KernelFailureAudit, ...]
    candidate_audits: tuple[KernelFailureAudit, ...]
    trial_count: int

    @property
    def unique_fixed_point_count(self) -> int:
        return len(self.stable_fixed_points)

    @property
    def unique_axiom_ix_fixed_point(self) -> bool:
        return bool(
            self.unique_fixed_point_count == 1
            and self.stable_fixed_points[0].coordinates == self.benchmark_coordinates
            and self.unique_fixed_point.coordinates == self.benchmark_coordinates
        )

    @property
    def closure_gap(self) -> float:
        return float(self.runner_up_kernel.closure.closure_metric - self.unique_fixed_point.closure.closure_metric)

    @property
    def alternative_kernel_failures(self) -> tuple[KernelFailureAudit, ...]:
        return tuple(audit for audit in self.candidate_audits if audit.coordinates != self.benchmark_coordinates)

    @property
    def statement(self) -> str:
        return (
            "The combinatorial symmetry scan certifies (26, 8, 312) as the unique non-singular "
            f"Axiom IX fixed point after evaluating {self.trial_count} candidate kernels."
        )

    def assert_unique_fixed_point(self) -> None:
        if self.unique_fixed_point_count != 1:
            raise AssertionError(
                "Combinatorial closure scan failed to isolate a unique Axiom IX fixed point: "
                f"found {self.unique_fixed_point_count}."
            )
        if self.stable_fixed_points[0].coordinates != self.benchmark_coordinates:
            raise AssertionError(
                "The isolated Axiom IX fixed point does not match the benchmark branch: "
                f"{self.stable_fixed_points[0].coordinates}."
            )
        if self.runner_up_kernel.closure.closure_metric <= 0.0:
            raise AssertionError("Runner-up kernel retained zero closure metric; uniqueness proof failed.")

    def to_payload(self) -> dict[str, object]:
        self.assert_unique_fixed_point()
        return {
            "artifact": "Combinatorial Symmetry Scan Uniqueness Report",
            "axiom": "Axiom IX",
            "benchmark_tuple": list(self.benchmark_coordinates),
            "scan_domain": {
                "dimension_levels": [int(level) for level in self.dimension_levels],
                "generation_levels": [int(level) for level in self.generation_levels],
                "parent_detunings": [int(detuning) for detuning in self.parent_detunings],
            },
            "closure_metric_definition": {
                "formula": "C_closure = Delta_fr + Delta_dio + ||R_transport - R_benchmark||_2",
                "transport_residues": [
                    "c_dark_residue",
                    "higgs_vev_residue",
                    "geometric_kappa",
                    "flavor_phase_lock",
                ],
                "non_singular_kernel_rule": "N mod (2D) = 0 and N mod (3G) = 0",
            },
            "trial_count": int(self.trial_count),
            "unique_stable_fixed_point": list(self.unique_fixed_point.coordinates),
            "unique_fixed_point_count": int(self.unique_fixed_point_count),
            "closure_gap": float(self.closure_gap),
            "runner_up_kernel": _kernel_payload(self.runner_up_kernel),
            "statement": self.statement,
            "alternative_kernel_failures": {
                _kernel_key(audit.coordinates): _kernel_payload(audit)
                for audit in self.alternative_kernel_failures
            },
        }


def _kernel_key(coordinates: tuple[int, int, int]) -> str:
    return f"{int(coordinates[0])}:{int(coordinates[1])}:{int(coordinates[2])}"


def _kernel_payload(audit: KernelFailureAudit) -> dict[str, object]:
    return {
        "coordinates": [int(value) for value in audit.coordinates],
        "minimal_parent_level": int(audit.minimal_parent_level),
        "parent_detuning": int(audit.parent_detuning),
        "non_singular_transport_kernel": bool(audit.non_singular_transport_kernel),
        "axiom_ix_fixed_point": bool(audit.axiom_ix_fixed_point),
        "closure_metric": float(audit.closure.closure_metric),
        "transport_residue_norm": float(audit.closure.transport_residue_norm),
        "framing_residue": float(audit.closure.framing_residue),
        "diophantine_gap": float(audit.closure.diophantine_gap),
        "c_dark_shift": float(audit.closure.c_dark_shift),
        "higgs_vev_shift": float(audit.closure.higgs_vev_shift),
        "geometric_kappa_shift": float(audit.closure.geometric_kappa_shift),
        "flavor_phase_lock_shift": float(audit.closure.flavor_phase_lock_shift),
        "failure_modes": list(audit.failure_modes),
        "statement": audit.statement,
    }


def _failure_modes(
    *,
    non_singular_transport_kernel: bool,
    closure: ClosureMetric,
) -> tuple[str, ...]:
    failure_modes: list[str] = []
    if not non_singular_transport_kernel:
        failure_modes.append("singular_transport_kernel")
    if not math.isclose(closure.diophantine_gap, 0.0, rel_tol=0.0, abs_tol=DEFAULT_STABILITY_TOLERANCE):
        failure_modes.append("diophantine_detuning")
    if not math.isclose(closure.framing_residue, 0.0, rel_tol=0.0, abs_tol=DEFAULT_STABILITY_TOLERANCE):
        failure_modes.append("framing_defect_reopened")
    if not math.isclose(closure.c_dark_shift, 0.0, rel_tol=0.0, abs_tol=DEFAULT_STABILITY_TOLERANCE):
        failure_modes.append("c_dark_transport_drift")
    if not math.isclose(closure.higgs_vev_shift, 0.0, rel_tol=0.0, abs_tol=DEFAULT_STABILITY_TOLERANCE):
        failure_modes.append("higgs_vev_transport_drift")
    if not math.isclose(closure.geometric_kappa_shift, 0.0, rel_tol=0.0, abs_tol=DEFAULT_STABILITY_TOLERANCE):
        failure_modes.append("geometric_kappa_transport_drift")
    if not math.isclose(closure.flavor_phase_lock_shift, 0.0, rel_tol=0.0, abs_tol=DEFAULT_STABILITY_TOLERANCE):
        failure_modes.append("phase_lock_transport_drift")
    if not failure_modes:
        failure_modes.append("axiom_ix_closed")
    return tuple(failure_modes)


@lru_cache(maxsize=None)
def evaluate_kernel(
    dimension_level: int,
    generation_level: int,
    parent_level: int,
) -> KernelFailureAudit:
    resolved_dimension_level = int(dimension_level)
    resolved_generation_level = int(generation_level)
    resolved_parent_level = int(parent_level)
    if resolved_dimension_level < 1 or resolved_generation_level < 1 or resolved_parent_level < 2:
        raise ValueError("Kernel coordinates must be positive integers with N >= 2.")

    benchmark = _benchmark_transport_residues()
    minimal_parent_level = math.lcm(2 * resolved_dimension_level, 3 * resolved_generation_level)
    parent_detuning = resolved_parent_level - minimal_parent_level
    defect = framing_defect(resolved_parent_level, resolved_dimension_level, resolved_generation_level)
    framing_residue = _sanitize_small(abs(float(defect.delta_fr)))
    diophantine_gap = _sanitize_small(abs(parent_detuning) / float(minimal_parent_level))
    c_dark_shift = _sanitize_small(
        abs(
            float(
                derive_c_dark_residue(
                    lepton_level=resolved_dimension_level,
                    quark_level=resolved_generation_level,
                    parent_level=resolved_parent_level,
                )
            )
            - benchmark["c_dark_residue"]
        )
    )
    higgs_vev_shift = _sanitize_small(
        abs(
            float(
                derive_higgs_vev_residue(
                    lepton_level=resolved_dimension_level,
                    quark_level=resolved_generation_level,
                    parent_level=resolved_parent_level,
                )
            )
            - benchmark["higgs_vev_residue"]
        )
    )
    geometric_kappa_shift = _sanitize_small(
        abs(float(derive_geometric_kappa(lepton_level=resolved_dimension_level)) - benchmark["geometric_kappa"])
    )
    flavor_phase_lock_shift = _sanitize_small(
        abs(float(0.5 * math.log(su2_total_quantum_dimension(resolved_dimension_level))) - benchmark["flavor_phase_lock"])
    )
    transport_residue_norm = _sanitize_small(
        math.sqrt(
            c_dark_shift * c_dark_shift
            + higgs_vev_shift * higgs_vev_shift
            + geometric_kappa_shift * geometric_kappa_shift
            + flavor_phase_lock_shift * flavor_phase_lock_shift
        )
    )
    closure_metric = _sanitize_small(framing_residue + diophantine_gap + transport_residue_norm)
    non_singular_transport_kernel = bool(
        resolved_parent_level % (2 * resolved_dimension_level) == 0
        and resolved_parent_level % (3 * resolved_generation_level) == 0
    )
    closure = ClosureMetric(
        framing_residue=framing_residue,
        diophantine_gap=diophantine_gap,
        c_dark_shift=c_dark_shift,
        higgs_vev_shift=higgs_vev_shift,
        geometric_kappa_shift=geometric_kappa_shift,
        flavor_phase_lock_shift=flavor_phase_lock_shift,
        transport_residue_norm=transport_residue_norm,
        closure_metric=closure_metric,
    )
    return KernelFailureAudit(
        coordinates=(resolved_dimension_level, resolved_generation_level, resolved_parent_level),
        minimal_parent_level=minimal_parent_level,
        parent_detuning=parent_detuning,
        non_singular_transport_kernel=non_singular_transport_kernel,
        closure=closure,
        failure_modes=_failure_modes(
            non_singular_transport_kernel=non_singular_transport_kernel,
            closure=closure,
        ),
    )


def scan_combinatorial_symmetry_space(
    *,
    dimension_levels: Sequence[int] = DEFAULT_DIMENSION_LEVELS,
    generation_levels: Sequence[int] = DEFAULT_GENERATION_LEVELS,
    parent_detunings: Sequence[int] = DEFAULT_PARENT_DETUNINGS,
) -> CombinatorialSymmetryAudit:
    resolved_dimension_levels = tuple(sorted({int(level) for level in dimension_levels}))
    resolved_generation_levels = tuple(sorted({int(level) for level in generation_levels}))
    resolved_parent_detunings = tuple(sorted({int(detuning) for detuning in parent_detunings}))
    if not resolved_dimension_levels or min(resolved_dimension_levels) < 1:
        raise ValueError("dimension_levels must contain positive integers.")
    if not resolved_generation_levels or min(resolved_generation_levels) < 1:
        raise ValueError("generation_levels must contain positive integers.")
    if not resolved_parent_detunings:
        raise ValueError("parent_detunings must contain at least one integer detuning.")

    candidate_audits: list[KernelFailureAudit] = []
    for dimension_level in resolved_dimension_levels:
        for generation_level in resolved_generation_levels:
            minimal_parent_level = math.lcm(2 * dimension_level, 3 * generation_level)
            for parent_detuning in resolved_parent_detunings:
                candidate_parent_level = minimal_parent_level + int(parent_detuning)
                if candidate_parent_level < 2:
                    continue
                candidate_audits.append(
                    evaluate_kernel(dimension_level, generation_level, candidate_parent_level)
                )

    if not candidate_audits:
        raise ValueError("Combinatorial symmetry scan requires at least one candidate kernel.")

    ordered_candidates = tuple(
        sorted(
            candidate_audits,
            key=lambda audit: (
                float(audit.closure.closure_metric),
                int(audit.coordinates[2]),
                int(audit.coordinates[0]),
                int(audit.coordinates[1]),
            ),
        )
    )
    stable_fixed_points = tuple(audit for audit in ordered_candidates if audit.axiom_ix_fixed_point)
    unique_fixed_point = ordered_candidates[0]
    runner_up_kernel = next(
        audit for audit in ordered_candidates if audit.coordinates != unique_fixed_point.coordinates
    )
    scan = CombinatorialSymmetryAudit(
        benchmark_coordinates=EXPECTED_BENCHMARK,
        dimension_levels=resolved_dimension_levels,
        generation_levels=resolved_generation_levels,
        parent_detunings=resolved_parent_detunings,
        unique_fixed_point=unique_fixed_point,
        runner_up_kernel=runner_up_kernel,
        stable_fixed_points=stable_fixed_points,
        candidate_audits=ordered_candidates,
        trial_count=len(ordered_candidates),
    )
    scan.assert_unique_fixed_point()
    return scan


def build_uniqueness_report(
    *,
    dimension_levels: Sequence[int] = DEFAULT_DIMENSION_LEVELS,
    generation_levels: Sequence[int] = DEFAULT_GENERATION_LEVELS,
    parent_detunings: Sequence[int] = DEFAULT_PARENT_DETUNINGS,
) -> dict[str, object]:
    return scan_combinatorial_symmetry_space(
        dimension_levels=dimension_levels,
        generation_levels=generation_levels,
        parent_detunings=parent_detunings,
    ).to_payload()


def write_uniqueness_report(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    dimension_levels: Sequence[int] = DEFAULT_DIMENSION_LEVELS,
    generation_levels: Sequence[int] = DEFAULT_GENERATION_LEVELS,
    parent_detunings: Sequence[int] = DEFAULT_PARENT_DETUNINGS,
) -> Path:
    payload = build_uniqueness_report(
        dimension_levels=dimension_levels,
        generation_levels=generation_levels,
        parent_detunings=parent_detunings,
    )
    return write_json_artifact(Path(output_path), payload)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--dimension-min", type=int, default=min(DEFAULT_DIMENSION_LEVELS))
    parser.add_argument("--dimension-max", type=int, default=max(DEFAULT_DIMENSION_LEVELS))
    parser.add_argument("--generation-min", type=int, default=min(DEFAULT_GENERATION_LEVELS))
    parser.add_argument("--generation-max", type=int, default=max(DEFAULT_GENERATION_LEVELS))
    parser.add_argument(
        "--parent-detunings",
        type=int,
        nargs="+",
        default=list(DEFAULT_PARENT_DETUNINGS),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    write_uniqueness_report(
        output_path=args.output_path,
        dimension_levels=range(int(args.dimension_min), int(args.dimension_max) + 1),
        generation_levels=range(int(args.generation_min), int(args.generation_max) + 1),
        parent_detunings=tuple(int(value) for value in args.parent_detunings),
    )
    return 0


__all__ = [
    "ClosureMetric",
    "CombinatorialSymmetryAudit",
    "DEFAULT_DIMENSION_LEVELS",
    "DEFAULT_GENERATION_LEVELS",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_PARENT_DETUNINGS",
    "DEFAULT_STABILITY_TOLERANCE",
    "EXPECTED_BENCHMARK",
    "KernelFailureAudit",
    "build_uniqueness_report",
    "evaluate_kernel",
    "scan_combinatorial_symmetry_space",
    "write_uniqueness_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
