from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pub.constants import (
        LEPTON_INTERVALS,
        LEPTON_LEVEL,
        MIXING_SECTOR_RIGIDITY_MESSAGE,
        PARENT_LEVEL,
        PRIMARY_BENCHMARK_AUDIT_SUCCESS_MESSAGE,
        QUARK_INTERVALS,
        QUARK_LEVEL,
        SVD_RIGIDITY_SHIELD_SIGMA_THRESHOLD,
        SVD_RIGIDITY_SHIELD_VEV_DEFORMATION_FRACTION,
    )
    from pub.topological_kernel import pdg_parameters
    from pub.tn import (
        _angle_interval_from_modulus_interval as angle_interval_from_modulus_interval,
        apply_higgs_vev_alignment_constraint,
        calculate_126_higgs_cg_correction,
        derive_boundary_bulk_interface,
        derive_lie_algebraic_vev_residue,
    )
else:
    from .constants import (
        LEPTON_INTERVALS,
        LEPTON_LEVEL,
        MIXING_SECTOR_RIGIDITY_MESSAGE,
        PARENT_LEVEL,
        PRIMARY_BENCHMARK_AUDIT_SUCCESS_MESSAGE,
        QUARK_INTERVALS,
        QUARK_LEVEL,
        SVD_RIGIDITY_SHIELD_SIGMA_THRESHOLD,
        SVD_RIGIDITY_SHIELD_VEV_DEFORMATION_FRACTION,
    )
    from .topological_kernel import pdg_parameters
    from .tn import (
        _angle_interval_from_modulus_interval as angle_interval_from_modulus_interval,
        apply_higgs_vev_alignment_constraint,
        calculate_126_higgs_cg_correction,
        derive_boundary_bulk_interface,
        derive_lie_algebraic_vev_residue,
    )


EXPECTED_BRANCH = (26, 8, 312)
EXPECTED_REPRESENTATIONAL_ADMISSIBILITY_RATIO = Fraction(64, 312)
EXPECTED_REPRESENTATIONAL_ADMISSIBILITY_DISPLAY = "64/312"
PMNS_LABELS = ("theta12", "theta13", "theta23")
CKM_LABELS = ("thetaC", "theta13^q", "theta23^q")


@dataclass(frozen=True)
class SectorSnapshot:
    singular_values: tuple[float, ...]
    angles_deg: tuple[float, float, float]
    unitary: np.ndarray
    dressed_matrix: np.ndarray


@dataclass(frozen=True)
class SweepPoint:
    ratio: float
    pmns_singular_values: tuple[float, ...]
    ckm_singular_values: tuple[float, ...]
    pmns_angle_shifts_deg: tuple[float, float, float]
    ckm_angle_shifts_deg: tuple[float, float, float]
    pmns_sigma_drifts: tuple[float, float, float]
    ckm_sigma_drifts: tuple[float, float, float]
    lepton_left_overlap_min: float
    lepton_right_overlap_min: float
    quark_left_overlap_min: float
    quark_right_overlap_min: float
    max_sigma_drift: float


@dataclass(frozen=True)
class RigiditySweepReport:
    branch: tuple[int, int, int]
    benchmark_ratio_fraction: Fraction
    benchmark_ratio_display: str
    benchmark_ratio: float
    sweep_min_ratio: float
    sweep_max_ratio: float
    sigma_threshold: float
    benchmark_pmns_angles_deg: tuple[float, float, float]
    benchmark_ckm_angles_deg: tuple[float, float, float]
    points: tuple[SweepPoint, ...]
    max_sigma_angle_drift: float
    min_lepton_overlap: tuple[float, float]
    min_quark_overlap: tuple[float, float]
    decoupling_proof: str
    verdict: str


def _minimum_singular_vector_overlap(baseline: np.ndarray, perturbed: np.ndarray) -> tuple[float, float]:
    baseline_left, baseline_singular_values, baseline_right_dag = np.linalg.svd(np.asarray(baseline, dtype=np.complex128))
    perturbed_left, perturbed_singular_values, perturbed_right_dag = np.linalg.svd(np.asarray(perturbed, dtype=np.complex128))
    baseline_right = baseline_right_dag.conjugate().T
    perturbed_right = perturbed_right_dag.conjugate().T

    def singular_blocks(singular_values: np.ndarray) -> tuple[slice, ...]:
        blocks: list[slice] = []
        start = 0
        while start < singular_values.size:
            stop = start + 1
            while stop < singular_values.size and math.isclose(
                float(singular_values[stop]),
                float(singular_values[start]),
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            ):
                stop += 1
            blocks.append(slice(start, stop))
            start = stop
        return tuple(blocks)

    def block_overlap(baseline_basis: np.ndarray, perturbed_basis: np.ndarray) -> float:
        principal_cosines = np.linalg.svd(baseline_basis.conjugate().T @ perturbed_basis, compute_uv=False)
        return float(np.min(principal_cosines, initial=1.0))

    baseline_blocks = singular_blocks(baseline_singular_values)
    perturbed_blocks = singular_blocks(perturbed_singular_values)
    baseline_block_sizes = tuple(block.stop - block.start for block in baseline_blocks)
    perturbed_block_sizes = tuple(block.stop - block.start for block in perturbed_blocks)

    if baseline_block_sizes != perturbed_block_sizes:
        left_overlap_matrix = np.abs(baseline_left.conjugate().T @ perturbed_left)
        right_overlap_matrix = np.abs(baseline_right.conjugate().T @ perturbed_right)
        left_overlap = min(float(np.min(np.max(left_overlap_matrix, axis=0))), float(np.min(np.max(left_overlap_matrix, axis=1))))
        right_overlap = min(float(np.min(np.max(right_overlap_matrix, axis=0))), float(np.min(np.max(right_overlap_matrix, axis=1))))
        return left_overlap, right_overlap

    left_overlap = min(
        block_overlap(baseline_left[:, block], perturbed_left[:, block])
        for block in baseline_blocks
    )
    right_overlap = min(
        block_overlap(baseline_right[:, block], perturbed_right[:, block])
        for block in baseline_blocks
    )
    return left_overlap, right_overlap


def _build_sector_snapshot(
    matrix: np.ndarray,
    *,
    relative_suppression: float,
    sector_exponent: float,
) -> SectorSnapshot:
    dressed_matrix, _, _ = apply_higgs_vev_alignment_constraint(
        np.asarray(matrix, dtype=np.complex128),
        relative_suppression,
        sector_exponent,
    )
    left_unitary, singular_values, right_dag = np.linalg.svd(np.asarray(dressed_matrix, dtype=np.complex128))
    unitary = np.asarray(left_unitary @ right_dag, dtype=np.complex128)
    theta12_deg, theta13_deg, theta23_deg, _, _ = pdg_parameters(unitary)
    return SectorSnapshot(
        singular_values=tuple(float(value) for value in singular_values),
        angles_deg=(float(theta12_deg), float(theta13_deg), float(theta23_deg)),
        unitary=unitary,
        dressed_matrix=np.asarray(dressed_matrix, dtype=np.complex128),
    )


def _sigma_drifts(angle_shifts_deg: tuple[float, float, float], intervals: Sequence[object]) -> tuple[float, float, float]:
    return tuple(float(abs(delta) / interval.sigma) for delta, interval in zip(angle_shifts_deg, intervals, strict=True))


def _build_ratio_grid(benchmark_ratio: float, deformation_fraction: float, steps: int) -> np.ndarray:
    if steps < 2:
        raise ValueError(f"steps must be at least 2, received {steps}")
    if deformation_fraction < 0.0:
        raise ValueError(f"deformation_fraction must be non-negative, received {deformation_fraction}")

    lower = benchmark_ratio * (1.0 - deformation_fraction)
    upper = benchmark_ratio * (1.0 + deformation_fraction)
    ratio_grid = np.linspace(lower, upper, int(steps), dtype=float)
    if not np.any(np.isclose(ratio_grid, benchmark_ratio, rtol=0.0, atol=1.0e-15)):
        ratio_grid = np.sort(np.append(ratio_grid, benchmark_ratio))
    return ratio_grid


def _build_decoupling_proof(
    *,
    ratio_display: str,
    max_sigma_angle_drift: float,
    min_lepton_overlap: tuple[float, float],
    min_quark_overlap: tuple[float, float],
) -> str:
    return "\n".join(
        (
            "1. Write each benchmark flavor map in SVD form Y_f = U_f Sigma_f V_f^dagger.",
            (
                "2. The scalar-sector deformation is implemented as "
                "Y_f(r) = U_f diag(d_f(r)) Sigma_f V_f^dagger with diag(d_f(r)) > 0 and "
                f"r centered on the exact benchmark ratio {ratio_display}."
            ),
            "3. Because the deformation acts only inside the diagonal singular-value block, the left/right singular vectors U_f and V_f are unchanged.",
            "4. PMNS/CKM angles are extracted from the unitary polar part U_f V_f^dagger, so the scalar sector can dress masses but cannot rotate flavor eigenvectors.",
            (
                "5. The explicit sweep confirms the algebra: minimum singular-subspace overlaps stay at "
                f"L={min_lepton_overlap[0]:.12f}, R={min_lepton_overlap[1]:.12f} in the PMNS sector and "
                f"L={min_quark_overlap[0]:.12f}, R={min_quark_overlap[1]:.12f} in the CKM sector, while the largest "
                f"sigma-weighted angle drift is only {max_sigma_angle_drift:.6e}."
            ),
        )
    )


def execute_svd_rigidity_shield(
    *,
    steps: int = 21,
    deformation_fraction: float = SVD_RIGIDITY_SHIELD_VEV_DEFORMATION_FRACTION,
) -> RigiditySweepReport:
    """Run the benchmark-centered SVD Rigidity Shield sweep."""

    branch = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))
    if branch != EXPECTED_BRANCH:
        raise AssertionError(
            f"Eigenvector Rigidity is benchmarked on the branch {EXPECTED_BRANCH}, received {branch}."
        )

    benchmark_ratio_fraction = derive_lie_algebraic_vev_residue(
        parent_level=PARENT_LEVEL,
        lepton_level=LEPTON_LEVEL,
        quark_level=QUARK_LEVEL,
    )
    if benchmark_ratio_fraction != EXPECTED_REPRESENTATIONAL_ADMISSIBILITY_RATIO:
        raise AssertionError(
            "Representational Admissibility drift detected: "
            f"expected {EXPECTED_REPRESENTATIONAL_ADMISSIBILITY_RATIO}, got {benchmark_ratio_fraction}."
        )

    benchmark_ratio = float(benchmark_ratio_fraction)
    benchmark_clebsch = calculate_126_higgs_cg_correction(target_suppression=benchmark_ratio)
    if not math.isclose(
        float(benchmark_clebsch.inverse_clebsch_126_suppression),
        benchmark_ratio,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise AssertionError("The benchmark inverse-Clebsch suppression no longer matches the 64/312 branch residue.")

    lepton_interface = derive_boundary_bulk_interface(level=LEPTON_LEVEL, sector="lepton")
    quark_interface = derive_boundary_bulk_interface(level=QUARK_LEVEL, sector="quark")

    benchmark_pmns = _build_sector_snapshot(
        lepton_interface.majorana_yukawa_texture,
        relative_suppression=benchmark_ratio,
        sector_exponent=-0.5,
    )
    benchmark_ckm = _build_sector_snapshot(
        quark_interface.framed_yukawa_texture,
        relative_suppression=benchmark_ratio,
        sector_exponent=+0.5,
    )

    lepton_intervals = (
        LEPTON_INTERVALS["theta12"],
        LEPTON_INTERVALS["theta13"],
        LEPTON_INTERVALS["theta23"],
    )
    quark_intervals = tuple(
        angle_interval_from_modulus_interval(QUARK_INTERVALS[key])
        for key in ("vus", "vub", "vcb")
    )

    sweep_ratios = _build_ratio_grid(benchmark_ratio, deformation_fraction, steps)
    sweep_points: list[SweepPoint] = []
    max_sigma_angle_drift = 0.0
    lepton_left_overlap_min = 1.0
    lepton_right_overlap_min = 1.0
    quark_left_overlap_min = 1.0
    quark_right_overlap_min = 1.0

    for ratio in sweep_ratios:
        pmns_snapshot = _build_sector_snapshot(
            lepton_interface.majorana_yukawa_texture,
            relative_suppression=float(ratio),
            sector_exponent=-0.5,
        )
        ckm_snapshot = _build_sector_snapshot(
            quark_interface.framed_yukawa_texture,
            relative_suppression=float(ratio),
            sector_exponent=+0.5,
        )

        pmns_angle_shifts_deg = tuple(
            float(current - benchmark)
            for current, benchmark in zip(pmns_snapshot.angles_deg, benchmark_pmns.angles_deg, strict=True)
        )
        ckm_angle_shifts_deg = tuple(
            float(current - benchmark)
            for current, benchmark in zip(ckm_snapshot.angles_deg, benchmark_ckm.angles_deg, strict=True)
        )
        pmns_sigma_drifts = _sigma_drifts(pmns_angle_shifts_deg, lepton_intervals)
        ckm_sigma_drifts = _sigma_drifts(ckm_angle_shifts_deg, quark_intervals)
        lepton_overlap = _minimum_singular_vector_overlap(
            benchmark_pmns.dressed_matrix,
            pmns_snapshot.dressed_matrix,
        )
        quark_overlap = _minimum_singular_vector_overlap(
            benchmark_ckm.dressed_matrix,
            ckm_snapshot.dressed_matrix,
        )
        point_max_sigma_drift = float(max((*pmns_sigma_drifts, *ckm_sigma_drifts), default=0.0))
        max_sigma_angle_drift = max(max_sigma_angle_drift, point_max_sigma_drift)
        lepton_left_overlap_min = min(lepton_left_overlap_min, lepton_overlap[0])
        lepton_right_overlap_min = min(lepton_right_overlap_min, lepton_overlap[1])
        quark_left_overlap_min = min(quark_left_overlap_min, quark_overlap[0])
        quark_right_overlap_min = min(quark_right_overlap_min, quark_overlap[1])

        sweep_points.append(
            SweepPoint(
                ratio=float(ratio),
                pmns_singular_values=pmns_snapshot.singular_values,
                ckm_singular_values=ckm_snapshot.singular_values,
                pmns_angle_shifts_deg=pmns_angle_shifts_deg,
                ckm_angle_shifts_deg=ckm_angle_shifts_deg,
                pmns_sigma_drifts=pmns_sigma_drifts,
                ckm_sigma_drifts=ckm_sigma_drifts,
                lepton_left_overlap_min=lepton_overlap[0],
                lepton_right_overlap_min=lepton_overlap[1],
                quark_left_overlap_min=quark_overlap[0],
                quark_right_overlap_min=quark_overlap[1],
                max_sigma_drift=point_max_sigma_drift,
            )
        )

    assert max_sigma_angle_drift < SVD_RIGIDITY_SHIELD_SIGMA_THRESHOLD, (
        "Eigenvector Rigidity violated under scalar-potential uncertainty: max sigma-angle drift must remain below "
        f"{SVD_RIGIDITY_SHIELD_SIGMA_THRESHOLD:.1e}, got {max_sigma_angle_drift:.3e}."
    )

    min_lepton_overlap = (lepton_left_overlap_min, lepton_right_overlap_min)
    min_quark_overlap = (quark_left_overlap_min, quark_right_overlap_min)
    decoupling_proof = _build_decoupling_proof(
        ratio_display=EXPECTED_REPRESENTATIONAL_ADMISSIBILITY_DISPLAY,
        max_sigma_angle_drift=max_sigma_angle_drift,
        min_lepton_overlap=min_lepton_overlap,
        min_quark_overlap=min_quark_overlap,
    )
    verdict = (
        "Stability Verdict: PASS — Eigenvector Rigidity is fundamentally intact under scalar potential uncertainty. "
        f"Max drift = {max_sigma_angle_drift:.6e}σ < {SVD_RIGIDITY_SHIELD_SIGMA_THRESHOLD:.1e}σ."
    )

    return RigiditySweepReport(
        branch=branch,
        benchmark_ratio_fraction=benchmark_ratio_fraction,
        benchmark_ratio_display=EXPECTED_REPRESENTATIONAL_ADMISSIBILITY_DISPLAY,
        benchmark_ratio=benchmark_ratio,
        sweep_min_ratio=float(sweep_ratios[0]),
        sweep_max_ratio=float(sweep_ratios[-1]),
        sigma_threshold=SVD_RIGIDITY_SHIELD_SIGMA_THRESHOLD,
        benchmark_pmns_angles_deg=benchmark_pmns.angles_deg,
        benchmark_ckm_angles_deg=benchmark_ckm.angles_deg,
        points=tuple(sweep_points),
        max_sigma_angle_drift=max_sigma_angle_drift,
        min_lepton_overlap=min_lepton_overlap,
        min_quark_overlap=min_quark_overlap,
        decoupling_proof=decoupling_proof,
        verdict=verdict,
    )


def format_rigidity_report(report: RigiditySweepReport, *, show_points: bool = False) -> str:
    """Render the SVD Rigidity Shield report as plain text."""

    lines = [
        "SVD Rigidity Shield",
        "===================",
        f"Branch: {report.branch}",
        (
            "Representational Admissibility: "
            f"{report.benchmark_ratio_display} = {report.benchmark_ratio:.12f}"
        ),
        (
            "VEV sweep: "
            f"[{report.sweep_min_ratio:.12f}, {report.sweep_max_ratio:.12f}] across {len(report.points)} benchmark-centered steps"
        ),
        (
            "Benchmark PMNS angles [deg]: "
            + ", ".join(f"{label}={value:.12f}" for label, value in zip(PMNS_LABELS, report.benchmark_pmns_angles_deg, strict=True))
        ),
        (
            "Benchmark CKM angles [deg]: "
            + ", ".join(f"{label}={value:.12f}" for label, value in zip(CKM_LABELS, report.benchmark_ckm_angles_deg, strict=True))
        ),
        (
            "Minimum PMNS singular-subspace overlaps (L, R): "
            f"({report.min_lepton_overlap[0]:.12f}, {report.min_lepton_overlap[1]:.12f})"
        ),
        (
            "Minimum CKM singular-subspace overlaps (L, R): "
            f"({report.min_quark_overlap[0]:.12f}, {report.min_quark_overlap[1]:.12f})"
        ),
        f"Maximum sigma-weighted angle drift: {report.max_sigma_angle_drift:.6e}",
        PRIMARY_BENCHMARK_AUDIT_SUCCESS_MESSAGE,
        MIXING_SECTOR_RIGIDITY_MESSAGE,
        "",
        "Decoupling Proof",
        "----------------",
        report.decoupling_proof,
    ]

    if show_points:
        lines.extend(("", "Sweep Samples", "-------------"))
        for point in report.points:
            pmns_max = max(point.pmns_sigma_drifts)
            ckm_max = max(point.ckm_sigma_drifts)
            lines.append(
                (
                    f"r={point.ratio:.12f}  max_sigma={point.max_sigma_drift:.6e}  "
                    f"PMNS_max={pmns_max:.6e}  CKM_max={ckm_max:.6e}"
                )
            )

    lines.extend(("", report.verdict))
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute the SVD Rigidity Shield stress test.")
    parser.add_argument(
        "--steps",
        type=int,
        default=21,
        help="Number of benchmark-centered VEV-ratio samples across the ±10%% sweep.",
    )
    parser.add_argument(
        "--deformation-fraction",
        type=float,
        default=SVD_RIGIDITY_SHIELD_VEV_DEFORMATION_FRACTION,
        help="Fractional sweep around 64/312; the benchmark uses 0.10.",
    )
    parser.add_argument(
        "--show-points",
        action="store_true",
        help="Print the per-step sweep summary in addition to the benchmark verdict.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = execute_svd_rigidity_shield(
        steps=int(args.steps),
        deformation_fraction=float(args.deformation_fraction),
    )
    print(format_rigidity_report(report, show_points=bool(args.show_points)))


if __name__ == "__main__":
    main()
