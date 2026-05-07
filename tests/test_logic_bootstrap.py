from __future__ import annotations

import json
from pathlib import Path

from shbt.logic.bootstrap import (
    EXPECTED_BENCHMARK,
    build_uniqueness_report,
    evaluate_kernel,
    scan_combinatorial_symmetry_space,
    write_uniqueness_report,
)


def test_evaluate_kernel_certifies_benchmark_axiom_ix_fixed_point() -> None:
    audit = evaluate_kernel(*EXPECTED_BENCHMARK)

    assert audit.coordinates == EXPECTED_BENCHMARK
    assert audit.non_singular_transport_kernel is True
    assert audit.axiom_ix_fixed_point is True
    assert audit.closure.closure_metric == 0.0
    assert audit.failure_modes == ("axiom_ix_closed",)


def test_evaluate_kernel_flags_detuned_kernel_failures() -> None:
    audit = evaluate_kernel(26, 8, 311)

    assert audit.coordinates == (26, 8, 311)
    assert audit.non_singular_transport_kernel is False
    assert audit.axiom_ix_fixed_point is False
    assert audit.closure.framing_residue > 0.0
    assert audit.closure.diophantine_gap > 0.0
    assert audit.closure.c_dark_shift > 0.0
    assert "singular_transport_kernel" in audit.failure_modes
    assert "diophantine_detuning" in audit.failure_modes
    assert "framing_defect_reopened" in audit.failure_modes


def test_scan_combinatorial_symmetry_space_proves_unique_benchmark_closure() -> None:
    scan = scan_combinatorial_symmetry_space(
        dimension_levels=range(25, 28),
        generation_levels=range(7, 10),
        parent_detunings=(-1, 0, 1),
    )

    assert scan.unique_axiom_ix_fixed_point is True
    assert scan.unique_fixed_point.coordinates == EXPECTED_BENCHMARK
    assert scan.stable_fixed_points == (scan.unique_fixed_point,)
    assert scan.runner_up_kernel.coordinates != EXPECTED_BENCHMARK
    assert scan.runner_up_kernel.closure.closure_metric > 0.0
    assert scan.closure_gap > 0.0


def test_write_uniqueness_report_writes_failure_mapping(tmp_path: Path) -> None:
    output_path = tmp_path / "uniqueness_report.json"

    written_path = write_uniqueness_report(
        output_path,
        dimension_levels=range(25, 28),
        generation_levels=range(8, 9),
        parent_detunings=(-1, 0, 1),
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert written_path == output_path
    assert payload["artifact"] == "Combinatorial Symmetry Scan Uniqueness Report"
    assert payload["benchmark_tuple"] == [26, 8, 312]
    assert payload["unique_stable_fixed_point"] == [26, 8, 312]
    assert payload["closure_gap"] > 0.0
    assert "26:8:311" in payload["alternative_kernel_failures"]
    assert payload["alternative_kernel_failures"]["26:8:311"]["axiom_ix_fixed_point"] is False


def test_build_uniqueness_report_uses_disclosed_scan_domain() -> None:
    payload = build_uniqueness_report(
        dimension_levels=range(26, 27),
        generation_levels=range(8, 9),
        parent_detunings=(-2, -1, 0, 1, 2),
    )

    assert payload["scan_domain"]["dimension_levels"] == [26]
    assert payload["scan_domain"]["generation_levels"] == [8]
    assert payload["scan_domain"]["parent_detunings"] == [-2, -1, 0, 1, 2]
    assert payload["unique_fixed_point_count"] == 1
    assert payload["statement"].startswith("The combinatorial symmetry scan certifies")
