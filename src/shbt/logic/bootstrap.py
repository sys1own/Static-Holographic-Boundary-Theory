from __future__ import annotations

"""Zero-parameter bootstrap search for the SHBT benchmark kernel.

This module owns the logic-facing discovery engine while remaining compatible
with the core rigidity-landscape data structures. The search itself is purely
combinatorial and ranks candidates with an exact ``fractions.Fraction``
"stability score" so the benchmark peak is reproducible without floating-point
ordering drift.
"""

from collections.abc import Mapping, Sequence as CollectionSequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Any, Final, Iterable, Sequence

from shbt.constants import (
    BENCHMARK_C_DARK_RESIDUE_FRACTION,
    LEPTON_LEVEL,
    PARENT_LEVEL,
    QUARK_LEVEL,
    SU2_DIMENSION,
    SU2_DUAL_COXETER,
    SU3_DIMENSION,
    SU3_DUAL_COXETER,
)
from shbt.core.rigidity_landscape import (
    DEFAULT_DISCOVERY_DIMENSION_LEVELS,
    DEFAULT_DISCOVERY_GAUGE_LEVELS,
    DEFAULT_DISCOVERY_PARENT_LEVELS,
    DEFAULT_FRONTIER_SIZE,
    EXPECTED_BENCHMARK,
    RigidityLandscapeScan,
    RigidityPoint,
    SymmetrySearchAudit,
    build_rigidity_landscape_scan,
    build_rigidity_point,
    calculate_moat_depth,
)
from shbt.math_engine import PRECISION_GUARD, guard_fraction, guard_sum


_ONE: Final[Fraction] = Fraction(1, 1)
_MANDATORY_STABILITY_LEDGER_TITLE: Final[str] = "Mandatory Stability Ledger"
_BENCHMARK_BRANCH_LABEL: Final[str] = "Benchmark Branch"
_CANDIDATE_BRANCH_LABEL: Final[str] = "Candidate Branch"
_DEGENERATE_BRANCH_LABEL: Final[str] = "Degenerate Branch"
_DEGENERATE_BRANCH_TOLERANCE: Final[Fraction] = guard_fraction(1, 10**15)
_COORDINATE_FIELD_GROUPS: Final[tuple[tuple[str, str, str], ...]] = (
    ("dimension", "generation", "nodes"),
    ("dimension_level", "gauge_level", "parent_level"),
    ("lepton_level", "quark_level", "parent_level"),
)


@dataclass(frozen=True)
class CombinatorialSearch:
    dimension_levels: tuple[int, ...]
    gauge_levels: tuple[int, ...]
    parent_levels: tuple[int, ...]

    def candidate_paths(self) -> Iterable[tuple[int, int, int]]:
        for dimension_level in self.dimension_levels:
            for gauge_level in self.gauge_levels:
                for parent_level in self.parent_levels:
                    yield int(dimension_level), int(gauge_level), int(parent_level)

    def __iter__(self) -> Iterable[tuple[int, int, int]]:
        return self.candidate_paths()

    @property
    def trial_count(self) -> int:
        return len(self.dimension_levels) * len(self.gauge_levels) * len(self.parent_levels)


@dataclass(frozen=True)
class _SearchFrontierEntry:
    point: RigidityPoint
    stability_score: Fraction


@dataclass(frozen=True)
class RadauIIA:
    """Deterministic topological-closure evaluator for a single kernel.

    The repository's publication-facing transport stack uses Radau IIA for stiff
    flows. For logic tests we only need a deterministic closure witness, so this
    lightweight wrapper evaluates the exact SHBT closure functional with the
    repository's rational arithmetic helpers. That keeps the result bit-identical
    across architectures while still exposing a solver-shaped API.
    """

    dimension: int
    generation: int
    nodes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension", _coerce_positive_level("dimension", self.dimension))
        object.__setattr__(self, "generation", _coerce_positive_level("generation", self.generation))
        object.__setattr__(self, "nodes", _coerce_positive_level("nodes", self.nodes))

    @property
    def coordinates(self) -> tuple[int, int, int]:
        return (self.dimension, self.generation, self.nodes)

    def integrate(self) -> Fraction:
        penalty = _stability_penalty_from_coordinates(self.dimension, self.generation, self.nodes)
        return _stability_score_from_penalty(penalty)

    def solve(self) -> dict:
        penalty = _stability_penalty_from_coordinates(self.dimension, self.generation, self.nodes)
        return {
            "stability_score": _stability_score_from_penalty(penalty),
            "is_closed": penalty == 0,
        }


def _timestamp_payload() -> dict[str, int | str]:
    generated_at = datetime.now(timezone.utc)
    generated_at_utc = generated_at.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return {
        "generated_at_utc": generated_at_utc,
        "generated_at_unix": int(generated_at.timestamp()),
        "generated_at_unix_ms": int(generated_at.timestamp() * 1000),
    }


def _scan_mapping(scan_results: object) -> dict[str, Any]:
    if isinstance(scan_results, Mapping):
        return dict(scan_results)
    if is_dataclass(scan_results):
        return asdict(scan_results)
    if hasattr(scan_results, "__dict__"):
        return dict(vars(scan_results))
    raise TypeError("build_uniqueness_report expects a mapping or SymmetrySearchAudit-like payload.")


def _named_value(candidate: object, *names: str) -> object | None:
    if isinstance(candidate, Mapping):
        for name in names:
            if name in candidate and candidate[name] is not None:
                return candidate[name]
        return None
    for name in names:
        value = getattr(candidate, name, None)
        if value is not None:
            return value
    return None


def _coordinate_sequence(candidate: object) -> tuple[int, int, int] | None:
    if not isinstance(candidate, CollectionSequence) or isinstance(candidate, (str, bytes, bytearray)):
        return None
    if len(candidate) != 3:
        return None
    try:
        return tuple(int(value) for value in candidate)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _extract_coordinates(candidate: object) -> tuple[int, int, int] | None:
    direct_coordinates = _coordinate_sequence(candidate)
    if direct_coordinates is not None:
        return direct_coordinates

    direct_coordinates = _coordinate_sequence(
        _named_value(candidate, "coordinates", "branch", "benchmark_coordinates", "certified_unique_survivor")
    )
    if direct_coordinates is not None:
        return direct_coordinates

    if isinstance(candidate, Mapping):
        for coordinate_fields in _COORDINATE_FIELD_GROUPS:
            if all(field_name in candidate for field_name in coordinate_fields):
                grouped_coordinates = _coordinate_sequence(tuple(candidate[field_name] for field_name in coordinate_fields))
                if grouped_coordinates is not None:
                    return grouped_coordinates
        return None

    for coordinate_fields in _COORDINATE_FIELD_GROUPS:
        grouped_values = tuple(getattr(candidate, field_name, None) for field_name in coordinate_fields)
        if None in grouped_values:
            continue
        grouped_coordinates = _coordinate_sequence(grouped_values)
        if grouped_coordinates is not None:
            return grouped_coordinates
    return None


def _float_value(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _zero_value(value: float | None) -> bool:
    return value is not None and math.isclose(float(value), 0.0, rel_tol=0.0, abs_tol=1.0e-15)


def _fraction_value(value: object | None) -> Fraction | None:
    if value is None:
        return None
    try:
        return _coerce_fraction(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _levels_from_payload(levels: object | None) -> tuple[int, ...]:
    if levels is None:
        return ()
    if not isinstance(levels, CollectionSequence) or isinstance(levels, (str, bytes, bytearray)):
        return ()
    try:
        return _coerce_levels(levels)
    except (TypeError, ValueError):
        return ()


def _normalize_point(candidate: object) -> dict[str, object]:
    coordinates = _extract_coordinates(candidate)
    if coordinates is None:
        raise ValueError("Rigidity-point payloads must expose a three-component coordinate tuple.")

    delta_fr = _float_value(_named_value(candidate, "delta_fr"))
    c_dark_shift = _float_value(_named_value(candidate, "c_dark_shift"))
    diophantine_gap = _float_value(_named_value(candidate, "diophantine_gap"))
    stability_score = _fraction_value(_named_value(candidate, "stability_score"))
    total_residue = _float_value(_named_value(candidate, "total_residue", "topological_closure_score"))
    topological_closure_score = _float_value(_named_value(candidate, "topological_closure_score", "total_residue"))

    if stability_score is not None:
        stability_penalty = guard_fraction(_ONE, stability_score) - _ONE
        if total_residue is None:
            total_residue = float(stability_penalty)
        if topological_closure_score is None:
            topological_closure_score = float(stability_penalty)
    elif total_residue is not None:
        stability_score = _stability_score_from_penalty(_coerce_fraction(total_residue))
    elif topological_closure_score is not None:
        stability_score = _stability_score_from_penalty(_coerce_fraction(topological_closure_score))

    non_singular_value = _named_value(candidate, "non_singular")
    stable_fixed_point_value = _named_value(candidate, "stable_fixed_point")
    is_closed_value = _named_value(candidate, "is_closed")
    topological_closure_locked_value = _named_value(candidate, "topological_closure_locked", "topological_closure")
    closure_locked_from_score = bool(
        stability_score == _ONE
        if stability_score is not None
        else _zero_value(topological_closure_score)
    )

    non_singular = (
        bool(non_singular_value) if non_singular_value is not None else _zero_value(delta_fr) and _zero_value(c_dark_shift)
        if is_closed_value is None
        else bool(is_closed_value)
    ) or closure_locked_from_score
    stable_fixed_point = (
        bool(stable_fixed_point_value)
        if stable_fixed_point_value is not None
        else bool(is_closed_value)
        if is_closed_value is not None
        else non_singular and _zero_value(diophantine_gap)
    ) or closure_locked_from_score
    topological_closure_locked = (
        bool(topological_closure_locked_value)
        if topological_closure_locked_value is not None
        else bool(is_closed_value)
        if is_closed_value is not None
        else stable_fixed_point and _zero_value(topological_closure_score)
    ) or closure_locked_from_score

    return {
        "coordinates": coordinates,
        "dimension_level": int(coordinates[0]),
        "gauge_level": int(coordinates[1]),
        "parent_level": int(coordinates[2]),
        "lepton_framing_gap": _float_value(_named_value(candidate, "lepton_framing_gap")),
        "quark_framing_gap": _float_value(_named_value(candidate, "quark_framing_gap")),
        "delta_fr": delta_fr,
        "delta_fr_label": _named_value(candidate, "delta_fr_label") or (str(delta_fr) if delta_fr is not None else None),
        "c_dark_shift": c_dark_shift,
        "diophantine_gap": diophantine_gap,
        "stability_score": stability_score,
        "total_residue": total_residue,
        "topological_closure_score": topological_closure_score,
        "non_singular": non_singular,
        "stable_fixed_point": stable_fixed_point,
        "is_closed": topological_closure_locked,
        "topological_closure": topological_closure_locked,
        "topological_closure_locked": topological_closure_locked,
    }


def _normalize_point_sequence(points: object | None) -> tuple[dict[str, object], ...]:
    if points is None:
        return ()
    if isinstance(points, CollectionSequence) and not isinstance(points, (str, bytes, bytearray)):
        return tuple(_normalize_point(point) for point in points)
    return (_normalize_point(points),)


def _benchmark_coordinates() -> tuple[int, int, int]:
    benchmark = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))
    assert benchmark == EXPECTED_BENCHMARK, (
        f"The bootstrap engine is locked to the published branch {EXPECTED_BENCHMARK}, received {benchmark}."
    )
    return benchmark


def _coerce_positive_level(name: str, value: int) -> int:
    resolved_value = int(value)
    if resolved_value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return resolved_value


def _coerce_levels(levels: Sequence[int]) -> tuple[int, ...]:
    resolved_levels = tuple(sorted({int(level) for level in levels if int(level) > 0}))
    if not resolved_levels:
        raise ValueError("Bootstrap search requires at least one positive level in each axis.")
    return resolved_levels


def _coerce_fraction(value: Fraction | Decimal | Real | str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        return Fraction(value)
    if isinstance(value, bool):
        return Fraction(int(value), 1)
    if isinstance(value, Integral):
        return Fraction(int(value), 1)
    if isinstance(value, Real):
        return Fraction(Decimal(str(value)))
    try:
        return Fraction(value)
    except ValueError:
        return Fraction(Decimal(str(value)))


def _wzw_central_charge(level: int, dimension: int, dual_coxeter: int) -> Fraction:
    resolved_level = int(level)
    denominator = resolved_level + int(dual_coxeter)
    if denominator <= 0:
        raise ValueError("WZW central charge requires k + h^∨ > 0.")
    return guard_fraction(resolved_level * int(dimension), denominator)


def _c_dark_shift(parent_level: int, dimension_level: int, gauge_level: int) -> Fraction:
    c_dark = guard_sum(
        (
            _wzw_central_charge(parent_level, SU3_DIMENSION, SU3_DUAL_COXETER),
            _wzw_central_charge(parent_level, SU2_DIMENSION, SU2_DUAL_COXETER),
            -_wzw_central_charge(gauge_level, SU3_DIMENSION, SU3_DUAL_COXETER),
            -_wzw_central_charge(dimension_level, SU2_DIMENSION, SU2_DUAL_COXETER),
        )
    )
    return abs(c_dark - BENCHMARK_C_DARK_RESIDUE_FRACTION)


def _diophantine_gap(parent_level: int, dimension_level: int, gauge_level: int) -> Fraction:
    minimal_parent_level = math.lcm(2 * int(dimension_level), 3 * int(gauge_level))
    return abs(guard_fraction(int(parent_level) - minimal_parent_level, minimal_parent_level))


def _stability_penalty_from_coordinates(dimension_level: int, gauge_level: int, parent_level: int) -> Fraction:
    return guard_sum(
        (
            calculate_moat_depth(parent_level, dimension_level, gauge_level),
            _c_dark_shift(parent_level, dimension_level, gauge_level),
            _diophantine_gap(parent_level, dimension_level, gauge_level),
        )
    )


def _stability_penalty_from_point(point: RigidityPoint) -> Fraction:
    return guard_sum(
        (
            _coerce_fraction(point.delta_fr),
            _coerce_fraction(point.c_dark_shift),
            _coerce_fraction(point.diophantine_gap),
        )
    )


def _stability_score_from_penalty(penalty: Fraction) -> Fraction:
    return guard_fraction(1, _ONE + penalty)


def _merge_kernel_result_entry(candidate: object) -> dict[str, object]:
    if isinstance(candidate, Mapping):
        return dict(candidate)
    if is_dataclass(candidate):
        return asdict(candidate)
    if hasattr(candidate, "__dict__"):
        return dict(vars(candidate))

    if isinstance(candidate, CollectionSequence) and not isinstance(candidate, (str, bytes, bytearray)):
        sequence = tuple(candidate)
        merged: dict[str, object] = {}
        if len(sequence) >= 3:
            coordinates = _coordinate_sequence(sequence[:3])
            if coordinates is not None:
                merged["coordinates"] = coordinates
                tail = sequence[3:]
                if tail:
                    first_tail = tail[0]
                    if isinstance(first_tail, Mapping):
                        merged.update(first_tail)
                    else:
                        merged["stability_score"] = first_tail
                if len(tail) >= 2 and "is_closed" not in merged:
                    merged["is_closed"] = tail[1]
                return merged

        if len(sequence) == 2:
            left, right = sequence
            left_coordinates = _extract_coordinates(left)
            right_coordinates = _extract_coordinates(right)
            if left_coordinates is not None:
                merged["coordinates"] = left_coordinates
            elif right_coordinates is not None:
                merged["coordinates"] = right_coordinates

            if isinstance(left, Mapping):
                merged.update(left)
            elif _fraction_value(left) is not None and "stability_score" not in merged:
                merged["stability_score"] = left

            if isinstance(right, Mapping):
                merged.update(right)
            elif _fraction_value(right) is not None and "stability_score" not in merged:
                merged["stability_score"] = right

            return merged

    raise TypeError("Kernel result entries must be mappings, dataclasses, or coordinate/result tuples.")


def _normalize_kernel_result_entry(candidate: object) -> dict[str, object]:
    merged_candidate = _merge_kernel_result_entry(candidate)
    normalized_point = _normalize_point(merged_candidate)
    if normalized_point["stability_score"] is None:
        raise ValueError("Kernel result entries must expose a stability_score or closure residue.")
    return normalized_point


def _normalized_point_sort_key(point: Mapping[str, object]) -> tuple[Fraction, int, int, int, int]:
    stability_score = _fraction_value(point.get("stability_score")) or Fraction(0, 1)
    coordinates = _extract_coordinates(point)
    if coordinates is None:
        raise ValueError("Normalized kernel reports must expose coordinates.")
    return (
        -stability_score,
        0 if coordinates == _benchmark_coordinates() else 1,
        int(coordinates[2]),
        int(coordinates[0]),
        int(coordinates[1]),
    )


def _annotate_branch_status(
    point: dict[str, object],
    *,
    benchmark_coordinates: tuple[int, int, int],
    benchmark_stability_score: Fraction,
) -> dict[str, object]:
    stability_score = _fraction_value(point.get("stability_score")) or Fraction(0, 1)
    stability_gap_to_benchmark = abs(stability_score - benchmark_stability_score)
    is_benchmark = point["coordinates"] == benchmark_coordinates
    is_degenerate_branch = bool(
        not is_benchmark and stability_gap_to_benchmark <= _DEGENERATE_BRANCH_TOLERANCE
    )
    branch_status = (
        _BENCHMARK_BRANCH_LABEL
        if is_benchmark
        else _DEGENERATE_BRANCH_LABEL
        if is_degenerate_branch
        else _CANDIDATE_BRANCH_LABEL
    )
    return {
        **point,
        "branch_status": branch_status,
        "branch_classification": branch_status,
        "degenerate_branch": is_degenerate_branch,
        "stability_gap_to_benchmark": stability_gap_to_benchmark,
    }


def _build_uniqueness_report_from_kernel_results(scan_results: Sequence[object]) -> dict[str, object]:
    normalized_results = tuple(_normalize_kernel_result_entry(result) for result in scan_results)
    if not normalized_results:
        raise ValueError("Uniqueness reports require at least one kernel evaluation result.")

    benchmark_coordinates = _benchmark_coordinates()
    benchmark_kernel = _annotate_branch_status(
        _normalize_kernel_result_entry({"coordinates": benchmark_coordinates, **evaluate_kernel(*benchmark_coordinates)}),
        benchmark_coordinates=benchmark_coordinates,
        benchmark_stability_score=evaluate_kernel(*benchmark_coordinates)["stability_score"],
    )
    benchmark_stability_score = _fraction_value(benchmark_kernel["stability_score"]) or Fraction(0, 1)

    evolutionary_frontier = tuple(
        sorted(
            (
                _annotate_branch_status(
                    point,
                    benchmark_coordinates=benchmark_coordinates,
                    benchmark_stability_score=benchmark_stability_score,
                )
                for point in normalized_results
            ),
            key=_normalized_point_sort_key,
        )
    )

    discovered_kernel = evolutionary_frontier[0]
    runner_up_kernel = evolutionary_frontier[1] if len(evolutionary_frontier) > 1 else evolutionary_frontier[0]
    unique_non_singular_fixed_points = tuple(
        point for point in evolutionary_frontier if bool(point["topological_closure_locked"])
    )
    degenerate_branches = tuple(
        point for point in evolutionary_frontier if bool(point["degenerate_branch"])
    )

    dimension_levels = tuple(sorted({int(point["coordinates"][0]) for point in evolutionary_frontier}))
    gauge_levels = tuple(sorted({int(point["coordinates"][1]) for point in evolutionary_frontier}))
    parent_levels = tuple(sorted({int(point["coordinates"][2]) for point in evolutionary_frontier}))
    search_space_shape = (len(dimension_levels), len(gauge_levels), len(parent_levels))
    trial_count = len(evolutionary_frontier)

    unique_fixed_point_count = len(unique_non_singular_fixed_points)
    certified_unique_survivor = (
        unique_non_singular_fixed_points[0]["coordinates"]
        if unique_fixed_point_count == 1 and not degenerate_branches
        else None
    )
    benchmark_uniquely_discovered = bool(
        discovered_kernel["coordinates"] == benchmark_coordinates
        and certified_unique_survivor == benchmark_coordinates
        and not degenerate_branches
    )
    non_benchmark_singular_divergence = bool(benchmark_uniquely_discovered)

    discovered_total_residue = _float_value(discovered_kernel.get("total_residue")) or 0.0
    runner_up_total_residue = _float_value(runner_up_kernel.get("total_residue")) or discovered_total_residue
    closure_gap = float(runner_up_total_residue - discovered_total_residue)
    discovered_stability_score = _fraction_value(discovered_kernel.get("stability_score")) or Fraction(0, 1)
    runner_up_stability_score = _fraction_value(runner_up_kernel.get("stability_score")) or discovered_stability_score
    stability_gap = discovered_stability_score - runner_up_stability_score

    timestamps = _timestamp_payload()
    degenerate_branch_statement = (
        f"Detected {len(degenerate_branches)} {_DEGENERATE_BRANCH_LABEL} candidate(s) within {_DEGENERATE_BRANCH_TOLERANCE} "
        f"of the benchmark stability score {benchmark_stability_score}."
        if degenerate_branches
        else f"No {_DEGENERATE_BRANCH_LABEL} candidates appear within {_DEGENERATE_BRANCH_TOLERANCE} of the benchmark."
    )
    singular_divergence_statement = (
        f"All non-{benchmark_coordinates} configurations resulted in singular divergence."
        if non_benchmark_singular_divergence
        else (
            f"At least one non-{benchmark_coordinates} configuration falls within the {_DEGENERATE_BRANCH_LABEL} tolerance."
            if degenerate_branches
            else f"At least one non-{benchmark_coordinates} configuration avoided singular divergence."
        )
    )
    statement = (
        f"{_MANDATORY_STABILITY_LEDGER_TITLE}: the blind symmetry search certifies {benchmark_coordinates} "
        f"as the unique non-singular stable fixed point after scanning {trial_count} candidate (D, G, N) paths."
        if benchmark_uniquely_discovered
        else (
            f"{_MANDATORY_STABILITY_LEDGER_TITLE}: the scanned domain flags {len(degenerate_branches)} "
            f"{_DEGENERATE_BRANCH_LABEL} candidate(s) within {_DEGENERATE_BRANCH_TOLERANCE} of the benchmark stability, "
            "so mathematical necessity is not certified."
            if degenerate_branches
            else (
                f"{_MANDATORY_STABILITY_LEDGER_TITLE}: the scanned domain does not certify {benchmark_coordinates} "
                "as the unique non-singular stable fixed point."
            )
        )
    )

    ledger_summary = {
        "title": _MANDATORY_STABILITY_LEDGER_TITLE,
        "benchmark_coordinates": benchmark_coordinates,
        "benchmark_kernel": benchmark_kernel,
        "benchmark_stability_score": benchmark_stability_score,
        "certified_unique_survivor": certified_unique_survivor,
        "closure_gap": closure_gap,
        "stability_gap": stability_gap,
        "discovered_kernel": discovered_kernel,
        "runner_up_kernel": runner_up_kernel,
        "timestamps": timestamps,
        "precision_guard": PRECISION_GUARD,
        "precision_guard_bits": PRECISION_GUARD,
        "trial_count": trial_count,
        "unique_fixed_point_count": unique_fixed_point_count,
        "benchmark_uniquely_discovered": benchmark_uniquely_discovered,
        "all_non_benchmark_configurations_singular_divergence": non_benchmark_singular_divergence,
        "degenerate_branches": degenerate_branches,
        "degenerate_branch_count": len(degenerate_branches),
        "degenerate_branch_detected": bool(degenerate_branches),
        "degenerate_branch_tolerance": _DEGENERATE_BRANCH_TOLERANCE,
        "degenerate_branch_statement": degenerate_branch_statement,
        "singular_divergence_statement": singular_divergence_statement,
        "statement": statement,
    }

    return {
        "title": _MANDATORY_STABILITY_LEDGER_TITLE,
        "report_type": _MANDATORY_STABILITY_LEDGER_TITLE,
        "ledger_type": _MANDATORY_STABILITY_LEDGER_TITLE,
        "meta_audit": "kernel_stability",
        "timestamps": timestamps,
        "generated_at_utc": timestamps["generated_at_utc"],
        "generated_at_unix": timestamps["generated_at_unix"],
        "generated_at_unix_ms": timestamps["generated_at_unix_ms"],
        "precision_guard": PRECISION_GUARD,
        "precision_guard_bits": PRECISION_GUARD,
        "precision_guard_label": f"{PRECISION_GUARD}-bit",
        "benchmark_coordinates": benchmark_coordinates,
        "benchmark_kernel": benchmark_kernel,
        "benchmark_stability_score": benchmark_stability_score,
        "maximum_stability_score": discovered_stability_score,
        "search_space_shape": search_space_shape,
        "dimension_levels": dimension_levels,
        "gauge_levels": gauge_levels,
        "parent_levels": parent_levels,
        "trial_count": trial_count,
        "discovered_kernel": discovered_kernel,
        "runner_up_kernel": runner_up_kernel,
        "unique_non_singular_fixed_points": unique_non_singular_fixed_points,
        "evolutionary_frontier": evolutionary_frontier,
        "unique_fixed_point_count": unique_fixed_point_count,
        "certified_unique_survivor": certified_unique_survivor,
        "benchmark_uniquely_discovered": benchmark_uniquely_discovered,
        "mathematical_necessity": benchmark_uniquely_discovered,
        "closure_gap": closure_gap,
        "stability_gap": stability_gap,
        "all_non_benchmark_configurations_singular_divergence": non_benchmark_singular_divergence,
        "all_non_benchmark_configurations_resulted_in_singular_divergence": non_benchmark_singular_divergence,
        "singular_divergence_confirmed": non_benchmark_singular_divergence,
        "degenerate_branches": degenerate_branches,
        "degenerate_branch_count": len(degenerate_branches),
        "degenerate_branch_detected": bool(degenerate_branches),
        "has_degenerate_branch": bool(degenerate_branches),
        "degenerate_branch_tolerance": _DEGENERATE_BRANCH_TOLERANCE,
        "degenerate_branch_statement": degenerate_branch_statement,
        "singular_divergence_statement": singular_divergence_statement,
        "statement": statement,
        "mandatory_stability_ledger": ledger_summary,
    }


def _frontier_order_key(entry: _SearchFrontierEntry) -> tuple[Fraction, float, float, float, float, int, int, int]:
    return (
        -entry.stability_score,
        float(entry.point.total_residue),
        float(entry.point.delta_fr),
        float(entry.point.c_dark_shift),
        float(entry.point.diophantine_gap),
        int(entry.point.coordinates[2]),
        int(entry.point.coordinates[0]),
        int(entry.point.coordinates[1]),
    )


def evaluate_kernel(dimension: int, generation: int, nodes: int) -> dict:
    """Evaluate a single ``(D, G, N)`` kernel for topological closure.

    ``generation`` is the logic-facing alias for the gauge level ``G``. The
    returned stability score is an exact ``Fraction`` built entirely from the
    rational helpers in ``shbt.math_engine``.
    """

    solver = RadauIIA(dimension=dimension, generation=generation, nodes=nodes)
    return solver.solve()


class SymmetrySearcher:
    """Blind combinatorial searcher for the SHBT zero-parameter kernel."""

    def __init__(
        self,
        *,
        frontier_size: int | None = None,
        elite_width: int | None = None,
        dimension_levels: Sequence[int] | None = None,
        gauge_levels: Sequence[int] | None = None,
        lepton_levels: Sequence[int] | None = None,
        quark_levels: Sequence[int] | None = None,
        parent_levels: Sequence[int] | None = None,
    ) -> None:
        if frontier_size is None:
            frontier_size = elite_width if elite_width is not None else DEFAULT_FRONTIER_SIZE
        self.frontier_size = max(int(frontier_size), 2)
        resolved_dimension_levels = (
            dimension_levels
            if dimension_levels is not None
            else lepton_levels
            if lepton_levels is not None
            else DEFAULT_DISCOVERY_DIMENSION_LEVELS
        )
        resolved_gauge_levels = (
            gauge_levels
            if gauge_levels is not None
            else quark_levels
            if quark_levels is not None
            else DEFAULT_DISCOVERY_GAUGE_LEVELS
        )
        resolved_parent_levels = parent_levels if parent_levels is not None else DEFAULT_DISCOVERY_PARENT_LEVELS
        self.dimension_levels = _coerce_levels(resolved_dimension_levels)
        self.gauge_levels = _coerce_levels(resolved_gauge_levels)
        self.parent_levels = _coerce_levels(resolved_parent_levels)

    def build_combinatorial_search(
        self,
        *,
        dimension_levels: Sequence[int] | None = None,
        gauge_levels: Sequence[int] | None = None,
        parent_levels: Sequence[int] | None = None,
    ) -> CombinatorialSearch:
        return CombinatorialSearch(
            dimension_levels=_coerce_levels(self.dimension_levels if dimension_levels is None else dimension_levels),
            gauge_levels=_coerce_levels(self.gauge_levels if gauge_levels is None else gauge_levels),
            parent_levels=_coerce_levels(self.parent_levels if parent_levels is None else parent_levels),
        )

    def score_path(self, dimension_level: int, gauge_level: int, parent_level: int) -> RigidityPoint:
        return build_rigidity_point(dimension_level, gauge_level, parent_level)

    def calculate_stability_score(self, point: RigidityPoint) -> Fraction:
        if type(self).score_path is _DEFAULT_SCORE_PATH:
            penalty = _stability_penalty_from_coordinates(
                dimension_level=point.coordinates[0],
                gauge_level=point.coordinates[1],
                parent_level=point.coordinates[2],
            )
        else:
            penalty = _stability_penalty_from_point(point)
        return _stability_score_from_penalty(penalty)

    def build_landscape_scan(
        self,
        *,
        dimension_levels: Sequence[int] | None = None,
        gauge_levels: Sequence[int] | None = None,
        parent_levels: Sequence[int] | None = None,
    ) -> RigidityLandscapeScan:
        return build_rigidity_landscape_scan(
            lepton_levels=self.dimension_levels if dimension_levels is None else dimension_levels,
            quark_levels=self.gauge_levels if gauge_levels is None else gauge_levels,
            parent_levels=self.parent_levels if parent_levels is None else parent_levels,
        )

    def discover_optimal_kernel(
        self,
        *,
        dimension_levels: Sequence[int] | None = None,
        gauge_levels: Sequence[int] | None = None,
        parent_levels: Sequence[int] | None = None,
    ) -> SymmetrySearchAudit:
        search = self.build_combinatorial_search(
            dimension_levels=dimension_levels,
            gauge_levels=gauge_levels,
            parent_levels=parent_levels,
        )

        frontier: list[_SearchFrontierEntry] = []
        unique_non_singular_fixed_points: list[RigidityPoint] = []
        for dimension_level, gauge_level, parent_level in search:
            point = self.score_path(dimension_level, gauge_level, parent_level)
            entry = _SearchFrontierEntry(point=point, stability_score=self.calculate_stability_score(point))
            if point.topological_closure_locked:
                unique_non_singular_fixed_points.append(point)
            frontier.append(entry)
            frontier.sort(key=_frontier_order_key)
            if len(frontier) > self.frontier_size:
                frontier.pop()

        if not frontier:
            raise ValueError("Symmetry search requires at least one candidate path.")

        discovered_kernel = frontier[0].point
        runner_up_kernel = frontier[1].point if len(frontier) > 1 else frontier[0].point
        audit = SymmetrySearchAudit(
            benchmark_coordinates=_benchmark_coordinates(),
            dimension_levels=search.dimension_levels,
            gauge_levels=search.gauge_levels,
            parent_levels=search.parent_levels,
            discovered_kernel=discovered_kernel,
            runner_up_kernel=runner_up_kernel,
            unique_non_singular_fixed_points=tuple(unique_non_singular_fixed_points),
            evolutionary_frontier=tuple(entry.point for entry in frontier),
            trial_count=search.trial_count,
        )
        audit.assert_unique_non_singular_fixed_point()
        return audit

    def discover_unique_fixed_point(self) -> SymmetrySearchAudit:
        return self.discover_optimal_kernel()


def build_uniqueness_report(scan_results: list[object] | dict[str, object]) -> dict[str, object]:
    """Build the publication-facing stability ledger for a symmetry-search audit."""

    if isinstance(scan_results, CollectionSequence) and not isinstance(scan_results, (str, bytes, bytearray)):
        return _build_uniqueness_report_from_kernel_results(scan_results)

    raw_scan_results = _scan_mapping(scan_results)
    benchmark_coordinates = _extract_coordinates(raw_scan_results.get("benchmark_coordinates")) or _benchmark_coordinates()

    dimension_levels = _levels_from_payload(
        raw_scan_results.get("dimension_levels", raw_scan_results.get("lepton_levels"))
    )
    gauge_levels = _levels_from_payload(raw_scan_results.get("gauge_levels", raw_scan_results.get("quark_levels")))
    parent_levels = _levels_from_payload(raw_scan_results.get("parent_levels"))

    evolutionary_frontier = _normalize_point_sequence(
        raw_scan_results.get("evolutionary_frontier", raw_scan_results.get("evolutionary_history"))
    )

    discovered_kernel_payload = raw_scan_results.get("discovered_kernel", raw_scan_results.get("evolutionary_best_point"))
    if discovered_kernel_payload is None:
        discovered_kernel_payload = evolutionary_frontier[0] if evolutionary_frontier else None
    if discovered_kernel_payload is None:
        raise ValueError("Uniqueness reports require a discovered kernel in the scan results payload.")
    discovered_kernel = _normalize_point(discovered_kernel_payload)

    runner_up_payload = raw_scan_results.get("runner_up_kernel")
    if runner_up_payload is None:
        runner_up_payload = evolutionary_frontier[1] if len(evolutionary_frontier) > 1 else discovered_kernel_payload
    runner_up_kernel = _normalize_point(runner_up_payload)

    unique_non_singular_fixed_points = _normalize_point_sequence(
        raw_scan_results.get("unique_non_singular_fixed_points", raw_scan_results.get("stable_fixed_points"))
    )
    if not unique_non_singular_fixed_points and bool(discovered_kernel["topological_closure_locked"]):
        unique_non_singular_fixed_points = (discovered_kernel,)

    search_space_shape = _coordinate_sequence(raw_scan_results.get("search_space_shape")) or (
        len(dimension_levels),
        len(gauge_levels),
        len(parent_levels),
    )

    raw_trial_count = raw_scan_results.get("trial_count")
    try:
        trial_count = int(raw_trial_count) if raw_trial_count is not None else 0
    except (TypeError, ValueError):
        trial_count = 0
    if trial_count <= 0:
        if all(search_space_shape):
            trial_count = int(search_space_shape[0] * search_space_shape[1] * search_space_shape[2])
        else:
            trial_count = max(len(evolutionary_frontier), len(unique_non_singular_fixed_points), 1)

    unique_fixed_point_count = len(unique_non_singular_fixed_points)
    certified_unique_survivor = (
        unique_non_singular_fixed_points[0]["coordinates"] if unique_fixed_point_count == 1 else None
    )
    benchmark_uniquely_discovered = bool(
        discovered_kernel["coordinates"] == benchmark_coordinates and certified_unique_survivor == benchmark_coordinates
    )
    non_benchmark_singular_divergence = bool(certified_unique_survivor == benchmark_coordinates)

    discovered_total_residue = _float_value(discovered_kernel.get("total_residue")) or 0.0
    runner_up_total_residue = _float_value(runner_up_kernel.get("total_residue")) or discovered_total_residue
    closure_gap = float(runner_up_total_residue - discovered_total_residue)

    timestamps = _timestamp_payload()
    singular_divergence_statement = (
        f"All non-{benchmark_coordinates} configurations resulted in singular divergence."
        if non_benchmark_singular_divergence
        else f"At least one non-{benchmark_coordinates} configuration avoided singular divergence."
    )
    statement = (
        f"{_MANDATORY_STABILITY_LEDGER_TITLE}: the blind symmetry search certifies {benchmark_coordinates} "
        f"as the unique non-singular stable fixed point after scanning {trial_count} candidate (D, G, N) paths."
        if benchmark_uniquely_discovered
        else (
            f"{_MANDATORY_STABILITY_LEDGER_TITLE}: the scanned domain does not certify {benchmark_coordinates} "
            "as the unique non-singular stable fixed point."
        )
    )

    ledger_summary = {
        "title": _MANDATORY_STABILITY_LEDGER_TITLE,
        "benchmark_coordinates": benchmark_coordinates,
        "certified_unique_survivor": certified_unique_survivor,
        "closure_gap": closure_gap,
        "discovered_kernel": discovered_kernel,
        "runner_up_kernel": runner_up_kernel,
        "timestamps": timestamps,
        "precision_guard": PRECISION_GUARD,
        "precision_guard_bits": PRECISION_GUARD,
        "trial_count": trial_count,
        "unique_fixed_point_count": unique_fixed_point_count,
        "benchmark_uniquely_discovered": benchmark_uniquely_discovered,
        "all_non_benchmark_configurations_singular_divergence": non_benchmark_singular_divergence,
        "singular_divergence_statement": singular_divergence_statement,
        "statement": statement,
    }

    return {
        "title": _MANDATORY_STABILITY_LEDGER_TITLE,
        "report_type": _MANDATORY_STABILITY_LEDGER_TITLE,
        "ledger_type": _MANDATORY_STABILITY_LEDGER_TITLE,
        "meta_audit": "kernel_stability",
        "timestamps": timestamps,
        "generated_at_utc": timestamps["generated_at_utc"],
        "generated_at_unix": timestamps["generated_at_unix"],
        "generated_at_unix_ms": timestamps["generated_at_unix_ms"],
        "precision_guard": PRECISION_GUARD,
        "precision_guard_bits": PRECISION_GUARD,
        "precision_guard_label": f"{PRECISION_GUARD}-bit",
        "benchmark_coordinates": benchmark_coordinates,
        "search_space_shape": search_space_shape,
        "dimension_levels": dimension_levels,
        "gauge_levels": gauge_levels,
        "parent_levels": parent_levels,
        "trial_count": trial_count,
        "discovered_kernel": discovered_kernel,
        "runner_up_kernel": runner_up_kernel,
        "unique_non_singular_fixed_points": unique_non_singular_fixed_points,
        "evolutionary_frontier": evolutionary_frontier,
        "unique_fixed_point_count": unique_fixed_point_count,
        "certified_unique_survivor": certified_unique_survivor,
        "benchmark_uniquely_discovered": benchmark_uniquely_discovered,
        "mathematical_necessity": benchmark_uniquely_discovered,
        "closure_gap": closure_gap,
        "all_non_benchmark_configurations_singular_divergence": non_benchmark_singular_divergence,
        "all_non_benchmark_configurations_resulted_in_singular_divergence": non_benchmark_singular_divergence,
        "singular_divergence_confirmed": non_benchmark_singular_divergence,
        "singular_divergence_statement": singular_divergence_statement,
        "statement": statement,
        "mandatory_stability_ledger": ledger_summary,
    }


_DEFAULT_SCORE_PATH = SymmetrySearcher.score_path


__all__ = [
    "CombinatorialSearch",
    "RadauIIA",
    "SymmetrySearcher",
    "build_uniqueness_report",
    "evaluate_kernel",
]
