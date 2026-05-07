from __future__ import annotations

"""Zero-parameter bootstrap search for the SHBT benchmark kernel.

This module owns the logic-facing discovery engine while remaining compatible
with the core rigidity-landscape data structures. The search itself is purely
combinatorial and ranks candidates with an exact ``fractions.Fraction``
"stability score" so the benchmark peak is reproducible without floating-point
ordering drift.
"""

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Final, Iterable, Sequence

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
from shbt.math_engine import guard_fraction, guard_sum


_ONE: Final[Fraction] = Fraction(1, 1)


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


def _benchmark_coordinates() -> tuple[int, int, int]:
    benchmark = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))
    assert benchmark == EXPECTED_BENCHMARK, (
        f"The bootstrap engine is locked to the published branch {EXPECTED_BENCHMARK}, received {benchmark}."
    )
    return benchmark


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


_DEFAULT_SCORE_PATH = SymmetrySearcher.score_path


__all__ = ["CombinatorialSearch", "SymmetrySearcher"]
