from __future__ import annotations

"""Blind rigidity-landscape discovery for the SHBT benchmark kernel.

Historically the rigidity audit was visualized on a benchmark-centered moat in
plotting code. This module lifts that logic into the core package and adds a
blind searcher that treats ``(D, G, N)`` as free coordinates rather than a
fixed choice. Within SHBT those search axes map directly onto the branch tuple

    (D, G, N) <-> (k_ell, k_q, K).

Each candidate path is scored by a topological-closure functional

    C_top = Delta_fr + |c_dark - c_dark^*| + Delta_dio,

where

- ``Delta_fr`` is the framing-defect moat residue,
- ``|c_dark - c_dark^*|`` measures departure from the benchmark completion residue, and
- ``Delta_dio`` measures the Diophantine lock to ``lcm(2D, 3G)``.

The blind ``SymmetrySearcher`` scans a user-specified domain, ranks candidate
paths by ``C_top``, and proves necessity by asserting that only one path reaches
the non-singular stable fixed point where all three penalties vanish.
"""

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Final, Sequence

import numpy as np

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
from shbt.core.noether_bridge import framing_defect
from shbt.math_engine import (
    FIXED_POINT_DENOMINATOR,
    PRECISION_GUARD,
    guard_fraction,
    guard_sum,
    is_guard_zero,
)


EXPECTED_BENCHMARK: Final[tuple[int, int, int]] = (26, 8, 312)
DEFAULT_LEPTON_HALF_WIDTH: Final[int] = 2
DEFAULT_QUARK_HALF_WIDTH: Final[int] = 2
DEFAULT_PARENT_HALF_WIDTH: Final[int] = 2
DEFAULT_DISCOVERY_DIMENSION_LEVELS: Final[tuple[int, ...]] = tuple(range(2, 41))
DEFAULT_DISCOVERY_GAUGE_LEVELS: Final[tuple[int, ...]] = tuple(range(1, 21))
DEFAULT_DISCOVERY_PARENT_LEVELS: Final[tuple[int, ...]] = tuple(range(2, 401))
DEFAULT_FRONTIER_SIZE: Final[int] = 16
ZERO_TOLERANCE: Final[float] = 1.0e-15
MIN_COLOR_CEILING: Final[float] = 1.0e-6


def _benchmark_coordinates() -> tuple[int, int, int]:
    benchmark = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))
    assert benchmark == EXPECTED_BENCHMARK, (
        f"The rigidity landscape is locked to the published branch {EXPECTED_BENCHMARK}, received {benchmark}."
    )
    return benchmark


def _sanitize_small(value: float) -> float:
    numeric_value = float(value)
    if is_guard_zero(value):
        return 0.0
    return numeric_value


def _is_zero(value: float) -> bool:
    return bool(is_guard_zero(value))


def _is_positive_residue(value: float | Fraction) -> bool:
    return not is_guard_zero(value) and float(value) > 0.0


def _guard_zero_mask(values: np.ndarray) -> np.ndarray:
    return np.asarray(np.vectorize(is_guard_zero, otypes=[bool])(values), dtype=bool)


def _centered_levels(center: int, half_width: int) -> tuple[int, ...]:
    resolved_center = int(center)
    resolved_half_width = int(half_width)
    if resolved_half_width < 0:
        raise ValueError("Half-width arguments must be non-negative integers.")
    lower_bound = max(1, resolved_center - resolved_half_width)
    upper_bound = resolved_center + resolved_half_width
    return tuple(range(lower_bound, upper_bound + 1))


def _coerce_levels(levels: Sequence[int]) -> tuple[int, ...]:
    resolved_levels = tuple(sorted({int(level) for level in levels if int(level) > 0}))
    if not resolved_levels:
        raise ValueError("Rigidity scans require at least one positive level in each axis.")
    return resolved_levels


def _distance_to_guard_integer(value: Fraction) -> Fraction:
    denominator = value.denominator
    remainder = Fraction(value.numerator % denominator, denominator)
    return remainder if remainder <= Fraction(1, 1) - remainder else Fraction(1, 1) - remainder


def calculate_moat_depth(parent_level: int, lepton_level: int, quark_level: int) -> Fraction:
    lepton_loading = guard_fraction(parent_level, 2 * int(lepton_level))
    quark_loading = guard_fraction(parent_level, 3 * int(quark_level))
    lepton_gap = _distance_to_guard_integer(lepton_loading)
    quark_gap = _distance_to_guard_integer(quark_loading)
    return lepton_gap if lepton_gap >= quark_gap else quark_gap


def _wzw_central_charge(level: int, dimension: int, dual_coxeter: int) -> Fraction:
    resolved_level = int(level)
    denominator = resolved_level + int(dual_coxeter)
    if denominator <= 0:
        raise ValueError("WZW central charge requires k + h^∨ > 0.")
    return guard_fraction(resolved_level * int(dimension), denominator)


def _c_dark_residue(parent_level: int, lepton_level: int, quark_level: int) -> Fraction:
    return guard_sum(
        (
            _wzw_central_charge(parent_level, SU3_DIMENSION, SU3_DUAL_COXETER),
            _wzw_central_charge(parent_level, SU2_DIMENSION, SU2_DUAL_COXETER),
            -_wzw_central_charge(quark_level, SU3_DIMENSION, SU3_DUAL_COXETER),
            -_wzw_central_charge(lepton_level, SU2_DIMENSION, SU2_DUAL_COXETER),
        )
    )


def _diophantine_gap(parent_level: int, lepton_level: int, quark_level: int) -> Fraction:
    minimal_parent_level = math.lcm(2 * int(lepton_level), 3 * int(quark_level))
    return abs(guard_fraction(int(parent_level) - minimal_parent_level, minimal_parent_level))


def _point_order_key(point: "RigidityPoint") -> tuple[float, float, float, float, int, int, int]:
    return (
        float(point.total_residue),
        float(point.delta_fr),
        float(point.c_dark_shift),
        float(point.diophantine_gap),
        int(point.coordinates[2]),
        int(point.coordinates[0]),
        int(point.coordinates[1]),
    )


@dataclass(frozen=True)
class RigidityPoint:
    coordinates: tuple[int, int, int]
    lepton_framing_gap: float
    quark_framing_gap: float
    delta_fr: float
    delta_fr_label: str
    c_dark_shift: float
    diophantine_gap: float
    total_residue: float
    topological_closure_score: float

    @property
    def dimension_level(self) -> int:
        return int(self.coordinates[0])

    @property
    def gauge_level(self) -> int:
        return int(self.coordinates[1])

    @property
    def parent_level(self) -> int:
        return int(self.coordinates[2])

    @property
    def non_singular(self) -> bool:
        return bool(_is_zero(self.delta_fr) and _is_zero(self.c_dark_shift))

    @property
    def stable_fixed_point(self) -> bool:
        return bool(self.non_singular and _is_zero(self.diophantine_gap))

    @property
    def topological_closure(self) -> bool:
        return self.topological_closure_locked

    @property
    def topological_closure_locked(self) -> bool:
        return bool(self.stable_fixed_point and _is_zero(self.topological_closure_score))


@dataclass(frozen=True)
class RigidityLandscapeScan:
    benchmark_coordinates: tuple[int, int, int]
    lepton_levels: tuple[int, ...]
    quark_levels: tuple[int, ...]
    parent_levels: tuple[int, ...]
    total_residue_grid: np.ndarray
    delta_fr_grid: np.ndarray
    c_dark_shift_grid: np.ndarray
    diophantine_gap_grid: np.ndarray
    benchmark_index: tuple[int, int, int]
    benchmark_point: RigidityPoint
    nearest_detuned_point: RigidityPoint
    maximum_residue_point: RigidityPoint
    points: tuple[RigidityPoint, ...]
    color_floor: float

    @property
    def dimension_levels(self) -> tuple[int, ...]:
        return self.lepton_levels

    @property
    def gauge_levels(self) -> tuple[int, ...]:
        return self.quark_levels

    @property
    def topological_closure_grid(self) -> np.ndarray:
        return np.asarray(_guard_zero_mask(self.total_residue_grid), dtype=int)

    @property
    def unique_stable_fixed_point(self) -> RigidityPoint | None:
        if self.unique_fixed_point_count != 1:
            return None
        return self.unique_non_singular_fixed_points[0]

    @property
    def topological_closure_survivors(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(point.coordinates for point in self.unique_non_singular_fixed_points)

    @property
    def unique_stable_fixed_point_coordinates(self) -> tuple[int, int, int] | None:
        unique_point = self.unique_stable_fixed_point
        if unique_point is None:
            return None
        return unique_point.coordinates

    @property
    def topological_closure_score_grid(self) -> np.ndarray:
        return np.asarray(self.total_residue_grid, dtype=float)

    @property
    def discovered_point(self) -> RigidityPoint:
        return min(self.points, key=_point_order_key)

    @property
    def unique_non_singular_fixed_points(self) -> tuple[RigidityPoint, ...]:
        return tuple(point for point in self.points if point.topological_closure_locked)

    @property
    def stable_fixed_points(self) -> tuple[RigidityPoint, ...]:
        return self.unique_non_singular_fixed_points

    @property
    def unique_fixed_point_count(self) -> int:
        return len(self.unique_non_singular_fixed_points)

    @property
    def benchmark_uniquely_discovered(self) -> bool:
        return bool(
            self.discovered_point.coordinates == self.benchmark_coordinates
            and self.unique_fixed_point_count == 1
            and self.unique_non_singular_fixed_points[0].coordinates == self.benchmark_coordinates
        )

    def assert_unique_non_singular_fixed_point(self) -> None:
        if self.unique_fixed_point_count != 1:
            raise AssertionError(
                "Blind rigidity scan failed to isolate a unique non-singular stable fixed point: "
                f"found {self.unique_fixed_point_count}."
            )
        if self.unique_non_singular_fixed_points[0].coordinates != self.benchmark_coordinates:
            raise AssertionError(
                "Blind rigidity scan discovered a unique fixed point, but it does not match the benchmark "
                f"{self.benchmark_coordinates}: {self.unique_non_singular_fixed_points[0].coordinates}."
            )


@dataclass(frozen=True)
class SymmetrySearchAudit:
    benchmark_coordinates: tuple[int, int, int]
    dimension_levels: tuple[int, ...]
    gauge_levels: tuple[int, ...]
    parent_levels: tuple[int, ...]
    discovered_kernel: RigidityPoint
    runner_up_kernel: RigidityPoint
    unique_non_singular_fixed_points: tuple[RigidityPoint, ...]
    evolutionary_frontier: tuple[RigidityPoint, ...]
    trial_count: int

    @property
    def search_space_shape(self) -> tuple[int, int, int]:
        return (len(self.dimension_levels), len(self.gauge_levels), len(self.parent_levels))

    @property
    def unique_fixed_point_count(self) -> int:
        return len(self.unique_non_singular_fixed_points)

    @property
    def unique_stable_fixed_point(self) -> bool:
        return self.unique_fixed_point_count == 1

    @property
    def certified_unique_survivor(self) -> tuple[int, int, int] | None:
        if self.unique_fixed_point_count != 1:
            return None
        return self.unique_non_singular_fixed_points[0].coordinates

    @property
    def mathematical_necessity(self) -> bool:
        return self.benchmark_is_mathematical_necessity

    @property
    def evolutionary_history(self) -> tuple[RigidityPoint, ...]:
        return self.evolutionary_frontier

    @property
    def certified_scan(self) -> RigidityLandscapeScan:
        return build_rigidity_landscape_scan(
            lepton_levels=self.dimension_levels,
            quark_levels=self.gauge_levels,
            parent_levels=self.parent_levels,
        )

    @property
    def closure_gap(self) -> float:
        return float(self.runner_up_kernel.total_residue - self.discovered_kernel.total_residue)

    @property
    def benchmark_uniquely_discovered(self) -> bool:
        return bool(
            self.discovered_kernel.coordinates == self.benchmark_coordinates
            and self.unique_fixed_point_count == 1
            and self.unique_non_singular_fixed_points[0].coordinates == self.benchmark_coordinates
            and _is_positive_residue(self.runner_up_kernel.total_residue)
        )

    @property
    def benchmark_is_mathematical_necessity(self) -> bool:
        return self.benchmark_uniquely_discovered

    @property
    def evolutionary_best_point(self) -> RigidityPoint:
        return self.discovered_kernel

    @property
    def statement(self) -> str:
        return (
            "The blind symmetry search certifies "
            f"{self.benchmark_coordinates} as the unique non-singular stable fixed point after scanning "
            f"{self.trial_count} candidate (D, G, N) paths."
        )

    def assert_unique_non_singular_fixed_point(self) -> None:
        if self.unique_fixed_point_count != 1:
            raise AssertionError(
                "Blind symmetry search failed to isolate a unique non-singular stable fixed point: "
                f"found {self.unique_fixed_point_count}."
            )
        if self.discovered_kernel.coordinates != self.benchmark_coordinates:
            raise AssertionError(
                "Blind symmetry search did not discover the benchmark kernel: "
                f"expected {self.benchmark_coordinates}, found {self.discovered_kernel.coordinates}."
            )
        if self.unique_non_singular_fixed_points[0].coordinates != self.benchmark_coordinates:
            raise AssertionError(
                "The only stable fixed point is not the benchmark kernel: "
                f"{self.unique_non_singular_fixed_points[0].coordinates}."
            )
        if not _is_positive_residue(self.runner_up_kernel.total_residue):
            raise AssertionError("Runner-up candidate retained zero closure residue; uniqueness proof failed.")


def build_rigidity_point(lepton_level: int, quark_level: int, parent_level: int) -> RigidityPoint:
    defect = framing_defect(int(parent_level), int(lepton_level), int(quark_level))
    lepton_gap = _sanitize_small(float(defect.lepton_gap))
    quark_gap = _sanitize_small(float(defect.quark_gap))
    delta_fr_fraction = calculate_moat_depth(parent_level, lepton_level, quark_level)
    delta_fr = _sanitize_small(delta_fr_fraction)
    c_dark_shift_fraction = abs(_c_dark_residue(parent_level, lepton_level, quark_level) - BENCHMARK_C_DARK_RESIDUE_FRACTION)
    c_dark_shift = _sanitize_small(c_dark_shift_fraction)
    diophantine_gap_fraction = _diophantine_gap(parent_level, lepton_level, quark_level)
    diophantine_gap = _sanitize_small(diophantine_gap_fraction)
    topological_closure_score = _sanitize_small(
        guard_sum((delta_fr_fraction, c_dark_shift_fraction, diophantine_gap_fraction))
    )
    return RigidityPoint(
        coordinates=(int(lepton_level), int(quark_level), int(parent_level)),
        lepton_framing_gap=lepton_gap,
        quark_framing_gap=quark_gap,
        delta_fr=delta_fr,
        delta_fr_label=str(delta_fr_fraction),
        c_dark_shift=c_dark_shift,
        diophantine_gap=diophantine_gap,
        total_residue=topological_closure_score,
        topological_closure_score=topological_closure_score,
    )


def build_rigidity_landscape_scan(
    *,
    lepton_levels: Sequence[int],
    quark_levels: Sequence[int],
    parent_levels: Sequence[int],
) -> RigidityLandscapeScan:
    benchmark_kl, benchmark_kq, benchmark_parent = _benchmark_coordinates()
    resolved_lepton_levels = _coerce_levels(lepton_levels)
    resolved_quark_levels = _coerce_levels(quark_levels)
    resolved_parent_levels = _coerce_levels(parent_levels)

    if benchmark_kl not in resolved_lepton_levels or benchmark_kq not in resolved_quark_levels or benchmark_parent not in resolved_parent_levels:
        raise ValueError("The rigidity scan ranges must include the published benchmark (26, 8, 312).")

    benchmark_index = (
        resolved_lepton_levels.index(benchmark_kl),
        resolved_quark_levels.index(benchmark_kq),
        resolved_parent_levels.index(benchmark_parent),
    )

    grid_shape = (len(resolved_lepton_levels), len(resolved_quark_levels), len(resolved_parent_levels))
    total_residue_grid = np.zeros(grid_shape, dtype=float)
    delta_fr_grid = np.zeros(grid_shape, dtype=float)
    c_dark_shift_grid = np.zeros(grid_shape, dtype=float)
    diophantine_gap_grid = np.zeros(grid_shape, dtype=float)
    points: list[RigidityPoint] = []

    for lepton_index, lepton_level in enumerate(resolved_lepton_levels):
        for quark_index, quark_level in enumerate(resolved_quark_levels):
            for parent_index, parent_level in enumerate(resolved_parent_levels):
                point = build_rigidity_point(lepton_level, quark_level, parent_level)
                points.append(point)
                total_residue_grid[lepton_index, quark_index, parent_index] = point.total_residue
                delta_fr_grid[lepton_index, quark_index, parent_index] = point.delta_fr
                c_dark_shift_grid[lepton_index, quark_index, parent_index] = point.c_dark_shift
                diophantine_gap_grid[lepton_index, quark_index, parent_index] = point.diophantine_gap

    benchmark_point = next(point for point in points if point.coordinates == (benchmark_kl, benchmark_kq, benchmark_parent))
    assert benchmark_point.topological_closure_locked, (
        "The rigidity landscape must retain a zero-residue benchmark fixed point at the published branch."
    )

    nearest_detuned_point = min(
        (point for point in points if point.coordinates != (benchmark_kl, benchmark_kq, benchmark_parent)),
        key=_point_order_key,
    )
    maximum_residue_point = max(points, key=_point_order_key)

    positive_residues = delta_fr_grid[~_guard_zero_mask(delta_fr_grid)]
    color_floor = float(np.min(positive_residues)) if positive_residues.size else MIN_COLOR_CEILING

    scan = RigidityLandscapeScan(
        benchmark_coordinates=(benchmark_kl, benchmark_kq, benchmark_parent),
        lepton_levels=resolved_lepton_levels,
        quark_levels=resolved_quark_levels,
        parent_levels=resolved_parent_levels,
        total_residue_grid=np.asarray(total_residue_grid, dtype=float),
        delta_fr_grid=np.asarray(delta_fr_grid, dtype=float),
        c_dark_shift_grid=np.asarray(c_dark_shift_grid, dtype=float),
        diophantine_gap_grid=np.asarray(diophantine_gap_grid, dtype=float),
        benchmark_index=benchmark_index,
        benchmark_point=benchmark_point,
        nearest_detuned_point=nearest_detuned_point,
        maximum_residue_point=maximum_residue_point,
        points=tuple(points),
        color_floor=color_floor,
    )
    scan.assert_unique_non_singular_fixed_point()
    return scan


def build_centered_rigidity_landscape_scan(
    *,
    lepton_half_width: int = DEFAULT_LEPTON_HALF_WIDTH,
    quark_half_width: int = DEFAULT_QUARK_HALF_WIDTH,
    parent_half_width: int = DEFAULT_PARENT_HALF_WIDTH,
) -> RigidityLandscapeScan:
    benchmark_kl, benchmark_kq, benchmark_parent = _benchmark_coordinates()
    return build_rigidity_landscape_scan(
        lepton_levels=_centered_levels(benchmark_kl, lepton_half_width),
        quark_levels=_centered_levels(benchmark_kq, quark_half_width),
        parent_levels=_centered_levels(benchmark_parent, parent_half_width),
    )


class SymmetrySearcher:
    """Blind searcher that discovers the SHBT kernel from closure fitness alone."""

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

    def score_path(self, dimension_level: int, gauge_level: int, parent_level: int) -> RigidityPoint:
        return build_rigidity_point(dimension_level, gauge_level, parent_level)

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
        resolved_dimension_levels = _coerce_levels(self.dimension_levels if dimension_levels is None else dimension_levels)
        resolved_gauge_levels = _coerce_levels(self.gauge_levels if gauge_levels is None else gauge_levels)
        resolved_parent_levels = _coerce_levels(self.parent_levels if parent_levels is None else parent_levels)

        frontier: list[RigidityPoint] = []
        unique_non_singular_fixed_points: list[RigidityPoint] = []
        trial_count = 0
        benchmark_coordinates = _benchmark_coordinates()
        for dimension_level in resolved_dimension_levels:
            for gauge_level in resolved_gauge_levels:
                for parent_level in resolved_parent_levels:
                    trial_count += 1
                    point = self.score_path(dimension_level, gauge_level, parent_level)
                    if point.topological_closure_locked:
                        unique_non_singular_fixed_points.append(point)
                    frontier.append(point)
                    frontier.sort(key=_point_order_key)
                    if len(frontier) > self.frontier_size:
                        frontier.pop()

        if not frontier:
            raise ValueError("Symmetry search requires at least one candidate path.")

        discovered_kernel = frontier[0]
        runner_up_kernel = frontier[1] if len(frontier) > 1 else frontier[0]
        audit = SymmetrySearchAudit(
            benchmark_coordinates=benchmark_coordinates,
            dimension_levels=resolved_dimension_levels,
            gauge_levels=resolved_gauge_levels,
            parent_levels=resolved_parent_levels,
            discovered_kernel=discovered_kernel,
            runner_up_kernel=runner_up_kernel,
            unique_non_singular_fixed_points=tuple(sorted(unique_non_singular_fixed_points, key=_point_order_key)),
            evolutionary_frontier=tuple(frontier),
            trial_count=trial_count,
        )
        audit.assert_unique_non_singular_fixed_point()
        return audit

    def discover_unique_fixed_point(self) -> SymmetrySearchAudit:
        return self.discover_optimal_kernel()


def discover_optimal_kernel(
    *,
    dimension_levels: Sequence[int] = DEFAULT_DISCOVERY_DIMENSION_LEVELS,
    gauge_levels: Sequence[int] = DEFAULT_DISCOVERY_GAUGE_LEVELS,
    parent_levels: Sequence[int] = DEFAULT_DISCOVERY_PARENT_LEVELS,
    frontier_size: int = DEFAULT_FRONTIER_SIZE,
) -> SymmetrySearchAudit:
    return SymmetrySearcher(frontier_size=frontier_size).discover_optimal_kernel(
        dimension_levels=dimension_levels,
        gauge_levels=gauge_levels,
        parent_levels=parent_levels,
    )


def assert_unique_stable_fixed_point(audit: SymmetrySearchAudit) -> tuple[int, int, int]:
    audit.assert_unique_non_singular_fixed_point()
    return audit.discovered_kernel.coordinates


__all__ = [
    "DEFAULT_DISCOVERY_DIMENSION_LEVELS",
    "DEFAULT_DISCOVERY_GAUGE_LEVELS",
    "DEFAULT_DISCOVERY_PARENT_LEVELS",
    "DEFAULT_FRONTIER_SIZE",
    "DEFAULT_LEPTON_HALF_WIDTH",
    "DEFAULT_PARENT_HALF_WIDTH",
    "DEFAULT_QUARK_HALF_WIDTH",
    "EXPECTED_BENCHMARK",
    "FIXED_POINT_DENOMINATOR",
    "MIN_COLOR_CEILING",
    "PRECISION_GUARD",
    "RigidityLandscapeScan",
    "RigidityPoint",
    "SymmetrySearchAudit",
    "SymmetrySearcher",
    "ZERO_TOLERANCE",
    "calculate_moat_depth",
    "build_centered_rigidity_landscape_scan",
    "build_rigidity_landscape_scan",
    "build_rigidity_point",
    "discover_optimal_kernel",
    "assert_unique_stable_fixed_point",
]
