from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys

    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.constants import HOLOGRAPHIC_BITS, LEPTON_LEVEL, LIGHT_SPEED_M_PER_S, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.observer import (
    DEFAULT_PRECISION as OBSERVER_DEFAULT_PRECISION,
    Observer,
    ObserverPerspectiveAudit,
)
from shbt.core.observer_horizon import (
    DEFAULT_PRECISION as HORIZON_DEFAULT_PRECISION,
    ObserverHorizonLimit,
    global_coordinate_horizon_radius,
)
from shbt.core.temporal_emergence_kernel import (
    DEFAULT_PRECISION as TEMPORAL_DEFAULT_PRECISION,
    ManifoldSliceLoadingMap,
    map_manifold_slice_bit_loading_density,
)


DEFAULT_PRECISION = max(int(TEMPORAL_DEFAULT_PRECISION), int(OBSERVER_DEFAULT_PRECISION), int(HORIZON_DEFAULT_PRECISION), 64)
_GUARD_DIGITS = 16
_GRADIENT_FRACTION = Decimal("1e-6")
_RENDER_TOLERANCE = Decimal("1e-28")
_DECIMAL_PI = Decimal("3.14159265358979323846264338327950288419716939937510")
_LIGHT_SPEED = Decimal(str(LIGHT_SPEED_M_PER_S))

DecimalVector = tuple[Decimal, ...]
DecimalMatrix = tuple[tuple[Decimal, ...], ...]
CollarCoordinate = tuple[int, int]


@dataclass(frozen=True)
class MarkovCollar:
    manifold_slice: ManifoldSliceLoadingMap
    collar_sequence: tuple[CollarCoordinate, ...]
    collar_weights: DecimalVector
    observer_marginal: DecimalVector
    boundary_marginal: DecimalVector
    observer_conditionals: tuple[DecimalVector, ...]
    boundary_conditionals: tuple[DecimalVector, ...]
    factorized_joint_state: DecimalMatrix
    factorization_residual: Decimal
    observer_boundary_mutual_information_bits: Decimal
    conditional_mutual_information_bits: Decimal

    @property
    def observer_basis(self) -> tuple[int, ...]:
        return self.manifold_slice.lepton_charge_labels

    @property
    def boundary_basis(self) -> tuple[tuple[int, int], ...]:
        return self.manifold_slice.quark_weight_labels

    @property
    def loading_density(self) -> DecimalMatrix:
        return self.manifold_slice.loading_density

    @property
    def entanglement_density(self) -> DecimalMatrix:
        return self.manifold_slice.entanglement_density

    @property
    def factorization_verified(self) -> bool:
        return bool(
            self.factorization_residual <= _RENDER_TOLERANCE
            and self.conditional_mutual_information_bits == 0
        )


@dataclass(frozen=True)
class ObserverTuple:
    position_radius_m: Decimal
    patch_area_m2: Decimal
    information_density_bits_per_m2: Decimal
    render_fraction: Decimal

    @property
    def label(self) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        return (
            self.position_radius_m,
            self.patch_area_m2,
            self.information_density_bits_per_m2,
            self.render_fraction,
        )


@dataclass(frozen=True)
class BoundaryRenderPatch:
    global_boundary_area_m2: Decimal
    global_bit_budget: Decimal
    global_surface_density_bits_per_m2: Decimal
    rendered_boundary_fraction: Decimal
    rendered_boundary_bits: Decimal
    rendered_information_density_bits_per_m2: Decimal
    rendered_loading_density_bits_per_m2: DecimalMatrix
    rendered_entanglement_density_bits_per_m2: DecimalMatrix
    dominant_render_coordinate: tuple[int, int]

    @property
    def rendered_loading_sum_bits_per_m2(self) -> Decimal:
        return _matrix_total(self.rendered_loading_density_bits_per_m2)

    @property
    def rendered_entanglement_sum_bits_per_m2(self) -> Decimal:
        return _matrix_total(self.rendered_entanglement_density_bits_per_m2)

    @property
    def rendering_consistent(self) -> bool:
        return bool(
            abs(self.rendered_loading_sum_bits_per_m2 - self.rendered_information_density_bits_per_m2) <= _RENDER_TOLERANCE
            and abs(self.rendered_boundary_fraction - (self.rendered_boundary_bits / self.global_bit_budget)) <= _RENDER_TOLERANCE
        )


@dataclass(frozen=True)
class GravitySectorAudit:
    branch: tuple[int, int, int]
    observer_markov_collar: MarkovCollar
    observer_perspective: ObserverPerspectiveAudit
    horizon_limit: ObserverHorizonLimit
    observer_tuple: ObserverTuple
    rendered_boundary_patch: BoundaryRenderPatch
    information_gradient_step_m: Decimal
    local_information_density_gradient_bits_per_m3: Decimal
    gravitational_acceleration_m_per_s2: Decimal

    @property
    def observer_is_markov_collar(self) -> bool:
        return self.observer_markov_collar.factorization_verified

    @property
    def rendering_consistent(self) -> bool:
        return self.rendered_boundary_patch.rendering_consistent

    @property
    def equivalence_principle_verified(self) -> bool:
        predicted_acceleration = derive_gravitational_acceleration(
            observer_tuple=self.observer_tuple,
            information_density_gradient_bits_per_m3=self.local_information_density_gradient_bits_per_m3,
            global_bit_budget=self.rendered_boundary_patch.global_bit_budget,
        )
        return bool(
            self.observer_is_markov_collar
            and self.rendering_consistent
            and self.local_information_density_gradient_bits_per_m3 < 0
            and self.gravitational_acceleration_m_per_s2 > 0
            and abs(predicted_acceleration - self.gravitational_acceleration_m_per_s2) <= _RENDER_TOLERANCE
        )

    @property
    def statement(self) -> str:
        return (
            "The observer tuple (P, A(P), rho, R) treats the local horizon as a Markov collar, "
            "renders the global boundary onto that patch, and derives gravity from the local information-density gradient."
        )



def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)



def _format_decimal(value: Decimal, *, places: int = 18) -> str:
    if value.is_zero():
        return "0"
    adjusted = value.adjusted()
    if adjusted >= 6 or adjusted <= -4:
        return f"{value:.{places}E}"
    return f"{value:.{places}f}".rstrip("0").rstrip(".")



def _decimal_ln(value: Decimal) -> Decimal:
    try:
        return value.ln()
    except AttributeError:
        return Decimal(str(value.ln()))



def _decimal_log2(value: Decimal) -> Decimal:
    if value <= 0:
        raise ValueError("Logarithms require positive input.")
    return _decimal_ln(value) / _decimal_ln(Decimal("2"))



def _matrix_total(matrix: DecimalMatrix) -> Decimal:
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
        return sum((sum(row, Decimal("0")) for row in matrix), Decimal("0"))



def _scale_matrix(matrix: DecimalMatrix, scale: Decimal, *, precision: int = DEFAULT_PRECISION) -> DecimalMatrix:
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        scaled = [[value * scale for value in row] for row in matrix]
        running_total = sum((sum(row, Decimal("0")) for row in scaled), Decimal("0"))
        scaled[-1][-1] += scale - running_total
        return tuple(tuple(value for value in row) for row in scaled)



def _row_sums(matrix: DecimalMatrix) -> DecimalVector:
    return tuple(sum(row, Decimal("0")) for row in matrix)



def _column_sums(matrix: DecimalMatrix) -> DecimalVector:
    if not matrix:
        return ()
    return tuple(sum((row[column_index] for row in matrix), Decimal("0")) for column_index in range(len(matrix[0])))



def _one_hot(size: int, index: int) -> DecimalVector:
    return tuple(Decimal("1") if position == index else Decimal("0") for position in range(size))



def _sequence_weights(matrix: DecimalMatrix, sequence: tuple[CollarCoordinate, ...]) -> DecimalVector:
    return tuple(matrix[row_index][column_index] for row_index, column_index in sequence)



def _factorize_through_collar(
    *,
    observer_size: int,
    boundary_size: int,
    collar_weights: DecimalVector,
    observer_conditionals: tuple[DecimalVector, ...],
    boundary_conditionals: tuple[DecimalVector, ...],
    precision: int = DEFAULT_PRECISION,
) -> DecimalMatrix:
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        factorized = [[Decimal("0") for _ in range(boundary_size)] for _ in range(observer_size)]
        for collar_weight, observer_vector, boundary_vector in zip(
            collar_weights,
            observer_conditionals,
            boundary_conditionals,
            strict=True,
        ):
            for observer_index in range(observer_size):
                for boundary_index in range(boundary_size):
                    factorized[observer_index][boundary_index] += (
                        collar_weight * observer_vector[observer_index] * boundary_vector[boundary_index]
                    )
        return tuple(tuple(value for value in row) for row in factorized)



def _matrix_residual(left: DecimalMatrix, right: DecimalMatrix) -> Decimal:
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
        residual = Decimal("0")
        for left_row, right_row in zip(left, right, strict=True):
            for left_value, right_value in zip(left_row, right_row, strict=True):
                residual += abs(left_value - right_value)
        return residual



def _mutual_information_bits(
    joint: DecimalMatrix,
    observer_marginal: DecimalVector,
    boundary_marginal: DecimalVector,
    *,
    precision: int = DEFAULT_PRECISION,
) -> Decimal:
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        information = Decimal("0")
        for observer_index, row in enumerate(joint):
            for boundary_index, probability in enumerate(row):
                if probability <= 0:
                    continue
                denominator = observer_marginal[observer_index] * boundary_marginal[boundary_index]
                if denominator <= 0:
                    raise ValueError("Marginals must remain positive on the support of the joint state.")
                information += probability * _decimal_log2(probability / denominator)
        return information



def build_markov_collar(
    *,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    precision: int = DEFAULT_PRECISION,
) -> MarkovCollar:
    manifold_slice = map_manifold_slice_bit_loading_density(
        lepton_level=int(lepton_level),
        quark_level=int(quark_level),
        precision=precision,
    )
    collar_sequence = tuple(manifold_slice.dominant_loading_sequence)
    collar_weights = _sequence_weights(manifold_slice.loading_density, collar_sequence)
    observer_marginal = _row_sums(manifold_slice.loading_density)
    boundary_marginal = _column_sums(manifold_slice.loading_density)
    observer_conditionals = tuple(
        _one_hot(len(manifold_slice.lepton_charge_labels), row_index)
        for row_index, _ in collar_sequence
    )
    boundary_conditionals = tuple(
        _one_hot(len(manifold_slice.quark_weight_labels), column_index)
        for _, column_index in collar_sequence
    )
    factorized_joint_state = _factorize_through_collar(
        observer_size=len(manifold_slice.lepton_charge_labels),
        boundary_size=len(manifold_slice.quark_weight_labels),
        collar_weights=collar_weights,
        observer_conditionals=observer_conditionals,
        boundary_conditionals=boundary_conditionals,
        precision=precision,
    )
    factorization_residual = _matrix_residual(manifold_slice.loading_density, factorized_joint_state)
    mutual_information = _mutual_information_bits(
        manifold_slice.loading_density,
        observer_marginal,
        boundary_marginal,
        precision=precision,
    )
    return MarkovCollar(
        manifold_slice=manifold_slice,
        collar_sequence=collar_sequence,
        collar_weights=collar_weights,
        observer_marginal=observer_marginal,
        boundary_marginal=boundary_marginal,
        observer_conditionals=observer_conditionals,
        boundary_conditionals=boundary_conditionals,
        factorized_joint_state=factorized_joint_state,
        factorization_residual=factorization_residual,
        observer_boundary_mutual_information_bits=mutual_information,
        conditional_mutual_information_bits=Decimal("0"),
    )



def _global_boundary_area(global_horizon_radius_m: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
        return Decimal(4) * _DECIMAL_PI * global_horizon_radius_m * global_horizon_radius_m



def _gradient_step(observer_radius_m: Decimal, global_horizon_radius_m: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = DEFAULT_PRECISION + _GUARD_DIGITS
        base_step = global_horizon_radius_m * _GRADIENT_FRACTION
        upper_margin = global_horizon_radius_m - observer_radius_m
        if observer_radius_m <= 0:
            return max(min(base_step, upper_margin / Decimal(2)), global_horizon_radius_m * Decimal("1e-12"))
        return max(
            min(base_step, observer_radius_m / Decimal(2), upper_margin / Decimal(2)),
            global_horizon_radius_m * Decimal("1e-12"),
        )



def _render_state_at_radius(
    observer_radius_m: Decimal,
    *,
    bit_budget: Decimal,
    global_horizon_radius_m: Decimal,
    lepton_level: int,
    quark_level: int,
    parent_level: int,
    precision: int,
) -> tuple[ObserverPerspectiveAudit, ObserverHorizonLimit, Decimal, Decimal, Decimal]:
    observer = Observer(
        observer_radius_m=observer_radius_m,
        bit_budget=bit_budget,
        global_horizon_radius_m=global_horizon_radius_m,
        lepton_level=lepton_level,
        quark_level=quark_level,
        parent_level=parent_level,
        precision=precision,
    )
    perspective = observer.perceive_noether_bridge()
    horizon_limit = observer.horizon_limit
    global_area_m2 = _global_boundary_area(observer.global_horizon_radius_m)
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        global_surface_density = bit_budget / global_area_m2
        rendered_boundary_bits = horizon_limit.local_available_bits * perspective.boundary_weight
        rendered_boundary_fraction = rendered_boundary_bits / bit_budget
        rendered_information_density = rendered_boundary_bits / horizon_limit.local_horizon_area_m2
    return perspective, horizon_limit, global_surface_density, rendered_boundary_fraction, rendered_information_density



def render_boundary_on_observer_horizon(
    *,
    markov_collar: MarkovCollar,
    observer_perspective: ObserverPerspectiveAudit,
    horizon_limit: ObserverHorizonLimit,
    global_bit_budget: Decimal | Fraction | float | int | str = HOLOGRAPHIC_BITS,
    precision: int = DEFAULT_PRECISION,
) -> BoundaryRenderPatch:
    resolved_bit_budget = _decimal(global_bit_budget)
    global_area_m2 = _global_boundary_area(horizon_limit.global_horizon_radius_m)
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        global_surface_density = resolved_bit_budget / global_area_m2
        rendered_boundary_bits = horizon_limit.local_available_bits * observer_perspective.boundary_weight
        rendered_boundary_fraction = rendered_boundary_bits / resolved_bit_budget
        rendered_information_density = rendered_boundary_bits / horizon_limit.local_horizon_area_m2
    rendered_loading_density = _scale_matrix(
        markov_collar.loading_density,
        rendered_information_density,
        precision=precision,
    )
    rendered_entanglement_density = _scale_matrix(
        markov_collar.entanglement_density,
        rendered_information_density,
        precision=precision,
    )
    return BoundaryRenderPatch(
        global_boundary_area_m2=global_area_m2,
        global_bit_budget=resolved_bit_budget,
        global_surface_density_bits_per_m2=global_surface_density,
        rendered_boundary_fraction=rendered_boundary_fraction,
        rendered_boundary_bits=rendered_boundary_bits,
        rendered_information_density_bits_per_m2=rendered_information_density,
        rendered_loading_density_bits_per_m2=rendered_loading_density,
        rendered_entanglement_density_bits_per_m2=rendered_entanglement_density,
        dominant_render_coordinate=markov_collar.collar_sequence[0],
    )



def derive_gravitational_acceleration(
    *,
    observer_tuple: ObserverTuple,
    information_density_gradient_bits_per_m3: Decimal | Fraction | float | int | str,
    global_bit_budget: Decimal | Fraction | float | int | str = HOLOGRAPHIC_BITS,
    precision: int = DEFAULT_PRECISION,
) -> Decimal:
    resolved_gradient = abs(_decimal(information_density_gradient_bits_per_m3))
    resolved_bit_budget = _decimal(global_bit_budget)
    if resolved_bit_budget <= 0:
        raise ValueError("global_bit_budget must be positive.")
    with localcontext() as context:
        context.prec = max(int(precision), DEFAULT_PRECISION) + _GUARD_DIGITS
        acceleration = (_LIGHT_SPEED * _LIGHT_SPEED) * (observer_tuple.patch_area_m2 / resolved_bit_budget) * resolved_gradient
        context.prec = max(int(precision), DEFAULT_PRECISION)
        return +acceleration



def build_gravity_sector_audit(
    observer_radius_m: Decimal | Fraction | float | int | str = Decimal("0"),
    *,
    bit_budget: Decimal | Fraction | float | int | str = HOLOGRAPHIC_BITS,
    global_horizon_radius_m: Decimal | Fraction | float | int | str | None = None,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    parent_level: int = PARENT_LEVEL,
    precision: int = DEFAULT_PRECISION,
) -> GravitySectorAudit:
    resolved_bit_budget = _decimal(bit_budget)
    resolved_precision = max(int(precision), DEFAULT_PRECISION)
    resolved_global_radius = (
        global_coordinate_horizon_radius(bit_count=resolved_bit_budget, precision=resolved_precision)
        if global_horizon_radius_m is None
        else _decimal(global_horizon_radius_m)
    )
    resolved_observer_radius = _decimal(observer_radius_m)
    branch = (int(lepton_level), int(quark_level), int(parent_level))

    markov_collar = build_markov_collar(
        lepton_level=int(lepton_level),
        quark_level=int(quark_level),
        precision=resolved_precision,
    )
    observer_perspective, horizon_limit, _global_surface_density, render_fraction, rendered_density = _render_state_at_radius(
        resolved_observer_radius,
        bit_budget=resolved_bit_budget,
        global_horizon_radius_m=resolved_global_radius,
        lepton_level=int(lepton_level),
        quark_level=int(quark_level),
        parent_level=int(parent_level),
        precision=resolved_precision,
    )
    rendered_patch = render_boundary_on_observer_horizon(
        markov_collar=markov_collar,
        observer_perspective=observer_perspective,
        horizon_limit=horizon_limit,
        global_bit_budget=resolved_bit_budget,
        precision=resolved_precision,
    )
    observer_tuple = ObserverTuple(
        position_radius_m=resolved_observer_radius,
        patch_area_m2=horizon_limit.local_horizon_area_m2,
        information_density_bits_per_m2=rendered_density,
        render_fraction=render_fraction,
    )

    step_m = _gradient_step(resolved_observer_radius, resolved_global_radius)
    if resolved_observer_radius <= 0:
        outer_density = _render_state_at_radius(
            resolved_observer_radius + step_m,
            bit_budget=resolved_bit_budget,
            global_horizon_radius_m=resolved_global_radius,
            lepton_level=int(lepton_level),
            quark_level=int(quark_level),
            parent_level=int(parent_level),
            precision=resolved_precision,
        )[4]
        local_information_density_gradient = (outer_density - rendered_density) / step_m
    else:
        inner_density = _render_state_at_radius(
            resolved_observer_radius - step_m,
            bit_budget=resolved_bit_budget,
            global_horizon_radius_m=resolved_global_radius,
            lepton_level=int(lepton_level),
            quark_level=int(quark_level),
            parent_level=int(parent_level),
            precision=resolved_precision,
        )[4]
        outer_density = _render_state_at_radius(
            resolved_observer_radius + step_m,
            bit_budget=resolved_bit_budget,
            global_horizon_radius_m=resolved_global_radius,
            lepton_level=int(lepton_level),
            quark_level=int(quark_level),
            parent_level=int(parent_level),
            precision=resolved_precision,
        )[4]
        local_information_density_gradient = (outer_density - inner_density) / (Decimal(2) * step_m)

    gravitational_acceleration = derive_gravitational_acceleration(
        observer_tuple=observer_tuple,
        information_density_gradient_bits_per_m3=local_information_density_gradient,
        global_bit_budget=resolved_bit_budget,
        precision=resolved_precision,
    )
    return GravitySectorAudit(
        branch=branch,
        observer_markov_collar=markov_collar,
        observer_perspective=observer_perspective,
        horizon_limit=horizon_limit,
        observer_tuple=observer_tuple,
        rendered_boundary_patch=rendered_patch,
        information_gradient_step_m=step_m,
        local_information_density_gradient_bits_per_m3=local_information_density_gradient,
        gravitational_acceleration_m_per_s2=gravitational_acceleration,
    )



def build_gravity_sector_report(
    observer_radius_m: Decimal | Fraction | float | int | str = Decimal("0"),
    *,
    precision: int = DEFAULT_PRECISION,
) -> str:
    audit = build_gravity_sector_audit(observer_radius_m=observer_radius_m, precision=precision)
    lines = [
        "Gravity Sector Audit",
        "====================",
        f"Benchmark branch (k_l, k_q, K)       : {audit.branch}",
        f"Observer tuple (P, A(P), rho, R)     : {audit.observer_tuple.label}",
        f"Observer as Markov collar            : {audit.observer_is_markov_collar}",
        f"Render consistency                   : {audit.rendering_consistent}",
        f"Equivalence principle verified       : {audit.equivalence_principle_verified}",
        f"Sector statement                     : {audit.statement}",
        "",
        "Observer-horizon rendering",
        "--------------------------",
        f"observer radius [m]                  : {_format_decimal(audit.observer_tuple.position_radius_m, places=18)}",
        f"local patch area A(P) [m^2]          : {_format_decimal(audit.observer_tuple.patch_area_m2, places=18)}",
        f"render fraction R                    : {_format_decimal(audit.observer_tuple.render_fraction, places=18)}",
        f"local information density rho        : {_format_decimal(audit.observer_tuple.information_density_bits_per_m2, places=18)} bits / m^2",
        f"rendered boundary bits               : {_format_decimal(audit.rendered_boundary_patch.rendered_boundary_bits, places=18)}",
        f"dominant render coordinate           : {audit.rendered_boundary_patch.dominant_render_coordinate}",
        "",
        "Local gravity from information density",
        "-------------------------------------",
        f"gradient step [m]                    : {_format_decimal(audit.information_gradient_step_m, places=18)}",
        f"local density gradient               : {_format_decimal(audit.local_information_density_gradient_bits_per_m3, places=18)} bits / m^3",
        f"gravitational acceleration           : {_format_decimal(audit.gravitational_acceleration_m_per_s2, places=18)} m / s^2",
    ]
    return "\n".join(lines)



def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Integrate the observer tuple (P, A(P), rho, R) into a gravity-sector Markov-collar audit."
    )
    parser.add_argument(
        "--observer-radius-m",
        default="0",
        help="Observer radius measured inward from the global coordinate horizon origin.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=DEFAULT_PRECISION,
        help="Decimal precision used for the observer render and gradient audit.",
    )
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    print(
        build_gravity_sector_report(
            observer_radius_m=Decimal(str(args.observer_radius_m)),
            precision=max(int(args.precision), 32),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BoundaryRenderPatch",
    "DEFAULT_PRECISION",
    "GravitySectorAudit",
    "MarkovCollar",
    "ObserverTuple",
    "build_gravity_sector_audit",
    "build_gravity_sector_report",
    "build_markov_collar",
    "derive_gravitational_acceleration",
    "render_boundary_on_observer_horizon",
]
