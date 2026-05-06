from __future__ import annotations

r"""Non-equilibrium bulk thermodynamics for the static SHBT boundary block.

This module interfaces directly with ``shbt.core.temporal_emergence_kernel`` to
convert the static entanglement data on the benchmark four-dimensional block
into a directed entropy-production chain on its emergent three-dimensional bulk
image. If ``\sigma`` is the branch-fixed hierarchical loading order and
``\rho_ent`` is the normalized entanglement density on the static manifold,
define

    S_n = \sum_{m=1}^n \rho_ent[\sigma_m],
    \Delta S_n = S_n - S_{n-1} = \rho_ent[\sigma_n] > 0,
    dT_bulk = \frac{1}{N} \sum_n dS_n.

Because every inter-layer increment ``\Delta S_n`` is strictly positive, the
bulk projection is necessarily non-equilibrium: the Arrow of Time is the unique
direction of increasing cumulative entanglement across the ordered hierarchy.
This is the precise sense in which a static 4D block produces a non-equilibrium
3D bulk in the SHBT bookkeeping.
"""

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

from shbt.constants import HOLOGRAPHIC_BITS, LEPTON_LEVEL, QUARK_LEVEL
from shbt.core.temporal_emergence_kernel import (
    DEFAULT_PRECISION as TEMPORAL_DEFAULT_PRECISION,
    TemporalEmergenceAudit,
    TemporalEmergencePoint,
    build_temporal_emergence_audit,
    calculate_arrow_of_time_gradient,
    derive_temporal_increment,
)


DEFAULT_PRECISION = max(int(TEMPORAL_DEFAULT_PRECISION), 64)
_GUARD_DIGITS = 16
_FLOW_TOLERANCE = Decimal("1e-15")


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(str(value))


def _format_decimal(value: Decimal, *, digits: int = 12) -> str:
    return f"{value:.{digits}E}" if value != 0 else "0"


@dataclass(frozen=True)
class BulkFlowLayer:
    layer_index: int
    manifold_coordinate: tuple[int, int]
    static_loading_density: Decimal
    static_entanglement_density: Decimal
    cumulative_entanglement_density: Decimal
    interlayer_entanglement_gradient: Decimal
    local_entropy_production_rate: Decimal
    local_temporal_gradient_km_s_mpc: Decimal

    @property
    def entropy_increases(self) -> bool:
        return self.interlayer_entanglement_gradient > 0


@dataclass(frozen=True)
class BulkFlowAudit:
    redshift: Decimal
    bit_budget: Decimal
    source_block_dimension: int
    emergent_bulk_dimension: int
    temporal_audit: TemporalEmergenceAudit
    temporal_point: TemporalEmergencePoint
    layers: tuple[BulkFlowLayer, ...]
    total_entropy_production_rate: Decimal
    layerwise_arrow_of_time_gradient_km_s_mpc: Decimal
    temporal_kernel_arrow_of_time_gradient_km_s_mpc: Decimal
    arrow_of_time_consistency_residual: Decimal
    entropy_to_time_identity_residual: Decimal
    entropy_rate_residual: Decimal
    metric_lock_residual: Decimal

    @property
    def entropy_rate_temporal_residual(self) -> Decimal:
        return derive_temporal_increment(self.entropy_rate_residual, self.bit_budget)

    @property
    def monotonic_entropy_growth(self) -> bool:
        if not self.layers:
            return False
        return bool(
            all(layer.entropy_increases for layer in self.layers)
            and self.layers[0].cumulative_entanglement_density > 0
            and all(
                self.layers[index].cumulative_entanglement_density
                > self.layers[index - 1].cumulative_entanglement_density
                for index in range(1, len(self.layers))
            )
        )

    @property
    def arrow_of_time_positive(self) -> bool:
        return self.layerwise_arrow_of_time_gradient_km_s_mpc > 0

    @property
    def consistent_with_temporal_emergence_kernel(self) -> bool:
        return bool(
            abs(self.arrow_of_time_consistency_residual) <= _FLOW_TOLERANCE
            and abs(self.entropy_to_time_identity_residual) <= _FLOW_TOLERANCE
            and abs(self.entropy_rate_temporal_residual) <= _FLOW_TOLERANCE
            and abs(self.metric_lock_residual) <= _FLOW_TOLERANCE
        )

    @property
    def static_block_projects_bulk(self) -> bool:
        return bool(
            self.source_block_dimension == 4
            and self.emergent_bulk_dimension == 3
            and self.monotonic_entropy_growth
            and self.arrow_of_time_positive
            and self.consistent_with_temporal_emergence_kernel
        )

    @property
    def statement(self) -> str:
        return (
            "A static 4D block produces a non-equilibrium 3D bulk through "
            "mandatory layer-by-layer entanglement growth."
        )


class BulkDynamics:
    def __init__(
        self,
        *,
        bit_budget: Decimal | Fraction | float | int | str = HOLOGRAPHIC_BITS,
        lepton_level: int = LEPTON_LEVEL,
        quark_level: int = QUARK_LEVEL,
        precision: int = DEFAULT_PRECISION,
    ) -> None:
        self.bit_budget = _decimal(bit_budget)
        self.lepton_level = int(lepton_level)
        self.quark_level = int(quark_level)
        self.precision = max(int(precision), DEFAULT_PRECISION)
        if self.bit_budget <= 0:
            raise ValueError("bit_budget must be positive.")

    def calculate_bulk_flow(
        self,
        *,
        redshift: Decimal | Fraction | float | int | str = Decimal("0"),
    ) -> BulkFlowAudit:
        resolved_redshift = _decimal(redshift)
        redshift_ladder = (
            (Decimal("0"), resolved_redshift)
            if resolved_redshift != 0
            else (resolved_redshift,)
        )
        temporal_audit = build_temporal_emergence_audit(
            bit_budget=self.bit_budget,
            redshifts=redshift_ladder,
            lepton_level=self.lepton_level,
            quark_level=self.quark_level,
            precision=self.precision,
        )
        if not temporal_audit.samples:
            raise RuntimeError("Temporal emergence audit did not return any samples.")

        temporal_point = next(
            (
                sample
                for sample in temporal_audit.samples
                if _decimal(sample.redshift) == resolved_redshift
            ),
            None,
        )
        if temporal_point is None:
            raise RuntimeError(f"No temporal emergence sample found for redshift {resolved_redshift}.")
        layers = self._build_hierarchical_layers(temporal_audit, temporal_point)

        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            layerwise_entropy_production_rate = sum(
                (layer.local_entropy_production_rate for layer in layers),
                Decimal("0"),
            )
            total_entropy_production_rate = temporal_point.total_entanglement_entropy_gradient
            layerwise_arrow_of_time_gradient = sum(
                (layer.local_temporal_gradient_km_s_mpc for layer in layers),
                Decimal("0"),
            )
            temporal_kernel_arrow_of_time_gradient = calculate_arrow_of_time_gradient(
                temporal_point.local_bit_loading_rate_density,
                temporal_audit.manifold_slice.dominant_loading_sequence,
                self.bit_budget,
                precision=self.precision,
            )
            entropy_to_time_identity_residual = (
                layerwise_arrow_of_time_gradient
                - derive_temporal_increment(
                    total_entropy_production_rate,
                    self.bit_budget,
                    precision=self.precision,
                )
            )
            arrow_of_time_consistency_residual = (
                layerwise_arrow_of_time_gradient - temporal_kernel_arrow_of_time_gradient
            )
            entropy_rate_residual = (
                layerwise_entropy_production_rate - total_entropy_production_rate
            )
            metric_lock_residual = (
                layerwise_arrow_of_time_gradient - temporal_point.derived_temporal_rate_km_s_mpc
            )

        return BulkFlowAudit(
            redshift=resolved_redshift,
            bit_budget=self.bit_budget,
            source_block_dimension=4,
            emergent_bulk_dimension=3,
            temporal_audit=temporal_audit,
            temporal_point=temporal_point,
            layers=layers,
            total_entropy_production_rate=total_entropy_production_rate,
            layerwise_arrow_of_time_gradient_km_s_mpc=layerwise_arrow_of_time_gradient,
            temporal_kernel_arrow_of_time_gradient_km_s_mpc=temporal_kernel_arrow_of_time_gradient,
            arrow_of_time_consistency_residual=arrow_of_time_consistency_residual,
            entropy_to_time_identity_residual=entropy_to_time_identity_residual,
            entropy_rate_residual=entropy_rate_residual,
            metric_lock_residual=metric_lock_residual,
        )

    def _build_hierarchical_layers(
        self,
        temporal_audit: TemporalEmergenceAudit,
        temporal_point: TemporalEmergencePoint,
    ) -> tuple[BulkFlowLayer, ...]:
        layers: list[BulkFlowLayer] = []
        cumulative_entanglement_density = Decimal("0")

        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            for layer_index, manifold_coordinate in enumerate(
                temporal_audit.manifold_slice.dominant_loading_sequence,
                start=1,
            ):
                row_index, column_index = manifold_coordinate
                static_loading_density = temporal_audit.manifold_slice.loading_density[row_index][column_index]
                static_entanglement_density = temporal_audit.manifold_slice.entanglement_density[row_index][column_index]
                previous_cumulative_density = cumulative_entanglement_density
                cumulative_entanglement_density += static_entanglement_density
                interlayer_entanglement_gradient = (
                    cumulative_entanglement_density - previous_cumulative_density
                )
                local_entropy_production_rate = temporal_point.local_entanglement_entropy_gradient[row_index][column_index]
                local_temporal_gradient = derive_temporal_increment(
                    local_entropy_production_rate,
                    self.bit_budget,
                    precision=self.precision,
                )
                layers.append(
                    BulkFlowLayer(
                        layer_index=layer_index,
                        manifold_coordinate=(row_index, column_index),
                        static_loading_density=static_loading_density,
                        static_entanglement_density=static_entanglement_density,
                        cumulative_entanglement_density=cumulative_entanglement_density,
                        interlayer_entanglement_gradient=interlayer_entanglement_gradient,
                        local_entropy_production_rate=local_entropy_production_rate,
                        local_temporal_gradient_km_s_mpc=local_temporal_gradient,
                    )
                )

        return tuple(layers)


def build_bulk_dynamics_audit(
    *,
    redshift: Decimal | Fraction | float | int | str = Decimal("0"),
    bit_budget: Decimal | Fraction | float | int | str = HOLOGRAPHIC_BITS,
    lepton_level: int = LEPTON_LEVEL,
    quark_level: int = QUARK_LEVEL,
    precision: int = DEFAULT_PRECISION,
) -> BulkFlowAudit:
    return BulkDynamics(
        bit_budget=bit_budget,
        lepton_level=lepton_level,
        quark_level=quark_level,
        precision=precision,
    ).calculate_bulk_flow(redshift=redshift)


def render_report(audit: BulkFlowAudit) -> str:
    verification_status = "PASS" if audit.static_block_projects_bulk else "CHECK"
    lines = [
        "Bulk Dynamics",
        "=============",
        "Static 4D boundary data are converted into a directed 3D bulk entropy flow.",
        "Operational identity: dT_bulk = sum_layers(dS_layer) / N.",
        "",
        f"Redshift z                           : {audit.redshift}",
        f"Bit budget N                        : {_format_decimal(audit.bit_budget)}",
        f"Source block dimension              : {audit.source_block_dimension}",
        f"Emergent bulk dimension             : {audit.emergent_bulk_dimension}",
        f"Total entropy production rate       : {_format_decimal(audit.total_entropy_production_rate)}",
        f"Layerwise Arrow-of-Time gradient    : {audit.layerwise_arrow_of_time_gradient_km_s_mpc:.12f}",
        f"Kernel Arrow-of-Time gradient       : {audit.temporal_kernel_arrow_of_time_gradient_km_s_mpc:.12f}",
        f"Arrow consistency residual          : {_format_decimal(audit.arrow_of_time_consistency_residual)}",
        f"Entropy-to-time identity residual   : {_format_decimal(audit.entropy_to_time_identity_residual)}",
        f"Metric lock residual                : {_format_decimal(audit.metric_lock_residual)}",
        f"Bulk emergence verification         : {verification_status}",
        "",
        "Hierarchical layers",
        "-------------------",
        "L  coord    dS_density        S_cumulative      dS/dt             dT",
        "--------------------------------------------------------------------------",
    ]
    lines.extend(
        f"{layer.layer_index:>1}  {layer.manifold_coordinate!s:<8}  "
        f"{layer.interlayer_entanglement_gradient:>16.12f}  "
        f"{layer.cumulative_entanglement_density:>16.12f}  "
        f"{layer.local_entropy_production_rate:>16.6E}  "
        f"{layer.local_temporal_gradient_km_s_mpc:>12.9f}"
        for layer in audit.layers
    )
    lines.extend(
        (
            "",
            audit.statement,
        )
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redshift", type=Decimal, default=Decimal("0"))
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    audit = build_bulk_dynamics_audit(
        redshift=args.redshift,
        precision=max(int(args.precision), 32),
    )
    print(render_report(audit))


__all__ = [
    "BulkDynamics",
    "BulkFlowAudit",
    "BulkFlowLayer",
    "DEFAULT_PRECISION",
    "build_bulk_dynamics_audit",
    "main",
    "parse_args",
    "render_report",
]


if __name__ == "__main__":
    main()
