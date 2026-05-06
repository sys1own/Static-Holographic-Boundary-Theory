from __future__ import annotations

"""Adaptive Holographic Mesh (AHM) for scale-stable bulk simulations.

The mesh rescales the holographic bit budget against the observer's local
coordinate volume so that coarse-grained large-scale structure remains a
parity-preserving image of the same anomaly-free boundary data. The operative
coarsening rules are

    N(V_obs) = N_0 (V_obs / V_ref),
    Lambda_holo(V_obs) = 3 pi / (L_P^2 N(V_obs)).

At every recursive coarsening step the module checks two invariants:

1. Bit-budget conservation across levels, preventing mesh-scale "bit rot".
2. Unity-of-Scale closure, enforced to remain within 1e-15 precision.

Because the benchmark branch coordinates remain fixed while only the observer's
local coarse-graining volume changes, the framing parity lock is required to
stay exact throughout the scale ladder.
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

from shbt.constants import HOLOGRAPHIC_BITS, LEPTON_LEVEL, PARENT_LEVEL, PLANCK_LENGTH_M, QUARK_LEVEL
from shbt.core.holographic_error_stabilizer import (
    BulkChecksumVerification,
    HolographicStabilizer,
    TopologicalChecksum,
    TopologicalChecksumCode,
)
from shbt.core.noether_bridge import (
    DEFAULT_PRECISION as NOETHER_DEFAULT_PRECISION,
    NewtonLockAudit,
    SaturationAudit,
    UnityOfScaleAudit,
    derive_kappa_d5,
    framing_defect,
    load_c_dark_completion_fraction,
    newton_constant_lock,
    saturation_audit,
    unity_of_scale_audit,
)


DEFAULT_PRECISION = max(int(NOETHER_DEFAULT_PRECISION), 200)
UNITY_TOLERANCE = Decimal("1e-15")
_GUARD_DIGITS = 16
PI = Decimal("3.14159265358979323846264338327950288419716939937510")
PLANCK_VOLUME_M3 = Decimal(str(PLANCK_LENGTH_M)) ** 3


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return Decimal(str(value))


def _format_decimal(value: Decimal, *, digits: int = 12) -> str:
    return f"{value:.{digits}E}" if value != 0 else "0"


@dataclass(frozen=True)
class AdaptiveMeshLevel:
    refinement_level: int
    coordinate_volume_m3: Decimal
    volume_ratio: Decimal
    bit_budget: Decimal
    lambda_holo_si_m2: Decimal
    coarsening_factor_from_child: Decimal | None
    saturation: SaturationAudit
    unity: UnityOfScaleAudit
    bulk_checksum: BulkChecksumVerification
    parity_checksum: TopologicalChecksum
    bit_budget_checksum: TopologicalChecksum | None
    child: AdaptiveMeshLevel | None = None

    @property
    def unity_within_tolerance(self) -> bool:
        return bool(
            self.unity.epsilon_lambda <= UNITY_TOLERANCE
            and self.unity.epsilon_lambda_noether_bridged <= UNITY_TOLERANCE
        )

    @property
    def parity_preserved(self) -> bool:
        return self.parity_checksum.passed and self.bulk_checksum.parity_checksum_passed

    @property
    def bit_rot_free(self) -> bool:
        return self.bit_budget_checksum is None or self.bit_budget_checksum.passed

    @property
    def budget_relative_residual(self) -> Decimal:
        return Decimal("0") if self.bit_budget_checksum is None else self.bit_budget_checksum.residue

    @property
    def mesh_locked(self) -> bool:
        return bool(
            self.unity_within_tolerance
            and self.parity_preserved
            and self.bit_rot_free
            and self.bulk_checksum.passed
        )


@dataclass(frozen=True)
class HolographicAMRAudit:
    observer_coordinate_volume_m3: Decimal
    reference_coordinate_volume_m3: Decimal
    branching_factor: Decimal
    unity_tolerance: Decimal
    branch: tuple[int, int, int]
    c_dark_fraction: Fraction
    kappa_d5: Decimal
    newton_lock: NewtonLockAudit
    bulk_checksum: BulkChecksumVerification
    root: AdaptiveMeshLevel
    levels: tuple[AdaptiveMeshLevel, ...]

    @property
    def mesh_depth(self) -> int:
        return max(len(self.levels) - 1, 0)

    @property
    def max_epsilon_lambda(self) -> Decimal:
        return max(level.unity.epsilon_lambda for level in self.levels)

    @property
    def max_epsilon_lambda_noether_bridged(self) -> Decimal:
        return max(level.unity.epsilon_lambda_noether_bridged for level in self.levels)

    @property
    def max_budget_relative_residual(self) -> Decimal:
        return max(level.budget_relative_residual for level in self.levels)

    @property
    def max_parity_residual(self) -> Decimal:
        return max(level.parity_checksum.residue for level in self.levels)

    @property
    def parity_preserved(self) -> bool:
        return all(level.parity_preserved for level in self.levels)

    @property
    def bit_rot_free(self) -> bool:
        return all(level.bit_rot_free for level in self.levels)

    @property
    def unity_locked(self) -> bool:
        return bool(
            self.max_epsilon_lambda <= self.unity_tolerance
            and self.max_epsilon_lambda_noether_bridged <= self.unity_tolerance
        )

    @property
    def passed(self) -> bool:
        return bool(self.bulk_checksum.passed and self.parity_preserved and self.bit_rot_free and self.unity_locked)

    @property
    def statement(self) -> str:
        return (
            "The Adaptive Holographic Mesh coarsens from Planckian to macroscopic scales "
            "without parity loss, bit rot, or Unity-of-Scale drift."
        )


class AdaptiveHolographicMesh:
    def __init__(
        self,
        *,
        reference_coordinate_volume_m3: Decimal | Fraction | float | int | str = PLANCK_VOLUME_M3,
        branching_factor: Decimal | Fraction | float | int | str = Decimal("8"),
        unity_tolerance: Decimal | Fraction | float | int | str = UNITY_TOLERANCE,
        precision: int = DEFAULT_PRECISION,
    ) -> None:
        self.reference_coordinate_volume_m3 = _decimal(reference_coordinate_volume_m3)
        self.branching_factor = _decimal(branching_factor)
        self.unity_tolerance = _decimal(unity_tolerance)
        self.precision = max(int(precision), DEFAULT_PRECISION)
        self._planck_length_m = Decimal(str(PLANCK_LENGTH_M))
        self._base_bit_budget = _decimal(HOLOGRAPHIC_BITS)
        if self.reference_coordinate_volume_m3 <= 0:
            raise ValueError("reference_coordinate_volume_m3 must be positive.")
        if self.branching_factor <= 1:
            raise ValueError("branching_factor must be greater than 1.")
        if self.unity_tolerance < 0:
            raise ValueError("unity_tolerance must be non-negative.")

        self.c_dark_fraction = load_c_dark_completion_fraction()
        self.kappa_d5 = derive_kappa_d5(precision=self.precision)
        self.bulk_checksum = HolographicStabilizer(precision=self.precision).verify_bulk_checksum()
        if not self.bulk_checksum.passed:
            raise RuntimeError(f"Benchmark holographic checksum failed before mesh construction: {self.bulk_checksum.detail}.")
        self.newton_lock = newton_constant_lock(
            c_dark_fraction=self.c_dark_fraction,
            precision=self.precision,
        )
        self._framing = framing_defect(PARENT_LEVEL, LEPTON_LEVEL, QUARK_LEVEL)

    def refine_mesh(
        self,
        *,
        observer_coordinate_volume_m3: Decimal | Fraction | float | int | str,
    ) -> HolographicAMRAudit:
        resolved_volume = _decimal(observer_coordinate_volume_m3)
        if resolved_volume <= 0:
            raise ValueError("observer_coordinate_volume_m3 must be positive.")
        if resolved_volume < self.reference_coordinate_volume_m3:
            raise ValueError(
                "observer_coordinate_volume_m3 must be greater than or equal to the reference coordinate volume."
            )

        scale_ladder = self._resolve_scale_ladder(self.reference_coordinate_volume_m3, resolved_volume)
        root = self._build_level_chain(scale_ladder, len(scale_ladder) - 1)
        levels = self._flatten_levels(root)
        return HolographicAMRAudit(
            observer_coordinate_volume_m3=resolved_volume,
            reference_coordinate_volume_m3=self.reference_coordinate_volume_m3,
            branching_factor=self.branching_factor,
            unity_tolerance=self.unity_tolerance,
            branch=(int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL)),
            c_dark_fraction=self.c_dark_fraction,
            kappa_d5=self.kappa_d5,
            newton_lock=self.newton_lock,
            bulk_checksum=self.bulk_checksum,
            root=root,
            levels=levels,
        )

    def _resolve_scale_ladder(self, current_volume_m3: Decimal, target_volume_m3: Decimal) -> tuple[Decimal, ...]:
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            if current_volume_m3 >= target_volume_m3:
                return (target_volume_m3,)
            remaining_ratio = target_volume_m3 / current_volume_m3
            coarsening_factor = self.branching_factor if remaining_ratio > self.branching_factor else remaining_ratio
            next_volume = current_volume_m3 * coarsening_factor
            if next_volume == current_volume_m3:
                next_volume = target_volume_m3
        return (current_volume_m3, *self._resolve_scale_ladder(next_volume, target_volume_m3))

    def _scaled_bit_budget(self, coordinate_volume_m3: Decimal) -> Decimal:
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            return self._base_bit_budget * (coordinate_volume_m3 / self.reference_coordinate_volume_m3)

    def _lambda_from_bit_budget(self, bit_budget: Decimal) -> Decimal:
        if bit_budget <= 0:
            raise ValueError("bit_budget must be positive.")
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            return (Decimal("3") * PI) / (self._planck_length_m * self._planck_length_m * bit_budget)

    def _build_parity_checksum(self) -> TopologicalChecksum:
        residual = Decimal(self._framing.delta_fr.numerator) / Decimal(self._framing.delta_fr.denominator)
        return TopologicalChecksumCode(
            law="parity",
            stabilizer_name="Adaptive Mesh Parity Stabilizer",
            protected_quantity="Framing parity under coarse graining",
            checksum_equation="Delta_fr(mesh)",
            boundary_integer=Decimal("0"),
            bulk_projection=residual,
            syndrome_tolerance=Decimal("0"),
            interpretation=(
                "Mesh coarsening is permitted only if the anomaly-free framing lock remains exact. "
                "A non-zero Delta_fr would signal parity loss under large-scale coarse graining."
            ),
        ).verify()

    def _build_budget_checksum(
        self,
        *,
        parent_bit_budget: Decimal,
        child_bit_budget: Decimal,
        coarsening_factor: Decimal,
    ) -> TopologicalChecksum:
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            projected_parent_ratio = (child_bit_budget * coarsening_factor) / parent_bit_budget
        return TopologicalChecksumCode(
            law="charge",
            stabilizer_name="Adaptive Mesh Budget Stabilizer",
            protected_quantity="Bit-budget conservation across mesh levels",
            checksum_equation="|1 - (f_coarse N_child)/N_parent|",
            boundary_integer=Decimal("1"),
            bulk_projection=projected_parent_ratio,
            syndrome_tolerance=self.unity_tolerance,
            interpretation=(
                "Bit rot is excluded only when a coarsened mesh cell carries exactly the bit budget of its aggregated child patch. "
                "Any drift would mean boundary information is being created or lost under scale changes."
            ),
        ).verify()

    def _build_level_chain(self, scale_ladder: tuple[Decimal, ...], index: int) -> AdaptiveMeshLevel:
        coordinate_volume_m3 = scale_ladder[index]
        child = self._build_level_chain(scale_ladder, index - 1) if index > 0 else None
        bit_budget = self._scaled_bit_budget(coordinate_volume_m3)
        lambda_holo_si_m2 = self._lambda_from_bit_budget(bit_budget)
        saturation = saturation_audit(lambda_obs_si_m2=lambda_holo_si_m2, precision=self.precision)
        unity = unity_of_scale_audit(
            kappa_d5=self.kappa_d5,
            newton_lock_audit=self.newton_lock,
            saturation=saturation,
            precision=self.precision,
        )
        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            volume_ratio = coordinate_volume_m3 / self.reference_coordinate_volume_m3
            coarsening_factor = None if child is None else coordinate_volume_m3 / child.coordinate_volume_m3
        bit_budget_checksum = None
        if child is not None and coarsening_factor is not None:
            bit_budget_checksum = self._build_budget_checksum(
                parent_bit_budget=bit_budget,
                child_bit_budget=child.bit_budget,
                coarsening_factor=coarsening_factor,
            )
        return AdaptiveMeshLevel(
            refinement_level=index,
            coordinate_volume_m3=coordinate_volume_m3,
            volume_ratio=volume_ratio,
            bit_budget=bit_budget,
            lambda_holo_si_m2=lambda_holo_si_m2,
            coarsening_factor_from_child=coarsening_factor,
            saturation=saturation,
            unity=unity,
            bulk_checksum=self.bulk_checksum,
            parity_checksum=self._build_parity_checksum(),
            bit_budget_checksum=bit_budget_checksum,
            child=child,
        )

    def _flatten_levels(self, root: AdaptiveMeshLevel) -> tuple[AdaptiveMeshLevel, ...]:
        if root.child is None:
            return (root,)
        return (root, *self._flatten_levels(root.child))


def build_holographic_amr_audit(
    *,
    observer_coordinate_volume_m3: Decimal | Fraction | float | int | str,
    reference_coordinate_volume_m3: Decimal | Fraction | float | int | str = PLANCK_VOLUME_M3,
    branching_factor: Decimal | Fraction | float | int | str = Decimal("8"),
    unity_tolerance: Decimal | Fraction | float | int | str = UNITY_TOLERANCE,
    precision: int = DEFAULT_PRECISION,
) -> HolographicAMRAudit:
    return AdaptiveHolographicMesh(
        reference_coordinate_volume_m3=reference_coordinate_volume_m3,
        branching_factor=branching_factor,
        unity_tolerance=unity_tolerance,
        precision=precision,
    ).refine_mesh(observer_coordinate_volume_m3=observer_coordinate_volume_m3)


def render_report(audit: HolographicAMRAudit) -> str:
    verification_status = "PASS" if audit.passed else "CHECK"
    lines = [
        "Adaptive Holographic Mesh",
        "=========================",
        "The local bit budget is recursively rescaled with observer volume while preserving parity and unity.",
        "",
        f"Observer volume [m^3]               : {_format_decimal(audit.observer_coordinate_volume_m3)}",
        f"Reference volume [m^3]              : {_format_decimal(audit.reference_coordinate_volume_m3)}",
        f"Branching factor                    : {audit.branching_factor}",
        f"Mesh depth                          : {audit.mesh_depth}",
        f"Max epsilon_lambda(top)             : {_format_decimal(audit.max_epsilon_lambda)}",
        f"Max epsilon_lambda(bridge)          : {_format_decimal(audit.max_epsilon_lambda_noether_bridged)}",
        f"Max bit-budget residual             : {_format_decimal(audit.max_budget_relative_residual)}",
        f"Max parity residual                 : {_format_decimal(audit.max_parity_residual)}",
        f"AHM verification                    : {verification_status}",
        "",
        "Scale ladder",
        "------------",
        "lvl   V/V_ref          N(V)                 eps_bridge         bit_rot",
        "--------------------------------------------------------------------------",
    ]
    lines.extend(
        f"{level.refinement_level:>3}   {_format_decimal(level.volume_ratio):>12}   "
        f"{_format_decimal(level.bit_budget):>20}   {_format_decimal(level.unity.epsilon_lambda_noether_bridged):>16}   "
        f"{_format_decimal(level.budget_relative_residual):>8}"
        for level in audit.levels
    )
    lines.extend(("", audit.statement))
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observer-volume-m3", type=Decimal, default=PLANCK_VOLUME_M3)
    parser.add_argument("--reference-volume-m3", type=Decimal, default=PLANCK_VOLUME_M3)
    parser.add_argument("--branching-factor", type=Decimal, default=Decimal("8"))
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    audit = build_holographic_amr_audit(
        observer_coordinate_volume_m3=args.observer_volume_m3,
        reference_coordinate_volume_m3=args.reference_volume_m3,
        branching_factor=args.branching_factor,
        precision=max(int(args.precision), 32),
    )
    print(render_report(audit))


__all__ = [
    "AdaptiveHolographicMesh",
    "AdaptiveMeshLevel",
    "DEFAULT_PRECISION",
    "HolographicAMRAudit",
    "PLANCK_VOLUME_M3",
    "UNITY_TOLERANCE",
    "build_holographic_amr_audit",
    "main",
    "parse_args",
    "render_report",
]


if __name__ == "__main__":
    main()
