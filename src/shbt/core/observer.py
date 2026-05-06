from __future__ import annotations

"""Observer-facing bulk/boundary projections of Noether-bridge residues.

This module combines the global Noether-bridge bookkeeping with the local
observer-horizon model already used elsewhere in the repository. The guiding
idea is compact:

- the holographic budget ``N`` is fixed globally by the cosmological boundary,
- a local observer only accesses a finite horizon sub-capacity of that budget,
  and
- the same bridge residues can therefore be partitioned into a boundary-facing
  share and a bulk-facing share without changing their invariant totals.

The frame shift is modeled by dressing the local logarithmic horizon exposure by
the remaining-horizon redshift factor. As the observer moves through the bulk,
the perceived residue budget tilts from boundary-dominated bookkeeping toward a
bulk-dressed image, while the total bit budget and the total residue amplitudes
remain conserved.
"""

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache

import mpmath

from shbt.constants import (
    HOLOGRAPHIC_BITS,
    LEPTON_LEVEL,
    PARENT_LEVEL,
    PLANCK2018_LAMBDA_SI_M2,
    QUARK_LEVEL,
)
from shbt.core import noether_bridge
from shbt.core.observer_horizon import (
    BENCHMARK_BRANCH,
    DEFAULT_PRECISION as HORIZON_DEFAULT_PRECISION,
    ObserverHorizonLimit,
    calculate_observer_horizon_limit,
    global_coordinate_horizon_radius,
)


DEFAULT_PRECISION = max(
    int(noether_bridge.DEFAULT_PRECISION),
    int(HORIZON_DEFAULT_PRECISION),
)
_GUARD_DIGITS = 16
_CONSERVATION_TOLERANCE = Decimal("1e-50")


def _decimal(value: Decimal | Fraction | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


def _fraction_to_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _mp_to_decimal(value: mpmath.mpf, *, precision: int) -> Decimal:
    return Decimal(mpmath.nstr(value, n=max(int(precision), DEFAULT_PRECISION)))


def _cache_key(value: Decimal) -> str:
    return format(value, "f") if value.is_finite() else str(value)


@dataclass(frozen=True)
class NoetherBridgeResidues:
    completion_residue: Decimal
    unity_closure_residue: Decimal
    framing_defect: Decimal


@dataclass(frozen=True)
class ObserverPerspectiveAudit:
    benchmark_branch: tuple[int, int, int]
    evaluated_branch: tuple[int, int, int]
    horizon_limit: ObserverHorizonLimit
    entropy_capacity_bits: Decimal
    hidden_entropy_capacity_bits: Decimal
    global_bit_budget: Decimal
    register_noise_floor: Decimal
    covariant_frame_shift: Decimal
    boundary_weight: Decimal
    bulk_weight: Decimal
    invariant_residues: NoetherBridgeResidues
    boundary_residues: NoetherBridgeResidues
    bulk_residues: NoetherBridgeResidues
    bit_budget_residual: Decimal
    completion_residue_residual: Decimal
    unity_closure_residue_residual: Decimal
    framing_defect_residual: Decimal

    @property
    def local_horizon_radius_m(self) -> Decimal:
        return self.horizon_limit.coordinate_horizon_radius_m

    @property
    def bit_budget_conserved(self) -> bool:
        tolerance = max(self.register_noise_floor, _CONSERVATION_TOLERANCE)
        return abs(self.bit_budget_residual) <= tolerance

    @property
    def residues_conserved(self) -> bool:
        tolerance = max(self.register_noise_floor, _CONSERVATION_TOLERANCE)
        return bool(
            abs(self.completion_residue_residual) <= tolerance
            and abs(self.unity_closure_residue_residual) <= tolerance
            and abs(self.framing_defect_residual) <= tolerance
        )

    def assert_global_conservation(self) -> None:
        if not self.bit_budget_conserved:
            raise AssertionError(
                "Observer frame breaks the global holographic bit budget: "
                f"residual={self.bit_budget_residual:.6e}."
            )
        if not self.residues_conserved:
            raise AssertionError(
                "Observer frame breaks Noether-bridge residue conservation: "
                f"completion={self.completion_residue_residual:.6e}, "
                f"unity={self.unity_closure_residue_residual:.6e}, "
                f"framing={self.framing_defect_residual:.6e}."
            )


@lru_cache(maxsize=None)
def _benchmark_bridge_invariants(
    bit_budget_key: str,
    lambda_anchor_key: str,
    precision: int,
) -> tuple[Decimal, Decimal, Decimal]:
    c_dark_fraction = noether_bridge.load_c_dark_completion_fraction()
    kappa_d5 = noether_bridge.derive_kappa_d5(precision=precision)
    snapshot = noether_bridge.high_precision_unity_of_scale_snapshot(
        bit_count=Decimal(bit_budget_key),
        kappa_d5=kappa_d5,
        c_dark_fraction=c_dark_fraction,
        lambda_obs_si_m2=Decimal(lambda_anchor_key),
        mpmath_dps=precision,
    )
    return (
        _mp_to_decimal(snapshot.c_dark, precision=precision),
        _mp_to_decimal(snapshot.epsilon_lambda_noether_bridged, precision=precision),
        _mp_to_decimal(snapshot.register_noise_floor, precision=precision),
    )


def _scale_residues(
    residues: NoetherBridgeResidues,
    weight: Decimal,
) -> NoetherBridgeResidues:
    return NoetherBridgeResidues(
        completion_residue=residues.completion_residue * weight,
        unity_closure_residue=residues.unity_closure_residue * weight,
        framing_defect=residues.framing_defect * weight,
    )


class Observer:
    """Local observer carrying a finite horizon radius and entropy capacity."""

    def __init__(
        self,
        observer_radius_m: Decimal | Fraction | float | int | str = Decimal("0"),
        *,
        bit_budget: Decimal | Fraction | float | int | str = HOLOGRAPHIC_BITS,
        global_horizon_radius_m: Decimal | Fraction | float | int | str | None = None,
        lambda_anchor_si_m2: Decimal | Fraction | float | int | str = PLANCK2018_LAMBDA_SI_M2,
        lepton_level: int = LEPTON_LEVEL,
        quark_level: int = QUARK_LEVEL,
        parent_level: int = PARENT_LEVEL,
        precision: int = DEFAULT_PRECISION,
    ) -> None:
        self.precision = max(int(precision), DEFAULT_PRECISION)
        self.global_bit_budget = _decimal(bit_budget)
        if self.global_bit_budget <= 0:
            raise ValueError("bit_budget must be positive.")

        self.lambda_anchor_si_m2 = _decimal(lambda_anchor_si_m2)
        self.evaluated_branch = (
            int(lepton_level),
            int(quark_level),
            int(parent_level),
        )
        self.global_horizon_radius_m = (
            global_coordinate_horizon_radius(
                bit_count=self.global_bit_budget,
                precision=self.precision,
            )
            if global_horizon_radius_m is None
            else _decimal(global_horizon_radius_m)
        )
        self._horizon_limit = self._resolve_horizon_limit(observer_radius_m)

    @property
    def observer_radius_m(self) -> Decimal:
        return self._horizon_limit.observer_radius_m

    @property
    def local_horizon_radius_m(self) -> Decimal:
        return self._horizon_limit.coordinate_horizon_radius_m

    @property
    def entropy_capacity_bits(self) -> Decimal:
        return self._horizon_limit.local_available_bits

    @property
    def bekenstein_hawking_entropy_bits(self) -> Decimal:
        return self._horizon_limit.bekenstein_hawking_entropy_bits

    @property
    def horizon_limit(self) -> ObserverHorizonLimit:
        return self._horizon_limit

    def move_to(
        self,
        observer_radius_m: Decimal | Fraction | float | int | str,
    ) -> ObserverPerspectiveAudit:
        self._horizon_limit = self._resolve_horizon_limit(observer_radius_m)
        return self.perceive_noether_bridge()

    def move_by(
        self,
        radial_step_m: Decimal | Fraction | float | int | str,
    ) -> ObserverPerspectiveAudit:
        return self.move_to(self.observer_radius_m + _decimal(radial_step_m))

    def perceive_noether_bridge(self) -> ObserverPerspectiveAudit:
        completion_residue, unity_closure_residue, register_noise_floor = _benchmark_bridge_invariants(
            _cache_key(self.global_bit_budget),
            _cache_key(self.lambda_anchor_si_m2),
            self.precision,
        )
        lepton_level, quark_level, parent_level = self.evaluated_branch
        framing_audit = noether_bridge.framing_defect(
            parent_level,
            lepton_level,
            quark_level,
        )

        with localcontext() as context:
            context.prec = self.precision + _GUARD_DIGITS
            invariant_residues = NoetherBridgeResidues(
                completion_residue=completion_residue,
                unity_closure_residue=unity_closure_residue,
                framing_defect=abs(_fraction_to_decimal(framing_audit.delta_fr)),
            )
            safe_remaining_fraction = max(
                self._horizon_limit.remaining_horizon_fraction,
                Decimal(1) / self.global_bit_budget,
            )
            covariant_frame_shift = (
                self._horizon_limit.log_horizon_loading_factor
                / safe_remaining_fraction
            )
            boundary_weight = Decimal(1) / (Decimal(1) + covariant_frame_shift)
            bulk_weight = covariant_frame_shift / (Decimal(1) + covariant_frame_shift)
            boundary_residues = _scale_residues(invariant_residues, boundary_weight)
            bulk_residues = _scale_residues(invariant_residues, bulk_weight)
            hidden_entropy_capacity = self.global_bit_budget - self._horizon_limit.local_available_bits
            bit_budget_residual = (
                self._horizon_limit.local_available_bits + hidden_entropy_capacity - self.global_bit_budget
            )
            completion_residue_residual = (
                boundary_residues.completion_residue
                + bulk_residues.completion_residue
                - invariant_residues.completion_residue
            )
            unity_closure_residue_residual = (
                boundary_residues.unity_closure_residue
                + bulk_residues.unity_closure_residue
                - invariant_residues.unity_closure_residue
            )
            framing_defect_residual = (
                boundary_residues.framing_defect
                + bulk_residues.framing_defect
                - invariant_residues.framing_defect
            )

        audit = ObserverPerspectiveAudit(
            benchmark_branch=BENCHMARK_BRANCH,
            evaluated_branch=self.evaluated_branch,
            horizon_limit=self._horizon_limit,
            entropy_capacity_bits=self._horizon_limit.local_available_bits,
            hidden_entropy_capacity_bits=hidden_entropy_capacity,
            global_bit_budget=self.global_bit_budget,
            register_noise_floor=register_noise_floor,
            covariant_frame_shift=+covariant_frame_shift,
            boundary_weight=+boundary_weight,
            bulk_weight=+bulk_weight,
            invariant_residues=invariant_residues,
            boundary_residues=boundary_residues,
            bulk_residues=bulk_residues,
            bit_budget_residual=+bit_budget_residual,
            completion_residue_residual=+completion_residue_residual,
            unity_closure_residue_residual=+unity_closure_residue_residual,
            framing_defect_residual=+framing_defect_residual,
        )
        audit.assert_global_conservation()
        return audit

    def assert_global_bit_budget_conserved(
        self,
        audit: ObserverPerspectiveAudit | None = None,
    ) -> ObserverPerspectiveAudit:
        resolved_audit = self.perceive_noether_bridge() if audit is None else audit
        resolved_audit.assert_global_conservation()
        return resolved_audit

    def _resolve_horizon_limit(
        self,
        observer_radius_m: Decimal | Fraction | float | int | str,
    ) -> ObserverHorizonLimit:
        return calculate_observer_horizon_limit(
            observer_radius_m,
            global_horizon_radius_m=self.global_horizon_radius_m,
            bit_count=self.global_bit_budget,
            precision=self.precision,
        )


__all__ = [
    "DEFAULT_PRECISION",
    "NoetherBridgeResidues",
    "Observer",
    "ObserverPerspectiveAudit",
]
