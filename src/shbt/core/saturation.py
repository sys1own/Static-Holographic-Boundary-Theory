from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import sys

from shbt.constants import HOLOGRAPHIC_NOISE_FLOOR, TREAT_N_AS_BOUNDARY_CONDITION


def _compat_noise_floor() -> Decimal:
    noether_bridge = sys.modules.get("shbt.core.noether_bridge")
    value = getattr(noether_bridge, "HOLOGRAPHIC_NOISE_FLOOR", HOLOGRAPHIC_NOISE_FLOOR)
    return Decimal(str(value))


def _compat_boundary_condition_flag() -> bool:
    noether_bridge = sys.modules.get("shbt.core.noether_bridge")
    return bool(getattr(noether_bridge, "TREAT_N_AS_BOUNDARY_CONDITION", TREAT_N_AS_BOUNDARY_CONDITION))


@dataclass(frozen=True)
class SaturationAudit:
    lambda_obs_si_m2: Decimal
    lambda_obs_ev2: Decimal
    holographic_bits_from_lambda: Decimal
    configured_holographic_bits: Decimal
    register_noise_floor: Decimal
    relative_mismatch: Decimal

    @property
    def success(self) -> bool:
        is_saturated = abs(self.relative_mismatch) < _compat_noise_floor()
        return is_saturated

    @property
    def passed(self) -> bool:
        return self.success

    @property
    def is_saturated(self) -> bool:
        return self.success

    @property
    def boundary_condition_locked(self) -> bool:
        if _compat_boundary_condition_flag():
            return True
        return self.is_saturated
