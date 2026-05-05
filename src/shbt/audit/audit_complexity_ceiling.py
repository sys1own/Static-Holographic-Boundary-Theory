from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.main import (
    HBAR_EV_SECONDS,
    HOLOGRAPHIC_BITS,
    LIGHT_SPEED_M_PER_S,
    LloydBridge,
    PLANCK_LENGTH_M,
    TopologicalVacuum,
    calculate_lloyds_limit_bound,
    verify_dark_energy_tension,
    verify_unitary_bounds,
)

DEFAULT_PRECISION = 80
_GUARD_DIGITS = 12


def _decimal(value: Decimal | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
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


@dataclass(frozen=True)
class ComplexityCeilingAudit:
    holographic_bits: Decimal
    rho_vac_surface_tension_ev4: Decimal
    horizon_radius_m: Decimal
    horizon_volume_m3: Decimal
    meter_to_ev_inverse: Decimal
    horizon_volume_ev_minus3: Decimal
    vacuum_surface_energy_ev: Decimal
    lloyd_limit_ops_per_second: Decimal
    live_lloyd_limit_ops_per_second: Decimal
    scalar_tilt: Decimal
    clock_skew: Decimal
    clock_skew_throttle_ops_per_second: Decimal
    complexity_growth_rate_ops_per_second: Decimal
    live_complexity_growth_rate_ops_per_second: Decimal
    complexity_utilization_fraction: Decimal
    live_complexity_utilization_fraction: Decimal
    unitary_bound_satisfied: bool
    universal_computational_limit_pass: bool


def derive_complexity_ceiling(*, precision: int = DEFAULT_PRECISION) -> ComplexityCeilingAudit:
    model = TopologicalVacuum()
    dark_energy_audit = verify_dark_energy_tension(model=model)
    unitary_audit = verify_unitary_bounds(model=model)
    cosmology_audit = model.derive_cosmology_audit()

    with localcontext() as context:
        context.prec = precision + _GUARD_DIGITS
        holographic_bits = _decimal(HOLOGRAPHIC_BITS)
        rho_vac_surface_tension_ev4 = _decimal(dark_energy_audit.rho_vac_surface_tension_ev4)
        pi_value = _decimal(math.pi)
        horizon_radius_m = _decimal(PLANCK_LENGTH_M) * (holographic_bits / pi_value).sqrt()
        horizon_volume_m3 = (Decimal(4) * pi_value / Decimal(3)) * horizon_radius_m**3
        meter_to_ev_inverse = Decimal(1) / (_decimal(HBAR_EV_SECONDS) * _decimal(LIGHT_SPEED_M_PER_S))
        horizon_volume_ev_minus3 = horizon_volume_m3 * meter_to_ev_inverse**3
        vacuum_surface_energy_ev = rho_vac_surface_tension_ev4 * horizon_volume_ev_minus3
        lloyd_limit_ops_per_second = (Decimal(2) * vacuum_surface_energy_ev) / (pi_value * _decimal(HBAR_EV_SECONDS))
        scalar_tilt = _decimal(cosmology_audit.n_s_locked)
        clock_skew = Decimal(1) - scalar_tilt
        clock_skew_throttle_ops_per_second = clock_skew * lloyd_limit_ops_per_second
        complexity_growth_rate_ops_per_second = scalar_tilt * lloyd_limit_ops_per_second
        complexity_utilization_fraction = complexity_growth_rate_ops_per_second / lloyd_limit_ops_per_second

        live_lloyd_limit_ops_per_second = _decimal(
            LloydBridge.limit_ops_per_second(
                dark_energy_audit.rho_vac_surface_tension_ev4,
                bit_count=model.bit_count,
            )
        )
        live_complexity_growth_rate_ops_per_second = _decimal(unitary_audit.complexity_growth_rate_ops_per_second)
        live_complexity_utilization_fraction = _decimal(unitary_audit.complexity_utilization_fraction)

        context.prec = precision

    tolerance_limit = Decimal("1e-12") * max(abs(live_lloyd_limit_ops_per_second), Decimal(1))
    tolerance_growth = Decimal("1e-12") * max(abs(live_complexity_growth_rate_ops_per_second), Decimal(1))
    tolerance_fraction = Decimal("1e-15")

    assert abs(lloyd_limit_ops_per_second - live_lloyd_limit_ops_per_second) <= tolerance_limit
    assert abs(complexity_growth_rate_ops_per_second - live_complexity_growth_rate_ops_per_second) <= tolerance_growth
    assert abs(complexity_utilization_fraction - live_complexity_utilization_fraction) <= tolerance_fraction
    assert complexity_utilization_fraction <= Decimal(1) + tolerance_fraction

    return ComplexityCeilingAudit(
        holographic_bits=+holographic_bits,
        rho_vac_surface_tension_ev4=+rho_vac_surface_tension_ev4,
        horizon_radius_m=+horizon_radius_m,
        horizon_volume_m3=+horizon_volume_m3,
        meter_to_ev_inverse=+meter_to_ev_inverse,
        horizon_volume_ev_minus3=+horizon_volume_ev_minus3,
        vacuum_surface_energy_ev=+vacuum_surface_energy_ev,
        lloyd_limit_ops_per_second=+lloyd_limit_ops_per_second,
        live_lloyd_limit_ops_per_second=+live_lloyd_limit_ops_per_second,
        scalar_tilt=+scalar_tilt,
        clock_skew=+clock_skew,
        clock_skew_throttle_ops_per_second=+clock_skew_throttle_ops_per_second,
        complexity_growth_rate_ops_per_second=+complexity_growth_rate_ops_per_second,
        live_complexity_growth_rate_ops_per_second=+live_complexity_growth_rate_ops_per_second,
        complexity_utilization_fraction=+complexity_utilization_fraction,
        live_complexity_utilization_fraction=+live_complexity_utilization_fraction,
        unitary_bound_satisfied=bool(unitary_audit.unitary_bound_satisfied),
        universal_computational_limit_pass=bool(unitary_audit.universal_computational_limit_pass),
    )


def build_complexity_ceiling_report(*, precision: int = DEFAULT_PRECISION) -> str:
    audit = derive_complexity_ceiling(precision=precision)
    lines = [
        "Complexity Ceiling Audit",
        "========================",
        "",
        "Stage XIII Lloyd Bridge",
        f"- N_holo = {_format_decimal(audit.holographic_bits, places=18)}",
        f"- rho_vac^surf = {_format_decimal(audit.rho_vac_surface_tension_ev4, places=24)} eV^4",
        f"- horizon radius = L_P * sqrt(N_holo/pi) = {_format_decimal(audit.horizon_radius_m, places=18)} m",
        f"- horizon volume = 4*pi*R^3/3 = {_format_decimal(audit.horizon_volume_m3, places=18)} m^3",
        f"- meter->eV^-1 = 1/(hbar*c) = {_format_decimal(audit.meter_to_ev_inverse, places=18)}",
        f"- horizon volume [eV^-3] = {_format_decimal(audit.horizon_volume_ev_minus3, places=18)}",
        f"- E_vac^surf = rho_vac^surf * V_hor = {_format_decimal(audit.vacuum_surface_energy_ev, places=18)} eV",
        f"- dC/dt = 2 E_vac^surf / (pi hbar) = {_format_decimal(audit.lloyd_limit_ops_per_second, places=18)} s^-1",
        f"- live LloydBridge check = {_format_decimal(audit.live_lloyd_limit_ops_per_second, places=18)} s^-1",
        "",
        "Clock-Skew Analysis",
        f"- n_s = {_format_decimal(audit.scalar_tilt, places=18)}",
        f"- clock-skew = 1 - n_s = {_format_decimal(audit.clock_skew, places=18)}",
        f"- skew throttle = (1 - n_s) * dC/dt = {_format_decimal(audit.clock_skew_throttle_ops_per_second, places=18)} s^-1",
        f"- realized growth = n_s * dC/dt = {_format_decimal(audit.complexity_growth_rate_ops_per_second, places=18)} s^-1",
        f"- live growth check = {_format_decimal(audit.live_complexity_growth_rate_ops_per_second, places=18)} s^-1",
        "",
        "Efficiency Audit",
        f"- utilization fraction = (realized growth)/(Lloyd limit) = {_format_decimal(audit.complexity_utilization_fraction, places=18)}",
        f"- live utilization check = {_format_decimal(audit.live_complexity_utilization_fraction, places=18)}",
        f"- utilization <= 1 = {audit.complexity_utilization_fraction <= Decimal(1)}",
        f"- unitary bound satisfied = {audit.unitary_bound_satisfied}",
        f"- universal computational limit pass = {audit.universal_computational_limit_pass}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the Stage XIII complexity ceiling of the 4D bulk.")
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION, help="Decimal precision used for the ceiling audit.")
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    print(build_complexity_ceiling_report(precision=max(args.precision, 32)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
