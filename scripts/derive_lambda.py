from __future__ import annotations

import argparse
import json
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

from derive_universe import (
    DEFAULT_PRECISION,
    decimal_pi,
    derive_kappa_d5,
    derive_mass_bridge,
    derive_unity_of_scale,
)
from shbt.constants import (
    BENCHMARK_DIAGNOSTICS_FILENAME,
    HOLOGRAPHIC_BITS,
    LEPTON_LEVEL,
    PARENT_LEVEL,
    PLANCK_LENGTH_M,
    QUARK_LEVEL,
)
from shbt.main import (
    derive_cosmology_anchor,
    holographic_lambda_scaling_identity_si_m2,
    holographic_surface_tension_lambda_si_m2,
    lambda_si_m2_to_ev2,
    verify_unity_of_scale,
)
from shbt.paths import ProjectPaths

_GUARD_DIGITS = 12
LIVE_BENCHMARK_AUDIT_SKIPPED_MESSAGE = "Benchmark artifacts not found; skipping live comparison audit"


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
class LambdaSurfaceDerivation:
    holographic_bits: Decimal
    planck_length_m: Decimal
    scaling_identity_si_m2: Decimal
    scaling_identity_ev2: Decimal
    lambda_holo_si_m2: Decimal
    lambda_holo_ev2: Decimal
    live_scaling_identity_si_m2: Decimal
    live_lambda_holo_si_m2: Decimal
    live_lambda_holo_ev2: Decimal
    scaling_identity_drift_si_m2: Decimal
    lambda_holo_drift_si_m2: Decimal
    lambda_holo_ev2_drift: Decimal
    surface_tension_prefactor: Decimal
    anchor_lambda_si_m2: Decimal
    anchor_lambda_ev2: Decimal
    anchor_ratio: Decimal
    deviation_percent: Decimal


@dataclass(frozen=True)
class CheckedInUnityPayload:
    source_path: str
    epsilon_lambda: Decimal
    exact_epsilon_lambda: Decimal
    numerical_residual: Decimal
    register_noise_floor: Decimal
    exact_register_noise_floor: Decimal
    passed: bool

    @property
    def comparison_tuple(self) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal, bool]:
        return (
            self.epsilon_lambda,
            self.exact_epsilon_lambda,
            self.numerical_residual,
            self.register_noise_floor,
            self.exact_register_noise_floor,
            self.passed,
        )


def derive_lambda_surface(*, precision: int = DEFAULT_PRECISION) -> LambdaSurfaceDerivation:
    with localcontext() as context:
        context.prec = precision + _GUARD_DIGITS
        holographic_bits = _decimal(HOLOGRAPHIC_BITS)
        planck_length_m = _decimal(PLANCK_LENGTH_M)
        scaling_identity_si_m2 = Decimal(1) / (holographic_bits * planck_length_m * planck_length_m)
        scaling_identity_ev2 = _decimal(lambda_si_m2_to_ev2(float(scaling_identity_si_m2)))
        surface_tension_prefactor = Decimal(3) * decimal_pi(context.prec)
        lambda_holo_si_m2 = surface_tension_prefactor * scaling_identity_si_m2
        lambda_holo_ev2 = _decimal(lambda_si_m2_to_ev2(float(lambda_holo_si_m2)))

        live_scaling_identity_si_m2 = _decimal(holographic_lambda_scaling_identity_si_m2())
        live_lambda_holo_si_m2 = _decimal(holographic_surface_tension_lambda_si_m2())
        live_lambda_holo_ev2 = _decimal(lambda_si_m2_to_ev2(float(live_lambda_holo_si_m2)))

        cosmology_anchor = derive_cosmology_anchor()
        anchor_lambda_si_m2 = _decimal(cosmology_anchor.lambda_si_m2)
        anchor_lambda_ev2 = _decimal(lambda_si_m2_to_ev2(cosmology_anchor.lambda_si_m2))
        anchor_ratio = lambda_holo_si_m2 / anchor_lambda_si_m2
        deviation_percent = Decimal(100) * abs(anchor_ratio - Decimal(1))

        scaling_identity_drift_si_m2 = abs(scaling_identity_si_m2 - live_scaling_identity_si_m2)
        lambda_holo_drift_si_m2 = abs(lambda_holo_si_m2 - live_lambda_holo_si_m2)
        lambda_holo_ev2_drift = abs(lambda_holo_ev2 - live_lambda_holo_ev2)

        context.prec = precision
        return LambdaSurfaceDerivation(
            holographic_bits=+holographic_bits,
            planck_length_m=+planck_length_m,
            scaling_identity_si_m2=+scaling_identity_si_m2,
            scaling_identity_ev2=+scaling_identity_ev2,
            lambda_holo_si_m2=+lambda_holo_si_m2,
            lambda_holo_ev2=+lambda_holo_ev2,
            live_scaling_identity_si_m2=+live_scaling_identity_si_m2,
            live_lambda_holo_si_m2=+live_lambda_holo_si_m2,
            live_lambda_holo_ev2=+live_lambda_holo_ev2,
            scaling_identity_drift_si_m2=+scaling_identity_drift_si_m2,
            lambda_holo_drift_si_m2=+lambda_holo_drift_si_m2,
            lambda_holo_ev2_drift=+lambda_holo_ev2_drift,
            surface_tension_prefactor=+surface_tension_prefactor,
            anchor_lambda_si_m2=+anchor_lambda_si_m2,
            anchor_lambda_ev2=+anchor_lambda_ev2,
            anchor_ratio=+anchor_ratio,
            deviation_percent=+deviation_percent,
        )


def _checked_in_diagnostics_paths() -> tuple[Path, ...]:
    results_dir = ProjectPaths.RESULTS
    if not results_dir.is_dir():
        return ()
    preferred_paths = (
        results_dir / "final" / BENCHMARK_DIAGNOSTICS_FILENAME,
        results_dir / BENCHMARK_DIAGNOSTICS_FILENAME,
    )
    existing_preferred_paths = tuple(path for path in preferred_paths if path.is_file())
    if existing_preferred_paths:
        return existing_preferred_paths
    for candidate in sorted(results_dir.rglob(BENCHMARK_DIAGNOSTICS_FILENAME)):
        if candidate.is_file():
            return (candidate,)
    return ()


def _display_diagnostics_path(diagnostics_path: Path) -> str:
    try:
        return str(diagnostics_path.relative_to(ProjectPaths.ROOT))
    except ValueError:
        return str(diagnostics_path)


def load_checked_in_unity_payloads() -> list[CheckedInUnityPayload]:
    payloads: list[CheckedInUnityPayload] = []
    diagnostics_paths = _checked_in_diagnostics_paths()

    if not diagnostics_paths:
        print(LIVE_BENCHMARK_AUDIT_SKIPPED_MESSAGE)
        return []

    for diagnostics_path in diagnostics_paths:
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"), parse_float=Decimal)

        unity_payload = diagnostics.get("unity_of_scale_identity")
        if not isinstance(unity_payload, dict):
            raise KeyError(
                f"Checked-in benchmark diagnostics at {diagnostics_path} do not expose a unity_of_scale_identity payload."
            )

        payloads.append(
            CheckedInUnityPayload(
                source_path=_display_diagnostics_path(diagnostics_path),
                epsilon_lambda=_decimal(unity_payload["epsilon_lambda"]),
                exact_epsilon_lambda=_decimal(unity_payload["exact_epsilon_lambda"]),
                numerical_residual=_decimal(unity_payload["numerical_residual"]),
                register_noise_floor=_decimal(unity_payload["register_noise_floor"]),
                exact_register_noise_floor=_decimal(unity_payload["exact_register_noise_floor"]),
                passed=bool(unity_payload["passed"]),
            )
        )

    if not payloads:
        print(LIVE_BENCHMARK_AUDIT_SKIPPED_MESSAGE)
        return []

    reference_payload = payloads[0]
    for payload in payloads[1:]:
        assert payload.comparison_tuple == reference_payload.comparison_tuple, (
            "Checked-in benchmark diagnostics no longer mirror the same unity-of-scale payload: "
            f"{reference_payload.source_path} != {payload.source_path}."
        )

    return payloads


def build_lambda_ledger(*, precision: int = DEFAULT_PRECISION) -> str:
    lambda_surface = derive_lambda_surface(precision=precision)
    kappa = derive_kappa_d5(precision=precision)
    mass = derive_mass_bridge(precision=precision, kappa=kappa.kappa)
    decimal_unity = derive_unity_of_scale(precision=precision, kappa=kappa.kappa, mass_bridge=mass)
    live_unity = verify_unity_of_scale()
    checked_in_payloads = load_checked_in_unity_payloads()

    live_epsilon_lambda = _decimal(live_unity["epsilon_lambda"])
    live_exact_epsilon_lambda = _decimal(live_unity["exact_epsilon_lambda"])
    live_numerical_residual = _decimal(live_unity["numerical_residual"])
    live_register_noise_floor = _decimal(live_unity["register_noise_floor"])
    live_exact_register_noise_floor = _decimal(live_unity["exact_register_noise_floor"])
    live_passed = bool(live_unity["passed"])

    for payload in checked_in_payloads:
        assert payload.epsilon_lambda == live_epsilon_lambda, (
            "Checked-in epsilon_Lambda no longer matches the live verifier payload: "
            f"{payload.source_path} reports {payload.epsilon_lambda}, live tn.py reports {live_epsilon_lambda}."
        )
        assert payload.exact_epsilon_lambda == live_exact_epsilon_lambda, (
            "Checked-in exact epsilon_Lambda no longer matches the live verifier payload: "
            f"{payload.source_path} reports {payload.exact_epsilon_lambda}, live tn.py reports {live_exact_epsilon_lambda}."
        )
        assert payload.numerical_residual == live_numerical_residual, (
            "Checked-in numerical unity residual no longer matches the live verifier payload: "
            f"{payload.source_path} reports {payload.numerical_residual}, live tn.py reports {live_numerical_residual}."
        )
        assert payload.register_noise_floor == live_register_noise_floor, (
            "Checked-in register noise floor no longer matches the live verifier payload: "
            f"{payload.source_path} reports {payload.register_noise_floor}, live tn.py reports {live_register_noise_floor}."
        )
        assert payload.exact_register_noise_floor == live_exact_register_noise_floor, (
            "Checked-in exact register noise floor no longer matches the live verifier payload: "
            f"{payload.source_path} reports {payload.exact_register_noise_floor}, live tn.py reports {live_exact_register_noise_floor}."
        )
        assert payload.passed == live_passed, (
            "Checked-in unity-of-scale pass/fail flag no longer matches the live verifier payload: "
            f"{payload.source_path} reports {payload.passed}, live tn.py reports {live_passed}."
        )

    lines = [
        "Lambda Ledger",
        "=============",
        "",
        "Branch Integers",
        f"- k_l = {LEPTON_LEVEL}",
        f"- k_q = {QUARK_LEVEL}",
        f"- K = {PARENT_LEVEL}",
        "",
        "Holographic Surface Tension",
        f"- N_holo = {_format_decimal(lambda_surface.holographic_bits, places=18)}",
        f"- L_P = {_format_decimal(lambda_surface.planck_length_m, places=24)} m",
        f"- 1/(L_P^2 N_holo) = {_format_decimal(lambda_surface.scaling_identity_si_m2, places=24)} m^(-2)",
        f"- 1/(L_P^2 N_holo) [eV^2] = {_format_decimal(lambda_surface.scaling_identity_ev2, places=24)} eV^2",
        f"- surface-tension prefactor = 3*pi = {_format_decimal(lambda_surface.surface_tension_prefactor, places=24)}",
        f"- Lambda_holo = 3*pi/(L_P^2 N_holo) = {_format_decimal(lambda_surface.lambda_holo_si_m2, places=24)} m^(-2)",
        f"- Lambda_holo [eV^2] = {_format_decimal(lambda_surface.lambda_holo_ev2, places=24)} eV^2",
        f"- tn.py live 1/(L_P^2 N_holo) = {_format_decimal(lambda_surface.live_scaling_identity_si_m2, places=24)} m^(-2)",
        f"- tn.py live Lambda_holo = {_format_decimal(lambda_surface.live_lambda_holo_si_m2, places=24)} m^(-2)",
        f"- tn.py live Lambda_holo [eV^2] = {_format_decimal(lambda_surface.live_lambda_holo_ev2, places=24)} eV^2",
        f"- scaling-identity drift = {_format_decimal(lambda_surface.scaling_identity_drift_si_m2, places=24)} m^(-2)",
        f"- Lambda_holo drift [m^(-2)] = {_format_decimal(lambda_surface.lambda_holo_drift_si_m2, places=24)} m^(-2)",
        f"- Lambda_holo drift [eV^2] = {_format_decimal(lambda_surface.lambda_holo_ev2_drift, places=24)} eV^2",
        "",
        "Planck 2018 Anchor",
        f"- Lambda_obs = {_format_decimal(lambda_surface.anchor_lambda_si_m2, places=24)} m^(-2)",
        f"- Lambda_obs [eV^2] = {_format_decimal(lambda_surface.anchor_lambda_ev2, places=24)} eV^2",
        f"- Lambda_holo/Lambda_obs = {_format_decimal(lambda_surface.anchor_ratio, places=24)}",
        f"- surface-tension deviation = {_format_decimal(lambda_surface.deviation_percent, places=24)} %",
        "",
        "Unity of Scale Closure",
        f"- kappa_D5 = {_format_decimal(kappa.kappa, places=24)}",
        f"- m_nu = kappa_D5 * M_P * N_holo^(-1/4) = {_format_decimal(mass.neutrino_floor_ev, places=24)} eV",
        f"- G_N = {_format_decimal(decimal_unity.g_newton_topological_ev_minus2, places=24)} eV^(-2)",
        f"- Lambda_holo(lhs) = {_format_decimal(decimal_unity.lambda_holo_ev2, places=24)} eV^2",
        f"- Lambda_holo(rhs) = (3*pi/kappa_D5^4) * G_N * m_nu^4 = {_format_decimal(decimal_unity.expanded_theorem_rhs_ev2, places=24)} eV^2",
        f"- Decimal spot-check lhs/rhs = {_format_decimal(decimal_unity.identity_ratio, places=24)}",
        f"- Decimal spot-check epsilon_Lambda = {_format_decimal(decimal_unity.epsilon_lambda, places=24)}",
        f"- Decimal audit tolerance = {_format_decimal(decimal_unity.decimal_tolerance, places=24)}",
        f"- tn.py exact epsilon_Lambda = {_format_decimal(live_exact_epsilon_lambda, places=24)}",
        f"- tn.py numerical residual = {_format_decimal(live_numerical_residual, places=24)}",
        f"- register noise floor = {_format_decimal(live_register_noise_floor, places=24)}",
        f"- exact register noise floor = {_format_decimal(live_exact_register_noise_floor, places=24)}",
        f"- closes in benchmark verifier = {live_passed}",
        "",
        "Checked-In Benchmark Payloads",
    ]

    if checked_in_payloads:
        lines.append(f"- mirrored checked-in payloads = {len(checked_in_payloads)}")
        for payload in checked_in_payloads:
            lines.append(
                "- "
                f"{payload.source_path}: epsilon_Lambda={_format_decimal(payload.epsilon_lambda, places=24)}, "
                f"exact_epsilon_Lambda={_format_decimal(payload.exact_epsilon_lambda, places=24)}, "
                f"numerical_residual={_format_decimal(payload.numerical_residual, places=24)}, "
                f"register_noise_floor={_format_decimal(payload.register_noise_floor, places=24)}, "
                f"passed={payload.passed}"
            )

        lines.extend(
            [
                f"- live exact epsilon_Lambda - checked-in exact epsilon_Lambda = {_format_decimal(live_exact_epsilon_lambda - checked_in_payloads[0].exact_epsilon_lambda, places=24)}",
                f"- live numerical residual - checked-in numerical residual = {_format_decimal(live_numerical_residual - checked_in_payloads[0].numerical_residual, places=24)}",
                f"- live register noise floor - checked-in register noise floor = {_format_decimal(live_register_noise_floor - checked_in_payloads[0].register_noise_floor, places=24)}",
                "- checked-in payloads mirror the live unity-of-scale benchmark = True",
            ]
        )
    else:
        lines.append("- mirrored checked-in payloads = 0")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive the holographic cosmological constant from the branch-fixed ledger.")
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION, help="Decimal precision used for the lambda ledger.")
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    print(build_lambda_ledger(precision=max(args.precision, DEFAULT_PRECISION)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
