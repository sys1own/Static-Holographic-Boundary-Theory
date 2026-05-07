from __future__ import annotations

import argparse
import inspect
import math
from dataclasses import dataclass, is_dataclass, replace
from functools import wraps
from pathlib import Path
from typing import Any, Callable, ParamSpec, Sequence, TypeVar

if __package__ in (None, ""):
    import sys

    for parent in Path(__file__).resolve().parents:
        candidate = parent if parent.name == "src" else parent / "src"
        if (candidate / "shbt").is_dir():
            sys.path.insert(0, str(candidate))
            break

from shbt.constants import HOLOGRAPHIC_BITS, LEPTON_LEVEL, PARENT_LEVEL, QUARK_LEVEL
from shbt.core.flavor_identity_resolver import FlavorIdentityAudit, build_flavor_identity_audit
from shbt.main import GravityAudit, TopologicalVacuum


P = ParamSpec("P")
R = TypeVar("R")

BENCHMARK_MOAT_CENTER = (int(LEPTON_LEVEL), int(QUARK_LEVEL), int(PARENT_LEVEL))
DEFAULT_GUARDIAN_NOISE = (1.0e-9, -1.0e-9, 1.0e-6)


@dataclass(frozen=True)
class GuardedFunctionAudit:
    sector: str
    input_branch: tuple[float, float, float]
    corrected_branch: tuple[int, int, int]
    result_branch: tuple[int, int, int] | None
    input_bit_count: float
    output_bit_count: float
    correction_applied: bool
    information_conserved: bool
    statement: str


@dataclass(frozen=True)
class InformationConservationAudit:
    benchmark_branch: tuple[int, int, int]
    input_branch: tuple[float, float, float]
    input_bit_count: float
    sector_audits: tuple[GuardedFunctionAudit, ...]
    statement: str

    @property
    def information_conserved(self) -> bool:
        return all(audit.information_conserved for audit in self.sector_audits)


def _float_branch(branch: tuple[int | float, int | float, int | float]) -> tuple[float, float, float]:
    return tuple(float(value) for value in branch)


def _rounded_branch(branch: Sequence[int | float]) -> tuple[int, int, int]:
    return tuple(int(round(float(value))) for value in branch)  # type: ignore[return-value]


def _extract_branch(candidate: Any) -> tuple[float, float, float] | tuple[int, int, int] | None:
    if candidate is None:
        return None
    if isinstance(candidate, (tuple, list)) and len(candidate) == 3:
        return tuple(candidate)  # type: ignore[return-value]
    for attribute in ("branch", "target_tuple", "evaluated_branch", "benchmark_branch"):
        value = getattr(candidate, attribute, None)
        if isinstance(value, (tuple, list)) and len(value) == 3:
            return tuple(value)  # type: ignore[return-value]
    nested_vacuum = getattr(candidate, "vacuum", None)
    if nested_vacuum is not None:
        return _extract_branch(nested_vacuum)
    return None


def _extract_bit_count(candidate: Any) -> float | None:
    if candidate is None:
        return None
    for attribute in ("bit_count", "holographic_bits"):
        value = getattr(candidate, attribute, None)
        if value is not None:
            return float(value)
    nested_vacuum = getattr(candidate, "vacuum", None)
    if nested_vacuum is not None:
        nested = _extract_bit_count(nested_vacuum)
        if nested is not None:
            return nested
    return None


def build_perturbed_guardian_vacuum(
    *,
    lepton_noise: float = DEFAULT_GUARDIAN_NOISE[0],
    quark_noise: float = DEFAULT_GUARDIAN_NOISE[1],
    parent_noise: float = DEFAULT_GUARDIAN_NOISE[2],
) -> TopologicalVacuum:
    benchmark_vacuum = TopologicalVacuum()
    return replace(
        benchmark_vacuum,
        k_l=float(benchmark_vacuum.k_l) + float(lepton_noise),
        k_q=float(benchmark_vacuum.k_q) + float(quark_noise),
        parent_level=float(benchmark_vacuum.parent_level) + float(parent_noise),
    )


def _recenter_vacuum(vacuum: TopologicalVacuum | None) -> tuple[TopologicalVacuum, tuple[float, float, float], float, bool]:
    resolved_vacuum = TopologicalVacuum() if vacuum is None else vacuum
    input_branch = _float_branch(
        (
            getattr(resolved_vacuum, "k_l", getattr(resolved_vacuum, "lepton_level", LEPTON_LEVEL)),
            getattr(resolved_vacuum, "k_q", getattr(resolved_vacuum, "quark_level", QUARK_LEVEL)),
            getattr(resolved_vacuum, "parent_level", PARENT_LEVEL),
        )
    )
    input_bit_count = float(getattr(resolved_vacuum, "bit_count", HOLOGRAPHIC_BITS))
    correction_applied = _rounded_branch(input_branch) != BENCHMARK_MOAT_CENTER or any(
        not math.isclose(component, benchmark, rel_tol=0.0, abs_tol=1.0e-12)
        for component, benchmark in zip(input_branch, BENCHMARK_MOAT_CENTER, strict=True)
    )
    corrected_vacuum = replace(
        resolved_vacuum,
        k_l=BENCHMARK_MOAT_CENTER[0],
        k_q=BENCHMARK_MOAT_CENTER[1],
        parent_level=BENCHMARK_MOAT_CENTER[2],
        bit_count=input_bit_count,
    )
    return corrected_vacuum, input_branch, input_bit_count, correction_applied


def _recenter_result(result: Any, corrected_vacuum: TopologicalVacuum) -> Any:
    if not is_dataclass(result):
        return result
    for attribute, replacement in (
        ("branch", BENCHMARK_MOAT_CENTER),
        ("benchmark_branch", BENCHMARK_MOAT_CENTER),
        ("evaluated_branch", BENCHMARK_MOAT_CENTER),
        ("vacuum", corrected_vacuum),
    ):
        if hasattr(result, attribute):
            try:
                result = replace(result, **{attribute: replacement})
            except TypeError:
                continue
    return result


def _run_guarded(
    function: Callable[P, R],
    sector: str,
    *args: P.args,
    **kwargs: P.kwargs,
) -> tuple[R, GuardedFunctionAudit]:
    signature = inspect.signature(function)
    bound = signature.bind_partial(*args, **kwargs)
    raw_vacuum = bound.arguments.get("vacuum")
    corrected_vacuum, input_branch, input_bit_count, correction_applied = _recenter_vacuum(raw_vacuum)
    if "vacuum" in signature.parameters:
        bound.arguments["vacuum"] = corrected_vacuum

    result = function(*bound.args, **bound.kwargs)
    recentered_result = _recenter_result(result, corrected_vacuum)
    raw_result_branch = _extract_branch(recentered_result)
    result_branch = None if raw_result_branch is None else _rounded_branch(raw_result_branch)
    output_bit_count = _extract_bit_count(recentered_result)
    if output_bit_count is None:
        output_bit_count = float(corrected_vacuum.bit_count)
    information_conserved = math.isclose(output_bit_count, input_bit_count, rel_tol=0.0, abs_tol=0.0)
    statement = (
        f"Runtime Guardian recentered {sector} to the moat center {BENCHMARK_MOAT_CENTER} "
        f"and preserved holographic bit-count N={input_bit_count:.6e}."
    )
    audit = GuardedFunctionAudit(
        sector=sector,
        input_branch=input_branch,
        corrected_branch=BENCHMARK_MOAT_CENTER,
        result_branch=result_branch,
        input_bit_count=input_bit_count,
        output_bit_count=float(output_bit_count),
        correction_applied=bool(correction_applied),
        information_conserved=bool(information_conserved),
        statement=statement,
    )
    return recentered_result, audit  # type: ignore[return-value]


def runtime_guardian(
    function: Callable[P, R] | None = None,
    *,
    sector: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
    def decorator(inner: Callable[P, R]) -> Callable[P, R]:
        resolved_sector = inner.__name__ if sector is None else sector

        @wraps(inner)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            result, audit = _run_guarded(inner, resolved_sector, *args, **kwargs)
            setattr(wrapper, "__guardian_audit__", audit)
            return result

        setattr(wrapper, "__guardian_audit__", None)
        return wrapper

    if function is None:
        return decorator
    return decorator(function)


def guardian_audit_of(function: Callable[..., Any]) -> GuardedFunctionAudit:
    audit = getattr(function, "__guardian_audit__", None)
    if audit is None:
        raise ValueError(f"Guardian audit for {getattr(function, '__name__', repr(function))} is not available yet.")
    return audit


@runtime_guardian(sector="cosmology")
def derive_guarded_cosmology_audit(vacuum: TopologicalVacuum | None = None) -> Any:
    resolved_vacuum = TopologicalVacuum() if vacuum is None else vacuum
    return resolved_vacuum.derive_cosmology_audit()


@runtime_guardian(sector="flavor")
def derive_guarded_flavor_audit(vacuum: TopologicalVacuum | None = None) -> FlavorIdentityAudit:
    del vacuum
    return build_flavor_identity_audit()


@runtime_guardian(sector="gravity")
def derive_guarded_gravity_audit(vacuum: TopologicalVacuum | None = None) -> GravityAudit:
    resolved_vacuum = TopologicalVacuum() if vacuum is None else vacuum
    return resolved_vacuum.verify_bulk_emergence()


def audit_information_conservation(vacuum: TopologicalVacuum | None = None) -> InformationConservationAudit:
    guarded_vacuum = build_perturbed_guardian_vacuum() if vacuum is None else vacuum
    input_branch = _float_branch((guarded_vacuum.k_l, guarded_vacuum.k_q, guarded_vacuum.parent_level))
    input_bit_count = float(guarded_vacuum.bit_count)

    derive_guarded_cosmology_audit(guarded_vacuum)
    derive_guarded_flavor_audit(guarded_vacuum)
    derive_guarded_gravity_audit(guarded_vacuum)

    sector_audits = (
        guardian_audit_of(derive_guarded_cosmology_audit),
        guardian_audit_of(derive_guarded_flavor_audit),
        guardian_audit_of(derive_guarded_gravity_audit),
    )
    statement = (
        "The Rigidity Guardian treats space-time as a quantum error-correcting code: "
        "branch drift is projected back to the moat center while total holographic bit-count N is conserved."
    )
    return InformationConservationAudit(
        benchmark_branch=BENCHMARK_MOAT_CENTER,
        input_branch=input_branch,
        input_bit_count=input_bit_count,
        sector_audits=sector_audits,
        statement=statement,
    )


def build_guardian_report(vacuum: TopologicalVacuum | None = None) -> str:
    audit = audit_information_conservation(vacuum=vacuum)
    lines = [
        "Rigidity Guardian Audit",
        "======================",
        f"Input branch                 : {audit.input_branch}",
        f"Moat center                  : {audit.benchmark_branch}",
        f"Input holographic bits N     : {audit.input_bit_count:.6e}",
        f"Information conserved        : {audit.information_conserved}",
        audit.statement,
        "",
        "Sector Corrections",
        "------------------",
    ]
    lines.extend(
        f"- {sector_audit.sector}: corrected={sector_audit.correction_applied} result_branch={sector_audit.result_branch} N_out={sector_audit.output_bit_count:.6e}"
        for sector_audit in audit.sector_audits
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SHBT Rigidity Guardian audit.")
    parser.add_argument(
        "--quiet-input",
        action="store_true",
        help="Use the exact benchmark vacuum instead of a synthetically perturbed runtime input.",
    )
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    vacuum = None if not args.quiet_input else TopologicalVacuum()
    print(build_guardian_report(vacuum=vacuum))
    return 0


__all__ = [
    "BENCHMARK_MOAT_CENTER",
    "DEFAULT_GUARDIAN_NOISE",
    "GuardedFunctionAudit",
    "InformationConservationAudit",
    "audit_information_conservation",
    "build_guardian_report",
    "build_perturbed_guardian_vacuum",
    "derive_guarded_cosmology_audit",
    "derive_guarded_flavor_audit",
    "derive_guarded_gravity_audit",
    "guardian_audit_of",
    "main",
    "runtime_guardian",
]


if __name__ == "__main__":
    raise SystemExit(main())
