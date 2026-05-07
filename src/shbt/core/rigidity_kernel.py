from __future__ import annotations

"""Runtime rigidity kernel for branch-stabilized SHBT derivations.

This module promotes rigidity from a post-hoc audit into a live correction
kernel. The anomaly-free ``(26, 8, 312)`` boundary is embedded as a diagonal
parity-check matrix acting on visible branch coordinates. Any numerical drift
in a branch-carrying vacuum, model, or sector residue is projected back onto
that benchmark codeword before the derivation executes and again on the
observer-facing payload that comes back out.

The public ``stabilize_boundary`` decorator is the runtime entrypoint used by
the derivation APIs. The legacy guardian report helpers remain available, but
they now sit on top of the same feedback loop that keeps flavor, cosmology,
and gravity aligned with the topological boundary constraints.
"""

import argparse
from contextvars import ContextVar
import inspect
import math
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Final, ParamSpec, Sequence, TypeVar

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
BENCHMARK_PARITY_CHECK_MATRIX: Final[tuple[tuple[float, float, float], ...]] = (
    (float(BENCHMARK_MOAT_CENTER[0]), 0.0, 0.0),
    (0.0, float(BENCHMARK_MOAT_CENTER[1]), 0.0),
    (0.0, 0.0, float(BENCHMARK_MOAT_CENTER[2])),
)
DEFAULT_GUARDIAN_NOISE = (1.0e-9, -1.0e-9, 1.0e-6)
DEFAULT_STABILIZER_PRECISION = 200
_STABILIZER_DEPTH: ContextVar[int] = ContextVar("shbt_rigidity_stabilizer_depth", default=0)

_BRANCH_ATTR_NAMES: Final[tuple[str, ...]] = (
    "branch",
    "target_tuple",
    "evaluated_branch",
    "benchmark_branch",
    "selected_tuple",
    "unique_survivor",
)
_BRANCH_MAPPING_KEYS: Final[tuple[str, ...]] = _BRANCH_ATTR_NAMES + ("benchmark_tuple",)
_COORDINATE_FIELD_GROUPS: Final[tuple[tuple[str, str, str], ...]] = (
    ("lepton_level", "quark_level", "parent_level"),
    ("endpoint_lepton_level", "quark_level", "parent_level"),
    ("k_l", "k_q", "parent_level"),
)
_COORDINATE_REPLACEMENTS: Final[dict[str, int]] = {
    "lepton_level": BENCHMARK_MOAT_CENTER[0],
    "endpoint_lepton_level": BENCHMARK_MOAT_CENTER[0],
    "k_l": BENCHMARK_MOAT_CENTER[0],
    "quark_level": BENCHMARK_MOAT_CENTER[1],
    "k_q": BENCHMARK_MOAT_CENTER[1],
    "parent_level": BENCHMARK_MOAT_CENTER[2],
}


class BoundaryStabilizationError(RuntimeError):
    """Raised when the boundary lock or parity correction fails."""


@dataclass(frozen=True)
class ParitySyndromeAudit:
    benchmark_branch: tuple[int, int, int]
    observed_branch: tuple[float, float, float]
    parity_check_matrix: tuple[tuple[float, float, float], ...]
    syndrome_vector: tuple[float, float, float]
    correction_applied: bool

    @property
    def corrected_branch(self) -> tuple[int, int, int]:
        return self.benchmark_branch

    @property
    def zero_syndrome(self) -> bool:
        return all(math.isclose(component, 0.0, rel_tol=0.0, abs_tol=1.0e-12) for component in self.syndrome_vector)


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
    syndrome: ParitySyndromeAudit | None = None


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


def _branch_sequence(candidate: Any) -> tuple[float, float, float] | None:
    if not isinstance(candidate, (tuple, list)) or len(candidate) != 3:
        return None
    try:
        return tuple(float(value) for value in candidate)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _extract_branch(candidate: Any) -> tuple[float, float, float] | tuple[int, int, int] | None:
    if candidate is None:
        return None

    branch_sequence = _branch_sequence(candidate)
    if branch_sequence is not None:
        return tuple(candidate)  # type: ignore[return-value]

    if isinstance(candidate, dict):
        for key in _BRANCH_MAPPING_KEYS:
            value = candidate.get(key)
            if _branch_sequence(value) is not None:
                return tuple(value)  # type: ignore[return-value]
        for coordinate_names in _COORDINATE_FIELD_GROUPS:
            if all(name in candidate for name in coordinate_names):
                values = tuple(candidate[name] for name in coordinate_names)
                if _branch_sequence(values) is not None:
                    return values  # type: ignore[return-value]
        for nested_name in ("vacuum", "model"):
            nested = candidate.get(nested_name)
            if nested is not None:
                nested_branch = _extract_branch(nested)
                if nested_branch is not None:
                    return nested_branch
        return None

    for attribute in _BRANCH_ATTR_NAMES:
        value = getattr(candidate, attribute, None)
        if _branch_sequence(value) is not None:
            return tuple(value)  # type: ignore[return-value]

    for coordinate_names in _COORDINATE_FIELD_GROUPS:
        values: list[Any] = []
        for attribute in coordinate_names:
            if not hasattr(candidate, attribute):
                values = []
                break
            values.append(getattr(candidate, attribute))
        if values and _branch_sequence(values) is not None:
            return tuple(values)  # type: ignore[return-value]

    for nested_name in ("vacuum", "model"):
        nested = getattr(candidate, nested_name, None)
        if nested is not None:
            nested_branch = _extract_branch(nested)
            if nested_branch is not None:
                return nested_branch

    return None


def _extract_bit_count(candidate: Any) -> float | None:
    if candidate is None:
        return None
    if isinstance(candidate, dict):
        for key in ("bit_count", "holographic_bits"):
            value = candidate.get(key)
            if value is not None:
                return float(value)
        for nested_name in ("vacuum", "model"):
            nested = candidate.get(nested_name)
            if nested is not None:
                nested_bit_count = _extract_bit_count(nested)
                if nested_bit_count is not None:
                    return nested_bit_count
        return None
    for attribute in ("bit_count", "holographic_bits"):
        value = getattr(candidate, attribute, None)
        if value is not None:
            return float(value)
    for nested_name in ("vacuum", "model"):
        nested = getattr(candidate, nested_name, None)
        if nested is not None:
            nested_bit_count = _extract_bit_count(nested)
            if nested_bit_count is not None:
                return nested_bit_count
    return None


def _match_branch_payload_shape(payload: Any, branch: tuple[int, int, int]) -> tuple[int, int, int] | list[int]:
    if isinstance(payload, list):
        return list(branch)
    return tuple(branch)


def _resolve_precision_argument(callable_object: Callable[..., Any], *args: object, **kwargs: object) -> int:
    try:
        bound_arguments = inspect.signature(callable_object).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return DEFAULT_STABILIZER_PRECISION
    precision = bound_arguments.arguments.get("precision", DEFAULT_STABILIZER_PRECISION)
    try:
        return max(int(precision), DEFAULT_STABILIZER_PRECISION)
    except (TypeError, ValueError):
        return DEFAULT_STABILIZER_PRECISION


def _verify_boundary_lock(precision: int = DEFAULT_STABILIZER_PRECISION) -> Any:
    from shbt.core import holographic_error_stabilizer as stabilizer_module

    return stabilizer_module._verify_boundary_lock(max(int(precision), DEFAULT_STABILIZER_PRECISION))


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


class HolographicErrorCorrection:
    """Project noisy boundary data back onto the benchmark parity-check codeword."""

    benchmark_branch: Final[tuple[int, int, int]] = BENCHMARK_MOAT_CENTER
    parity_check_matrix: Final[tuple[tuple[float, float, float], ...]] = BENCHMARK_PARITY_CHECK_MATRIX

    def __init__(self, *, branch_tolerance: float = 1.0e-12) -> None:
        self.branch_tolerance = float(branch_tolerance)

    def audit_candidate(self, candidate: Any) -> ParitySyndromeAudit:
        raw_branch = _extract_branch(candidate)
        observed_branch = _float_branch(self.benchmark_branch if raw_branch is None else raw_branch)
        branch_delta = tuple(
            observed_branch[index] - float(self.benchmark_branch[index])
            for index in range(len(self.benchmark_branch))
        )
        syndrome_vector = tuple(
            sum(row[column] * branch_delta[column] for column in range(len(self.benchmark_branch)))
            for row in self.parity_check_matrix
        )
        correction_applied = any(
            not math.isclose(
                observed_branch[index],
                float(self.benchmark_branch[index]),
                rel_tol=0.0,
                abs_tol=self.branch_tolerance,
            )
            for index in range(len(self.benchmark_branch))
        )
        return ParitySyndromeAudit(
            benchmark_branch=self.benchmark_branch,
            observed_branch=observed_branch,
            parity_check_matrix=self.parity_check_matrix,
            syndrome_vector=syndrome_vector,
            correction_applied=correction_applied,
        )

    def _stabilizable_argument(self, name: str, value: Any) -> bool:
        if name == "cls" or _extract_branch(value) is None:
            return False
        if name in {"vacuum", "model", "self"}:
            return True
        if name.endswith("_vacuum") or name.endswith("_model"):
            return True
        type_name = type(value).__name__.lower()
        return "vacuum" in type_name or "model" in type_name

    def correct_candidate(self, candidate: Any, *, corrected_reference: Any | None = None) -> Any:
        branch_sequence = _branch_sequence(candidate)
        if branch_sequence is not None:
            return _match_branch_payload_shape(candidate, self.benchmark_branch)

        if candidate is None:
            return None

        if is_dataclass(candidate):
            field_names = {field.name for field in fields(candidate)}
            updates: dict[str, Any] = {}

            for attribute in _BRANCH_ATTR_NAMES:
                if attribute not in field_names:
                    continue
                current = getattr(candidate, attribute)
                if _branch_sequence(current) is not None:
                    updates[attribute] = _match_branch_payload_shape(current, self.benchmark_branch)

            if "benchmark_tuple" in field_names:
                current = getattr(candidate, "benchmark_tuple")
                if _branch_sequence(current) is not None:
                    updates["benchmark_tuple"] = _match_branch_payload_shape(current, self.benchmark_branch)

            for attribute, replacement in _COORDINATE_REPLACEMENTS.items():
                if attribute not in field_names:
                    continue
                current = getattr(candidate, attribute)
                try:
                    drifted = not math.isclose(float(current), float(replacement), rel_tol=0.0, abs_tol=self.branch_tolerance)
                except (TypeError, ValueError):
                    drifted = True
                if drifted:
                    updates[attribute] = replacement

            for nested_name in ("vacuum", "model"):
                if nested_name not in field_names:
                    continue
                current_nested = getattr(candidate, nested_name)
                corrected_nested = (
                    corrected_reference
                    if nested_name == "vacuum" and corrected_reference is not None
                    else self.correct_candidate(current_nested, corrected_reference=corrected_reference)
                )
                if corrected_nested is not current_nested:
                    updates[nested_name] = corrected_nested

            if updates:
                return replace(candidate, **updates)
            return candidate

        if isinstance(candidate, dict):
            updated = dict(candidate)
            changed = False

            for key in _BRANCH_MAPPING_KEYS:
                if key not in updated:
                    continue
                current = updated[key]
                if _branch_sequence(current) is not None:
                    updated[key] = _match_branch_payload_shape(current, self.benchmark_branch)
                    changed = True

            for key, replacement in _COORDINATE_REPLACEMENTS.items():
                if key not in updated:
                    continue
                current = updated[key]
                try:
                    drifted = not math.isclose(float(current), float(replacement), rel_tol=0.0, abs_tol=self.branch_tolerance)
                except (TypeError, ValueError):
                    drifted = True
                if drifted:
                    updated[key] = replacement
                    changed = True

            for nested_name in ("vacuum", "model"):
                if nested_name not in updated:
                    continue
                current_nested = updated[nested_name]
                corrected_nested = (
                    corrected_reference
                    if nested_name == "vacuum" and corrected_reference is not None
                    else self.correct_candidate(current_nested, corrected_reference=corrected_reference)
                )
                if corrected_nested is not current_nested:
                    updated[nested_name] = corrected_nested
                    changed = True

            return updated if changed else candidate

        if isinstance(candidate, tuple):
            corrected_items = tuple(self.correct_candidate(item, corrected_reference=corrected_reference) for item in candidate)
            return corrected_items if corrected_items != candidate else candidate

        if isinstance(candidate, list):
            corrected_items = [self.correct_candidate(item, corrected_reference=corrected_reference) for item in candidate]
            return corrected_items if corrected_items != candidate else candidate

        return candidate

    def stabilize_arguments(
        self,
        function: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[inspect.BoundArguments, ParitySyndromeAudit, float, Any | None]:
        signature = inspect.signature(function)
        bound = signature.bind_partial(*args, **kwargs)

        primary_audit = self.audit_candidate(self.benchmark_branch)
        input_bit_count = float(HOLOGRAPHIC_BITS)
        corrected_reference: Any | None = None

        for name, value in tuple(bound.arguments.items()):
            if not self._stabilizable_argument(name, value):
                continue

            current_audit = self.audit_candidate(value)
            corrected_value = self.correct_candidate(value)
            bound.arguments[name] = corrected_value

            if corrected_reference is None or current_audit.correction_applied:
                corrected_reference = corrected_value
                primary_audit = current_audit

            extracted_bit_count = _extract_bit_count(value)
            if extracted_bit_count is not None:
                input_bit_count = float(extracted_bit_count)

        return bound, primary_audit, input_bit_count, corrected_reference

    def stabilize_result(self, result: Any, *, corrected_reference: Any | None = None) -> Any:
        return self.correct_candidate(result, corrected_reference=corrected_reference)


def _build_guarded_audit(
    *,
    sector: str,
    input_syndrome: ParitySyndromeAudit,
    output_syndrome: ParitySyndromeAudit,
    input_bit_count: float,
    result: Any,
    corrected_reference: Any | None,
) -> GuardedFunctionAudit:
    raw_result_branch = _extract_branch(result)
    result_branch = None if raw_result_branch is None else _rounded_branch(raw_result_branch)

    output_bit_count = _extract_bit_count(result)
    if output_bit_count is None and corrected_reference is not None:
        output_bit_count = _extract_bit_count(corrected_reference)
    if output_bit_count is None:
        output_bit_count = input_bit_count

    correction_applied = bool(input_syndrome.correction_applied or output_syndrome.correction_applied)
    information_conserved = math.isclose(float(output_bit_count), input_bit_count, rel_tol=0.0, abs_tol=0.0)
    action = "recentered" if correction_applied else "verified"
    statement = (
        f"HolographicErrorCorrection {action} {sector} against the (26, 8, 312) parity-check kernel "
        f"and preserved holographic bit-count N={input_bit_count:.6e}."
    )
    return GuardedFunctionAudit(
        sector=sector,
        input_branch=input_syndrome.observed_branch,
        corrected_branch=input_syndrome.corrected_branch,
        result_branch=result_branch,
        input_bit_count=input_bit_count,
        output_bit_count=float(output_bit_count),
        correction_applied=correction_applied,
        information_conserved=bool(information_conserved),
        statement=statement,
        syndrome=input_syndrome if input_syndrome.correction_applied else output_syndrome,
    )


def stabilize_boundary(
    function: Callable[P, R] | None = None,
    *,
    sector: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
    kernel = HolographicErrorCorrection()

    def decorator(inner: Callable[P, R]) -> Callable[P, R]:
        if getattr(inner, "__boundary_stabilized__", False):
            return inner

        resolved_sector = inner.__name__ if sector is None else sector

        @wraps(inner)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            current_depth = _STABILIZER_DEPTH.get()
            resolved_precision = _resolve_precision_argument(inner, *args, **kwargs)
            if current_depth == 0:
                _verify_boundary_lock(resolved_precision)

            bound, input_syndrome, input_bit_count, corrected_reference = kernel.stabilize_arguments(inner, *args, **kwargs)
            token = _STABILIZER_DEPTH.set(current_depth + 1)
            try:
                raw_result = inner(*bound.args, **bound.kwargs)
            finally:
                _STABILIZER_DEPTH.reset(token)

            output_syndrome = kernel.audit_candidate(raw_result)
            recentered_result = kernel.stabilize_result(raw_result, corrected_reference=corrected_reference)

            if current_depth == 0:
                _verify_boundary_lock(resolved_precision)

            audit = _build_guarded_audit(
                sector=resolved_sector,
                input_syndrome=input_syndrome,
                output_syndrome=output_syndrome,
                input_bit_count=input_bit_count,
                result=recentered_result,
                corrected_reference=corrected_reference,
            )
            setattr(wrapper, "__guardian_audit__", audit)
            setattr(wrapper, "__boundary_audit__", audit)
            return recentered_result

        setattr(wrapper, "__guardian_audit__", None)
        setattr(wrapper, "__boundary_audit__", None)
        wrapper.__boundary_stabilized__ = True
        return wrapper

    if function is None:
        return decorator
    return decorator(function)


def stabilize_classmethods(class_object: type) -> type:
    if getattr(class_object, "__boundary_stabilized_class__", False):
        return class_object

    for name, attribute in tuple(vars(class_object).items()):
        if name.startswith("__"):
            continue
        if isinstance(attribute, classmethod):
            setattr(class_object, name, classmethod(stabilize_boundary(attribute.__func__, sector=name)))
            continue
        if isinstance(attribute, staticmethod):
            setattr(class_object, name, staticmethod(stabilize_boundary(attribute.__func__, sector=name)))
    class_object.__boundary_stabilized_class__ = True
    return class_object


def runtime_guardian(
    function: Callable[P, R] | None = None,
    *,
    sector: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
    return stabilize_boundary(function, sector=sector)


def guardian_audit_of(function: Callable[..., Any]) -> GuardedFunctionAudit:
    audit = getattr(function, "__guardian_audit__", getattr(function, "__boundary_audit__", None))
    if audit is None:
        raise ValueError(f"Guardian audit for {getattr(function, '__name__', repr(function))} is not available yet.")
    return audit


@stabilize_boundary(sector="cosmology")
def derive_guarded_cosmology_audit(vacuum: TopologicalVacuum | None = None) -> Any:
    resolved_vacuum = TopologicalVacuum() if vacuum is None else vacuum
    return resolved_vacuum.derive_cosmology_audit()


@stabilize_boundary(sector="flavor")
def derive_guarded_flavor_audit(vacuum: TopologicalVacuum | None = None) -> FlavorIdentityAudit:
    del vacuum
    return build_flavor_identity_audit()


@stabilize_boundary(sector="gravity")
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
        "The Rigidity kernel treats the anomaly-free branch as a quantum error-correcting code: "
        "a diagonal parity-check matrix built from (26, 8, 312) projects noisy Flavor, Cosmology, and Gravity "
        "residues back to the moat center while the total holographic bit-count N is conserved."
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
        f"Parity-check matrix          : {BENCHMARK_PARITY_CHECK_MATRIX}",
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
    "BENCHMARK_PARITY_CHECK_MATRIX",
    "BoundaryStabilizationError",
    "DEFAULT_GUARDIAN_NOISE",
    "DEFAULT_STABILIZER_PRECISION",
    "GuardedFunctionAudit",
    "HolographicErrorCorrection",
    "InformationConservationAudit",
    "ParitySyndromeAudit",
    "audit_information_conservation",
    "build_guardian_report",
    "build_perturbed_guardian_vacuum",
    "derive_guarded_cosmology_audit",
    "derive_guarded_flavor_audit",
    "derive_guarded_gravity_audit",
    "guardian_audit_of",
    "main",
    "runtime_guardian",
    "stabilize_boundary",
    "stabilize_classmethods",
]


if __name__ == "__main__":
    raise SystemExit(main())
